"""Audit Phase 3b (circuit attribution) and Phase 3c (attention entropy)
under three ablation variants: baseline_k20, confirmed (Phase 2b LANGUAGE only),
top_A_k20 (Deng-style).

Key fix vs first attempt: store per-prompt attention ENTROPY (shape (H,))
rather than raw attention probs (shape (H, T)) so we can stack across
varying-length prompts.
"""
import os, sys
sys.path.insert(0, '/workspace/nlp-project')
os.environ['PATH'] = os.environ['HOME'] + '/.local/bin:' + os.environ.get('PATH','')

from dotenv import load_dotenv; load_dotenv()
import torch, numpy as np, pandas as pd
from tqdm import tqdm
from scipy.stats import wilcoxon

from src.config import TARGET_LANGUAGES, SAE_SUBSET_LAYERS, SAE_WIDTH_16K, RESULTS_DIR, MODEL_ID
from src.data import load_mgsm
from src.model import load_saes_at_layers, get_decoder_layers
from src.intervention import directional_ablation, get_sae_decoder_directions
from src.extraction import encode_activations_through_sae

torch.manual_seed(0); np.random.seed(0)
PRIMARY_LAYER = 17
DOWNSTREAM = [22, 29]
ATTN_LAYERS = [17, 22, 29]
N_PROBS = 20

print('Loading model with eager attention + 4 SAEs...')
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map='auto',
    token=os.environ['HF_TOKEN'], attn_implementation='eager',
)
model.eval()
DECODER_LAYERS = get_decoder_layers(model)
saes = load_saes_at_layers(layers=SAE_SUBSET_LAYERS, width=SAE_WIDTH_16K, l0_target='medium')

phase1  = torch.load(RESULTS_DIR / 'phase1_features.pt', weights_only=False)
phase2b = torch.load(RESULTS_DIR / 'phase2_ablation.pt', weights_only=False)
intersection = phase1['intersection_features']
top_A = phase1['top_features_A']
confirmed_language = phase2b['confirmed_language']
best_k = 20

def select_baseline(lang):
    out = list(confirmed_language[lang])
    for f in intersection[PRIMARY_LAYER][lang]:
        if f not in out: out.append(f)
    for f in top_A[PRIMARY_LAYER][lang]:
        if f not in out: out.append(f)
    return out[:best_k]

VARIANTS = {
    'baseline_k20': {l: select_baseline(l) for l in TARGET_LANGUAGES},
    'confirmed':    {l: list(confirmed_language[l]) for l in TARGET_LANGUAGES},
    'top_A_k20':    {l: list(top_A[PRIMARY_LAYER][l][:best_k]) for l in TARGET_LANGUAGES},
}

def make_hook(directions, input_length):
    pos = input_length - 1
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            hs = output; tup = False
        else:
            hs = output[0]; tup = True
        if hs.dim() == 3 and hs.shape[1] > pos:
            hs[:, pos, :] = directional_ablation(hs[:, pos, :], directions)
        return (hs,) + output[1:] if tup else hs
    return hook

def attn_entropy_last_query(p_layer):
    """p_layer: (1, H, T, T). Return (H,) entropy at q=T-1."""
    p = p_layer[0, :, -1, :].clamp_min(1e-12)
    return -(p * p.log()).sum(dim=-1)

def forward_collect(prompt, ablation_dirs=None):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]
    handles = []
    if ablation_dirs is not None and len(ablation_dirs) > 0:
        handles.append(DECODER_LAYERS[PRIMARY_LAYER].register_forward_hook(
            make_hook(ablation_dirs.to(model.device), input_len)))
    try:
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, output_attentions=True, use_cache=False)
    finally:
        for h in handles: h.remove()
    resid = {L: out.hidden_states[L+1][:, -1, :].cpu().float() for L in DOWNSTREAM}
    # KEY FIX: compute entropy now (collapses T dimension), stash (H,) tensors.
    ent = {L: attn_entropy_last_query(out.attentions[L]).cpu().float() for L in ATTN_LAYERS}
    return resid, ent

mgsm = load_mgsm(TARGET_LANGUAGES)
def make_prompt(q):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)

print('\nClean pass...')
clean_data = {}
for lang in TARGET_LANGUAGES:
    rs = {L: [] for L in DOWNSTREAM}
    es = {L: [] for L in ATTN_LAYERS}
    for i in tqdm(range(N_PROBS), desc=f'clean {lang}', leave=False):
        prompt = make_prompt(mgsm[lang][i]['question'])
        r, e = forward_collect(prompt, ablation_dirs=None)
        for L in DOWNSTREAM:  rs[L].append(r[L])
        for L in ATTN_LAYERS: es[L].append(e[L])
    clean_data[lang] = {
        'resid': {L: torch.stack(rs[L]).squeeze(1) for L in DOWNSTREAM},
        'ent':   {L: torch.stack(es[L]) for L in ATTN_LAYERS},
    }

all_3b = {}
all_3c = []

for vname, vdict in VARIANTS.items():
    print(f'\n========= {vname} =========')
    all_3b[vname] = {}
    for lang in TARGET_LANGUAGES:
        feats = vdict[lang]
        if not feats:
            print(f'  {lang}: empty — skipped'); continue
        dirs = get_sae_decoder_directions(saes[PRIMARY_LAYER], feats).to(model.device)
        rs = {L: [] for L in DOWNSTREAM}
        es = {L: [] for L in ATTN_LAYERS}
        for i in tqdm(range(N_PROBS), desc=f'{vname} {lang}', leave=False):
            prompt = make_prompt(mgsm[lang][i]['question'])
            r, e = forward_collect(prompt, ablation_dirs=dirs)
            for L in DOWNSTREAM:  rs[L].append(r[L])
            for L in ATTN_LAYERS: es[L].append(e[L])
        ablt_resid = {L: torch.stack(rs[L]).squeeze(1) for L in DOWNSTREAM}
        ablt_ent   = {L: torch.stack(es[L]) for L in ATTN_LAYERS}

        # 3b: top-30 abs deltas at each downstream layer
        edges = []
        for L in DOWNSTREAM:
            sae_d = saes[L]
            f_clean = encode_activations_through_sae(clean_data[lang]['resid'][L], sae_d)
            f_ablt  = encode_activations_through_sae(ablt_resid[L], sae_d)
            mean_clean = f_clean.mean(dim=0)
            mean_ablt  = f_ablt.mean(dim=0)
            delta = mean_ablt - mean_clean
            top = torch.topk(delta.abs(), k=30)
            for j in range(30):
                idx = int(top.indices[j])
                edges.append({
                    'downstream_layer': L, 'feature': idx,
                    'mean_clean': float(mean_clean[idx]),
                    'mean_ablated': float(mean_ablt[idx]),
                    'delta': float(delta[idx]),
                    'abs_delta': float(top.values[j]),
                })
        all_3b[vname][lang] = pd.DataFrame(edges)

        # 3c: entropy delta per layer
        for L in ATTN_LAYERS:
            ec = clean_data[lang]['ent'][L]  # (N, H)
            ea = ablt_ent[L]
            d = (ea - ec).flatten().numpy()
            try: stat, p = wilcoxon(ea.flatten().numpy(), ec.flatten().numpy())
            except ValueError: stat, p = 0.0, 1.0
            all_3c.append({
                'variant': vname, 'lang': lang, 'layer': L,
                'mean_dent': float(d.mean()), 'p_value': float(p),
            })
        df = all_3b[vname][lang]
        head_l22 = df[df.downstream_layer == 22].head(3)
        head_l29 = df[df.downstream_layer == 29].head(3)
        print(f'  {lang}: top L22 = {[(int(r.feature), round(r.delta,1)) for _,r in head_l22.iterrows()]}, '
              f'top L29 = {[(int(r.feature), round(r.delta,1)) for _,r in head_l29.iterrows()]}')

out = {
    '3b_edges': {v: {l: df.to_dict('records') for l, df in d.items()} for v, d in all_3b.items()},
    '3c_summary': all_3c,
    'variant_inputs': VARIANTS,
    'config': {'n_probs': N_PROBS, 'primary_layer': PRIMARY_LAYER,
               'downstream': DOWNSTREAM, 'attn_layers': ATTN_LAYERS},
}
torch.save(out, RESULTS_DIR / 'phase3bc_controls.pt')
print(f'\nSaved {RESULTS_DIR}/phase3bc_controls.pt')

print('\n=== 3c entropy delta heatmap (mean_dent, last query) ===')
df3c = pd.DataFrame(all_3c)
piv = df3c.pivot_table(index=['variant','layer'], columns='lang', values='mean_dent')
print(piv.round(4).to_string())

print('\n=== 3c significance count ===')
for v in df3c['variant'].unique():
    for L in ATTN_LAYERS:
        cell = df3c[(df3c.variant==v) & (df3c.layer==L)]
        n_sig = (cell['p_value'] < 0.05).sum()
        print(f'  {v:14s} L{L}: {n_sig}/{len(cell)} langs significant (p<0.05)')

print('\n=== 3b top-30 feature overlap: baseline_k20 vs confirmed (per layer, lang) ===')
for lang in TARGET_LANGUAGES:
    if lang not in all_3b['confirmed']:
        print(f'  {lang}: confirmed empty — skipped'); continue
    for L in DOWNSTREAM:
        b_feats = set(all_3b['baseline_k20'][lang][all_3b['baseline_k20'][lang].downstream_layer==L]['feature'])
        c_feats = set(all_3b['confirmed'   ][lang][all_3b['confirmed'   ][lang].downstream_layer==L]['feature'])
        ovl = len(b_feats & c_feats)
        print(f'  {lang} L{L}: |baseline∩confirmed| = {ovl}/30')

print('\n=== 3b: are top deltas dominated by feature 96 / aligned features? ===')
SUSPECT_L17_FEATS_TO_CHECK = [96, 34, 234, 9983, 1135]  # known f=96-aligned features per Phase 3a alignment
for vname in ['baseline_k20', 'confirmed', 'top_A_k20']:
    for lang in TARGET_LANGUAGES:
        if lang not in all_3b[vname]: continue
        df = all_3b[vname][lang]
        # Note: these features live in L22/L29 SAEs, not L17. So this isn't a
        # direct alignment check — just whether the same indices appear.
