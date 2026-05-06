"""Run two more capacity controls to settle whether the f=96 release
in confirmed-en/es/zh is alignment-mediated or a real effect.

NEW VARIANTS:
  - f96_clean: confirmed-LANGUAGE only, with any feature whose decoder direction
    has |cosine| > 0.1 with W_enc[96] OR W_dec[96] excluded.
  - max_aligned_k20: top-20 features most aligned with W_enc[96].

Reuses the clean residuals collected in 05b (cached on disk via partial save).
"""
import os, sys, gc, time
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
N_DEV = 50

print('Loading model + SAE...')
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ['HF_TOKEN'])
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16, device_map='auto',
    token=os.environ['HF_TOKEN'],
)
model.eval()
DECODER_LAYERS = get_decoder_layers(model)

saes = load_saes_at_layers(layers=SAE_SUBSET_LAYERS, width=SAE_WIDTH_16K, l0_target='medium')
sae17 = saes[17]

# Load existing artifacts
phase1  = torch.load(RESULTS_DIR / 'phase1_features.pt', weights_only=False)
phase2b = torch.load(RESULTS_DIR / 'phase2_ablation.pt', weights_only=False)
ctrl    = torch.load(RESULTS_DIR / 'phase3a_controls.pt', weights_only=False)

intersection = phase1['intersection_features']
top_A        = phase1['top_features_A']
reasoning_features_per_layer = phase1['reasoning_features']
confirmed_language = phase2b['confirmed_language']
best_k = 20

REASONING_FEATS_L17 = reasoning_features_per_layer[17] or list(range(50))
print(f'reasoning candidates: {len(REASONING_FEATS_L17)}')

# Build f=96-aligned exclusion set
W_enc = sae17.W_enc.data.float()
W_dec = sae17.W_dec.data.float()
enc96 = W_enc[:, 96]; enc96_unit = enc96 / enc96.norm()
dec96 = W_dec[96];    dec96_unit = dec96 / dec96.norm()
W_dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True)
cos_enc = (W_dec_norm @ enc96_unit).abs()
cos_dec = (W_dec_norm @ dec96_unit).abs()
EXCLUDE = set((cos_enc > 0.1).nonzero().squeeze(-1).tolist()) | set((cos_dec > 0.1).nonzero().squeeze(-1).tolist())
print(f'features excluded (|cos|>0.1 with enc96 or dec96): {len(EXCLUDE)}')

# Build new variants
def select_lang_clean(lang, k=best_k, layer=PRIMARY_LAYER):
    out = list(confirmed_language[lang])
    for f in intersection[layer][lang]:
        if f not in out: out.append(f)
    for f in top_A[layer][lang]:
        if f not in out: out.append(f)
    return [f for f in out if f not in EXCLUDE][:k]

f96_clean_set = {l: select_lang_clean(l) for l in TARGET_LANGUAGES}
print('\nf96_clean variant:')
for l, fs in f96_clean_set.items():
    print(f'  {l}: k={len(fs)}, feats={fs}')

# max_aligned_k20: same set across all langs (since it's about f=96 directly)
top20_aligned = torch.topk(cos_enc, k=20).indices.tolist()
max_aligned_set = {l: list(top20_aligned) for l in TARGET_LANGUAGES}
print(f'\nmax_aligned_k20 (same across langs): {top20_aligned}')

NEW_VARIANTS = {'f96_clean': f96_clean_set, 'max_aligned_k20': max_aligned_set}

# Forward helper
def make_ablation_hook(directions, input_length):
    pos = input_length - 1
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            hs = output; is_tuple = False
        else:
            hs = output[0]; is_tuple = True
        if hs.dim() == 3 and hs.shape[1] > pos:
            hs[:, pos, :] = directional_ablation(hs[:, pos, :], directions)
        if is_tuple: return (hs,) + output[1:]
        return hs
    return hook

def forward_l17(prompt, ablation_dirs=None):
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_len = inputs['input_ids'].shape[1]
    handles = []
    if ablation_dirs is not None and len(ablation_dirs) > 0:
        handles.append(DECODER_LAYERS[PRIMARY_LAYER].register_forward_hook(
            make_ablation_hook(ablation_dirs.to(model.device), input_len)))
    try:
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=False)
    finally:
        for h in handles: h.remove()
    return out.hidden_states[PRIMARY_LAYER + 1][:, -1, :].cpu().float()

mgsm = load_mgsm(TARGET_LANGUAGES)
def make_prompt(q):
    return tokenizer.apply_chat_template(
        [{'role': 'user', 'content': q}], tokenize=False, add_generation_prompt=True)

# Step A: clean residuals (one pass per lang)
print('\nClean pass...')
clean_l17 = {}
for lang in TARGET_LANGUAGES:
    rs = []
    for i in tqdm(range(N_DEV), desc=f'clean {lang}', leave=False):
        rs.append(forward_l17(make_prompt(mgsm[lang][i]['question'])))
    clean_l17[lang] = torch.stack(rs).squeeze(1)

# Step B: per-variant ablated + summary
all_results = {}
for vname, vdict in NEW_VARIANTS.items():
    print(f'\n========= {vname} =========')
    vsum = {}
    for lang in TARGET_LANGUAGES:
        feats = vdict[lang]
        if len(feats) == 0:
            print(f'  {lang}: empty — skipped'); continue
        dirs = get_sae_decoder_directions(sae17, feats).to(model.device)
        ablt_l17 = []
        for i in tqdm(range(N_DEV), desc=f'{vname} {lang}', leave=False):
            ablt_l17.append(forward_l17(make_prompt(mgsm[lang][i]['question']), ablation_dirs=dirs))
        ablt_l17 = torch.stack(ablt_l17).squeeze(1)
        clean_enc = encode_activations_through_sae(clean_l17[lang], sae17)
        ablt_enc  = encode_activations_through_sae(ablt_l17, sae17)
        clean_r = clean_enc[:, REASONING_FEATS_L17].numpy()
        ablt_r  = ablt_enc[:, REASONING_FEATS_L17].numpy()
        rows = []
        for j, fidx in enumerate(REASONING_FEATS_L17):
            cf, af = clean_r[:, j], ablt_r[:, j]
            diff = af - cf
            try: stat, p = wilcoxon(af, cf) if not np.allclose(diff, 0) else (0, 1.0)
            except ValueError: stat, p = 0, 1.0
            rows.append({'feature': int(fidx),
                         'mean_clean': float(cf.mean()), 'mean_ablated': float(af.mean()),
                         'mean_delta': float(diff.mean()), 'p_value': float(p)})
        df = pd.DataFrame(rows)
        f96 = df[df.feature == 96]
        f96_str = ''
        if not f96.empty:
            r = f96.iloc[0]
            f96_str = f' | f=96: clean={r.mean_clean:.3f} ablated={r.mean_ablated:.3f} Δ={r.mean_delta:+.3f}'
        n_up   = int((df['mean_delta'] > 0).sum())
        n_down = int((df['mean_delta'] < 0).sum())
        n_sig  = int((df['p_value'] < 0.05).sum())
        print(f'  {lang}: k={len(feats)} {n_up}↑ {n_down}↓ {n_sig} sig{f96_str}')
        vsum[lang] = df
    all_results[vname] = vsum

# Save merged
merged = dict(ctrl)
merged_variants = dict(merged.get('variants', {}))
merged_inputs   = dict(merged.get('variant_inputs', {}))
for vname, vsum in all_results.items():
    merged_variants[vname] = {l: df.to_dict('records') for l, df in vsum.items()}
    merged_inputs[vname]   = NEW_VARIANTS[vname]

# Rebuild summary
rows = []
for vname, vsum in merged_variants.items():
    for lang, recs in vsum.items():
        df = pd.DataFrame(recs)
        rows.append({
            'variant': vname, 'lang': lang,
            'k': len(merged_inputs[vname][lang]),
            'n_up': int((df['mean_delta'] > 0).sum()),
            'n_down': int((df['mean_delta'] < 0).sum()),
            'n_sig': int((df['p_value'] < 0.05).sum()),
            'mean_delta_avg': float(df['mean_delta'].mean()),
        })
        f96 = df[df.feature == 96]
        if not f96.empty:
            rows[-1].update({
                'f96_clean': float(f96.iloc[0].mean_clean),
                'f96_ablated': float(f96.iloc[0].mean_ablated),
                'f96_delta': float(f96.iloc[0].mean_delta),
                'f96_p': float(f96.iloc[0].p_value),
            })
summary_df = pd.DataFrame(rows)
merged['variants'] = merged_variants
merged['variant_inputs'] = merged_inputs
merged['summary_table'] = summary_df.to_dict('records')

torch.save(merged, RESULTS_DIR / 'phase3a_controls.pt')
summary_df.to_csv(RESULTS_DIR / 'phase3a_controls_summary.csv', index=False)
print(f'\nMerged saved.')

print('\n=== ALL variants — f=96 release ===')
piv = summary_df.pivot(index='variant', columns='lang', values='f96_delta')
print(piv.round(1).to_string())
