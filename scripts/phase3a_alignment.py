"""
Subspace alignment analysis: explains the per-(variant, lang) f=96 release
in terms of how much each ablation subspace overlaps with f=96's
encoder/decoder directions.

Hypothesis: f=96 release magnitude ∝ alignment between the ablation subspace
and f=96's encoder direction W_enc[96].
"""
import os, sys
sys.path.insert(0, '/workspace/nlp-project')
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from src.config import RESULTS_DIR, SAE_SUBSET_LAYERS, SAE_WIDTH_16K, TARGET_LANGUAGES
from src.model import load_saes_at_layers

print('Loading L17 SAE...')
saes = load_saes_at_layers(layers=[17], width=SAE_WIDTH_16K, l0_target='medium')
sae17 = saes[17]
W_enc = sae17.W_enc.data.float()    # (d_model, d_sae)
W_dec = sae17.W_dec.data.float()    # (d_sae, d_model)
print(f'W_enc {W_enc.shape}, W_dec {W_dec.shape}')

# f=96's encoder direction (the row/column SAE uses to read out f=96)
enc96 = W_enc[:, 96]            # (d_model,)
dec96 = W_dec[96]               # (d_model,)
print(f'enc96 norm = {enc96.norm().item():.4f}, dec96 norm = {dec96.norm().item():.4f}')

# Bias: SAEs typically have an encoder bias too
b_enc96 = sae17.b_enc.data.float()[96].item()
print(f'encoder bias for f=96 = {b_enc96:.4f}')
print(f'enc96 cosine with dec96 = {torch.dot(enc96, dec96).item() / (enc96.norm() * dec96.norm()).item():.4f}')

# Load variants
ctrl = torch.load(RESULTS_DIR / 'phase3a_controls.pt', weights_only=False)
variant_inputs = ctrl['variant_inputs']
summary_rows = ctrl['summary_table']
print('\nVariants:', list(variant_inputs.keys()))

def alignment_metrics(feat_indices):
    """For an ablation set, compute:
    - rank of subspace (≤k)
    - ||Q^T enc96|| / ||enc96||  (fraction of enc96's mass in ablation subspace)
    - ||Q^T dec96|| / ||dec96||
    - max single-feature cosine with enc96 (which feature is most aligned)
    """
    if not feat_indices:
        return {'rank': 0, 'enc96_alignment': 0.0, 'dec96_alignment': 0.0, 'max_cos_enc96': 0.0, 'best_feat': None}
    D = W_dec[feat_indices].T  # (d_model, k) - decoder COLUMNS, the directions we project out
    Q, _ = torch.linalg.qr(D, mode='reduced')
    rank = Q.shape[1]
    enc96_proj = (Q.T @ enc96).norm() / enc96.norm()
    dec96_proj = (Q.T @ dec96).norm() / dec96.norm()
    # Per-feature cosine with enc96
    cos = torch.einsum('df,d->f', D / D.norm(dim=0, keepdim=True), enc96 / enc96.norm())
    max_cos = cos.abs().max().item()
    best_feat = feat_indices[cos.abs().argmax().item()]
    return {
        'rank': rank,
        'enc96_alignment': enc96_proj.item(),
        'dec96_alignment': dec96_proj.item(),
        'max_cos_enc96': max_cos,
        'best_feat': int(best_feat),
    }

# Compute alignment for every (variant, lang)
rows = []
for vname, vdict in variant_inputs.items():
    for lang in TARGET_LANGUAGES:
        feats = vdict[lang]
        m = alignment_metrics(feats)
        # Find matching f96_delta from summary
        match = [r for r in summary_rows if r['variant'] == vname and r['lang'] == lang]
        f96_delta = match[0].get('f96_delta', float('nan')) if match else float('nan')
        rows.append({
            'variant': vname, 'lang': lang, 'k': len(feats),
            **m, 'f96_delta': f96_delta,
        })
df = pd.DataFrame(rows)
df = df.sort_values(['lang', 'variant'])
print('\n=== Alignment vs f96 release ===')
print(df.to_string(index=False))

# Correlation across all (variant, lang) pairs that have both data
valid = df.dropna(subset=['f96_delta', 'enc96_alignment'])
valid = valid[valid['rank'] > 0]
if len(valid) > 0:
    r_enc = valid['enc96_alignment'].corr(valid['f96_delta'])
    r_dec = valid['dec96_alignment'].corr(valid['f96_delta'])
    r_maxcos = valid['max_cos_enc96'].corr(valid['f96_delta'])
    print(f'\nCorrelations (n={len(valid)}):')
    print(f'  enc96_alignment ↔ f96_delta: r = {r_enc:+.3f}')
    print(f'  dec96_alignment ↔ f96_delta: r = {r_dec:+.3f}')
    print(f'  max_cos_enc96   ↔ f96_delta: r = {r_maxcos:+.3f}')

# Now build a "max-aligned random" control: pick 20 features whose decoder
# directions have the highest cosine with W_enc[96]. Predict: this releases
# f=96 more than baseline_k20. We don't run the ablation here (would need
# the full notebook), just identify the set so we can check whether we
# already happen to ablate them in the existing variants.
print('\n=== Top-20 features whose dec direction is most aligned with W_enc[96] ===')
W_dec_norm = W_dec / W_dec.norm(dim=1, keepdim=True)
enc96_unit = enc96 / enc96.norm()
cos_all = W_dec_norm @ enc96_unit
top20 = torch.topk(cos_all.abs(), k=20)
top_aligned = [int(i) for i in top20.indices.tolist()]
print('Indices:', top_aligned)
print('Cosines:', [f'{v:+.3f}' for v in top20.values.tolist()])
# Cross-check: do any of these appear in our variants?
print('\nOverlap of these top-20-aligned features with existing ablation sets:')
for vname, vdict in variant_inputs.items():
    for lang in TARGET_LANGUAGES:
        ovl = set(vdict[lang]) & set(top_aligned)
        if ovl:
            print(f'  {vname:14s} {lang}: {sorted(ovl)}')

# Also check W_dec[96] cosine alignment (different question: directions that
# write to f=96, when ablated, would change f=96's input via gradient)
print('\n=== Top-20 features whose dec direction is most aligned with W_dec[96] ===')
dec96_unit = dec96 / dec96.norm()
cos_dec = W_dec_norm @ dec96_unit
top20d = torch.topk(cos_dec.abs(), k=20)
top_aligned_dec = [int(i) for i in top20d.indices.tolist()]
print('Indices:', top_aligned_dec)
print('Cosines:', [f'{v:+.3f}' for v in top20d.values.tolist()])
for vname, vdict in variant_inputs.items():
    for lang in TARGET_LANGUAGES:
        ovl = set(vdict[lang]) & set(top_aligned_dec)
        if ovl:
            print(f'  {vname:14s} {lang}: {sorted(ovl)}')

# Save
out = {
    'alignment_table': df.to_dict('records'),
    'top20_aligned_to_enc96': top_aligned,
    'top20_aligned_to_dec96': top_aligned_dec,
    'enc96': enc96, 'dec96': dec96, 'b_enc96': b_enc96,
}
torch.save(out, RESULTS_DIR / 'phase3a_alignment.pt')
print(f'\nSaved {RESULTS_DIR}/phase3a_alignment.pt')
