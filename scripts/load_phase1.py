"""Load and inspect results/phase1_features.pt."""

import torch
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

phase1 = torch.load(RESULTS_DIR / "phase1_features.pt", map_location="cpu", weights_only=False)

print("Keys:", list(phase1.keys()))
print()

# top_features_A: {layer: tensor of feature indices}
top_A = phase1["top_features_A"]
print("=== top_features_A (Method A — monolinguality metric) ===")
for layer, feats in top_A.items():
    print(f"  layer {layer}: {feats.shape if hasattr(feats, 'shape') else len(feats)} features")

# top_features_B: {layer: tensor of feature indices}
top_B = phase1["top_features_B"]
print("\n=== top_features_B (Method B — supervised probe) ===")
for layer, feats in top_B.items():
    print(f"  layer {layer}: {feats.shape if hasattr(feats, 'shape') else len(feats)} features")

# intersection_features: {layer: {lang: [feat_indices]}}
intersection = phase1["intersection_features"]
print("\n=== intersection_features (A ∩ B) ===")
for layer, lang_dict in intersection.items():
    if isinstance(lang_dict, dict):
        for lang, feats in lang_dict.items():
            n = len(feats) if hasattr(feats, "__len__") else feats
            print(f"  layer {layer}, lang {lang}: {n} features")
    else:
        n = len(lang_dict) if hasattr(lang_dict, "__len__") else lang_dict
        print(f"  layer {layer}: {n} features")

# reasoning_features: {layer: ...}
reasoning = phase1["reasoning_features"]
print("\n=== reasoning_features (Method C — cross-lingual) ===")
for layer, feats in reasoning.items():
    n = feats.shape if hasattr(feats, "shape") else len(feats)
    print(f"  layer {layer}: {n} features")

# probe_accuracies
print("\n=== probe_accuracies ===")
probe_acc = phase1["probe_accuracies"]
if isinstance(probe_acc, dict):
    for layer, acc in probe_acc.items():
        print(f"  layer {layer}: {acc}")
else:
    print(f"  {probe_acc}")

# probe_importances
print("\n=== probe_importances ===")
probe_imp = phase1["probe_importances"]
for layer, imp in probe_imp.items():
    shape = imp.shape if hasattr(imp, "shape") else len(imp)
    print(f"  layer {layer}: {shape}")

# jaccard_AB
print("\n=== jaccard_AB ===")
jaccard = phase1["jaccard_AB"]
if isinstance(jaccard, dict):
    for k, v in jaccard.items():
        print(f"  {k}: {v}")
else:
    print(f"  {jaccard}")
