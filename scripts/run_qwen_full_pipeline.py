"""Full Qwen2.5-7B-Instruct pipeline: Phases 1, 2a, 2b, 3.

Replicates the Gemma 3 4B IT analysis (notebooks 02–05) on Qwen2.5-7B-Instruct
with BatchTopK SAEs. Designed for an H100 80GB GPU with batched generation
throughout.

Usage:
    python scripts/run_qwen_full_pipeline.py
    python scripts/run_qwen_full_pipeline.py --skip-phase 1      # resume from phase 2a
    python scripts/run_qwen_full_pipeline.py --resume-from 2b    # skip 1+2a, start at 2b
    python scripts/run_qwen_full_pipeline.py --batch-size 4      # reduce if OOM
"""

import argparse
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path and .env is loaded
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from src.config import (
    QWEN_D_MODEL,
    QWEN_MODEL_ID,
    QWEN_N_LAYERS,
    QWEN_SAE_SUBSET_LAYERS,
    QWEN_SAE_TRAINER,
    QWEN_SAE_WIDTH,
    QWEN_ZHAO_HIGHER_LAYERS,
    QWEN_ZHAO_MIDDLE_LAYERS,
    RESULTS_DIR,
    TARGET_LANGUAGES,
)
from src.data import compute_accuracy, load_mgsm, parse_answer_number
from src.extraction import encode_activations_batchtopk, extract_residual_activations
from src.intervention import (
    directional_ablation,
    get_sae_decoder_directions,
    run_generate_with_hooks_batched,
)
from src.monolinguality import (
    compute_monolinguality,
    identify_language_features,
    identify_reasoning_features,
    probe_language_features,
    train_language_probe,
)
from src.svd_baseline import compute_language_subspace, generate_with_svd_batched

torch.manual_seed(0)
np.random.seed(0)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRIMARY_LAYER = 11          # ~40% depth in 28-layer Qwen, analogous to L17 in Gemma
DOWNSTREAM_LAYERS = [19, 27]
TOP_K = 50                  # per-language feature selection
TOP_PER_LANG = 5            # candidates for causal labeling
N_LABEL_DEV = 25            # causal labeling accuracy dev split
N_DEV = 50                  # per-language dev split
N_TEST = 250                # full MGSM per language
K_VALUES = [1, 5, 10, 20]
MAX_NEW_TOKENS = 384
N_ATTENTION = 30            # problems for attention entropy (batch_size=1)
N_CIRCUIT = 10              # problems for circuit attribution

ZHAO_GRID = [
    (0.2, -0.2, 3),
    (0.1, -0.2, 3),
    (0.3, -0.2, 3),
    (0.2,  0.0, 3),
]
RANKS = [2, 3, 4]

DOCS_DIR = REPO_ROOT / "docs"
FINDINGS_PATH = DOCS_DIR / "findings.md"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
_last_commit_time = time.time()


def auto_commit(message: str, force: bool = False):
    """Commit results + findings and push. Also commits if 30+ min since last."""
    global _last_commit_time
    elapsed = time.time() - _last_commit_time
    if not force and elapsed < 1800:
        return
    try:
        import glob as glob_mod
        pt_files = glob_mod.glob(str(RESULTS_DIR / "qwen_*.pt"))
        files_to_add = pt_files + [str(FINDINGS_PATH)]
        files_to_add = [f for f in files_to_add if Path(f).exists()]
        if not files_to_add:
            return
        subprocess.run(
            ["git", "add"] + files_to_add,
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=REPO_ROOT, check=False, capture_output=True,
        )
        subprocess.run(
            ["git", "push"],
            cwd=REPO_ROOT, check=False, capture_output=True,
        )
        _last_commit_time = time.time()
        print(f"[auto-commit] {message}")
    except Exception as e:
        print(f"[auto-commit] failed: {e}")


def append_findings(text: str):
    """Append text to docs/findings.md."""
    DOCS_DIR.mkdir(exist_ok=True)
    with open(FINDINGS_PATH, "a") as f:
        f.write("\n" + text + "\n")


def make_prompt(tokenizer, question: str) -> str:
    """Build chat-templated prompt for Qwen."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def evaluate_outputs(outputs: list[str], gold_answers: list[float]) -> dict:
    """Parse model outputs and compute accuracy."""
    predictions = [parse_answer_number(o) for o in outputs]
    correct = [
        pred is not None and abs(pred - gold) < 1e-6
        for pred, gold in zip(predictions, gold_answers)
    ]
    return {
        "accuracy": compute_accuracy(predictions, gold_answers),
        "predictions": predictions,
        "outputs": outputs,
        "correct": correct,
    }


def batched_baseline_eval(
    model, tokenizer, prompts: list[str], batch_size: int = 8, device: str = "cuda",
) -> list[str]:
    """Batched generation with no hooks (baseline eval)."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Baseline gen"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        input_lens = inputs["attention_mask"].sum(dim=1)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            )
        for j, gen in enumerate(gen_ids):
            il = input_lens[j].item()
            all_outputs.append(tokenizer.decode(gen[il:], skip_special_tokens=True))
    return all_outputs


def avg_acc(per_lang_results: dict) -> float:
    return float(np.mean([r["accuracy"] for r in per_lang_results.values()]))


def bootstrap_ci(correct_list, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    arr = np.asarray(correct_list, dtype=float)
    boots = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(n_boot)]
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def select_top_features(lang, k, confirmed_language, intersection, top_A, layer=PRIMARY_LAYER):
    """Prefer causally-confirmed LANGUAGE features, then A∩B, then top-A."""
    out = list(confirmed_language.get(lang, []))
    for f in intersection[layer].get(lang, []):
        if f not in out:
            out.append(f)
    for f in top_A[layer].get(lang, []):
        if f not in out:
            out.append(f)
    return out[:k]


# ---------------------------------------------------------------------------
# Phase 1 — Feature Identification
# ---------------------------------------------------------------------------
def run_phase1(model, tokenizer, saes, mgsm, batch_size=16):
    ckpt = RESULTS_DIR / "qwen_phase1_features.pt"
    if ckpt.exists():
        print(f"[Phase 1] Loading checkpoint from {ckpt}")
        return torch.load(ckpt, weights_only=False)

    print("\n" + "=" * 60)
    print("PHASE 1 — Feature Identification")
    print("=" * 60)

    # Build prompts
    prompts_by_lang = {
        lang: [make_prompt(tokenizer, ex["question"]) for ex in mgsm[lang]]
        for lang in TARGET_LANGUAGES
    }

    # 1. Extract residual activations
    activations = {layer: {} for layer in QWEN_SAE_SUBSET_LAYERS}
    for lang in TARGET_LANGUAGES:
        print(f"\n=== {lang}: extracting activations ===")
        acts = extract_residual_activations(
            model, tokenizer, prompts_by_lang[lang],
            layers=QWEN_SAE_SUBSET_LAYERS,
            batch_size=batch_size,
            positions="last",
        )
        for layer in QWEN_SAE_SUBSET_LAYERS:
            activations[layer][lang] = acts[layer]
            print(f"  layer {layer:>2}: shape={tuple(acts[layer].shape)}")
        torch.cuda.empty_cache()

    # 2. Encode through BatchTopK SAEs
    feature_acts = {layer: {} for layer in QWEN_SAE_SUBSET_LAYERS}
    for layer in QWEN_SAE_SUBSET_LAYERS:
        ae = saes[layer]
        for lang in TARGET_LANGUAGES:
            feats = encode_activations_batchtopk(activations[layer][lang], ae, batch_size=64)
            feature_acts[layer][lang] = feats
        sample = feature_acts[layer][TARGET_LANGUAGES[0]]
        n_active = (sample > 0).float().sum(dim=-1).mean().item()
        print(f"  layer {layer:>2}: shape {tuple(sample.shape)}, mean active={n_active:.1f}")
    torch.cuda.empty_cache()

    # 3. Method A — monolinguality
    monolinguality_per_layer = {}
    top_features_A = {layer: {} for layer in QWEN_SAE_SUBSET_LAYERS}
    for layer in QWEN_SAE_SUBSET_LAYERS:
        mono = compute_monolinguality(feature_acts[layer])
        monolinguality_per_layer[layer] = mono
        top_features_A[layer] = identify_language_features(mono, top_k=TOP_K)
        print(f"  layer {layer:>2}:")
        for lang in TARGET_LANGUAGES:
            top1 = top_features_A[layer][lang][0]
            score = mono[lang][top1].item()
            print(f"    {lang}: top1 f={top1}, ν={score:.4f}")

    # 4. Method B — supervised probe
    probe_accuracies = {}
    top_features_B = {layer: {} for layer in QWEN_SAE_SUBSET_LAYERS}
    for layer in QWEN_SAE_SUBSET_LAYERS:
        clf, importances = train_language_probe(feature_acts[layer], max_iter=2000)
        top_features_B[layer] = probe_language_features(
            clf, importances, sorted(TARGET_LANGUAGES), top_k=TOP_K,
        )
        X = np.concatenate(
            [feature_acts[layer][l].float().numpy() for l in sorted(TARGET_LANGUAGES)]
        )
        y = np.concatenate([np.full(250, i) for i in range(len(TARGET_LANGUAGES))])
        acc = clf.score(X, y)
        probe_accuracies[layer] = acc
        print(f"  layer {layer:>2}: probe accuracy = {acc:.3f}")

    # 5. Method C — reasoning features
    reasoning_features = {}
    for layer in QWEN_SAE_SUBSET_LAYERS:
        reason = identify_reasoning_features(feature_acts[layer], threshold=0.1)
        reasoning_features[layer] = reason
        print(f"  layer {layer:>2}: {len(reason)} reasoning candidates")

    # 6. A∩B intersection
    intersection_features = {
        layer: {
            lang: sorted(set(top_features_A[layer][lang]) & set(top_features_B[layer][lang]))
            for lang in TARGET_LANGUAGES
        }
        for layer in QWEN_SAE_SUBSET_LAYERS
    }

    # Print intersection sizes
    print("\nIntersection sizes (A∩B):")
    for layer in QWEN_SAE_SUBSET_LAYERS:
        sizes = {l: len(intersection_features[layer][l]) for l in TARGET_LANGUAGES}
        print(f"  layer {layer:>2}: {sizes}")

    # Jaccard
    print("\nJaccard(A, B):")
    for layer in QWEN_SAE_SUBSET_LAYERS:
        jaccards = {l: jaccard(top_features_A[layer][l], top_features_B[layer][l])
                    for l in TARGET_LANGUAGES}
        print(f"  layer {layer:>2}: {jaccards}")

    # Save
    payload = {
        "config": {
            "model_id": QWEN_MODEL_ID,
            "layers": QWEN_SAE_SUBSET_LAYERS,
            "sae_width": QWEN_SAE_WIDTH,
            "trainer": QWEN_SAE_TRAINER,
            "languages": TARGET_LANGUAGES,
            "top_k": TOP_K,
        },
        "monolinguality": {
            layer: {lang: monolinguality_per_layer[layer][lang] for lang in TARGET_LANGUAGES}
            for layer in QWEN_SAE_SUBSET_LAYERS
        },
        "top_features_A": top_features_A,
        "top_features_B": top_features_B,
        "intersection_features": intersection_features,
        "reasoning_features": reasoning_features,
        "probe_accuracies": probe_accuracies,
    }
    torch.save(payload, ckpt)
    print(f"\nPhase 1 saved to {ckpt}")

    # Findings
    findings = "\n---\n\n## Qwen2.5-7B — Phase 1: Feature Identification\n\n"
    findings += f"**Model:** `{QWEN_MODEL_ID}` ({QWEN_N_LAYERS} layers, d_model={QWEN_D_MODEL})\n"
    findings += f"**SAEs:** BatchTopK from `andyrdt/saes-qwen2.5-7b-instruct`, layers {QWEN_SAE_SUBSET_LAYERS}\n\n"
    findings += "**Probe accuracies:**\n\n| Layer | Accuracy |\n|-------|----------|\n"
    for layer in QWEN_SAE_SUBSET_LAYERS:
        findings += f"| {layer} | {probe_accuracies[layer]:.3f} |\n"
    findings += "\n**A∩B intersection sizes:**\n\n| Layer | " + " | ".join(TARGET_LANGUAGES) + " |\n"
    findings += "|-------" + "|----" * len(TARGET_LANGUAGES) + "|\n"
    for layer in QWEN_SAE_SUBSET_LAYERS:
        sizes = [str(len(intersection_features[layer][l])) for l in TARGET_LANGUAGES]
        findings += f"| {layer} | " + " | ".join(sizes) + " |\n"
    findings += f"\n**Reasoning candidates:** " + ", ".join(
        f"L{l}={len(reasoning_features[l])}" for l in QWEN_SAE_SUBSET_LAYERS
    ) + "\n"
    append_findings(findings)
    auto_commit("data(qwen): Phase 1 — feature identification complete", force=True)

    return payload


# ---------------------------------------------------------------------------
# Phase 2a — Zhao SVD Baseline
# ---------------------------------------------------------------------------
def run_phase2a(model, tokenizer, mgsm, phase1, batch_size=8):
    ckpt = RESULTS_DIR / "qwen_phase2a_zhao.pt"
    if ckpt.exists():
        print(f"[Phase 2a] Loading checkpoint from {ckpt}")
        return torch.load(ckpt, weights_only=False)

    print("\n" + "=" * 60)
    print("PHASE 2a — Zhao SVD Baseline")
    print("=" * 60)

    prompts_by_lang = {
        lang: [make_prompt(tokenizer, ex["question"]) for ex in mgsm[lang]]
        for lang in TARGET_LANGUAGES
    }
    golds_by_lang = {
        lang: [ex["answer_number"] for ex in mgsm[lang]]
        for lang in TARGET_LANGUAGES
    }

    # 1. Per-language mean residuals at all 28 layers
    all_layers = list(range(QWEN_N_LAYERS))
    per_lang_mean = {layer: {} for layer in all_layers}
    for lang in TARGET_LANGUAGES:
        print(f"\n=== {lang}: extracting mean residuals at all {QWEN_N_LAYERS} layers ===")
        acts = extract_residual_activations(
            model, tokenizer, prompts_by_lang[lang],
            layers=all_layers, batch_size=batch_size, positions="last",
        )
        for layer in all_layers:
            per_lang_mean[layer][lang] = acts[layer].float().mean(dim=0)
        torch.cuda.empty_cache()

    # 2. SVD decomposition
    M_s_by_rank_layer = {r: {} for r in RANKS}
    for r in RANKS:
        for layer in all_layers:
            _, M_s = compute_language_subspace(per_lang_mean[layer], rank=r)
            M_s_by_rank_layer[r][layer] = M_s
    print("SVD decomposition done.")

    # 3. Baseline eval on full 250×5 — batched
    print("\n--- Baseline eval (batched) ---")
    baseline_results = {}
    for lang in TARGET_LANGUAGES:
        outputs = batched_baseline_eval(
            model, tokenizer, prompts_by_lang[lang], batch_size=batch_size,
        )
        baseline_results[lang] = evaluate_outputs(outputs, golds_by_lang[lang])
        print(f"  {lang}: {baseline_results[lang]['accuracy']:.3f}")
    baseline_avg = avg_acc(baseline_results)
    print(f"  Baseline avg: {baseline_avg:.3f}")
    auto_commit("data(qwen): Phase 2a — baseline eval done", force=False)

    # 4. Dev sweep (4 configs on 50-prompt dev) — batched SVD
    print("\n--- SVD dev sweep ---")
    dev_prompts = {l: prompts_by_lang[l][:N_DEV] for l in TARGET_LANGUAGES}
    dev_golds = {l: golds_by_lang[l][:N_DEV] for l in TARGET_LANGUAGES}

    sweep_results = {}
    for lam_mid, lam_hi, rank in ZHAO_GRID:
        key = (lam_mid, lam_hi, rank)
        print(f"\n  config: λ_mid={lam_mid}, λ_hi={lam_hi}, rank={rank}")
        sweep_results[key] = {}
        M_s_per_layer = M_s_by_rank_layer[rank]
        for lang in TARGET_LANGUAGES:
            outputs = generate_with_svd_batched(
                model, tokenizer, dev_prompts[lang],
                M_s_per_layer=M_s_per_layer,
                lambda_middle=lam_mid, lambda_higher=lam_hi,
                max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
                middle_layers=QWEN_ZHAO_MIDDLE_LAYERS,
                higher_layers=QWEN_ZHAO_HIGHER_LAYERS,
            )
            sweep_results[key][lang] = evaluate_outputs(outputs, dev_golds[lang])
            print(f"    {lang}: {sweep_results[key][lang]['accuracy']:.3f}")
        sweep_results[key]["avg"] = np.mean(
            [sweep_results[key][l]["accuracy"] for l in TARGET_LANGUAGES]
        )
        print(f"    AVG: {sweep_results[key]['avg']:.3f}")

    best_key = max(sweep_results.keys(), key=lambda k: sweep_results[k]["avg"])
    best_lm, best_lh, best_rank = best_key
    print(f"\n  Best config: λ_mid={best_lm}, λ_hi={best_lh}, rank={best_rank}")

    # 5. Best config on full 250×5 — batched SVD
    print("\n--- Best SVD config on full test set ---")
    zhao_test = {}
    M_s_per_layer = M_s_by_rank_layer[best_rank]
    for lang in TARGET_LANGUAGES:
        outputs = generate_with_svd_batched(
            model, tokenizer, prompts_by_lang[lang],
            M_s_per_layer=M_s_per_layer,
            lambda_middle=best_lm, lambda_higher=best_lh,
            max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
            middle_layers=QWEN_ZHAO_MIDDLE_LAYERS,
            higher_layers=QWEN_ZHAO_HIGHER_LAYERS,
        )
        zhao_test[lang] = evaluate_outputs(outputs, golds_by_lang[lang])
        print(f"  {lang}: baseline={baseline_results[lang]['accuracy']:.3f} → "
              f"zhao={zhao_test[lang]['accuracy']:.3f}")
    zhao_avg = avg_acc(zhao_test)
    print(f"  Zhao avg: {zhao_avg:.3f} (Δ={zhao_avg - baseline_avg:+.3f})")

    # 6. Language fidelity
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    LANG_OK = {"en": {"en"}, "zh": {"zh-cn", "zh-tw"}, "es": {"es"}, "bn": {"bn"}, "sw": {"sw"}}

    def language_fidelity(outputs, target_lang):
        ok = total = 0
        for o in outputs:
            s = (o or "")[:300].strip()
            if not s:
                continue
            try:
                d = detect(s)
            except Exception:
                continue
            total += 1
            if d in LANG_OK[target_lang]:
                ok += 1
        return ok / total if total else 0.0

    fidelity = {}
    for lang in TARGET_LANGUAGES:
        fidelity[lang] = {
            "baseline": language_fidelity(baseline_results[lang]["outputs"], lang),
            "zhao": language_fidelity(zhao_test[lang]["outputs"], lang),
        }
        print(f"  {lang}: fidelity baseline={fidelity[lang]['baseline']:.3f}, "
              f"zhao={fidelity[lang]['zhao']:.3f}")

    # Save
    payload = {
        "config": {
            "middle_layers": QWEN_ZHAO_MIDDLE_LAYERS,
            "higher_layers": QWEN_ZHAO_HIGHER_LAYERS,
            "ranks_searched": RANKS,
            "grid": ZHAO_GRID,
            "best_lambda_middle": best_lm,
            "best_lambda_higher": best_lh,
            "best_rank": best_rank,
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_dev": N_DEV, "n_test": N_TEST,
        },
        "per_lang_mean": per_lang_mean,
        "M_s_by_rank_layer": M_s_by_rank_layer,
        "baseline_results": baseline_results,
        "sweep_results": sweep_results,
        "zhao_test": zhao_test,
        "language_fidelity": fidelity,
        "baseline_avg": baseline_avg,
        "zhao_avg": zhao_avg,
    }
    torch.save(payload, ckpt)
    print(f"\nPhase 2a saved to {ckpt}")

    # Findings
    findings = "\n---\n\n## Qwen2.5-7B — Phase 2a: Zhao SVD Baseline\n\n"
    findings += f"**Best config:** λ_mid={best_lm}, λ_hi={best_lh}, rank={best_rank}\n\n"
    findings += "| Language | Baseline | Zhao | Δ |\n|----------|----------|------|---|\n"
    for lang in TARGET_LANGUAGES:
        b = baseline_results[lang]["accuracy"]
        z = zhao_test[lang]["accuracy"]
        findings += f"| {lang} | {b:.3f} | {z:.3f} | {z - b:+.3f} |\n"
    findings += f"| **Avg** | **{baseline_avg:.3f}** | **{zhao_avg:.3f}** | **{zhao_avg - baseline_avg:+.3f}** |\n"
    append_findings(findings)
    auto_commit("data(qwen): Phase 2a — Zhao SVD baseline complete", force=True)

    return payload


# ---------------------------------------------------------------------------
# Phase 2b — Full Causal Ablation
# ---------------------------------------------------------------------------
def run_phase2b(model, tokenizer, saes, mgsm, phase1, phase2a, batch_size=8):
    ckpt = RESULTS_DIR / "qwen_phase2b_full.pt"
    if ckpt.exists():
        print(f"[Phase 2b] Loading checkpoint from {ckpt}")
        return torch.load(ckpt, weights_only=False)

    print("\n" + "=" * 60)
    print("PHASE 2b — Full Causal Ablation")
    print("=" * 60)

    from src.evaluation import compute_perplexity
    from src.model import get_decoder_layers

    DECODER_LAYERS = get_decoder_layers(model)

    intersection = phase1["intersection_features"]
    top_A = phase1["top_features_A"]
    baseline_results = phase2a["baseline_results"]
    zhao_test = phase2a["zhao_test"]

    prompts_by_lang = {
        lang: [make_prompt(tokenizer, ex["question"]) for ex in mgsm[lang]]
        for lang in TARGET_LANGUAGES
    }
    golds_by_lang = {
        lang: [ex["answer_number"] for ex in mgsm[lang]]
        for lang in TARGET_LANGUAGES
    }
    dev_prompts = {l: prompts_by_lang[l][:N_DEV] for l in TARGET_LANGUAGES}
    dev_golds = {l: golds_by_lang[l][:N_DEV] for l in TARGET_LANGUAGES}

    sae_primary = saes[PRIMARY_LAYER]

    # --- Step 1: Causal feature labeling ---
    print("\n--- Causal feature labeling ---")

    # Candidates: top-5 A∩B per language at PRIMARY_LAYER
    candidates = {}
    for lang in TARGET_LANGUAGES:
        inter = intersection[PRIMARY_LAYER].get(lang, [])
        if len(inter) >= TOP_PER_LANG:
            cand = inter[:TOP_PER_LANG]
        else:
            extra = [f for f in top_A[PRIMARY_LAYER].get(lang, []) if f not in inter]
            cand = (list(inter) + extra)[:TOP_PER_LANG]
        candidates[lang] = cand
        print(f"  {lang}: candidates = {cand}")

    # Label-dev baseline accuracy (ablation-free, batched)
    label_dev_prompts = {l: prompts_by_lang[l][:N_LABEL_DEV] for l in TARGET_LANGUAGES}
    label_dev_golds = {l: golds_by_lang[l][:N_LABEL_DEV] for l in TARGET_LANGUAGES}

    label_baseline_acc = {}
    for lang in TARGET_LANGUAGES:
        outputs = batched_baseline_eval(model, tokenizer, label_dev_prompts[lang], batch_size=batch_size)
        r = evaluate_outputs(outputs, label_dev_golds[lang])
        label_baseline_acc[lang] = r["accuracy"]
        print(f"  {lang}: label-dev baseline acc = {r['accuracy']:.3f}")

    # Baseline perplexity (held-out questions 200-249)
    ppl_texts = {l: [mgsm[l][i]["question"] for i in range(200, 250)] for l in TARGET_LANGUAGES}
    baseline_ppl = {}
    for lang in TARGET_LANGUAGES:
        ppls = compute_perplexity(model, tokenizer, ppl_texts[lang])
        baseline_ppl[lang] = float(np.mean(ppls))
        print(f"  {lang}: baseline ppl = {baseline_ppl[lang]:.2f}")

    # Per-feature ablation: accuracy delta + perplexity delta
    def ablation_config_dirs(layer, feat_indices):
        if not feat_indices:
            return {}
        dirs = get_sae_decoder_directions(sae_primary, feat_indices, sae_type="batchtopk")
        return {layer: dirs.to(model.device)}

    def perplexity_with_ablation(layer, feat_indices, text_lang):
        cfg = ablation_config_dirs(layer, feat_indices)
        if not cfg:
            return baseline_ppl[text_lang]
        handles = []
        for L, dirs in cfg.items():
            dirs_dev = dirs.to(model.device)

            def make_hook(d):
                def hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        output[:] = directional_ablation(output, d)
                        return output
                    hs = output[0]
                    hs[:] = directional_ablation(hs, d)
                    return (hs,) + output[1:]
                return hook

            handles.append(DECODER_LAYERS[L].register_forward_hook(make_hook(dirs_dev)))
        try:
            ppls = compute_perplexity(model, tokenizer, ppl_texts[text_lang])
        finally:
            for h in handles:
                h.remove()
        return float(np.mean(ppls))

    causal_labels = {}
    for lang in TARGET_LANGUAGES:
        for feat in candidates[lang]:
            # (i) Accuracy delta — single-feature ablation, batched
            hook_config = {PRIMARY_LAYER: (sae_primary, [feat])}
            outputs = run_generate_with_hooks_batched(
                model, tokenizer, label_dev_prompts[lang],
                hook_config=hook_config, method="directional",
                max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
                sae_type="batchtopk",
            )
            r = evaluate_outputs(outputs, label_dev_golds[lang])
            acc_delta = r["accuracy"] - label_baseline_acc[lang]

            # (ii) Perplexity delta
            ppl_with = perplexity_with_ablation(PRIMARY_LAYER, [feat], lang)
            ppl_delta = ppl_with - baseline_ppl[lang]

            # Tagging
            ppl_threshold = 0.05 * baseline_ppl[lang]
            acc_threshold = 0.04
            hit_lang = ppl_delta > ppl_threshold
            hit_arith = acc_delta < -acc_threshold

            if hit_lang and not hit_arith:
                tag = "LANGUAGE"
            elif hit_arith and not hit_lang:
                tag = "REASONING"
            elif hit_lang and hit_arith:
                tag = "SHARED"
            else:
                tag = "JUNK"

            causal_labels[(lang, feat)] = {
                "tag": tag, "acc_delta": acc_delta, "ppl_delta": ppl_delta,
            }
            print(f"  {lang} f={feat}: Δacc={acc_delta:+.3f}, Δppl={ppl_delta:+.2e} → {tag}")

    confirmed_language = {
        lang: [feat for (l, feat), v in causal_labels.items()
               if l == lang and v["tag"] == "LANGUAGE"]
        for lang in TARGET_LANGUAGES
    }
    tag_counts = Counter(v["tag"] for v in causal_labels.values())
    print(f"\nTag counts: {dict(tag_counts)}")
    for lang in TARGET_LANGUAGES:
        print(f"  {lang}: confirmed LANGUAGE = {confirmed_language[lang]}")
    auto_commit("data(qwen): Phase 2b — causal labeling done", force=True)

    # --- Step 2: H1 sweep ---
    print("\n--- H1 sweep on dev ---")
    h1_sweep = {}
    for k in K_VALUES:
        h1_sweep[k] = {}
        for lang in TARGET_LANGUAGES:
            feats = select_top_features(lang, k, confirmed_language, intersection, top_A)
            hook_config = {PRIMARY_LAYER: (sae_primary, feats)}
            outputs = run_generate_with_hooks_batched(
                model, tokenizer, dev_prompts[lang],
                hook_config=hook_config, method="directional",
                max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
                sae_type="batchtopk",
            )
            h1_sweep[k][lang] = evaluate_outputs(outputs, dev_golds[lang])
            print(f"  k={k:>2} {lang}: {h1_sweep[k][lang]['accuracy']:.3f}")
        h1_sweep[k]["avg"] = np.mean([h1_sweep[k][l]["accuracy"] for l in TARGET_LANGUAGES])
        print(f"  k={k:>2} AVG: {h1_sweep[k]['avg']:.3f}")
    best_k = max(K_VALUES, key=lambda k: h1_sweep[k]["avg"])
    print(f"\n  Best k on dev: {best_k} (avg {h1_sweep[best_k]['avg']:.3f})")
    auto_commit("data(qwen): Phase 2b — H1 sweep done", force=False)

    # --- Step 3: H1 controls ---
    print("\n--- H1 controls (random + Deng-style) on dev ---")
    rng = np.random.default_rng(0)
    random_features = rng.choice(QWEN_SAE_WIDTH, size=best_k, replace=False).tolist()

    ctrl_random = {}
    for lang in TARGET_LANGUAGES:
        hook_config = {PRIMARY_LAYER: (sae_primary, random_features)}
        outputs = run_generate_with_hooks_batched(
            model, tokenizer, dev_prompts[lang],
            hook_config=hook_config, method="directional",
            max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
            sae_type="batchtopk",
        )
        ctrl_random[lang] = evaluate_outputs(outputs, dev_golds[lang])
        print(f"  random k={best_k} {lang}: {ctrl_random[lang]['accuracy']:.3f}")

    ctrl_deng = {}
    for lang in TARGET_LANGUAGES:
        feats = top_A[PRIMARY_LAYER].get(lang, [])[:best_k]
        hook_config = {PRIMARY_LAYER: (sae_primary, feats)}
        outputs = run_generate_with_hooks_batched(
            model, tokenizer, dev_prompts[lang],
            hook_config=hook_config, method="directional",
            max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
            sae_type="batchtopk",
        )
        ctrl_deng[lang] = evaluate_outputs(outputs, dev_golds[lang])
        print(f"  deng k={best_k} {lang}: {ctrl_deng[lang]['accuracy']:.3f}")

    # --- Step 4: H1 final on full 250×5 ---
    print("\n--- H1 final on full test set ---")
    h1_test = {}
    for lang in TARGET_LANGUAGES:
        feats = select_top_features(lang, best_k, confirmed_language, intersection, top_A)
        hook_config = {PRIMARY_LAYER: (sae_primary, feats)}
        outputs = run_generate_with_hooks_batched(
            model, tokenizer, prompts_by_lang[lang],
            hook_config=hook_config, method="directional",
            max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
            sae_type="batchtopk",
        )
        h1_test[lang] = evaluate_outputs(outputs, golds_by_lang[lang])
        print(f"  {lang}: baseline={baseline_results[lang]['accuracy']:.3f} → "
              f"sae={h1_test[lang]['accuracy']:.3f} "
              f"(zhao={zhao_test[lang]['accuracy']:.3f})")
    h1_test_avg = avg_acc(h1_test)
    print(f"  H1 avg: {h1_test_avg:.3f}")
    auto_commit("data(qwen): Phase 2b — H1 final done", force=True)

    # --- Step 5: H2 language fidelity ---
    print("\n--- H2 language fidelity ---")
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    LANG_OK = {"en": {"en"}, "zh": {"zh-cn", "zh-tw"}, "es": {"es"}, "bn": {"bn"}, "sw": {"sw"}}

    def language_fidelity(outputs, target_lang):
        ok = total = 0
        for o in outputs:
            s = (o or "")[:300].strip()
            if not s:
                continue
            try:
                d = detect(s)
            except Exception:
                continue
            total += 1
            if d in LANG_OK[target_lang]:
                ok += 1
        return ok / total if total else 0.0

    fidelity = {}
    for lang in TARGET_LANGUAGES:
        fidelity[lang] = {
            "baseline": language_fidelity(baseline_results[lang]["outputs"], lang),
            "zhao": language_fidelity(zhao_test[lang]["outputs"], lang),
            "sae": language_fidelity(h1_test[lang]["outputs"], lang),
        }
        print(f"  {lang}: baseline={fidelity[lang]['baseline']:.3f}, "
              f"zhao={fidelity[lang]['zhao']:.3f}, sae={fidelity[lang]['sae']:.3f}")

    # --- Step 6: H3 layer-wise ---
    print("\n--- H3 layer-wise ---")
    K_H3 = 10
    h3_results = {}
    for layer in QWEN_SAE_SUBSET_LAYERS:
        h3_results[layer] = {}
        sae_layer = saes[layer]
        for lang in TARGET_LANGUAGES:
            inter = intersection[layer].get(lang, [])
            extras = [f for f in top_A[layer].get(lang, []) if f not in inter]
            feats = (list(inter) + extras)[:K_H3]
            hook_config = {layer: (sae_layer, feats)}
            outputs = run_generate_with_hooks_batched(
                model, tokenizer, dev_prompts[lang],
                hook_config=hook_config, method="directional",
                max_new_tokens=MAX_NEW_TOKENS, batch_size=batch_size,
                sae_type="batchtopk",
            )
            h3_results[layer][lang] = evaluate_outputs(outputs, dev_golds[lang])
            print(f"  L{layer} {lang}: {h3_results[layer][lang]['accuracy']:.3f}")
        h3_results[layer]["avg"] = np.mean(
            [h3_results[layer][l]["accuracy"] for l in TARGET_LANGUAGES]
        )
        print(f"  L{layer} AVG: {h3_results[layer]['avg']:.3f}")

    # --- Step 7: Bootstrap CIs ---
    print("\n--- Bootstrap CIs ---")
    ci_summary = {}
    for lang in TARGET_LANGUAGES:
        ci_summary[lang] = {
            "baseline": bootstrap_ci(baseline_results[lang]["correct"]),
            "zhao": bootstrap_ci(zhao_test[lang]["correct"]),
            "sae": bootstrap_ci(h1_test[lang]["correct"]),
        }
        print(f"  {lang}: baseline {ci_summary[lang]['baseline']} | "
              f"zhao {ci_summary[lang]['zhao']} | sae {ci_summary[lang]['sae']}")

    # Save
    payload = {
        "config": {
            "primary_layer": PRIMARY_LAYER, "k_values": K_VALUES, "best_k": best_k,
            "k_h3": K_H3, "n_dev": N_DEV, "n_test": N_TEST,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "causal_labels": causal_labels,
        "confirmed_language": confirmed_language,
        "baseline_ppl": baseline_ppl,
        "h1_sweep": h1_sweep,
        "ctrl_random": ctrl_random,
        "ctrl_deng": ctrl_deng,
        "h1_test": h1_test,
        "fidelity": fidelity,
        "h3_results": h3_results,
        "ci_summary": ci_summary,
        "best_k": best_k,
        "h1_test_avg": h1_test_avg,
    }
    torch.save(payload, ckpt)
    print(f"\nPhase 2b saved to {ckpt}")

    # Findings
    findings = "\n---\n\n## Qwen2.5-7B — Phase 2b: Causal Ablation\n\n"
    findings += f"**Primary layer:** {PRIMARY_LAYER}, **Best k:** {best_k}\n\n"
    findings += "**Causal label counts:** " + ", ".join(
        f"{t}={c}" for t, c in tag_counts.items()
    ) + "\n\n"
    findings += "**H1 final (full 250×5):**\n\n"
    findings += "| Lang | LANG count | Baseline | Zhao | SAE | Δ vs base |\n"
    findings += "|------|-----------|----------|------|-----|----------|\n"
    for lang in TARGET_LANGUAGES:
        n_lang = len(confirmed_language[lang])
        b = baseline_results[lang]["accuracy"]
        z = zhao_test[lang]["accuracy"]
        s = h1_test[lang]["accuracy"]
        findings += f"| {lang} | {n_lang} | {b:.3f} | {z:.3f} | {s:.3f} | {s - b:+.3f} |\n"
    findings += f"| **avg** | | **{phase2a['baseline_avg']:.3f}** | **{phase2a['zhao_avg']:.3f}** | **{h1_test_avg:.3f}** | **{h1_test_avg - phase2a['baseline_avg']:+.3f}** |\n"
    findings += "\n**H3 layer-wise (dev):**\n\n| Layer | avg | Δ vs baseline |\n|-------|-----|---------------|\n"
    dev_baseline_avg = phase2a["baseline_avg"]  # approximate
    for layer in QWEN_SAE_SUBSET_LAYERS:
        a = h3_results[layer]["avg"]
        findings += f"| L{layer} | {a:.3f} | {a - dev_baseline_avg:+.3f} |\n"
    append_findings(findings)
    auto_commit("data(qwen): Phase 2b — full causal ablation complete", force=True)

    return payload


# ---------------------------------------------------------------------------
# Phase 3 — Interaction Analysis
# ---------------------------------------------------------------------------
def run_phase3(model, tokenizer, saes, mgsm, phase1, phase2b, batch_size=8):
    ckpt = RESULTS_DIR / "qwen_phase3_full.pt"
    if ckpt.exists():
        print(f"[Phase 3] Loading checkpoint from {ckpt}")
        return torch.load(ckpt, weights_only=False)

    print("\n" + "=" * 60)
    print("PHASE 3 — Interaction Analysis")
    print("=" * 60)

    from src.model import get_decoder_layers

    DECODER_LAYERS = get_decoder_layers(model)

    intersection = phase1["intersection_features"]
    top_A = phase1["top_features_A"]
    reasoning_features = phase1["reasoning_features"]
    confirmed_language = phase2b["confirmed_language"]
    best_k = phase2b["best_k"]

    # Feature selection matching Phase 2b
    ablation_features = {
        l: select_top_features(l, best_k, confirmed_language, intersection, top_A)
        for l in TARGET_LANGUAGES
    }
    print("Ablation features per language:")
    for l in TARGET_LANGUAGES:
        print(f"  {l}: {ablation_features[l]}")

    def make_ablation_hook(directions, input_length):
        pos = input_length - 1
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                hs = output
                is_tuple = False
            else:
                hs = output[0]
                is_tuple = True
            if hs.dim() == 3 and hs.shape[1] > pos:
                hs[:, pos, :] = directional_ablation(hs[:, pos, :], directions)
            return hs if not is_tuple else (hs,) + output[1:]
        return hook

    def forward_collect(prompt, layers_to_collect, ablation_layer=None,
                        ablation_dirs=None, want_attentions=False):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]
        handles = []
        if ablation_layer is not None and ablation_dirs is not None:
            handles.append(DECODER_LAYERS[ablation_layer].register_forward_hook(
                make_ablation_hook(ablation_dirs.to(model.device), input_len)))
        try:
            with torch.no_grad():
                out = model(
                    **inputs, output_hidden_states=True,
                    output_attentions=want_attentions, use_cache=False,
                )
        finally:
            for h in handles:
                h.remove()
        resid = {layer: out.hidden_states[layer + 1][:, -1, :].cpu().float()
                 for layer in layers_to_collect}
        attns = None
        if want_attentions:
            attns = [a.detach().cpu().float() for a in out.attentions]
        return {"resid": resid, "attns": attns, "input_len": input_len}

    # --- 3a: Capacity competition ---
    print("\n--- 3a: Capacity competition ---")
    REASONING_FEATS = reasoning_features.get(PRIMARY_LAYER, [])
    if not REASONING_FEATS:
        print("  WARNING: no reasoning candidates at primary layer, using first 50 features")
        REASONING_FEATS = list(range(50))
    print(f"  Tracking {len(REASONING_FEATS)} reasoning features at L{PRIMARY_LAYER}")

    sae_primary = saes[PRIMARY_LAYER]
    clean_feats = {lang: [] for lang in TARGET_LANGUAGES}
    ablated_feats = {lang: [] for lang in TARGET_LANGUAGES}

    for lang in TARGET_LANGUAGES:
        dirs = get_sae_decoder_directions(
            sae_primary, ablation_features[lang], sae_type="batchtopk"
        ).to(model.device)
        print(f"\n  {lang}: ablating {len(ablation_features[lang])} features at L{PRIMARY_LAYER}")
        for i in tqdm(range(N_DEV), desc=f"capacity {lang}"):
            prompt = make_prompt(tokenizer, mgsm[lang][i]["question"])
            clean = forward_collect(prompt, [PRIMARY_LAYER])
            ablt = forward_collect(prompt, [PRIMARY_LAYER],
                                   ablation_layer=PRIMARY_LAYER, ablation_dirs=dirs)
            c_f = encode_activations_batchtopk(clean["resid"][PRIMARY_LAYER], sae_primary)[0]
            a_f = encode_activations_batchtopk(ablt["resid"][PRIMARY_LAYER], sae_primary)[0]
            clean_feats[lang].append(c_f[REASONING_FEATS])
            ablated_feats[lang].append(a_f[REASONING_FEATS])
        clean_feats[lang] = torch.stack(clean_feats[lang])
        ablated_feats[lang] = torch.stack(ablated_feats[lang])
    torch.cuda.empty_cache()

    # Statistical tests
    capacity_summary = {}
    for lang in TARGET_LANGUAGES:
        rows = []
        cf = clean_feats[lang].numpy()
        af = ablated_feats[lang].numpy()
        for j, feat_idx in enumerate(REASONING_FEATS):
            diff = af[:, j] - cf[:, j]
            if np.allclose(diff, 0):
                p, stat = 1.0, 0.0
            else:
                try:
                    stat, p = wilcoxon(af[:, j], cf[:, j])
                except ValueError:
                    stat, p = 0.0, 1.0
            rows.append({
                "feature": int(feat_idx),
                "mean_clean": float(cf[:, j].mean()),
                "mean_ablated": float(af[:, j].mean()),
                "mean_delta": float(diff.mean()),
                "p_value": float(p),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        capacity_summary[lang] = df
        n_sig = int((df["p_value"] < 0.05).sum())
        print(f"  {lang}: {n_sig} significant of {len(df)}, "
              f"top Δ={df['mean_delta'].abs().max():.1f}")

    auto_commit("data(qwen): Phase 3a — capacity competition done", force=True)

    # --- 3c: Attention disruption ---
    # Reload model with eager attention if needed
    print("\n--- 3c: Attention disruption ---")
    print("  Reloading model with attn_implementation='eager'...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    del model
    torch.cuda.empty_cache()

    model_eager = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        token=os.environ.get("HF_TOKEN"), attn_implementation="eager",
    )
    model_eager.eval()
    DECODER_LAYERS_EAGER = get_decoder_layers(model_eager)

    # Redefine forward_collect for eager model
    def forward_collect_eager(prompt, layers_to_collect, ablation_layer=None,
                              ablation_dirs=None, want_attentions=False):
        inputs = tokenizer(prompt, return_tensors="pt").to(model_eager.device)
        input_len = inputs["input_ids"].shape[1]
        handles = []
        if ablation_layer is not None and ablation_dirs is not None:
            pos = input_len - 1
            dirs_dev = ablation_dirs.to(model_eager.device)

            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    hs = output
                    is_tuple = False
                else:
                    hs = output[0]
                    is_tuple = True
                if hs.dim() == 3 and hs.shape[1] > pos:
                    hs[:, pos, :] = directional_ablation(hs[:, pos, :], dirs_dev)
                return hs if not is_tuple else (hs,) + output[1:]

            handles.append(DECODER_LAYERS_EAGER[ablation_layer].register_forward_hook(hook))
        try:
            with torch.no_grad():
                out = model_eager(
                    **inputs, output_hidden_states=True,
                    output_attentions=want_attentions, use_cache=False,
                )
        finally:
            for h in handles:
                h.remove()
        resid = {layer: out.hidden_states[layer + 1][:, -1, :].cpu().float()
                 for layer in layers_to_collect}
        attns = None
        if want_attentions:
            attns = [a.detach().cpu().float() for a in out.attentions]
        return {"resid": resid, "attns": attns, "input_len": input_len}

    def attention_entropy_last_query(attn_layer):
        p = attn_layer[0, :, -1, :].clamp_min(1e-12)
        return -(p * p.log()).sum(dim=-1)

    ATTN_LAYERS = QWEN_SAE_SUBSET_LAYERS
    entropy_clean = {layer: {l: [] for l in TARGET_LANGUAGES} for layer in ATTN_LAYERS}
    entropy_ablated = {layer: {l: [] for l in TARGET_LANGUAGES} for layer in ATTN_LAYERS}

    for lang in TARGET_LANGUAGES:
        dirs = get_sae_decoder_directions(
            sae_primary, ablation_features[lang], sae_type="batchtopk"
        ).to(model_eager.device)
        for i in tqdm(range(N_ATTENTION), desc=f"attn {lang}"):
            prompt = make_prompt(tokenizer, mgsm[lang][i]["question"])
            clean = forward_collect_eager(prompt, [], want_attentions=True)
            ablt = forward_collect_eager(
                prompt, [], ablation_layer=PRIMARY_LAYER,
                ablation_dirs=dirs, want_attentions=True,
            )
            for layer in ATTN_LAYERS:
                entropy_clean[layer][lang].append(
                    attention_entropy_last_query(clean["attns"][layer])
                )
                entropy_ablated[layer][lang].append(
                    attention_entropy_last_query(ablt["attns"][layer])
                )
        for layer in ATTN_LAYERS:
            entropy_clean[layer][lang] = torch.stack(entropy_clean[layer][lang])
            entropy_ablated[layer][lang] = torch.stack(entropy_ablated[layer][lang])

    # Attention summary
    attention_summary = []
    for layer in ATTN_LAYERS:
        for lang in TARGET_LANGUAGES:
            c = entropy_clean[layer][lang]
            a = entropy_ablated[layer][lang]
            delta = (a - c).mean().item()
            try:
                stat, p = wilcoxon(a.flatten().numpy(), c.flatten().numpy())
            except ValueError:
                stat, p = 0.0, 1.0
            attention_summary.append({
                "layer": layer, "lang": lang,
                "mean_delta_entropy": delta, "p_value": float(p),
            })
    import pandas as pd
    attn_df = pd.DataFrame(attention_summary)
    print("\nAttention entropy deltas:")
    print(attn_df.pivot(index="layer", columns="lang", values="mean_delta_entropy").round(4))

    auto_commit("data(qwen): Phase 3c — attention disruption done", force=True)

    # --- 3b: Circuit interference ---
    print("\n--- 3b: Circuit interference ---")
    # Use the eager model (still loaded)
    circuit_results = {lang: [] for lang in TARGET_LANGUAGES}
    TOP_EDGE = 30

    for lang in TARGET_LANGUAGES:
        dirs = get_sae_decoder_directions(
            sae_primary, ablation_features[lang], sae_type="batchtopk"
        ).to(model_eager.device)
        print(f"\n  circuit attribution: {lang}")
        for i in tqdm(range(N_CIRCUIT), desc=f"circuit {lang}"):
            prompt = make_prompt(tokenizer, mgsm[lang][i]["question"])
            clean = forward_collect_eager(prompt, DOWNSTREAM_LAYERS)
            ablt = forward_collect_eager(
                prompt, DOWNSTREAM_LAYERS,
                ablation_layer=PRIMARY_LAYER, ablation_dirs=dirs,
            )
            per_layer_edges = {}
            for dl in DOWNSTREAM_LAYERS:
                sae_d = saes[dl]
                f_clean = encode_activations_batchtopk(clean["resid"][dl], sae_d)[0]
                f_ablt = encode_activations_batchtopk(ablt["resid"][dl], sae_d)[0]
                delta = (f_ablt - f_clean).abs()
                top = torch.topk(delta, k=TOP_EDGE)
                per_layer_edges[dl] = [
                    {"feature": int(top.indices[j]),
                     "delta": float(f_ablt[top.indices[j]] - f_clean[top.indices[j]]),
                     "abs_delta": float(top.values[j])}
                    for j in range(TOP_EDGE)
                ]
            circuit_results[lang].append({
                "problem_idx": i, "edges": per_layer_edges,
            })

    # Aggregate
    edge_strength = defaultdict(list)
    for lang in TARGET_LANGUAGES:
        for record in circuit_results[lang]:
            for dl, edges in record["edges"].items():
                for e in edges:
                    key = (lang, dl, e["feature"])
                    edge_strength[key].append(e["abs_delta"])

    edge_rows = []
    for (lang, layer, feat), vs in edge_strength.items():
        edge_rows.append({
            "lang": lang, "downstream_layer": layer, "downstream_feature": feat,
            "mean_abs_delta": float(np.mean(vs)), "n_problems": len(vs),
        })
    edge_df = pd.DataFrame(edge_rows).sort_values("mean_abs_delta", ascending=False)
    print("\nTop 10 downstream effects:")
    print(edge_df.head(10).to_string(index=False))

    # Save
    payload = {
        "config": {
            "primary_layer": PRIMARY_LAYER,
            "downstream_layers": DOWNSTREAM_LAYERS,
            "attention_layers": ATTN_LAYERS,
            "n_dev": N_DEV, "n_attention": N_ATTENTION, "n_circuit": N_CIRCUIT,
            "best_k": best_k,
            "reasoning_features": REASONING_FEATS,
        },
        "capacity": {
            "clean_feats": {l: t for l, t in clean_feats.items()},
            "ablated_feats": {l: t for l, t in ablated_feats.items()},
            "summary": {l: capacity_summary[l].to_dict("records") for l in TARGET_LANGUAGES},
        },
        "attention": {
            "entropy_clean": {layer: dict(entropy_clean[layer]) for layer in ATTN_LAYERS},
            "entropy_ablated": {layer: dict(entropy_ablated[layer]) for layer in ATTN_LAYERS},
            "summary": attn_df.to_dict("records"),
        },
        "circuit": {
            "per_problem": circuit_results,
            "edges": edge_df.to_dict("records"),
        },
    }
    torch.save(payload, ckpt)
    print(f"\nPhase 3 saved to {ckpt}")

    # Findings
    findings = "\n---\n\n## Qwen2.5-7B — Phase 3: Interaction Analysis\n\n"
    findings += "**3a Capacity competition:**\n\n"
    for lang in TARGET_LANGUAGES:
        df = capacity_summary[lang]
        n_sig = int((df["p_value"] < 0.05).sum())
        findings += f"- {lang}: {n_sig} sig features, top |Δ|={df['mean_delta'].abs().max():.1f}\n"
    findings += "\n**3c Attention entropy delta (ablated − clean):**\n\n"
    findings += "| Layer | " + " | ".join(TARGET_LANGUAGES) + " |\n"
    findings += "|-------" + "|------" * len(TARGET_LANGUAGES) + "|\n"
    for layer in ATTN_LAYERS:
        vals = []
        for lang in TARGET_LANGUAGES:
            row = attn_df[(attn_df["layer"] == layer) & (attn_df["lang"] == lang)]
            vals.append(f"{row['mean_delta_entropy'].values[0]:+.3f}")
        findings += f"| L{layer} | " + " | ".join(vals) + " |\n"
    findings += f"\n**3b Circuit interference:** {len(edge_df)} unique edges captured.\n"
    if len(edge_df) > 0:
        top = edge_df.iloc[0]
        findings += f"Top edge: {top['lang']} L{int(top['downstream_layer'])} f={int(top['downstream_feature'])} (|Δ|={top['mean_abs_delta']:.0f})\n"
    append_findings(findings)
    auto_commit("data(qwen): Phase 3 — interaction analysis complete", force=True)

    return payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Full Qwen2.5-7B pipeline")
    parser.add_argument("--skip-phase", type=str, nargs="+", default=[],
                        help="Phase numbers to skip (e.g., '1' '2a')")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Start from this phase (e.g., '2b'). Loads checkpoints for earlier phases.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for generation (default 8)")
    parser.add_argument("--extract-batch-size", type=int, default=16,
                        help="Batch size for activation extraction (default 16)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    skip = set(args.skip_phase)
    if args.resume_from:
        phase_order = ["1", "2a", "2b", "3"]
        idx = phase_order.index(args.resume_from)
        skip = skip | set(phase_order[:idx])

    print(f"Config: batch_size={args.batch_size}, extract_batch_size={args.extract_batch_size}")
    print(f"Skipping phases: {skip or 'none'}")

    assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set. Add it to .env"

    # Load model + SAEs
    print("\nLoading Qwen model and tokenizer...")
    from src.model import load_model_and_tokenizer, load_qwen_saes_at_layers
    model, tokenizer = load_model_and_tokenizer(model_id=QWEN_MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading Qwen SAEs...")
    saes = load_qwen_saes_at_layers(layers=QWEN_SAE_SUBSET_LAYERS, trainer=QWEN_SAE_TRAINER)

    print("Loading MGSM...")
    mgsm = load_mgsm(TARGET_LANGUAGES)

    t_start = time.time()

    # Phase 1
    if "1" not in skip:
        phase1 = run_phase1(model, tokenizer, saes, mgsm, batch_size=args.extract_batch_size)
    else:
        phase1 = torch.load(RESULTS_DIR / "qwen_phase1_features.pt", weights_only=False)
        print("[Phase 1] Skipped (loaded checkpoint)")

    # Phase 2a
    if "2a" not in skip:
        phase2a = run_phase2a(model, tokenizer, mgsm, phase1, batch_size=args.batch_size)
    else:
        phase2a = torch.load(RESULTS_DIR / "qwen_phase2a_zhao.pt", weights_only=False)
        print("[Phase 2a] Skipped (loaded checkpoint)")

    # Phase 2b
    if "2b" not in skip:
        phase2b = run_phase2b(model, tokenizer, saes, mgsm, phase1, phase2a, batch_size=args.batch_size)
    else:
        phase2b = torch.load(RESULTS_DIR / "qwen_phase2b_full.pt", weights_only=False)
        print("[Phase 2b] Skipped (loaded checkpoint)")

    # Phase 3
    if "3" not in skip:
        phase3 = run_phase3(model, tokenizer, saes, mgsm, phase1, phase2b, batch_size=args.batch_size)
    else:
        print("[Phase 3] Skipped")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"ALL PHASES COMPLETE — total {elapsed / 3600:.1f}h")
    print(f"{'=' * 60}")
    auto_commit("data(qwen): full pipeline complete", force=True)


if __name__ == "__main__":
    main()
