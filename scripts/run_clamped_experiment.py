"""Phase 2: Clamped vs directional feature ablation on MGSM.

Runs 7 conditions × 2 methods (clamped, directional) on 50 MGSM problems
per language (en, zh, es). Saves results to results/phase2_clamped_vs_directional.pt.

Usage:
    python scripts/run_clamped_experiment.py [--batch-size 8] [--n-per-lang 50]
"""

import argparse
import time

import numpy as np
import torch

from src.config import RESULTS_DIR, SAE_WIDTH_16K
from src.data import format_prompt_gemma_it, load_mgsm, parse_answer_number, compute_accuracy
from src.intervention import run_generate_with_hooks_batched
from src.model import load_model_and_tokenizer, load_sae


# ── Feature sets (hand-picked, validated on Neuronpedia) ──────────────

FEATURES = {
    "en_specific": [203, 486, 513, 2127, 2426, 13456],
    "zh_specific": [828, 12299, 154, 2037, 9094],
    "es_specific": [2508, 5451, 2134, 9794, 1051],
    "polysemantic_lang": [5816, 2208, 1349, 7439],
    "math_numbers": [1367, 7910, 1858, 14019, 14379, 6544, 565, 812],
}

# Random control (seed=42, exclude all chosen features)
all_chosen = set()
for v in FEATURES.values():
    all_chosen.update(v)
rng = np.random.default_rng(42)
pool = [i for i in range(SAE_WIDTH_16K) if i not in all_chosen]
FEATURES["random"] = rng.choice(pool, size=5, replace=False).tolist()

# Which languages each condition ablates on
CONDITIONS = {
    "en_specific": ["en"],
    "zh_specific": ["zh"],
    "es_specific": ["es"],
    "random": ["en", "zh", "es"],
    "polysemantic_lang": ["en", "zh", "es"],
    "math_numbers": ["en", "zh", "es"],
}

LAYER = 17
LANGS = ["en", "zh", "es"]
METHODS = ["clamped", "directional"]


def evaluate_outputs(outputs: list[str], gold_answers: list[float]) -> dict:
    """Parse outputs and compute accuracy."""
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
        "n_correct": sum(correct),
        "n_total": len(correct),
    }


def run_baseline(model, tokenizer, prompts_by_lang, golds_by_lang, args):
    """Run baseline (no ablation) for all languages."""
    print("\n=== Baseline (no ablation) ===")
    baseline = {}
    for lang in LANGS:
        print(f"  {lang}: {len(prompts_by_lang[lang])} problems")
        outputs = run_generate_with_hooks_batched(
            model, tokenizer,
            prompts_by_lang[lang],
            hook_config={},
            method="clamped",  # no hooks, method irrelevant
            max_new_tokens=384,
            batch_size=args.batch_size,
            device=args.device,
        )
        baseline[lang] = evaluate_outputs(outputs, golds_by_lang[lang])
        print(f"    accuracy: {baseline[lang]['accuracy']:.1%}")
    return baseline


def run_condition(model, tokenizer, sae, condition_name, feat_ids, langs_to_ablate,
                  prompts_by_lang, golds_by_lang, method, args):
    """Run a single condition with a specific method."""
    results = {}
    hook_config = {LAYER: (sae, feat_ids)}

    for lang in langs_to_ablate:
        outputs = run_generate_with_hooks_batched(
            model, tokenizer,
            prompts_by_lang[lang],
            hook_config=hook_config,
            method=method,
            max_new_tokens=384,
            batch_size=args.batch_size,
            device=args.device,
        )
        results[lang] = evaluate_outputs(outputs, golds_by_lang[lang])
        print(f"    {lang}: {results[lang]['accuracy']:.1%} "
              f"({results[lang]['n_correct']}/{results[lang]['n_total']})")

    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Clamped vs directional ablation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-per-lang", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    args = parser.parse_args()

    print(f"Config: batch_size={args.batch_size}, n_per_lang={args.n_per_lang}, "
          f"device={args.device}, layer={LAYER}")
    print(f"Random control features: {FEATURES['random']}")

    # ── Load model + SAE ──────────────────────────────────────────────
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(device_map="auto")

    print(f"Loading SAE for layer {LAYER}...")
    sae, _, _ = load_sae(LAYER, width=SAE_WIDTH_16K, device=args.device)

    # ── Load data ─────────────────────────────────────────────────────
    print("Loading MGSM data...")
    mgsm = load_mgsm(LANGS)

    prompts_by_lang = {}
    golds_by_lang = {}
    for lang in LANGS:
        examples = mgsm[lang][:args.n_per_lang]
        prompts_by_lang[lang] = [format_prompt_gemma_it(ex["question"]) for ex in examples]
        golds_by_lang[lang] = [ex["answer_number"] for ex in examples]
        print(f"  {lang}: {len(examples)} examples loaded")

    # ── Run experiments ───────────────────────────────────────────────
    t_start = time.time()

    baseline = run_baseline(model, tokenizer, prompts_by_lang, golds_by_lang, args)

    all_results = {method: {} for method in METHODS}

    for method in METHODS:
        print(f"\n{'='*60}")
        print(f"METHOD: {method}")
        print(f"{'='*60}")

        for cond_name, langs_to_ablate in CONDITIONS.items():
            feat_ids = FEATURES[cond_name]
            print(f"\n  --- {cond_name} (features={feat_ids}) ---")

            results = run_condition(
                model, tokenizer, sae,
                cond_name, feat_ids, langs_to_ablate,
                prompts_by_lang, golds_by_lang,
                method, args,
            )
            all_results[method][cond_name] = results

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # ── Save results ──────────────────────────────────────────────────
    output_path = RESULTS_DIR / "phase2_clamped_vs_directional.pt"

    # Strip raw outputs to save space (keep predictions + correct flags)
    def slim(result_dict):
        return {
            lang: {k: v for k, v in res.items() if k != "outputs"}
            for lang, res in result_dict.items()
        }

    save_data = {
        "config": {
            "methods": METHODS,
            "layer": LAYER,
            "n_per_lang": args.n_per_lang,
            "features": FEATURES,
            "conditions": CONDITIONS,
            "batch_size": args.batch_size,
        },
        "baseline": slim(baseline),
        "clamped": {cond: slim(res) for cond, res in all_results["clamped"].items()},
        "directional": {cond: slim(res) for cond, res in all_results["directional"].items()},
    }

    torch.save(save_data, output_path)
    print(f"\nResults saved to {output_path}")

    # ── Print summary table ───────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Condition':<22} {'Lang':>4} {'Baseline':>9} {'Clamped':>9} {'Direct.':>9}")
    print("-"*70)

    for cond_name, langs_to_ablate in CONDITIONS.items():
        for lang in langs_to_ablate:
            base_acc = baseline[lang]["accuracy"]
            clamp_acc = all_results["clamped"][cond_name][lang]["accuracy"]
            dir_acc = all_results["directional"][cond_name][lang]["accuracy"]
            print(f"{cond_name:<22} {lang:>4} {base_acc:>8.1%} {clamp_acc:>8.1%} {dir_acc:>8.1%}")


if __name__ == "__main__":
    main()
