"""Fetch Neuronpedia auto-interp labels for Phase 3a ablation features.

Reads feature indices from neuronpedia_review.md and queries the Neuronpedia
API to get the auto-interpretation description for each feature.

Usage:
    python scripts/fetch_neuronpedia_labels.py
    python scripts/fetch_neuronpedia_labels.py --output results/neuronpedia_labels.json
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

# Neuronpedia API config
MODEL_ID = "gemma-3-4b-it"
SAE_ID = "17-gemmascope-2-res-16k"
API_BASE = "https://www.neuronpedia.org/api/feature"
RATE_LIMIT_DELAY = 0.5  # seconds between requests to be polite


def parse_features_from_md(md_path: Path) -> dict[str, list[int]]:
    """Parse feature indices from the neuronpedia_review.md file.

    Returns:
        Dict mapping language code to list of feature indices.
    """
    text = md_path.read_text()
    features_by_lang = {}

    # Split by language sections
    lang_pattern = re.compile(r"^## (\w+) \(target:", re.MULTILINE)
    sections = lang_pattern.split(text)

    # sections alternates: [preamble, lang1, content1, lang2, content2, ...]
    for i in range(1, len(sections), 2):
        lang = sections[i]
        content = sections[i + 1]

        # Extract feature indices from table rows: | # | <feature> | ...
        row_pattern = re.compile(r"^\|\s*\d+\s*\|\s*(\d+)\s*\|", re.MULTILINE)
        indices = [int(m.group(1)) for m in row_pattern.finditer(content)]
        features_by_lang[lang] = indices

    return features_by_lang


def fetch_feature_label(feature_index: int) -> dict:
    """Fetch the Neuronpedia auto-interp label for a single feature.

    Returns:
        Dict with 'index', 'explanation', 'pos_str' (top activating tokens).
    """
    url = f"{API_BASE}/{MODEL_ID}/{SAE_ID}/{feature_index}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Extract explanations
    explanations = data.get("explanations", [])
    if explanations:
        # Take the first (usually best/most recent) explanation
        label = explanations[0].get("description", None)
    else:
        label = None

    # Also grab top positive-activating tokens for context
    pos_str = data.get("pos_str", [])[:10]

    return {
        "index": feature_index,
        "explanation": label,
        "pos_tokens": pos_str,
        "frac_nonzero": data.get("frac_nonzero"),
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch Neuronpedia labels for review features")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path.home() / "Downloads" / "neuronpedia_review.md",
        help="Path to the neuronpedia_review.md file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/neuronpedia_labels.json"),
        help="Output JSON file path",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"Parsing features from: {args.input}")
    features_by_lang = parse_features_from_md(args.input)

    # Deduplicate feature indices (some are shared across languages)
    all_indices = sorted(set(idx for indices in features_by_lang.values() for idx in indices))
    print(f"Found {len(all_indices)} unique features across {len(features_by_lang)} languages")

    # Fetch labels
    labels = {}
    failed = []
    for i, idx in enumerate(all_indices):
        print(f"  [{i+1}/{len(all_indices)}] Feature {idx}...", end=" ", flush=True)
        try:
            result = fetch_feature_label(idx)
            labels[idx] = result
            desc = result["explanation"] or "(no explanation)"
            print(desc[:60])
        except requests.HTTPError as e:
            print(f"FAILED ({e.response.status_code})")
            failed.append(idx)
            labels[idx] = {"index": idx, "explanation": None, "pos_tokens": [], "error": str(e)}
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(idx)
            labels[idx] = {"index": idx, "explanation": None, "pos_tokens": [], "error": str(e)}

        time.sleep(RATE_LIMIT_DELAY)

    # Organize output by language
    output = {
        "metadata": {
            "model_id": MODEL_ID,
            "sae_id": SAE_ID,
            "layer": 17,
            "api_base": API_BASE,
            "n_features": len(all_indices),
            "n_failed": len(failed),
        },
        "labels": {str(k): v for k, v in labels.items()},
        "by_language": {},
    }

    for lang, indices in features_by_lang.items():
        output["by_language"][lang] = [
            {**labels.get(idx, {"index": idx, "explanation": None}), "index": idx}
            for idx in indices
        ]

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY BY LANGUAGE")
    print("=" * 80)
    for lang, indices in features_by_lang.items():
        print(f"\n--- {lang} ---")
        for idx in indices:
            info = labels.get(idx, {})
            desc = info.get("explanation") or "NO LABEL"
            tokens = info.get("pos_tokens", [])[:5]
            token_str = ", ".join(f'"{t}"' for t in tokens) if tokens else ""
            print(f"  {idx:>6}: {desc:<50} [{token_str}]")

    if failed:
        print(f"\nFailed features ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
