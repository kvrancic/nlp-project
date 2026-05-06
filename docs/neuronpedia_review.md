# Neuronpedia Manual Review — Phase 3a Ablation Features

We need to manually verify which features in the Phase 3a ablation set are *actually* language-specific
according to Neuronpedia's auto-interpretation. The Phase 3a result (capacity competition,
ablating each language's "language" features releases reasoning features at L17/L22/L29)
is only mechanistically meaningful if the ablated features are genuinely language features.

## Source

- **Layer:** 17 (primary ablation layer)
- **SAE:** `gemma-scope-2-4b-it-res`, width 16k, l0 medium (Gemma Scope 2)
- **Selection logic:** confirmed-LANGUAGE (Phase 2b causal) → A∩B (Method A ∩ Method B from Phase 1) → Method-A backfill, capped at 20 per language.
- **Neuronpedia URL pattern (try first):** `https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/<FEATURE>`
  If 404s, try variants: `gemma-2-4b-it`, `17-gemmascope-res-16k`. The model + SAE hosting on Neuronpedia for Gemma Scope 2 may differ — check what loads.

## Review protocol

For each feature below:
1. Open Neuronpedia URL.
2. Read the auto-interp label and look at the top-activating tokens.
3. Mark in the `Verdict` column:
   - `LANG-<X>` if it's clearly specific to language X (e.g. "Spanish words", "Latin script tokens", "Devanagari script", "Mandarin")
   - `LANG-multi` if it's about *multiple* specific languages (still language but not target-specific)
   - `MIXED` if partially language but also code/numbers/punctuation
   - `NON-LANG` if it's reasoning, math, code, formatting, BOS, attention sink, position, etc.
   - `?` if Neuronpedia has no data or the label is ambiguous

The cleanest result for the paper would be: drop everything except `LANG-<target>` matches, then re-run Phase 3a on those subsets.

Causal-test labels in `[brackets]` are from Phase 2b's ablation-and-perplexity test (gold-standard for our pipeline; Neuronpedia is a useful second opinion).

---

## en (target: English)

| # | Feature | Phase 2b causal | Neuronpedia verdict | Notes |
|---|---------|-----------------|---------------------|-------|
| 1 | 153   | LANGUAGE | | |
| 2 | 166   | LANGUAGE | | |
| 3 | 203   | LANGUAGE | | |
| 4 | 443   | LANGUAGE | | |
| 5 | 486   | LANGUAGE | | |
| 6 | 513   | – | | |
| 7 | 565   | – | | |
| 8 | 812   | – | | |
| 9 | 864   | – | | |
| 10 | 866   | – | | |
| 11 | 909   | – | | |
| 12 | 2127  | – | | |
| 13 | 2346  | – | | |
| 14 | 3836  | – | | |
| 15 | 4040  | – | | |
| 16 | 6854  | – | | |
| 17 | 7723  | – | | |
| 18 | 9983  | – | | |
| 19 | 12174 | – | | |
| 20 | 34    | – | | shared with es |

---

## zh (target: Chinese)

| # | Feature | Phase 2b causal | Neuronpedia verdict | Notes |
|---|---------|-----------------|---------------------|-------|
| 1 | 154   | LANGUAGE | | also in bn ablation set |
| 2 | 828   | LANGUAGE | | |
| 3 | 2037  | LANGUAGE | | |
| 4 | 1349  | SHARED | | also in sw ablation set |
| 5 | 5298  | SHARED | | |
| 6 | 12299 | – | | |
| 7 | 48    | – | | shared with es |
| 8 | 382   | – | | shared with es |
| 9 | 225   | – | | shared with es |
| 10 | 234   | – | | |
| 11 | 3     | – | | shared with es |
| 12 | 346   | – | | shared with es |
| 13 | 50    | – | | shared with es |
| 14 | 13    | – | | shared with es |
| 15 | 133   | – | | |
| 16 | 90    | – | | |
| 17 | 464   | – | | |
| 18 | 43    | – | | |
| 19 | 143   | – | | |
| 20 | 331   | – | | |

---

## es (target: Spanish)

| # | Feature | Phase 2b causal | Neuronpedia verdict | Notes |
|---|---------|-----------------|---------------------|-------|
| 1 | 107   | LANGUAGE | | |
| 2 | 279   | LANGUAGE | | |
| 3 | 471   | LANGUAGE | | |
| 4 | 2508  | JUNK | | known not-language |
| 5 | 5451  | REASONING | | known not-language |
| 6 | 5641  | – | | shared with sw |
| 7 | 6364  | – | | shared with sw |
| 8 | 8252  | – | | |
| 9 | 12712 | – | | |
| 10 | 225   | – | | shared with zh |
| 11 | 48    | – | | shared with zh |
| 12 | 497   | – | | |
| 13 | 50    | – | | shared with zh |
| 14 | 382   | – | | shared with zh |
| 15 | 346   | – | | shared with zh |
| 16 | 146   | – | | shared with sw |
| 17 | 489   | – | | |
| 18 | 13    | – | | shared with zh |
| 19 | 3     | – | | shared with zh |
| 20 | 34    | – | | shared with en |

---

## bn (target: Bengali)

| # | Feature | Phase 2b causal | Neuronpedia verdict | Notes |
|---|---------|-----------------|---------------------|-------|
| 1 | 154   | SHARED | | also in zh ablation set |
| 2 | 883   | SHARED | | |
| 3 | 898   | SHARED | | |
| 4 | 1008  | SHARED | | |
| 5 | 1066  | SHARED | | |
| 6 | 1301  | – | | |
| 7 | 1944  | – | | |
| 8 | 2390  | – | | |
| 9 | 2706  | – | | |
| 10 | 3565  | – | | |
| 11 | 4070  | – | | |
| 12 | 4213  | – | | |
| 13 | 5050  | – | | |
| 14 | 5660  | – | | |
| 15 | 13513 | – | | |
| 16 | 13823 | – | | |
| 17 | 421   | – | | |
| 18 | 63    | – | | |
| 19 | 1073  | – | | |
| 20 | 379   | – | | |

NOTE: bn has **zero** Phase 2b-confirmed LANGUAGE features at L17. Especially urgent to flag any
that Neuronpedia clearly tags as "Bengali" / "Devanagari script" so we have at least 1–2 confirmed
features to put bn in the controls run; otherwise bn is omitted from the cleanest control variant.

---

## sw (target: Swahili)

| # | Feature | Phase 2b causal | Neuronpedia verdict | Notes |
|---|---------|-----------------|---------------------|-------|
| 1 | 356   | LANGUAGE | | |
| 2 | 728   | LANGUAGE | | |
| 3 | 280   | SHARED | | |
| 4 | 659   | SHARED | | |
| 5 | 1349  | SHARED | | also in zh ablation set |
| 6 | 5641  | – | | shared with es |
| 7 | 5838  | – | | |
| 8 | 6364  | – | | shared with es |
| 9 | 7607  | – | | |
| 10 | 9303  | – | | |
| 11 | 14587 | – | | |
| 12 | 146   | – | | shared with es |
| 13 | 96    | – | | **the "released reasoning" feature in the Phase 3a headline — verify it is NOT released by its own ablation** |
| 14 | 242   | – | | |
| 15 | 1263  | – | | |
| 16 | 268   | – | | |
| 17 | 510   | – | | |
| 18 | 351   | – | | |
| 19 | 102   | – | | |
| 20 | 409   | – | | |

---

## What to fill in for the controls notebook

After review, fill in the dict below in `notebooks/05b_capacity_controls.ipynb` cell `LANGUAGE_CONFIRMED`:

```python
LANGUAGE_CONFIRMED = {
    'en': [...],   # features marked LANG-en in this doc
    'zh': [...],   # features marked LANG-zh
    'es': [...],   # features marked LANG-es
    'bn': [...],   # features marked LANG-bn (could be empty)
    'sw': [...],   # features marked LANG-sw
}
```

The controls notebook will then run all five variants (confirmed-only, random k=20, top-A only, etc.) and report whether the feature-96-release headline survives or was an artifact of contamination.
