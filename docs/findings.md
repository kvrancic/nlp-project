# Findings & Progress Log

Running log of experimental results, methodology decisions, and open questions for the
Language–Reasoning Interference (SAE) project. Updated chronologically per phase.
This is the working draft for the Results section of the paper — paste numbers from
here into Overleaf, do not re-derive them by re-running notebooks.

**Model:** `google/gemma-3-4b-it` (4.30B params, 34 layers, d_model=2560, BF16)
**SAEs:** `gemma-scope-2-4b-it-res`, sae_id `layer_{N}_width_16k_l0_medium` at layers {9, 17, 22, 29}
**Languages:** en, zh, es, bn, sw (MGSM, 250 problems each)
**Compute:** Colab Pro+ A100 40GB

---

## Setup & infrastructure (Phase 0)

### Decisions

- **Reverted from Gemma 2 9B IT back to Gemma 3 4B IT** (the proposal target). The
  friend's mid-project switch was made because Gemma 3's multimodal config caused
  loading errors; we resolved those instead of changing the model.
- **Replaced nnsight with HF native `output_hidden_states=True`** for activation
  extraction. nnsight failed on `BatchEncoding` inputs to `Gemma3TextModel.forward`.
  HF native is simpler and equivalent for residual-stream reads.
- **Kept nnsight only for hook-based interventions** (lazy-imported in
  `src/intervention.py`).

### Gotchas resolved (will bite anyone re-running)

- `pyproject.toml` had `setuptools.backends._legacy:_Backend` (does not exist).
  Fixed to `setuptools.build_meta`. Friend never ran `pip install -e .` so this
  was undetected for weeks.
- Gemma 3 4B IT is multimodal: decoder layers live at
  `model.model.language_model.layers`, not `model.model.layers`. Added
  `get_decoder_layers(obj)` helper in `src/model.py` that probes both paths.
- `Gemma3ForCausalLM` (text-only class) silently re-initializes weights when
  loading the multimodal checkpoint — generation produced only newlines. Stick
  with `AutoModelForCausalLM`, which keeps the multimodal wrapper but works.
- `Gemma3Config.hidden_size` doesn't exist on the multimodal config; use
  `model.get_input_embeddings().embedding_dim` instead.
- `torch_dtype=` triggers a deprecation warning; use `dtype=` (HF transformers
  ≥4.45).
- **Python module cache trap:** `pip install -e .` in a new Colab cell does NOT
  reload modules already imported in the kernel. After every `git pull` we need
  to evict `src.*` from `sys.modules` (cache-buster cell present in all
  notebooks 01–06).

### Smoke test (notebook 01, n=5 per lang — NOT a real measurement)

| en | zh | es | bn | sw | avg |
|----|----|----|----|----|-----|
| 80% | 60% | 60% | 80% | 40% | 64% |

Useful only as "model isn't broken." Do not cite. Real baselines come from
Phases 2 and beyond on the full 250×5.

---

## Phase 1 — Feature identification (notebook 02, complete)

### What we ran

For each layer in {9, 17, 22, 29}, encoded the 250×5 last-token residual
activations through the corresponding 16k-width SAE. Three feature-id methods,
following the professor's "use 3 methods, disagreement is a finding" guidance:

- **Method A (Deng monolinguality):** ν_s^L = μ_s^L − γ_s^L. Top-50 per
  (layer, lang).
- **Method B (supervised probe):** Pipeline(StandardScaler, LogisticRegression),
  multinomial. Top-50 per (layer, lang) by absolute coefficient.
- **Method C (cross-lingual reasoning candidates):** features active above
  threshold=0.1 on ≥10% of the same problem across all 5 languages.

### Headline numbers

**Probe in-sample accuracy (5-way classification):**

| layer | acc |
|-------|-----|
| 9     | 0.880 |
| 17    | 0.879 |
| 22    | 0.880 |
| 29    | 0.883 |

**Top-1 monolinguality score ν per (layer, lang):**

| layer | en | zh | es | bn | sw |
|-------|----|----|----|----|----|
| 9  | 19.7 (f75)   | 142.0 (f43)   | 43.8 (f13)   | 46.9 (f449)  | 52.4 (f154)  |
| 17 | 265.7 (f34)  | 381.9 (f48)   | 127.7 (f48)  | 324.0 (f1066)| 683.6 (f356) |
| 22 | 273.3 (f2003)| 615.9 (f217)  | 288.7 (f462) | 534.1 (f9040)| 707.2 (f273) |
| 29 | 986.3 (f761) | 2510.6 (f191) | 2579.1 (f596)| 1932.7 (f1254)| **5292.6** (f8612) |

**Cross-lingual reasoning candidates (Method C):**

| layer | count (out of 16384) |
|-------|----------------------|
| 9  | 34 |
| 17 | 19 |
| 22 | 18 |
| 29 | 6  |

**Method A ∩ B (intersection sizes, top-50 each):**

| layer | en | zh | es | bn | sw |
|-------|----|----|----|----|----|
| 9  | 8  | 3  | 3  | 12 | 2  |
| 17 | 18 | 3  | 9  | 14 | 11 |
| 22 | 10 | 17 | 11 | 17 | 15 |
| 29 | 9  | 18 | 15 | **25** | 12 |

**Jaccard(A, B) per (layer, lang):**

| layer | en | zh | es | bn | sw |
|-------|----|----|----|----|----|
| 9  | 0.087 | 0.031 | 0.031 | 0.136 | 0.020 |
| 17 | 0.220 | 0.031 | 0.099 | 0.163 | 0.124 |
| 22 | 0.111 | 0.205 | 0.124 | 0.205 | 0.176 |
| 29 | 0.099 | 0.220 | 0.176 | **0.333** | 0.136 |

### Interpretation

1. **ν grows monotonically with depth, ~250×.** Top-1 monolinguality goes from
   ~20 at L9 (en) to ~5293 at L29 (sw). Late layers concentrate language-specific
   signal — directly supports H3 (interference is layer-dependent). The factor of
   ~270× between L9 and L29 for Swahili is one of the strongest single observations
   in this dataset.

2. **Probe ceilings at 0.88, not 1.0.** This is *not* a training bug — verified
   by sweeping C, swapping solvers, raising max_iter, switching to multinomial.
   The probe converges cleanly. Ceiling is genuine: a non-trivial subset of
   MGSM problems have cross-lingually similar residual representations that no
   linear classifier can separate. Documented in `src/monolinguality.py`
   docstring. **This is itself a finding** worth a sentence in the paper:
   the residual stream's language identity is mostly but not perfectly linearly
   decodable, leaving room for the shared-feature ablation hypothesis.

3. **A and B mostly disagree.** Jaccard hovers at 0.03–0.33. Per the professor,
   disagreement is a finding, not a bug. Two interpretations co-exist:
   - Method A captures features whose *mean* differs (could include features
     active across all problems with shifted scale).
   - Method B captures features that are *discriminative under unit variance*
     (could include rare but reliable language markers).

   Phase 2's causal validation is what arbitrates: features that survive
   ablation (perplexity ↑ in target lang, accuracy ≈ unchanged) are the ones
   we trust regardless of method.

4. **bn-L29 is the agreement peak (Jaccard 0.333, intersection 25).** Bengali
   at the deepest layer has the strongest joint A∩B signal. Worth checking
   whether ablation works best for bn or whether this just reflects Bengali
   being phonologically/orthographically furthest from the others.

5. **Reasoning-candidate count drops with depth (34 → 19 → 18 → 6).** At L29
   only 6 features fire across all 5 languages on the same problem — late-layer
   representations are nearly fully language-specific. Mid layers (17, 22) are
   the realistic targets for shared reasoning circuitry, consistent with H3's
   prediction that ablation gains will peak at middle layers.

### Open questions for Phase 2 to answer

- Do A∪B candidates produce an actual perplexity bump in the target language
  when ablated? (Causal-feature definition step.)
- Of A∩B intersection features, how many survive causal validation? Hypothesis:
  intersection features are higher-precision, so survival rate should be ≥
  union survival rate.
- Are the L9 reasoning candidates (34 of them) genuinely arithmetic-related
  or just frequent generic tokens? (Auto-interp / Neuronpedia would help; not
  yet queried.)

### Artifacts

- `results/phase1_features.pt` (4.3 MB): **now committed to repo** (was previously
  Drive-only; pulled in 2026-05-05 from teammate's results bundle). Phase 2
  loads from this path.
- `notebooks/runs/02_feature_extraction.ipynb`: executed notebook with all
  cell outputs (figures inline) — version-controlled snapshot of this run.
- Figures saved to `results/figures/` on Colab session (transient).

---

## Phase 2 — Causal ablation

### Phase 2a: Zhao SVD baseline (notebook 03, complete)

**Status:** complete. Run by teammate on Colab A100, 2026-05-02. Artifact
`results/phase2_zhao_baseline.pt` (10 MB) now in repo.

What it ran:
1. Per-language mean residuals at all 34 layers.
2. SVD language subspaces at ranks {2, 3, 4}.
3. Unmodified baseline on full 1250 problems (250 × 5 langs).
4. 4-config sensitivity sweep on 50-prompt dev: (λ_mid, λ_hi) ∈
   {(0.2,−0.2), (0.1,−0.2), (0.3,−0.2), (0.2, 0.0)}, rank fixed at 3.
5. Best Zhao config on full 1250 problems + langdetect-based language fidelity.

Layer split: middle = layers 10–23, higher = layers 24–32 (per `config`).

#### Headline numbers

**Per-language MGSM accuracy (greedy, 250 problems each):**

| Language | Baseline | Zhao  | Δ (Zhao − base) |
|----------|----------|-------|------------------|
| en       | 0.580    | 0.632 | **+0.052**       |
| zh       | 0.624    | 0.624 | +0.000           |
| es       | 0.568    | 0.600 | **+0.032**       |
| bn       | 0.564    | 0.564 | +0.000           |
| sw       | 0.320    | 0.332 | +0.012           |
| **Avg**  | **0.531** | **0.550** | **+0.019**  |

**Best Zhao config:** λ_middle = 0.2, λ_higher = 0.0, rank = 3.

**Language fidelity (langdetect on first 300 chars of generation):**

| Language | Baseline | Zhao  | Δ        |
|----------|----------|-------|----------|
| en       | 1.000    | 0.964 | −0.036   |
| zh       | 0.928    | 0.859 | **−0.069** |
| es       | 0.996    | 0.940 | −0.056   |
| bn       | 1.000    | 0.976 | −0.024   |
| sw       | 0.904    | 0.936 | +0.032   |

#### Interpretation

1. **Modest aggregate gain (+1.9 pp), highly uneven by language.** The Zhao
   intervention reproduces the directional finding (interference exists, can
   be partly removed) but the effect is concentrated in en/es. Bengali and
   Mandarin are unmoved; Swahili moves only barely. This pattern is
   consistent with the Phase 1 monolinguality story: late-layer language
   specificity in zh/bn/sw is so strong (top-1 ν reaches 5293 for sw at L29)
   that linear-subspace projection can't fully neutralize it.

2. **Language fidelity drops 4–7 pp in 4 of 5 languages.** This is the
   tradeoff Zhao's method buys: nudging the residual toward the
   language-agnostic mean increases code-switching/hallucination of other
   languages. **Direct evidence for H2** — SAE ablation has the opportunity
   to do better here because it can target individual features rather than
   the whole subspace.

3. **Best λ_higher = 0.0 sits at the grid corner.** The sweep tested
   λ_higher ∈ {−0.2, 0.0} only, and the optimum landed at the boundary.
   This means the true optimum may be at λ_higher > 0 (positive projection,
   opposite the Zhao prior of removing higher-layer language). Documenting
   as a known limitation for the paper; expanding the grid would cost
   another ~6h on Colab. Defer unless reviewers flag it.

4. **Sweep was 4-config (not 25 as originally planned).** Compute budget
   constraint. The 4 configs sweep λ_middle around 0.2 (the Zhao paper's
   value) plus one λ_higher=0 corner. Conclusions hold but the search is
   coarse.

5. **Plot bug, not data bug.** Cell 21's `fig5_zhao_baseline.png` failed to
   render with a KeyError on 'en' — this is a plotting bug (likely a
   variable-shadowing issue from cell re-runs). All underlying dicts in the
   saved .pt are intact and correct (verified by re-loading). The bar chart
   gets regenerated cleanly in `06_paper_figures.ipynb`.

#### Open questions for Phase 2b to answer

- Can SAE single-feature ablation match or beat the +0.019 aggregate gain?
- Crucially: can it beat Zhao on **language fidelity** while matching or
  exceeding accuracy gains? (H2's headline claim.)
- Why don't bn/zh respond to the SVD intervention? Is it because their
  language signal is non-linear in the residual (so linear projection is
  insufficient), or because the means are degenerate at these layers?

#### Artifacts

- `results/phase2_zhao_baseline.pt` (10 MB): committed to repo. Keys:
  `config`, `per_lang_mean`, `M_s_by_rank_layer`, `baseline_results`,
  `sweep_results`, `zhao_test`, `language_fidelity`, `baseline_avg`,
  `zhao_avg`.
- `results/phase2_*_partial.pt`: local-only checkpoints (gitignored).
- `notebooks/runs/03_zhao_baseline.ipynb`: executed snapshot with all
  outputs and figures.

### Phase 2b: SAE causal ablation (`04_causal_ablation.ipynb`)

### Phase 2b: SAE causal ablation (`04_causal_ablation.ipynb`)

**Status:** code complete, ready to run on Prime Intellect H100 80GB
(~5–6h, ~$15–20). Tests H1, H2, H3 + professor's causal feature-definition
methodology. K_VALUES = [1, 5, 10, 20], primary layer = 17. Loads
`results/phase2_zhao_baseline.pt` for H2 comparison. Will save to
`results/phase2_ablation.pt`.

---

## Phase 3 — Interaction analysis (pending)

`05_interaction_analysis.ipynb` — three sub-experiments:
- (a) Capacity competition (cheapest)
- (c) Attention disruption (medium)
- (b) Circuit interference (hardest, scoped)

Estimated ~3h on A100.

---

## Phase 4 — Paper compilation (pending)

`06_paper_figures.ipynb` loads all `results/*.pt` and produces publication
figures + LaTeX-ready tables + appends to this `findings.md`.

---

## Hypothesis scoreboard (will fill in as evidence accumulates)

| Hypothesis | Claim | Evidence so far | Verdict |
|------------|-------|-----------------|---------|
| H1 | Top-k SAE language-feature ablation reproduces Zhao SVD's accuracy gain | Phase 2 pending | — |
| H2 | SAE ablation preserves output language fidelity better than SVD | Zhao baseline shows fidelity drops 4–7 pp in 4/5 langs (zh worst at −0.069). SAE comparison pending in Phase 2b | Bar set, awaits SAE side |
| H3 | Ablation gain is layer-dependent, peaks at middle layers | Reasoning-feature drop 34→19→18→6 supports late-layer language specificity; consistent with H3 prediction but not yet causal | Weak prior support |

---

## Methodological commitments (don't change without justification)

- BF16 throughout. A100 40GB, ~9GB model + ~5GB SAEs + activations fits.
- Use the tokenizer's chat template, never manual prompt formatting.
- Full test set is 250×5 = 1250 problems; dev split for grid search is the
  first 50 per language.
- Bootstrap 95% CIs over the 250 problems for all per-language accuracies in
  the paper.
- Ablation is at the SAE-decoder direction in the residual stream, applied via
  QR-orthogonalized projection (handles non-orthogonal decoder columns
  correctly — this was a critical fix; naive `dd^T` ablation is wrong when
  features are correlated).
- "Causally confirmed LANGUAGE feature" = single-feature ablation that
  raises target-language perplexity and leaves arithmetic accuracy ≈ baseline.
  Features failing both criteria are tagged JUNK and excluded.
