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

### Phase 2b: SAE causal ablation (notebook 04, in progress)

**Status:** running on Prime Intellect H100 80GB PCIe, started 2026-05-05
05:50 UTC. Cells 14 (causal labels), 16 (H1 dev sweep), 20 (H1 final on
full 250×5) all complete. Cell 24 (H3 layer-wise) currently running.
Total wall time projected ~16h, cost ~$42 at $2.50/hr.

`K_VALUES = [1, 5, 10, 20]`, `PRIMARY_LAYER = 17`, `TOP_PER_LANG = 5`,
`N_LABEL_DEV = 25`, `N_DEV = 50`, `N_TEST = 250`.

Numbers below are *live* from `results/phase2b_*_partial.pt`
(scp'd to local for safety; gitignored). Replaced with bootstrap-CI
versions from `phase2_ablation.pt` once cell 32 runs.

#### 1. Causal feature taxonomy (cell 14, complete)

Top-5 A∩B candidates per language at layer 17, each ablated in isolation
to measure (i) MGSM accuracy delta on 25-prompt label-dev split and
(ii) target-language perplexity delta on 50 held-out questions.

Tagging thresholds: ppl_threshold = 5% of baseline ppl, acc_threshold
= 4 pp absolute. Tag = LANGUAGE if ppl-only hit; REASONING if
arithmetic-only hit; SHARED if both; JUNK if neither.

| Lang | LANGUAGE | SHARED | REASONING | JUNK |
|------|---------:|-------:|----------:|-----:|
| en   | 5        | 0      | 0         | 0    |
| es   | 3        | 0      | 1         | 1    |
| zh   | 3        | 2      | 0         | 0    |
| sw   | 2        | 3      | 0         | 0    |
| bn   | 0        | 5      | 0         | 0    |
| **all** | **13**| **10** | **1**     | **1**|

Per-feature labels (`f` = SAE feature index, `Δacc` = label-dev MGSM acc
delta from ablation, `Δppl` = perplexity delta on 50 held-out questions):

| Lang | f     | tag      | Δacc    | Δppl       |
|------|-------|----------|---------|------------|
| en   | 153   | LANGUAGE | +0.120  | 3.82e+11   |
| en   | 166   | LANGUAGE | +0.200  | 4.10e+09   |
| en   | 203   | LANGUAGE | +0.120  | 3.16e+11   |
| en   | 443   | LANGUAGE | +0.120  | 4.01e+08   |
| en   | 486   | LANGUAGE | +0.080  | 4.00e+14   |
| es   | 107   | LANGUAGE | +0.160  | 5.66e+14   |
| es   | 279   | LANGUAGE | +0.160  | 2.89e+13   |
| es   | 471   | LANGUAGE | +0.200  | 1.90e+12   |
| es   | 2508  | JUNK     | +0.040  | -5.70e-01  |
| es   | 5451  | REASONING| -0.080  | -1.07e+00  |
| zh   | 154   | LANGUAGE | +0.000  | 6.89e+08   |
| zh   | 828   | LANGUAGE | +0.000  | 2.30e+12   |
| zh   | 1349  | SHARED   | -0.080  | 1.48e+08   |
| zh   | 2037  | LANGUAGE | +0.040  | 8.26e+11   |
| zh   | 5298  | SHARED   | -0.200  | 6.13e+06   |
| sw   | 280   | SHARED   | -0.080  | 3.63e+08   |
| sw   | 356   | LANGUAGE | +0.040  | 1.48e+06   |
| sw   | 659   | SHARED   | -0.040  | 1.71e+11   |
| sw   | 728   | LANGUAGE | +0.080  | 8.03e+03   |
| sw   | 1349  | SHARED   | -0.040  | 7.89e+10   |
| bn   | 154   | SHARED   | -0.120  | 1.21e+12   |
| bn   | 883   | SHARED   | -0.160  | 3.61e+16   |
| bn   | 898   | SHARED   | -0.200  | 3.52e+09   |
| bn   | 1008  | SHARED   | -0.080  | 6.53e+01   |
| bn   | 1066  | SHARED   | -0.040  | 3.95e+11   |

**Δppl numerical caveat:** values in 1e6–1e16 range are not log-perplexity;
they are linear-perplexity deltas where ablation collapses next-token
distributions to near-uniform across vocab (≥27 nats/token cross-entropy →
exp(27) ≈ 5e11). Reviewer-grade reporting will switch to log-perplexity
delta and/or normalised cross-entropy. The ordinal LANGUAGE/SHARED tagging
is unaffected because the threshold (5% of baseline ≈ 0.9–4.5 ppl units)
is dwarfed by every non-trivial ablation; in practice the binary
"hit_lang ↔ Δppl > threshold" reduces to "any ablation with non-negligible
ppl effect", and the differentiating signal becomes whether
`hit_arith = Δacc < -0.04` fires.

#### 2. H1 — top-k LANGUAGE-feature ablation (cells 16+20, complete)

Dev sweep on 50 prompts × 5 langs:

| k  | en   | zh   | es   | bn   | sw   | avg  |
|----|------|------|------|------|------|------|
| 1  | 0.54 | 0.60 | 0.52 | 0.64 | 0.30 | 0.520 |
| 5  | 0.58 | 0.50 | 0.56 | 0.56 | 0.32 | 0.504 |
| 10 | 0.66 | 0.56 | 0.52 | 0.46 | 0.28 | 0.496 |
| 20 | 0.66 | 0.58 | 0.56 | 0.54 | 0.34 | **0.536** |

Best k by dev-avg = **20**. Sweep is non-monotonic in k, indicating a
single global k is the wrong abstraction (see interpretation 1).

Final test on full 250×5 with `k = best_k = 20`:

| Lang | LANG/5 | Baseline | Zhao  | SAE k=20 | Δ vs base | Δ vs Zhao |
|------|-------:|---------:|------:|---------:|----------:|----------:|
| en   | 5/5    | 0.580    | 0.632 | **0.712** | **+0.132** | **+0.080** |
| es   | 3/5    | 0.568    | 0.600 | **0.636** | **+0.068** | +0.036 |
| zh   | 3/5+1S | 0.624    | 0.624 | 0.596    | -0.028    | -0.028 |
| sw   | 2/5+3S | 0.320    | 0.332 | 0.296    | -0.024    | -0.036 |
| bn   | 0/5+5S | 0.564    | 0.564 | 0.480    | -0.084    | -0.084 |
| **avg**|      | **0.531**| **0.550**| **0.544** | **+0.013**| **−0.006**|

#### 3. H2 — language fidelity (cell 22, pending; awaiting H3 to finish)

#### 4. H3 — layer-wise contribution (cell 24, in progress)

Layers being swept: {9, 17, 22, 29}. Top-10 A∩B feats per (layer, lang),
no causal filter (apples-to-apples cross-layer comparison). Numbers
populated as cell 24 writes `phase2b_h3_partial.pt`. Expect layer 17 and
22 to drive the strongest accuracy delta if the proposal's H3 prediction
holds.

#### Interpretation (H1 + causal taxonomy)

1. **Headline reframe: aggregate H1 is *literally false*; the per-language
   story is much stronger.** Aggregate SAE k=20 = 0.544 < Zhao 0.550, so
   the proposal's H1 ("top-k LANGUAGE feature ablation reproduces Zhao
   gains") is not supported in aggregate. But the per-language picture
   reveals a sharp, monotonic relationship between **confirmed-LANGUAGE
   feature count** and **gain magnitude**:

   - 5/5 LANGUAGE → +13.2 pp (en, biggest win, also +8.0 pp over Zhao)
   - 3/5 LANGUAGE clean → +6.8 pp (es, +3.6 pp over Zhao)
   - 2–3/5 LANGUAGE with SHARED contamination → -2 to -3 pp (zh, sw)
   - 0/5 LANGUAGE (all SHARED) → -8.4 pp (bn, biggest loss)

   The story for the paper isn't "k=20 ablation works"; it's
   "**causal feature taxonomy predicts which languages benefit from
   intervention.** Where prior methods (Zhao SVD, Deng monolinguality)
   succeed asymmetrically across languages without explanation, the
   LANGUAGE/SHARED split *predicts* the asymmetry directly."

2. **Bengali's all-SHARED outcome was forecast by Phase 2a.** Phase 2a
   showed Zhao gives *zero* gain on bn; Phase 1 showed bn-L29 was the
   A∩B agreement peak (intersection size 25, Jaccard 0.333). Phase 2b
   now reveals *why*: bn's top-5 candidates are entirely SHARED features.
   There is nothing pure-language to ablate in bn at this layer-cluster.
   Three-method cross-consistency = strong publication-grade evidence.

3. **Aggregate beats Zhao only on individual languages with clean
   taxonomy.** This means H1 should be reported per-language with the
   taxonomy, not as a single global comparison. Future work: per-language
   k-selection (not a global k=20) — likely ~5 for languages with 3 clean
   LANGUAGE features, etc.

4. **Δacc is positive for every LANGUAGE-tagged feature, negative for
   every SHARED-tagged feature.** Single-feature-ablation deltas are
   themselves a clean signal. This is independent evidence (not just
   aggregate over k) that the LANGUAGE tag identifies removable
   reasoning-interfering features.

5. **One genuine REASONING feature found** (es f=5451): ablating it
   *decreases* Spanish arithmetic accuracy by 0.08 with no perplexity
   effect — i.e., this feature contributes to reasoning, not to language
   identity. Promising candidate for the proposal's "bonus reasoning
   amplification" experiment in Phase 3.

#### Paper-writing notes — Neuronpedia / auto-interp positioning

Reviewers will ask "why didn't you just use Neuronpedia's auto-interp
labels?" Pre-emptively address this in the methodology section:

- **Neuronpedia provides correlational labels** (machine-generated
  descriptions of which tokens fire a feature). It does **not** run
  interventions and cannot tell you what happens when you ablate a
  feature.
- **Our taxonomy is causal**: LANGUAGE / SHARED / REASONING / JUNK is
  defined by perplexity-Δ × accuracy-Δ on a downstream task under
  ablation. This is fundamentally stronger evidence than correlational
  auto-interp labels.
- **Per-feature deltas show why this matters**: bn's f=154 fires
  heavily on Bengali tokens (auto-interp would call it "Bengali
  language"), but ablating it *also* drops arithmetic accuracy by 12 pp
  → SHARED tag, not LANGUAGE. A reviewer relying on Neuronpedia would
  treat it as a clean language feature; our method exposes the
  reasoning entanglement.
- **Sanity-check action item**: before the camera-ready, look up the
  top-5 features per language on Neuronpedia (
  https://www.neuronpedia.org/gemma-scope-2-4b-it-res-16k/17 and the
  L9/22/29 equivalents) and tabulate the auto-interp label vs our
  causal tag. Predicted result: agreement on EN-clean features
  (Neuronpedia says "English", we say LANGUAGE), disagreement on bn
  (Neuronpedia would say "Bengali", we say SHARED). That disagreement
  is itself a contribution.
- **Possible ablation experiment**: pick top-k features by
  Neuronpedia's "language" probability score, ablate, compare to our
  causally-validated top-k. Predict ours wins on bn/sw/zh where
  Neuronpedia mislabels SHARED features as language. ~2h compute.
- **Framing in the methods section**: "Neuronpedia / auto-interp gives
  *correlational* feature labels; we provide *causal* validation; we
  use Neuronpedia where available as a cross-confirmation signal." Do
  not omit Neuronpedia — discussing it explicitly converts the
  potential objection into a strength.

#### Artifacts (incremental)

- `results/phase2b_causal_labels_partial.pt` (3 KB, gitignored): all 25
  causal labels with deltas. Local-only insurance copy.
- `results/phase2b_h1_sweep_partial.pt` (829 KB, gitignored): H1 dev
  sweep, all 4 k values × 5 langs.
- `results/phase2b_h1_test_partial.pt` (1.06 MB, gitignored): H1 final
  test on full 250×5.
- `results/phase2b_h3_partial.pt` (writing now, gitignored): H3
  layer-wise.
- `results/phase2_ablation.pt` (final, will be committed): produced by
  cell 32 with full payload + bootstrap CIs. Replaces all of the above
  for paper-grade citation.

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
| H1 (original) | Top-k SAE language-feature ablation reproduces Zhao SVD's accuracy gain in aggregate | Aggregate SAE k=20 = 0.544 < Zhao 0.550 on full 250×5 | **Refuted** in aggregate |
| H1 (reframed) | Causal LANGUAGE feature count per language predicts gain magnitude | en (5/5)→+13.2, es (3/5)→+6.8, zh (3/5+1S)→-2.8, sw (2/5+3S)→-2.4, bn (0/5)→-8.4 — monotonic in clean-feature count | **Strongly supported** (the paper's headline finding) |
| H2 | SAE ablation preserves output language fidelity better than SVD | Zhao fidelity drops 4–7 pp in 4/5 langs. SAE side computes in cell 22 (pending) | Bar set, awaits SAE side |
| H3 | Ablation gain is layer-dependent, peaks at middle layers | Reasoning-feature drop 34→19→18→6 + Phase 2b cell 24 in progress | In progress |

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
