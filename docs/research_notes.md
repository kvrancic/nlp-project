# Research Notes

Detailed methodology dump for context preservation across sessions.

---

## 1. Zhao et al. (2025) — "When Less Language is More"

**Paper:** arXiv:2505.15257v2 (NeurIPS 2025)
**Code:** https://github.com/MuyuenLP/Language-Reasoning-Disentangle

### Method: Language Subspace Probing (Algorithm 1)

**Step 1:** Collect mean hidden states per language.
- For each language l, compute: `m_l = (1/n) * sum_i(e^i_l)`
- `e^i_l` = final-token hidden state from i-th sample in language l
- They used 7,500 MATH problems translated to 10 languages via Google Translate
- Concatenate: `M ∈ R^{d × L}` (columns are per-language means)

**Step 2:** SVD decomposition.
```
M'_a = column mean of M                    # language-agnostic
M'_s, S, Gamma' = Top-r SVD(M - M'_a * 1^T) # language-specific basis
M' = M'_a * 1^T + M'_s * diag(S) * Gamma'^T  # reconstruction
```

**Step 3:** Re-orthogonalize.
```
M_a = normalize(M' @ 1)                     # clean agnostic direction
M_s, _, Gamma = Top-r SVD(M' - M_a * 1^T)   # clean specific basis
```

**Step 4:** Inference-time projection.
```
h_hat = h - λ * M_s @ M_s^T @ h
```
- λ ∈ [0, 0.4] at middle layers (remove language)
- λ ∈ [-0.4, 0] at higher layers (re-inject for fidelity)
- Only applied to the final input token at each layer

### Models Tested (all IT variants, none Gemma)
1. Qwen-2.5-Instruct-3B (36 layers, middle=12-26, higher=27-35)
2. Qwen-2.5-Instruct-7B (28 layers, middle=10-19, higher=20-27)
3. Qwen-3-1.7B-Thinking
4. Qwen-3-4B-Thinking
5. Qwen-3-8B-Thinking
6. DeepSeek-R1-Distill-Qwen-7B
7. DeepSeek-R1-Distill-LLaMA-8B (32 layers, middle=12-22, higher=23-31)
8. DeepSeek-R1-Distill-Qwen-14B
9. GLM-Z1-9B
10. QwQ-32B

### Key Results on MGSM
- Average improvement: +0.5 to +3.5 points across models
- Largest gains on low-resource languages (Swahili: sometimes doubles)
- English stays stable or slightly improves
- Russian on Qwen-2.5-3B: +12.8 points
- Uses vLLM for inference, GlotLID for language detection

### Important for us
- They did NOT test on Gemma — our application is novel
- Their layer ranges suggest middle ~30-70% depth, higher ~70-97%
- For Gemma 3 4B (34 layers): middle ≈ 10-23, higher ≈ 24-32
- Rank r < L (number of languages). With L=5, try r ∈ {2, 3, 4}

---

## 2. Deng et al. (2025) — "Unveiling Language-Specific Features"

**Paper:** arXiv:2505.05111v2 (ACL 2025)
**Code:** https://github.com/Aatrox103/multilingual-llm-features

### Monolinguality Metric
```
ν_s^L = μ_s^L - γ_s^L

μ_s^L = mean activation of feature s on language L data
γ_s^L = mean activation of feature s across all OTHER languages
```
- Higher ν = more language-specific
- Computed on first 100 data points from Flores-10

### Feature Identification
- Used Gemma 2 2B/9B with Gemma Scope (v1) SAEs
- Residual stream SAEs
- For Gemma Scope, chose SAE with **second smallest L0 value** per layer
- Top-1 feature per language typically has dramatically higher ν than the rest
- Some languages have **synergistic features** — ablating together yields greater effect

### Directional Ablation
```
x' = x - d̂ * d̂^T * x
```
where d̂ = unit vector of SAE decoder column (W_dec[feature_idx] normalized)

- Ablated top-2 language-specific features per language per layer
- Measured cross-entropy loss change on target language vs others
- Key finding: ablation impacts CE loss much more for target language

### Steering Vectors
- Used language-specific feature activations as gating signal
- Applied steering vector only to tokens where features fire (non-zero activation)

---

## 3. Gemma Scope 2 — SAEs for Gemma 3

**Technical paper:** storage.googleapis.com/deepmind-media/...Gemma_Scope_2_Technical_Paper.pdf
**Released:** December 19, 2025

### Available SAEs for Gemma 3 4B IT

| Type | Layers | Widths | L0 targets |
|------|--------|--------|------------|
| SAE (all sites) | All 34 | 16k, 256k | ~10, ~100 |
| SAE (subset) | {9,17,22,29} | 16k, 64k, 256k, 1m | ~10, ~50, ~150 |
| Transcoder | All 34 | 16k, 256k | ~10, ~100 |
| Crosscoder | {9,17,22,29} | 64k, 256k, 512k, 1m | ~50, ~150 |

Sites: `resid_post`, `attn_out`, `mlp_out`

### HuggingFace Repos
- PT residual: `google/gemma-scope-2-4b-pt-res`
- IT residual: `google/gemma-scope-2-4b-it-res`
- Also: `-att`, `-mlp` variants

### SAELens Loading
```python
from sae_lens import SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="google/gemma-scope-2-4b-it-res",
    sae_id="layer_9/width_16k/average_l0_50",
    device="cuda",
)
# Encode: features = sae.encode(activation)  # (batch, d_sae)
# Decode: reconstruction = sae.decode(features)  # (batch, d_model)
# Decoder weights: sae.W_dec  # (d_sae, d_model)
```

**IMPORTANT:** The exact sae_id format needs verification against the actual HF repo. The naming convention may differ from what's documented. Check the repo contents before running.

### Architecture Notes (Gemma 3 4B)
- d_model = 2560
- 34 transformer layers
- Interleaved local/global attention (1 in 6 layers is global)
- GQA with QK-norm
- Vocab size: 262k (multilingual SentencePiece)

---

## 4. MGSM Dataset

**HuggingFace:** `juletxara/mgsm`
**Paper:** Shi et al. (NeurIPS 2022) "Language Models are Multilingual Chain-of-Thought Reasoners"

- 250 test problems + 8 few-shot examples per language
- Grade-school math, professionally translated from GSM8K
- 11 languages: en, es, fr, de, ru, zh, ja, th, sw, bn, te
- Our 5 targets: en, zh, es, bn, sw (3 families, 3 scripts, high-to-low resource)

```python
from datasets import load_dataset
ds = load_dataset("juletxara/mgsm", "bn", split="test")
# Fields: question, answer, answer_number
```

---

## 5. nnsight API

**Paper:** ICLR 2025
**Docs:** nnsight.net

### Basic Usage with Gemma 3
```python
from nnsight import NNsight
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", ...)
nn_model = NNsight(model)

with nn_model.trace(inputs, scan=False, validate=False):
    # Access residual stream at layer 9
    resid = nn_model.model.layers[9].output[0]
    saved_resid = resid.save()

    # Modify residual stream
    resid[:, -1, :] = modified_activation

    # Access attention output
    attn_out = nn_model.model.layers[9].self_attn.o_proj.output
```

### Key Patterns
- `scan=False, validate=False` needed for Gemma 3 (non-standard architecture)
- `.save()` to capture values
- Direct assignment to modify activations
- For generation with interventions, use PyTorch forward hooks instead

---

## 6. Professor's Feedback (Key Methodological Points)

1. **Three identification methods needed:**
   - Monolinguality metric (correlational, Deng et al.)
   - Supervised language probe (train classifier on SAE activations)
   - Causal ablation (ablate feature, check arithmetic vs perplexity)

2. **Disagreement between methods is a finding** — investigate what the feature actually is

3. **Causal definition is strongest:** A "language feature" should hurt perplexity in that language but not arithmetic when ablated

4. **Auto-interp labels from Neuronpedia** as additional validation

5. **"Must be published if done right"** — high quality bar, but negative results also valuable

---

## 7. Key Reference Code Repos

| What | URL |
|------|-----|
| Zhao et al. baseline | https://github.com/MuyuenLP/Language-Reasoning-Disentangle |
| Deng et al. features | https://github.com/Aatrox103/multilingual-llm-features |
| SAELens | https://github.com/jbloomAus/SAELens |
| nnsight | https://github.com/ndif-team/nnsight |
| circuit-tracer | https://github.com/anthropics/circuit-tracer |
| Gemma Scope 2 demo | https://www.neuronpedia.org/gemma-scope-2 |
| Gemma 3 + SAE blog | https://www.ibigford.dev/blog/gemma3-behavioral-circuits-sparse-autoencoders |
