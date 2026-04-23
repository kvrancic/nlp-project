# From Subspaces to Features: Mechanistic Analysis of Language-Reasoning Interference via Sparse Autoencoders

**MIT 6.8610 Graduate NLP | Spring 2026**
Karlo Vrancic, Anastasia Ahani, Riley Kong

## Motivation

Removing a model's own learned representations at inference time should degrade performance. [Zhao et al. (NeurIPS 2025)](https://arxiv.org/abs/2412.09535) showed the opposite: projecting language-specific components out of LLM hidden states consistently *improves* multilingual reasoning across 10 models and 11 languages, with Swahili gains exceeding 10%. This implies that language-specific representations actively interfere with reasoning circuits. **Why this interference occurs remains mechanistically unexplained.**

Zhao et al.'s SVD-based projection operates at the subspace level -- it cannot identify *which individual features* drive the interference or explain *how* they disrupt reasoning. Meanwhile, [Deng et al. (ACL 2025)](https://arxiv.org/abs/2501.02373) showed that SAE features in multilingual LLMs naturally separate into language-specific and language-agnostic categories, and [Chou et al. (2025)](https://arxiv.org/abs/2507.13410) achieved 90% language-switching success by modifying a single SAE feature. Neither study examines reasoning.

**We bridge these two lines of work.** Using Gemma Scope 2 pre-trained SAEs on Gemma 3 4B, we:
1. Identify which individual SAE features are language-specific vs. reasoning-specific via contrastive multilingual prompts
2. Test whether targeted ablation of language-specific features reproduces Zhao et al.'s reasoning gains with greater surgical precision
3. Characterize the interference mechanism at the feature level

## Research Design

### Phase 1: Contrastive Feature Identification

Extract SAE feature activations from Gemma 3 4B processing identical MGSM math problems in five typologically diverse languages (English, Chinese, Spanish, Bengali, Swahili -- 3 language families, 3 scripts, high-to-low resource). Classify features using:

- **Monolinguality metric** (Deng et al.): For feature *s* and language *L*, compute `v_s^L = mu_s^L - gamma_s^L`, where `mu_s^L` is the mean activation in *L* and `gamma_s^L` is the mean across other languages
- **Supervised language probe**: Logistic regression predicting language from SAE feature activations
- **Cross-lingual stability**: Features activated across all five languages for the same problem identify reasoning computations

This produces a per-layer feature taxonomy: **language-specific**, **reasoning-specific**, and **shared**.

### Phase 2: Causal Validation

Test three hypotheses:
- **H1**: SAE-targeted ablation of language-specific features reproduces Zhao et al.'s reasoning improvements
- **H2**: Targeted ablation achieves a better reasoning-fidelity trade-off than SVD projection (fewer features removed, more precisely)
- **H3**: Middle-layer language features (layers 17-22, ~50-65% depth) account for the majority of interference, consistent with the three-phase processing model (early=encode language, middle=reason, late=decode)

Baselines: unmodified model, Zhao et al. SVD (replicated), random-feature ablation, Deng et al. language-only ablation.

### Phase 3: Feature Interaction Analysis

Trace information flow between language-specific and reasoning-specific features using Sparse Feature Circuits. Three competing hypotheses:
- **(a) Capacity competition**: Language features compete for representation capacity via superposition
- **(b) Circuit interference**: Language features trigger downstream circuits irrelevant to reasoning
- **(c) Attention disruption**: Language features disrupt attention patterns that route information to reasoning circuits

Additionally test whether *amplifying* reasoning-specific features via activation steering outperforms *ablating* language-specific features.

## Stack

| Component | Tool |
|---|---|
| Model | Gemma 3 4B IT (`google/gemma-3-4b-it`) |
| SAEs | Gemma Scope 2 via SAELens (`google/gemma-scope-2-4b-it-res`) |
| Interventions | nnsight + PyTorch hooks (not TransformerLens -- no Gemma 3 support) |
| Dataset | MGSM (250 problems x 5 languages, inference-only) |
| Compute | Colab Pro+ (A100) for experiments, local M3 Pro for development |
| Precision | BF16 throughout |

## Evaluation

**Metrics:**
- Feature identification: monolinguality scores, cross-language activation correlations per layer
- Causal validation: MGSM accuracy under each intervention, language fidelity (LaBSE similarity), per-language and per-layer breakdowns
- Interaction analysis: causal effect sizes, feature-to-feature attribution strengths via integrated gradients
- All results report bootstrap 95% confidence intervals over the 250 problems

## Project Structure

```
src/
  config.py          # Hyperparameters, model/SAE/dataset constants
  data.py            # MGSM loading and prompt formatting
  model.py           # Gemma 3 4B + Gemma Scope 2 SAE loading
  extraction.py      # Residual stream extraction (nnsight) + SAE encoding
  monolinguality.py  # Feature classification (monolinguality, probe, reasoning ID)
  intervention.py    # Directional ablation via PyTorch hooks
  svd_baseline.py    # Zhao et al. SVD baseline replication
  evaluation.py      # MGSM accuracy + perplexity evaluation
notebooks/
  01_setup_verify.ipynb  # Colab notebook verifying all components
docs/
  proposal.tex       # Full research proposal
  research_notes.md  # Methodology reference notes
  profs-feedback.txt # Advisor feedback
results/             # Saved .pt tensors (gitignored)
```

## Setup

```bash
# Clone and install
git clone <repo-url>
cd nlp-project
pip install -e .

# Configure HuggingFace access
# 1. Accept the Gemma 3 license at https://huggingface.co/google/gemma-3-4b-it
# 2. Create .env from template
cp .env.example .env
# 3. Add your HF_TOKEN to .env
```

For Colab, upload notebooks directly -- the setup cell handles installation and reads `HF_TOKEN` from Colab secrets.

## Key References

1. W. Zhao et al., "When Less Language is More: Language-Reasoning Disentanglement," *NeurIPS*, 2025.
2. B. Deng et al., "Unveiling Language-Specific Features in LLMs via Sparse Autoencoders," *ACL*, 2025.
3. C.-T. Chou et al., "Causal Language Control via Sparse Feature Steering," *arXiv:2507.13410*, 2025.
4. C. McDougall et al., "Gemma Scope 2," Google DeepMind, 2025.
5. F. Shi et al., "Language Models are Multilingual Chain-of-Thought Reasoners," *NeurIPS*, 2022.
6. S. Marks et al., "Sparse Feature Circuits," *ICLR*, 2025.
