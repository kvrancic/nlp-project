# Project: Language-Reasoning Interference via SAEs

## What This Is
MIT 6.8610 project analyzing why language-specific representations interfere with reasoning in multilingual LLMs, using Sparse Autoencoders on Gemma 3 4B IT.

## Stack
- Model: `google/gemma-3-4b-it`
- SAEs: Gemma Scope 2 via SAELens (`google/gemma-scope-2-4b-it-res`)
- Interventions: nnsight (NOT TransformerLens — it doesn't support Gemma 3)
- Compute: Colab Pro+ (A100) for experiments, local 36GB M3 Pro for development

## Code Conventions
- All source modules in `src/`
- Notebooks in `notebooks/` (numbered, Colab-ready)
- Scripts in `scripts/` for batch execution
- Results saved as .pt tensors in `results/`
- LaTeX compiled in Overleaf, not locally

## Key Details
- MGSM dataset: `juletxara/mgsm`, 5 languages: en, zh, es, bn, sw
- SAE layers: {9, 17, 22, 29} for 64k width, all 34 layers for 16k
- Gemma 3 4B has d_model=2560, 34 transformer layers
- Use BF16 precision throughout

## Running
- `pip install -e .` to install in dev mode
- Set HF_TOKEN in `.env` before running anything
- Main experiments run on Colab; upload notebooks directly
