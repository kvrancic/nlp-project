"""Central configuration for the project."""

from pathlib import Path

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------- Model ----------
MODEL_ID = "google/gemma-3-4b-it"
MODEL_DTYPE = "bfloat16"
D_MODEL = 2560
N_LAYERS = 34

# ---------- SAE ----------
# Gemma Scope 2 residual-stream SAEs for Gemma 3 4B IT
# SAELens release name (NOT the HuggingFace repo_id)
SAE_RELEASE_RES = "gemma-scope-2-4b-it-res"       # subset layers {9, 17, 22, 29}
SAE_RELEASE_RES_ALL = "gemma-scope-2-4b-it-res-all"  # all 34 layers

# Subset layers where 65k/262k-width SAEs are available
SAE_SUBSET_LAYERS = [9, 17, 22, 29]

# All 34 layers have 16k-width SAEs
SAE_ALL_LAYERS = list(range(N_LAYERS))

# SAE widths available (numeric values for feature counts)
SAE_WIDTH_16K = 16384
SAE_WIDTH_65K = 65536
SAE_WIDTH_262K = 262144

# Default sparsity level
SAE_L0_TARGET = "medium"  # options: "small", "medium", "big"

# ---------- Dataset ----------
TARGET_LANGUAGES = ["en", "zh", "es", "bn", "sw"]
ALL_MGSM_LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]

# ---------- Zhao et al. baseline ----------
# Layer ranges for SVD intervention (estimated for 34-layer model)
# Middle layers: ~30-70% depth → layers 10-23
# Higher layers: ~70-97% depth → layers 24-32
ZHAO_MIDDLE_LAYERS = list(range(10, 24))
ZHAO_HIGHER_LAYERS = list(range(24, 33))
ZHAO_LAMBDA_RANGE_MIDDLE = (0.0, 0.4)    # positive = remove language
ZHAO_LAMBDA_RANGE_HIGHER = (-0.4, 0.0)   # negative = re-inject language

# ---------- Qwen2.5-7B-IT ----------
QWEN_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
QWEN_D_MODEL = 3584
QWEN_N_LAYERS = 28

# andyrdt BatchTopK SAEs (residual stream)
QWEN_SAE_REPO = "andyrdt/saes-qwen2.5-7b-instruct"
QWEN_SAE_LAYERS = [3, 7, 11, 15, 19, 23, 27]      # all available
QWEN_SAE_SUBSET_LAYERS = [7, 11, 19, 27]            # ~25%, ~40%, ~68%, ~96% depth
QWEN_SAE_WIDTH = 131072
QWEN_SAE_TRAINER = "trainer_1"   # k=64 (moderate sparsity)

# Zhao layer ranges for 28-layer model (~30-70% = 8-19, ~70-97% = 20-27)
QWEN_ZHAO_MIDDLE_LAYERS = list(range(8, 20))
QWEN_ZHAO_HIGHER_LAYERS = list(range(20, 27))

# ---------- Experiment ----------
BATCH_SIZE = 4  # for activation extraction on A100
MAX_NEW_TOKENS = 512  # for MGSM generation
