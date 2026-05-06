"""Model and SAE loading utilities."""

import json
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_DTYPE,
    MODEL_ID,
    QWEN_SAE_REPO,
    QWEN_SAE_TRAINER,
    SAE_L0_TARGET,
    SAE_RELEASE_RES,
    SAE_SUBSET_LAYERS,
    SAE_WIDTH_16K,
    SAE_WIDTH_65K,
)

load_dotenv()


# Gemma 3 4B IT is multimodal: model.model is Gemma3Model, which wraps both
# language_model (Gemma3TextModel with .layers) and vision_tower. Other HF
# decoder LMs put .layers directly on model.model. This getter probes both.
_LAYER_PATHS = (
    ("model", "layers"),                    # Llama / Gemma 2 / Mistral / etc.
    ("model", "language_model", "layers"),  # Gemma 3 multimodal CausalLM
    ("language_model", "model", "layers"),  # alternative multimodal wrapping
)


def get_decoder_layers(obj):
    """Return the ModuleList of transformer decoder layers from a HF model
    or any wrapper that proxies attribute access (e.g. an nnsight NNsight).
    """
    for path in _LAYER_PATHS:
        try:
            cur = obj
            for attr in path:
                cur = getattr(cur, attr)
            return cur
        except AttributeError:
            continue
    raise AttributeError(
        f"Could not find decoder layers on {type(obj).__name__}. "
        f"Tried paths: {[' .'.join(p) for p in _LAYER_PATHS]}"
    )


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    device_map: str = "auto",
    dtype: str = MODEL_DTYPE,
) -> tuple:
    """Load Gemma 3 4B IT model and tokenizer.

    Gemma 3 4B IT is multimodal (SigLIP vision tower + text decoder).
    AutoModelForCausalLM resolves to `Gemma3ForConditionalGeneration` which
    knows how to map the checkpoint's `language_model.model.layers.X` prefix.
    Using `Gemma3ForCausalLM` directly is tempting (skips vision tower) but
    it silently re-inits all weights because the prefix mapping is missing
    -- generation produces gibberish. So we eat the ~400MB vision tower load
    and use get_decoder_layers() to navigate the multimodal nesting.

    Returns:
        (model, tokenizer) tuple.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN not set. Create a .env file with your HuggingFace token. "
            "You must also accept the Gemma 3 license at: "
            "https://huggingface.co/google/gemma-3-4b-it"
        )

    torch_dtype = getattr(torch, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device_map,
        token=token,
    )
    model.eval()
    return model, tokenizer


def load_sae(
    layer: int,
    width: int = SAE_WIDTH_16K,
    l0_target: str = SAE_L0_TARGET,
    release: str = SAE_RELEASE_RES,
    device: str = "cuda",
):
    """Load a single Gemma Scope 2 SAE for a specific layer.

    Args:
        layer: Transformer layer index (0-33).
        width: SAE width (16384 or 65536).
        l0_target: Sparsity level ("small", "medium", "big").
        release: HuggingFace repo for SAEs.
        device: Device to load SAE on.

    Returns:
        (sae, cfg_dict, sparsity) tuple from SAELens.
    """
    from sae_lens import SAE

    # Gemma Scope 2 SAE ID format: layer_{N}_width_{W}_l0_{SIZE}
    width_str = f"{width // 1000}k"
    sae_id = f"layer_{layer}_width_{width_str}_l0_{l0_target}"

    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    return sae, cfg_dict, sparsity


def load_saes_at_layers(
    layers: list[int] | None = None,
    width: int = SAE_WIDTH_65K,
    l0_target: str = SAE_L0_TARGET,
    device: str = "cuda",
) -> dict:
    """Load SAEs for multiple layers.

    Args:
        layers: List of layer indices. Defaults to SAE_SUBSET_LAYERS.
        width: SAE width.
        l0_target: Sparsity level.
        device: Device.

    Returns:
        Dict mapping layer index to SAE object.
    """
    if layers is None:
        layers = SAE_SUBSET_LAYERS

    saes = {}
    for layer in layers:
        sae, _, _ = load_sae(layer, width=width, l0_target=l0_target, device=device)
        saes[layer] = sae
        print(f"  Loaded SAE for layer {layer} (width={width}, l0={l0_target})")
    return saes


# ---------- Qwen2.5-7B-IT BatchTopK SAEs ----------


def load_qwen_sae(
    layer: int,
    trainer: str = QWEN_SAE_TRAINER,
    repo: str = QWEN_SAE_REPO,
    device: str = "cuda",
):
    """Load a BatchTopK SAE for Qwen2.5-7B-IT from HuggingFace.

    These SAEs are from andyrdt/saes-qwen2.5-7b-instruct and use the
    dictionary_learning library's AutoEncoder format.

    Args:
        layer: Transformer layer index (from QWEN_SAE_LAYERS).
        trainer: Trainer variant (trainer_0=k32, trainer_1=k64,
            trainer_2=k128, trainer_3=k256).
        repo: HuggingFace repo ID.
        device: Device to load SAE on.

    Returns:
        (ae, config) tuple — AutoEncoder object and its config dict.
    """
    from huggingface_hub import hf_hub_download

    subdir = f"resid_post_layer_{layer}/{trainer}"
    ae_path = hf_hub_download(repo, f"{subdir}/ae.pt", repo_type="model")
    cfg_path = hf_hub_download(repo, f"{subdir}/config.json", repo_type="model")

    with open(cfg_path) as f:
        config = json.load(f)

    # Try dictionary_learning library first, fall back to raw state dict
    try:
        from dictionary_learning import AutoEncoder
        ae = AutoEncoder.from_pretrained(ae_path, device=device)
    except (ImportError, RuntimeError):
        # Fallback: load raw state dict and wrap in a simple module.
        # RuntimeError covers state_dict key mismatches between
        # dictionary_learning versions and the checkpoint format.
        ae = _load_batchtopk_raw(ae_path, config, device)

    return ae, config


def _load_batchtopk_raw(ae_path: str, config: dict, device: str):
    """Fallback loader when dictionary_learning is not installed.

    Loads the state dict and wraps it in a minimal module that exposes
    .encode() and .decoder.weight for compatibility.
    """
    import torch.nn as nn

    state = torch.load(ae_path, map_location=device, weights_only=True)

    class BatchTopKSAE(nn.Module):
        def __init__(self, state_dict, k):
            super().__init__()
            self.encoder = nn.Linear(
                state_dict["encoder.weight"].shape[1],
                state_dict["encoder.weight"].shape[0],
            )
            self.decoder = nn.Linear(
                state_dict["decoder.weight"].shape[1],
                state_dict["decoder.weight"].shape[0],
                bias=False,
            )
            # Load weights
            self.encoder.weight = nn.Parameter(state_dict["encoder.weight"])
            self.encoder.bias = nn.Parameter(state_dict["encoder.bias"])
            self.decoder.weight = nn.Parameter(state_dict["decoder.weight"])
            # Pre-bias: some checkpoints use "pre_bias", others "b_dec"
            if "pre_bias" in state_dict:
                self.register_buffer("pre_bias", state_dict["pre_bias"])
            elif "b_dec" in state_dict:
                self.register_buffer("pre_bias", state_dict["b_dec"])
            else:
                self.register_buffer("pre_bias", torch.zeros(self.encoder.weight.shape[1]))
            self.k = k

        def encode(self, x):
            x_centered = x - self.pre_bias
            pre_acts = self.encoder(x_centered)
            # BatchTopK: keep top-k activations per sample
            topk_vals, topk_idx = pre_acts.topk(self.k, dim=-1)
            acts = torch.zeros_like(pre_acts)
            acts.scatter_(-1, topk_idx, torch.relu(topk_vals))
            return acts

    # k can come from config or from the checkpoint itself
    if "k" in state and isinstance(state["k"], (int, torch.Tensor)):
        k = int(state["k"]) if isinstance(state["k"], torch.Tensor) else state["k"]
    else:
        k = config.get("k", 64)
    model = BatchTopKSAE(state, k).to(device)
    model.eval()
    return model


def load_qwen_saes_at_layers(
    layers: list[int] | None = None,
    trainer: str = QWEN_SAE_TRAINER,
    device: str = "cuda",
) -> dict:
    """Load Qwen BatchTopK SAEs for multiple layers.

    Args:
        layers: List of layer indices. Defaults to QWEN_SAE_SUBSET_LAYERS.
        trainer: Trainer variant.
        device: Device.

    Returns:
        Dict mapping layer index to AutoEncoder object.
    """
    from src.config import QWEN_SAE_SUBSET_LAYERS

    if layers is None:
        layers = QWEN_SAE_SUBSET_LAYERS

    saes = {}
    for layer in layers:
        ae, config = load_qwen_sae(layer, trainer=trainer, device=device)
        saes[layer] = ae
        print(f"  Loaded Qwen SAE for layer {layer} (trainer={trainer}, k={config.get('k', '?')})")
    return saes


