"""Model and SAE loading utilities."""

import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import (
    MODEL_DTYPE,
    MODEL_ID,
    SAE_L0_TARGET,
    SAE_RELEASE_RES,
    SAE_SUBSET_LAYERS,
    SAE_WIDTH_16K,
    SAE_WIDTH_65K,
)

load_dotenv()


def load_model_and_tokenizer(
    model_id: str = MODEL_ID,
    device_map: str = "auto",
    dtype: str = MODEL_DTYPE,
) -> tuple:
    """Load Gemma 3 4B IT model and tokenizer.

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
        torch_dtype=torch_dtype,
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


