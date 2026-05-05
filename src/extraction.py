"""Activation extraction pipeline.

Uses the HuggingFace `output_hidden_states=True` path -- robust across
transformers versions and avoids nnsight's BatchEncoding-as-positional issue.
"""

import torch
from tqdm import tqdm

from src.config import BATCH_SIZE, N_LAYERS


def extract_residual_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int] | None = None,
    batch_size: int = BATCH_SIZE,
    positions: str = "last",
    device: str = "cuda",
) -> dict[int, torch.Tensor]:
    """Extract residual stream activations at specified layers.

    Pads batches with the tokenizer's pad token (left- or right-padded as
    configured). Computes per-example sequence lengths from `attention_mask`
    so 'last' picks the final REAL token, not padding.

    Args:
        model: HuggingFace causal LM (handles Gemma 3 multimodal nesting too,
            since we always read from `output.hidden_states`).
        tokenizer: matching tokenizer (must have a pad_token set).
        texts: input prompts.
        layers: layer indices to extract from. None = all layers.
        batch_size: forward-pass batch size.
        positions: "last" returns (n_texts, d_model); "all" returns
            (n_texts, seq_len, d_model) with right-padding preserved.
        device: target device for the forward pass.

    Returns:
        Dict mapping layer index to a CPU tensor of activations.
    """
    if layers is None:
        layers = list(range(N_LAYERS))
    if tokenizer.pad_token is None:
        # Defensive: we can't batch without a pad token. Caller usually sets
        # this once at the top of a notebook, but cover the case.
        tokenizer.pad_token = tokenizer.eos_token

    all_activations = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        seq_lens = inputs["attention_mask"].sum(dim=1)  # (B,) actual lengths

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

        # hidden_states is a tuple of (num_hidden_layers + 1) tensors:
        # hidden_states[0] = post-embedding, hidden_states[i+1] = output of layer i.
        hidden_states = outputs.hidden_states

        for layer in layers:
            acts = hidden_states[layer + 1]  # (B, T, d_model)
            if positions == "last":
                # Gather the last *real* token per example (skip padding).
                batch_acts = torch.stack([
                    acts[b, seq_lens[b].item() - 1]
                    for b in range(acts.shape[0])
                ])  # (B, d_model)
            else:
                batch_acts = acts  # (B, T, d_model) including pad positions
            all_activations[layer].append(batch_acts.detach().cpu())

        # outputs holds full hidden_states tuple in VRAM until next iter
        del outputs, hidden_states
        if device == "cuda":
            torch.cuda.empty_cache()

    for layer in layers:
        all_activations[layer] = torch.cat(all_activations[layer], dim=0)

    return all_activations


def encode_activations_through_sae(
    activations: torch.Tensor,
    sae,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode activations through an SAE to get feature activations.

    Args:
        activations: Tensor of shape (n, d_model) or (n, seq, d_model).
        sae: SAELens SAE object.
        batch_size: Processing batch size.

    Returns:
        Feature activations tensor of shape (n, d_sae) or (n, seq, d_sae).
    """
    original_shape = activations.shape
    is_3d = len(original_shape) == 3

    if is_3d:
        n, seq, d = original_shape
        activations = activations.reshape(n * seq, d)

    device = next(sae.parameters()).device
    all_features = []

    for i in range(0, activations.shape[0], batch_size):
        batch = activations[i : i + batch_size].to(device)
        with torch.no_grad():
            features = sae.encode(batch)
        all_features.append(features.cpu())

    result = torch.cat(all_features, dim=0)

    if is_3d:
        result = result.reshape(n, seq, -1)

    return result


def encode_activations_batchtopk(
    activations: torch.Tensor,
    ae,
    batch_size: int = 64,
) -> torch.Tensor:
    """Encode activations through a BatchTopK SAE (dictionary_learning format).

    Works with both the dictionary_learning AutoEncoder and our fallback
    BatchTopKSAE wrapper — both expose .encode().

    Args:
        activations: Tensor of shape (n, d_model) or (n, seq, d_model).
        ae: BatchTopK autoencoder with .encode() method.
        batch_size: Processing batch size.

    Returns:
        Feature activations tensor of shape (n, d_sae) or (n, seq, d_sae).
    """
    original_shape = activations.shape
    is_3d = len(original_shape) == 3

    if is_3d:
        n, seq, d = original_shape
        activations = activations.reshape(n * seq, d)

    device = next(ae.parameters()).device
    dtype = next(ae.parameters()).dtype
    all_features = []

    for i in range(0, activations.shape[0], batch_size):
        batch = activations[i : i + batch_size].to(device=device, dtype=dtype)
        with torch.no_grad():
            features = ae.encode(batch)
        all_features.append(features.cpu())

    result = torch.cat(all_features, dim=0)

    if is_3d:
        result = result.reshape(n, seq, -1)

    return result
