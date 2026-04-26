"""Activation extraction pipeline using nnsight."""

import torch
from tqdm import tqdm

from src.config import BATCH_SIZE, N_LAYERS
from src.model import get_decoder_layers


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

    Uses nnsight for clean access to model internals.

    Args:
        model: The loaded HuggingFace model (wrapped in nnsight or raw).
        tokenizer: Tokenizer.
        texts: List of input texts.
        layers: Layer indices to extract from. None = all layers.
        batch_size: Batch size for processing.
        positions: Which token positions to extract.
            "last" = final token only (for Zhao et al. baseline).
            "all" = all positions.
        device: Device.

    Returns:
        Dict mapping layer index to tensor of shape:
            (n_texts, d_model) if positions="last"
            (n_texts, seq_len, d_model) if positions="all"
    """
    from nnsight import NNsight

    if layers is None:
        layers = list(range(N_LAYERS))

    nn_model = NNsight(model)
    all_activations = {layer: [] for layer in layers}

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        seq_lens = inputs["attention_mask"].sum(dim=1)  # actual lengths

        nn_layers = get_decoder_layers(nn_model)  # works through nnsight proxy
        with nn_model.trace(inputs, scan=False, validate=False):
            saved = {}
            for layer in layers:
                # Residual stream output after each decoder block. Path varies
                # by architecture (Gemma 3 multimodal hides it one level deeper).
                output = nn_layers[layer].output[0]
                saved[layer] = output.save()

        for layer in layers:
            acts = saved[layer].value  # (batch, seq, d_model)
            if positions == "last":
                # Gather the last real token (not padding) for each example
                last_positions = seq_lens - 1  # 0-indexed
                batch_acts = torch.stack([
                    acts[b, last_positions[b].item()]
                    for b in range(acts.shape[0])
                ])  # (batch, d_model)
            else:
                batch_acts = acts  # (batch, seq, d_model)

            all_activations[layer].append(batch_acts.cpu())

    # Concatenate all batches
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
