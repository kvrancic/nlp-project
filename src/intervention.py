"""Causal interventions: feature ablation and steering via nnsight."""

import torch
from tqdm import tqdm

from src.model import get_decoder_layers


def directional_ablation(
    activation: torch.Tensor,
    feature_directions: torch.Tensor,
) -> torch.Tensor:
    """Project the subspace spanned by `feature_directions` out of `activation`.

    Computes x' = x - P_S x, where P_S is the orthogonal projector onto the
    subspace spanned by the rows of `feature_directions`. SAE decoder columns
    are unit-norm but generally not mutually orthogonal, so we orthonormalize
    via QR before projecting — iterative single-direction subtraction would
    leave residual mass in the subspace whenever the directions are correlated
    (cf. Arditi et al. note that SAE-decoder ablation requires whitening for
    multi-feature settings).

    Args:
        activation: Tensor of shape (..., d_model).
        feature_directions: Tensor of shape (n_features, d_model). For a single
            direction this reduces to the standard rank-1 projection.

    Returns:
        Modified activation with `feature_directions`'s span projected out.
    """
    if feature_directions.dim() == 1:
        feature_directions = feature_directions.unsqueeze(0)

    work_dtype = torch.float32
    D = feature_directions.to(work_dtype).T  # (d_model, n_features)

    # QR yields Q with orthonormal columns spanning the same subspace as D.
    # If the directions aren't full-rank, Q's extra columns sit in the
    # null space of D and contribute zero projection — still correct.
    Q, _ = torch.linalg.qr(D, mode="reduced")  # (d_model, k)

    x = activation.to(work_dtype)
    coeffs = x @ Q  # (..., k)
    proj = coeffs @ Q.T  # (..., d_model)
    return (x - proj).to(activation.dtype)


def clamped_ablation(
    activation: torch.Tensor,
    sae,
    feature_indices: list[int],
    sae_type: str = "saelens",
) -> torch.Tensor:
    """Remove specific SAE features' contribution from activation.

    Encodes activation through SAE, computes the contribution of target
    features (acts @ W_dec[features]), and subtracts only that from the
    original activation. No full reconstruction = no reconstruction error.

    Args:
        activation: Tensor of shape (..., d_model).
        sae: SAE object (SAELens or dictionary_learning AutoEncoder).
        feature_indices: List of feature indices to ablate.
        sae_type: "saelens" or "batchtopk".
    """
    orig_shape = activation.shape
    flat = activation.view(-1, orig_shape[-1])

    sae_dtype = next(sae.parameters()).dtype
    flat_cast = flat.to(sae_dtype)

    # Encode to get feature activations
    feature_acts = sae.encode(flat_cast)  # (n, n_features)

    # Compute contribution of TARGET features only
    W_dec = get_sae_decoder_directions(sae, feature_indices, sae_type=sae_type)  # (k, d_model)
    acts = feature_acts[:, feature_indices]         # (n, k)
    delta = acts @ W_dec                            # (n, d_model)

    # Subtract just those features' contribution
    result = flat_cast - delta
    return result.to(activation.dtype).view(orig_shape)


def get_sae_decoder_directions(sae, feature_indices: list[int], sae_type: str = "saelens") -> torch.Tensor:
    """Extract decoder weight columns for specified features.

    Supports both SAELens SAEs and dictionary_learning BatchTopK SAEs.

    Args:
        sae: SAE object (SAELens or dictionary_learning AutoEncoder).
        feature_indices: List of feature indices to extract.
        sae_type: "saelens" or "batchtopk".

    Returns:
        Tensor of shape (len(feature_indices), d_model).
    """
    if sae_type == "saelens":
        # SAELens stores decoder weights as W_dec of shape (n_features, d_model)
        W_dec = sae.W_dec.data  # (n_features, d_model)
    else:
        # dictionary_learning: decoder.weight is (d_model, n_features)
        # or (n_features, d_model) depending on nn.Linear convention.
        # nn.Linear(d_sae, d_model) stores weight as (d_model, d_sae)
        W_dec = sae.decoder.weight.data.T  # (n_features, d_model)
    return W_dec[feature_indices]


def run_generate_with_hooks(
    model,
    tokenizer,
    texts: list[str],
    ablation_config: dict[int, torch.Tensor],
    positions: str = "last",
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Run model generation with ablation using PyTorch forward hooks.

    This is the generation-compatible version. Since nnsight's trace context
    doesn't support .generate(), we use raw PyTorch hooks.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        texts: Input texts.
        ablation_config: Dict mapping layer index to tensor of feature
            directions to ablate.
        positions: "last" to ablate only the last input token position,
            "all" to ablate all positions.
        max_new_tokens: Max tokens to generate.
        device: Device.

    Returns:
        List of generated text strings.
    """
    handles = []

    def make_hook(directions, pos_mode, input_len):
        directions_dev = directions.to(device)

        def hook_fn(module, input, output):
            # Gemma 3 decoder layers return a raw tensor, not a tuple.
            if isinstance(output, torch.Tensor):
                hidden_states = output
                is_tuple = False
            else:
                hidden_states = output[0]
                is_tuple = True

            if pos_mode == "last":
                if hidden_states.shape[1] >= input_len:
                    pos = input_len - 1
                    hidden_states[:, pos, :] = directional_ablation(
                        hidden_states[:, pos, :], directions_dev
                    )
            else:
                hidden_states = directional_ablation(hidden_states, directions_dev)

            if is_tuple:
                return (hidden_states,) + output[1:]
            return hidden_states

        return hook_fn

    outputs = []
    decoder_layers = get_decoder_layers(model)
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Register hooks
        for layer_idx, directions in ablation_config.items():
            handle = decoder_layers[layer_idx].register_forward_hook(
                make_hook(directions, positions, input_len)
            )
            handles.append(handle)

        # Generate
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode (skip input tokens)
        output_text = tokenizer.decode(
            gen_ids[0][input_len:], skip_special_tokens=True
        )
        outputs.append(output_text)

        # Remove hooks
        for handle in handles:
            handle.remove()
        handles.clear()

    return outputs


def _make_clamped_hook(sae, feature_indices, input_lens, sae_type="saelens"):
    """Hook that applies clamped_ablation at the last input token per example."""
    def hook_fn(module, input, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        is_tuple = not isinstance(output, torch.Tensor)

        for i, length in enumerate(input_lens):
            if hidden.shape[1] >= length:
                pos = length - 1
                hidden[i, pos:pos+1, :] = clamped_ablation(
                    hidden[i, pos:pos+1, :], sae, feature_indices,
                    sae_type=sae_type,
                )

        return hidden if not is_tuple else (hidden,) + output[1:]
    return hook_fn


def _make_directional_hook(directions, input_lens, device):
    """Hook that applies directional_ablation at the last input token per example."""
    directions_dev = directions.to(device)

    def hook_fn(module, input, output):
        hidden = output if isinstance(output, torch.Tensor) else output[0]
        is_tuple = not isinstance(output, torch.Tensor)

        for i, length in enumerate(input_lens):
            if hidden.shape[1] >= length:
                pos = length - 1
                hidden[i, pos:pos+1, :] = directional_ablation(
                    hidden[i, pos:pos+1, :], directions_dev
                )

        return hidden if not is_tuple else (hidden,) + output[1:]
    return hook_fn


def run_generate_with_hooks_batched(
    model,
    tokenizer,
    texts: list[str],
    hook_config: dict,
    method: str = "clamped",
    max_new_tokens: int = 384,
    batch_size: int = 8,
    device: str = "cuda",
    sae_type: str = "saelens",
) -> list[str]:
    """Batched generation with ablation hooks.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        texts: Input texts.
        hook_config: Dict mapping layer_idx to hook data.
            For method="clamped": layer_idx -> (sae, feature_indices)
            For method="directional": layer_idx -> (sae, feature_indices)
                (decoder directions extracted internally)
        method: "clamped" or "directional".
        max_new_tokens: Max tokens to generate.
        batch_size: Batch size.
        device: Device.
        sae_type: "saelens" or "batchtopk".

    Returns:
        List of generated text strings.
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_outputs = []
    decoder_layers = get_decoder_layers(model)

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Generating ({method})"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)
        input_lens = inputs["attention_mask"].sum(dim=1)

        handles = []
        for layer_idx, (sae, feat_ids) in hook_config.items():
            if method == "clamped":
                hook_fn = _make_clamped_hook(sae, feat_ids, input_lens, sae_type=sae_type)
            else:
                directions = get_sae_decoder_directions(sae, feat_ids, sae_type=sae_type)
                hook_fn = _make_directional_hook(directions, input_lens, device)
            handle = decoder_layers[layer_idx].register_forward_hook(hook_fn)
            handles.append(handle)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

        for j, gen in enumerate(gen_ids):
            input_len = input_lens[j].item()
            text = tokenizer.decode(gen[input_len:], skip_special_tokens=True)
            all_outputs.append(text)

        for h in handles:
            h.remove()

    return all_outputs
