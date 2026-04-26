"""Causal interventions: feature ablation and steering via nnsight."""

import torch


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


def get_sae_decoder_directions(sae, feature_indices: list[int]) -> torch.Tensor:
    """Extract decoder weight columns for specified features.

    Args:
        sae: SAELens SAE object.
        feature_indices: List of feature indices to extract.

    Returns:
        Tensor of shape (len(feature_indices), d_model).
    """
    # SAELens stores decoder weights as W_dec of shape (n_features, d_model)
    W_dec = sae.W_dec.data  # (n_features, d_model)
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
            hidden_states = output[0]  # (batch, seq, d_model)

            if pos_mode == "last":
                # During generation, the input grows. We only want to
                # ablate at the original last input position.
                if hidden_states.shape[1] >= input_len:
                    pos = input_len - 1
                    hidden_states[:, pos, :] = directional_ablation(
                        hidden_states[:, pos, :], directions_dev
                    )
            else:
                hidden_states = directional_ablation(hidden_states, directions_dev)
                output = (hidden_states,) + output[1:]

            return (hidden_states,) + output[1:]

        return hook_fn

    outputs = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        # Register hooks
        for layer_idx, directions in ablation_config.items():
            handle = model.model.layers[layer_idx].register_forward_hook(
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
