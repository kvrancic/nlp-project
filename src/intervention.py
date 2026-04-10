"""Causal interventions: feature ablation and steering via nnsight."""

import torch
from nnsight import NNsight


def directional_ablation(
    activation: torch.Tensor,
    feature_directions: torch.Tensor,
) -> torch.Tensor:
    """Ablate specific SAE feature directions from an activation.

    From Deng et al. / Arditi et al.:
        x' = x - sum_i (d̂_i * d̂_i^T * x)
    where d̂_i is the unit vector of SAE decoder column for feature i.

    Args:
        activation: Tensor of shape (..., d_model).
        feature_directions: Tensor of shape (n_features, d_model) — the
            decoder columns for features to ablate (not necessarily unit vectors;
            we normalize internally).

    Returns:
        Modified activation with feature directions projected out.
    """
    # Normalize to unit vectors
    directions = feature_directions / feature_directions.norm(dim=-1, keepdim=True)

    result = activation.clone()
    for d in directions:
        # Project out this direction: x' = x - (d^T x) * d
        proj = torch.einsum("...d, d -> ...", result, d)
        result = result - proj.unsqueeze(-1) * d

    return result


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


def run_with_ablation(
    model,
    tokenizer,
    texts: list[str],
    ablation_config: dict[int, torch.Tensor],
    positions: str = "last",
    max_new_tokens: int = 512,
    device: str = "cuda",
) -> list[str]:
    """Run model generation with feature ablation at specified layers.

    Uses nnsight to hook into the model's forward pass and modify
    residual stream activations.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        texts: Input texts.
        ablation_config: Dict mapping layer index to tensor of feature
            directions to ablate, shape (n_features, d_model).
        positions: "last" to ablate only the final input token,
            "all" to ablate at all positions.
        max_new_tokens: Max tokens to generate.
        device: Device.

    Returns:
        List of generated text outputs.
    """
    nn_model = NNsight(model)
    outputs = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]

        with nn_model.trace(inputs, scan=False, validate=False):
            for layer, directions in ablation_config.items():
                directions = directions.to(device)
                resid = nn_model.model.layers[layer].output[0]

                if positions == "last":
                    # Only ablate the final token position
                    last_pos_act = resid[:, -1, :]
                    ablated = directional_ablation(last_pos_act, directions)
                    resid[:, -1, :] = ablated
                else:
                    # Ablate all positions
                    ablated = directional_ablation(resid, directions)
                    nn_model.model.layers[layer].output[0][:] = ablated

            # Get output logits
            logits = nn_model.output.logits.save()

        # For generation, we need a different approach — nnsight trace
        # doesn't support generate() directly. We'll use hooks instead.
        # This is a forward-pass-only version for logit analysis.
        # For full generation with ablation, see run_generate_with_hooks below.

    return outputs


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
