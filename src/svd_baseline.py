"""Zhao et al. (2025) SVD-based language-reasoning disentanglement baseline.

Implements Algorithm 1 from "When Less Language is More":
1. Compute per-language mean hidden states at each layer
2. SVD decomposition to identify language-specific subspace
3. Inference-time projection to remove language-specific components

Reference: https://github.com/MuyuenLP/Language-Reasoning-Disentangle
"""

import torch

from src.config import (
    ZHAO_HIGHER_LAYERS,
    ZHAO_LAMBDA_RANGE_HIGHER,
    ZHAO_LAMBDA_RANGE_MIDDLE,
    ZHAO_MIDDLE_LAYERS,
)


def compute_language_subspace(
    per_language_activations: dict[str, torch.Tensor],
    rank: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute language-specific subspace via SVD (Algorithm 1).

    Args:
        per_language_activations: Dict mapping language code to tensor of
            shape (d_model,) — the mean hidden state for that language
            at a specific layer.
        rank: Rank r of the language-specific subspace. Must be < n_languages.

    Returns:
        (M_a, M_s) tuple:
            M_a: Language-agnostic component, shape (d_model,)
            M_s: Language-specific subspace basis, shape (d_model, rank)
    """
    languages = sorted(per_language_activations.keys())
    d_model = per_language_activations[languages[0]].shape[0]

    # Build matrix M: columns are per-language mean embeddings
    # M shape: (d_model, n_languages)
    M = torch.stack([per_language_activations[l] for l in languages], dim=1).float()

    # Step 1: Language-agnostic = column mean
    M_a_prime = M.mean(dim=1, keepdim=True)  # (d_model, 1)

    # Step 2: Subtract agnostic, take top-r SVD of residual
    residual = M - M_a_prime  # (d_model, n_languages)
    U, S, Vt = torch.linalg.svd(residual, full_matrices=False)
    M_s_prime = U[:, :rank]  # (d_model, rank)

    # Step 3: Reconstruct
    Gamma_prime = Vt[:rank, :]  # (rank, n_languages)
    M_prime = M_a_prime + M_s_prime @ (torch.diag(S[:rank]) @ Gamma_prime)

    # Step 4: Re-orthogonalize
    ones = torch.ones(len(languages), 1, device=M.device)  # (n_languages, 1)
    M_a_vec = M_prime @ ones  # (d_model, 1)
    M_a = M_a_vec / M_a_vec.norm()  # normalized

    # Step 5: Final SVD on re-orthogonalized residual
    # Project out the M_a direction from each column of M_prime so that
    # the resulting M_s is orthogonal to M_a.
    proj_coeffs = M_a.T @ M_prime  # (1, n_languages)
    residual2 = M_prime - M_a @ proj_coeffs  # (d_model, n_languages)
    U2, S2, _ = torch.linalg.svd(residual2, full_matrices=False)
    M_s = U2[:, :rank]  # (d_model, rank)

    return M_a.squeeze(), M_s


def project_out_language(
    hidden_state: torch.Tensor,
    M_s: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Project out language-specific component from hidden state.

    h_hat = h - λ * M_s @ M_s^T @ h

    Args:
        hidden_state: Tensor of shape (..., d_model).
        M_s: Language-specific subspace basis, shape (d_model, rank).
        lam: Scaling factor. Positive removes language, negative re-injects.

    Returns:
        Modified hidden state.
    """
    # M_s @ M_s^T @ h = projection onto language subspace
    proj = hidden_state @ M_s  # (..., rank)
    proj = proj @ M_s.T  # (..., d_model)
    return hidden_state - lam * proj


def create_svd_hooks(
    M_s_per_layer: dict[int, torch.Tensor],
    lambda_middle: float,
    lambda_higher: float,
    middle_layers: list[int] | None = None,
    higher_layers: list[int] | None = None,
    input_length: int | None = None,
    device: str = "cuda",
) -> dict:
    """Create forward hooks for SVD-based intervention.

    Args:
        M_s_per_layer: Dict mapping layer index to M_s tensor.
        lambda_middle: λ for middle layers (positive, removes language).
        lambda_higher: λ for higher layers (negative, re-injects).
        middle_layers: Layer indices for middle range.
        higher_layers: Layer indices for higher range.
        input_length: If provided, only modify the last input token position.
        device: Device.

    Returns:
        Dict mapping layer index to hook function.
    """
    if middle_layers is None:
        middle_layers = ZHAO_MIDDLE_LAYERS
    if higher_layers is None:
        higher_layers = ZHAO_HIGHER_LAYERS

    hooks = {}

    for layer_idx, M_s in M_s_per_layer.items():
        if layer_idx in middle_layers:
            lam = lambda_middle
        elif layer_idx in higher_layers:
            lam = lambda_higher
        else:
            continue  # skip layers outside intervention range

        def make_hook(m_s_raw, l, dev):
            m_s = None

            def hook_fn(module, input, output):
                nonlocal m_s
                hidden_states = output if isinstance(output, torch.Tensor) else output[0]
                if m_s is None:
                    m_s = m_s_raw.to(dtype=hidden_states.dtype, device=dev)
                if input_length is not None:
                    pos = input_length - 1
                    if hidden_states.shape[1] > pos:
                        hidden_states[:, pos, :] = project_out_language(
                            hidden_states[:, pos, :], m_s, l
                        )
                else:
                    projected = project_out_language(hidden_states, m_s, l)
                    hidden_states.copy_(projected)
            return hook_fn

        hooks[layer_idx] = make_hook(M_s, lam, device)

    return hooks


def generate_with_svd_intervention(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    M_s_per_layer: dict[int, torch.Tensor],
    lambda_middle: float,
    lambda_higher: float,
    max_new_tokens: int = 512,
    middle_layers: list[int] | None = None,
    higher_layers: list[int] | None = None,
    device: str = "cuda",
) -> torch.Tensor:
    """Generate with SVD projection applied during the prefill pass only.

    Avoids forward hook + model.generate() incompatibilities by running
    the hooked prefill as a plain forward pass, then continuing generation
    unhooked using the resulting KV cache.
    """
    from src.model import get_decoder_layers

    if middle_layers is None:
        middle_layers = ZHAO_MIDDLE_LAYERS
    if higher_layers is None:
        higher_layers = ZHAO_HIGHER_LAYERS

    decoder_layers = get_decoder_layers(model)
    input_length = input_ids.shape[1]

    hook_dict = create_svd_hooks(
        M_s_per_layer=M_s_per_layer,
        lambda_middle=lambda_middle,
        lambda_higher=lambda_higher,
        input_length=input_length,
        middle_layers=middle_layers,
        higher_layers=higher_layers,
        device=device,
    )

    handles = [
        decoder_layers[layer].register_forward_hook(fn)
        for layer, fn in hook_dict.items()
    ]

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
    finally:
        for h in handles:
            h.remove()

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)

    generated = [next_token]
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens - 1):
        if next_token.item() == eos_id:
            break
        new_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=new_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        attention_mask = new_mask

    return torch.cat([input_ids, torch.cat(generated, dim=1)], dim=1)


def grid_search_lambda(
    lambdas_middle: list[float] | None = None,
    lambdas_higher: list[float] | None = None,
    n_steps: int = 9,
) -> list[tuple[float, float]]:
    """Generate grid of (lambda_middle, lambda_higher) pairs to search.

    Args:
        lambdas_middle: Explicit list of middle λ values.
        lambdas_higher: Explicit list of higher λ values.
        n_steps: Number of steps if generating automatically.

    Returns:
        List of (lambda_middle, lambda_higher) tuples.
    """
    if lambdas_middle is None:
        low, high = ZHAO_LAMBDA_RANGE_MIDDLE
        step = (high - low) / (n_steps - 1)
        lambdas_middle = [low + i * step for i in range(n_steps)]

    if lambdas_higher is None:
        low, high = ZHAO_LAMBDA_RANGE_HIGHER
        step = (high - low) / (n_steps - 1)
        lambdas_higher = [low + i * step for i in range(n_steps)]

    return [(lm, lh) for lm in lambdas_middle for lh in lambdas_higher]
