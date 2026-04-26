"""Evaluation utilities: accuracy, language fidelity, perplexity."""

import torch
from tqdm import tqdm

from src.data import compute_accuracy, parse_answer_number
from src.model import get_decoder_layers


def evaluate_mgsm(
    model,
    tokenizer,
    questions: list[str],
    gold_answers: list[float],
    max_new_tokens: int = 512,
    batch_size: int = 1,
    device: str = "cuda",
    hooks: list | None = None,
) -> dict:
    """Evaluate model on MGSM problems.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        questions: List of formatted prompts.
        gold_answers: List of gold numeric answers.
        max_new_tokens: Max generation length.
        batch_size: Batch size for generation.
        device: Device.
        hooks: Optional list of (layer_idx, hook_fn) tuples to register.

    Returns:
        Dict with keys:
            'accuracy': float
            'predictions': list of parsed numeric answers
            'outputs': list of raw model output strings
            'correct': list of booleans
    """
    # Register hooks if provided
    handles = []
    if hooks:
        decoder_layers = get_decoder_layers(model)
        for layer_idx, hook_fn in hooks:
            handle = decoder_layers[layer_idx].register_forward_hook(hook_fn)
            handles.append(handle)

    predictions = []
    outputs = []

    try:
        for i in tqdm(range(0, len(questions), batch_size), desc="Evaluating"):
            batch_qs = questions[i : i + batch_size]

            for q in batch_qs:
                inputs = tokenizer(q, return_tensors="pt").to(device)

                with torch.no_grad():
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )

                input_len = inputs["input_ids"].shape[1]
                output_text = tokenizer.decode(
                    gen_ids[0][input_len:], skip_special_tokens=True
                )
                outputs.append(output_text)
                predictions.append(parse_answer_number(output_text))
    finally:
        # Always clean up hooks
        for handle in handles:
            handle.remove()

    accuracy = compute_accuracy(predictions, gold_answers)
    correct = [
        pred is not None and abs(pred - gold) < 1e-6
        for pred, gold in zip(predictions, gold_answers)
    ]

    return {
        "accuracy": accuracy,
        "predictions": predictions,
        "outputs": outputs,
        "correct": correct,
    }


def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    device: str = "cuda",
    max_length: int = 512,
) -> list[float]:
    """Compute per-text perplexity.

    Used for causal feature identification: ablating a language-specific
    feature should increase perplexity on that language's text.

    Args:
        model: HuggingFace model.
        tokenizer: Tokenizer.
        texts: List of texts.
        device: Device.
        max_length: Max sequence length.

    Returns:
        List of perplexity values (one per text).
    """
    perplexities = []

    for text in tqdm(texts, desc="Computing perplexity"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss  # cross-entropy loss
            ppl = torch.exp(loss).item()

        perplexities.append(ppl)

    return perplexities


def evaluate_per_language(
    results_by_lang: dict[str, dict],
) -> dict:
    """Aggregate per-language evaluation results.

    Args:
        results_by_lang: Dict mapping language code to evaluation result dict
            (output of evaluate_mgsm).

    Returns:
        Summary dict with per-language accuracies and overall average.
    """
    summary = {}
    accuracies = []

    for lang, result in results_by_lang.items():
        summary[lang] = {
            "accuracy": result["accuracy"],
            "n_correct": sum(result["correct"]),
            "n_total": len(result["correct"]),
        }
        accuracies.append(result["accuracy"])

    summary["average"] = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return summary
