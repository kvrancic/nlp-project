"""MGSM dataset loading, prompt formatting, and answer parsing."""

import re
import urllib.request

from src.config import TARGET_LANGUAGES

# Original MGSM data from Google Research (TSV format, no dependencies needed)
MGSM_BASE_URL = (
    "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_{lang}.tsv"
)


def load_mgsm(languages: list[str] | None = None) -> dict[str, list[dict]]:
    """Load MGSM test set directly from Google Research GitHub.

    Downloads TSV files — no `datasets` library needed (avoids numpy
    binary incompatibility issues on Colab).

    Returns:
        Dict mapping language code to list of examples.
        Each example has keys: 'question', 'answer', 'answer_number', 'language'.
    """
    if languages is None:
        languages = TARGET_LANGUAGES

    data = {}
    for lang in languages:
        url = MGSM_BASE_URL.format(lang=lang)
        response = urllib.request.urlopen(url)
        lines = response.read().decode("utf-8").strip().split("\n")

        examples = []
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            question = parts[0].strip()
            answer_raw = parts[1].strip()

            # The Google Research TSV has just the bare numeric answer
            try:
                answer_number = float(answer_raw.replace(",", ""))
            except ValueError:
                answer_number = None

            examples.append({
                "question": question,
                "answer_number": answer_number,
                "language": lang,
            })
        data[lang] = examples
    return data


def format_prompt_gemma_it(question: str, few_shot_examples: list[dict] | None = None) -> str:
    """Format a question for Gemma 3 IT using the chat template.

    Uses the Gemma 3 IT chat format:
    <start_of_turn>user
    {question}<end_of_turn>
    <start_of_turn>model
    """
    messages = []

    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({"role": "user", "content": ex["question"]})
            messages.append({"role": "assistant", "content": ex["answer"]})

    messages.append({"role": "user", "content": question})

    # Build the raw chat format (tokenizer.apply_chat_template is preferred
    # when a tokenizer is available, but this works for display/debugging)
    parts = []
    for msg in messages:
        if msg["role"] == "user":
            parts.append(f"<start_of_turn>user\n{msg['content']}<end_of_turn>")
        elif msg["role"] == "assistant":
            parts.append(f"<start_of_turn>model\n{msg['content']}<end_of_turn>")
    parts.append("<start_of_turn>model\n")
    return "\n".join(parts)


def parse_answer_number(model_output: str) -> float | None:
    """Extract the final numeric answer from model output.

    MGSM answers are integers. The model typically outputs chain-of-thought
    reasoning followed by a final answer. We look for common patterns:
    - "The answer is X"
    - "#### X"
    - The last number in the output
    """
    text = model_output.strip()

    # Pattern 1: "#### <number>"
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return _parse_num(match.group(1))

    # Pattern 2: "The answer is <number>" (multilingual variants)
    answer_patterns = [
        r"[Tt]he answer is\s*([\d,]+(?:\.\d+)?)",
        r"[Aa]nswer[:\s]*([\d,]+(?:\.\d+)?)",
        r"答案[是为：:]\s*([\d,]+(?:\.\d+)?)",        # Chinese
        r"[Rr]espuesta[:\s]*([\d,]+(?:\.\d+)?)",       # Spanish
        r"উত্তর[:\s]*([\d,]+(?:\.\d+)?)",                # Bengali
        r"[Jj]ibu[:\s]*([\d,]+(?:\.\d+)?)",            # Swahili
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            return _parse_num(match.group(1))

    # Fallback: last number in text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return _parse_num(numbers[-1])

    return None


def _parse_num(s: str) -> float:
    """Parse a number string, handling commas."""
    return float(s.replace(",", ""))


def compute_accuracy(predictions: list[float | None], gold: list[float]) -> float:
    """Compute exact-match accuracy on numeric answers."""
    correct = 0
    for pred, g in zip(predictions, gold):
        if pred is not None and abs(pred - g) < 1e-6:
            correct += 1
    return correct / len(gold) if gold else 0.0
