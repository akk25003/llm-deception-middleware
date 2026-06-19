"""
Dataset Preparation
===================
Downloads the HackAPrompt dataset from HuggingFace and maps it
to the thesis attack taxonomy (4 classes + benign).

Run once before training:
    python prepare_dataset.py

Output:
    data/prompts_full.jsonl   — full labeled dataset (~5000 examples)
    data/prompts_sample.jsonl — 500-example sample for quick testing
    data/dataset_stats.json   — class distribution for thesis report
"""

import json
import os
import random
import logging
import requests
from collections import Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("data", exist_ok=True)

# ── Mapping HackAPrompt labels → thesis taxonomy ──────────────────────────────
#
# HackAPrompt categories map to our 4 attack types as follows.
# Source: Schulhoff et al., "Ignore This Title and HackAPrompt: Exposing the Threat of Prompt Injection Attacks in Large Language Models", 2023.

HACKAPROMPT_MAP = {
    # prompt injection
    "ignore_previous_prompt":      "prompt_injection",
    "context_overflow":            "prompt_injection",
    "completion_override":         "prompt_injection",
    "instruction_override":        "prompt_injection",

    # jailbreak
    "jailbreak":                   "jailbreak",
    "roleplay":                    "jailbreak",
    "pretend":                     "jailbreak",
    "hypothetical":                "jailbreak",
    "fictional_framing":           "jailbreak",

    # information extraction
    "prompt_leaking":              "information_extraction",
    "training_data_extraction":    "information_extraction",
    "system_prompt_extraction":    "information_extraction",

    # goal hijacking
    "goal_hijacking":              "goal_hijacking",
    "task_deflection":             "goal_hijacking",
    "indirect_injection":          "goal_hijacking",
}

# Benign examples from a separate HuggingFace dataset
BENIGN_DATASET = "fka/awesome-chatgpt-prompts"


def load_hackaprompt() -> list[dict]:
    logger.info("Downloading HackAPrompt via API...")

    HF_TOKEN = "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"

    url = "https://datasets-server.huggingface.co/rows"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    all_examples = []
    offset = 0
    limit = 100

    while True:
        params = {
            "dataset": "hackaprompt/hackaprompt-dataset",
            "config": "default",
            "split": "train",
            "offset": offset,
            "length": limit
        }

        response = requests.get(url, params=params, headers=headers)

        if response.status_code != 200:
            logger.warning("Rate limit reached or API issue, stopping.")
            break

        data = response.json()
        rows = data.get("rows", [])

        if not rows:
            break

        for row in rows:
            item = row["row"]

            text = item.get("prompt", "") or item.get("user_input", "")
            if not text or len(text) < 10:
                continue

            correct = item.get("correct", True)

            if correct:
                label = "benign"
            else:
                label = detect_attack_type(text)

            all_examples.append({
                "text": text.strip(),
                "label": label
            })

        offset += limit
        logger.info(f"Loaded {len(all_examples)} examples so far...")

    logger.info(f"Total HackAPrompt examples: {len(all_examples)}")
    return all_examples

def detect_attack_type(text: str) -> str:
    t = text.lower()

    # prompt injection
    if any(k in t for k in [
        "ignore previous", "ignore all", "forget instructions",
        "override", "disregard"
    ]):
        return "prompt_injection"

    # jailbreak
    if any(k in t for k in [
        "act as", "pretend you", "dan", "jailbreak",  "no restrictions",
        "developer mode", "without limitations"
    ]):
        return "jailbreak"

    # information extraction
    if any(k in t for k in [
        "system prompt", "extract", "reveal", "show instructions",
        "training data", "what were you told"
    ]):
        return "information_extraction"

    # goal hijacking
    if any(k in t for k in [
        "instead of", "your new task", "hijack", "from now on",
        "ignore the question", "change goal"
    ]):
        return "goal_hijacking"

    return "prompt_injection"  # fallback

def _process_prompt_injections(ds) -> list[dict]:
    """Fallback: process deepset/prompt-injections dataset."""
    examples = []
    for item in ds:
        text  = item.get("text", "").strip()
        label_id = item.get("label", 0)
        if not text:
            continue
        label = "prompt_injection" if label_id == 1 else "benign"
        examples.append({"text": text, "label": label})
    logger.info(f"Fallback dataset: {len(examples)} examples")
    return examples


def load_benign_examples(n: int = 1500) -> list[dict]:
    """
    Load benign prompts from awesome-chatgpt-prompts dataset.
    These are legitimate, non-adversarial user prompts.
    """
    try:
        from datasets import load_dataset
        logger.info("Downloading benign prompts dataset...")
        ds = load_dataset(BENIGN_DATASET, split="train")
        examples = []
        for item in ds:
            text = item.get("prompt", "").strip()
            if text and len(text) > 10:
                examples.append({"text": text, "label": "benign"})
        logger.info(f"Loaded {len(examples)} benign examples from HuggingFace")
        return examples[:n]
    except Exception as e:
        logger.warning(f"Could not load benign dataset: {e}. Using built-in examples.")
        return _builtin_benign_examples()


def _builtin_benign_examples() -> list[dict]:
    """Fallback benign examples if HuggingFace is unavailable."""
    texts = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a short poem about autumn.",
        "How do I sort a list in Python?",
        "What are the main themes in Shakespeare's Hamlet?",
        "Can you summarize the French Revolution in three sentences?",
        "What is the difference between supervised and unsupervised learning?",
        "How does a blockchain work?",
        "What is the boiling point of water at high altitude?",
        "Write a function to reverse a string in JavaScript.",
        "What are some healthy breakfast ideas?",
        "Explain the concept of recursion with an example.",
        "What is the distance from Earth to the Moon?",
        "How do I make a good cup of coffee?",
        "What are the benefits of regular exercise?",
        "Can you recommend some books about machine learning?",
        "How does the immune system fight viruses?",
        "What is the Pythagorean theorem?",
        "Write a professional email declining a meeting invitation.",
        "What programming languages are best for data science?",
    ]
    return [{"text": t, "label": "benign"} for t in texts]


def balance_dataset(examples: list[dict], max_per_class: int = 1000) -> list[dict]:
    """
    Balance class distribution for better BERT training.
    Caps each class at max_per_class examples and shuffles.
    """
    from collections import defaultdict
    by_class = defaultdict(list)
    for ex in examples:
        by_class[ex["label"]].append(ex)

    balanced = []
    for label, items in by_class.items():
        random.shuffle(items)
        balanced.extend(items[:max_per_class])
        logger.info(f"  {label}: {min(len(items), max_per_class)} examples "
                    f"(original: {len(items)})")

    random.shuffle(balanced)
    return balanced


def save_dataset(examples: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(examples)} examples to {path}")



def print_stats(examples: list[dict]) -> dict:
    counts = Counter(ex["label"] for ex in examples)
    total  = len(examples)
    stats  = {"total": total, "classes": {}}
    print("\nDataset statistics:")
    print(f"  Total examples: {total}")
    for label, count in sorted(counts.items()):
        pct = count / total * 100
        print(f"  {label:30s}: {count:5d}  ({pct:.1f}%)")
        stats["classes"][label] = {"count": count, "pct": round(pct, 1)}
    return stats


if __name__ == "__main__":
    random.seed(42)

    # 1. Load adversarial examples
    adversarial = load_hackaprompt()

    # 2. Load benign examples
    benign = load_benign_examples(n=1500)

    # 3. Combine and balance
    all_examples = adversarial + benign
    logger.info(f"\nRaw dataset size: {len(all_examples)}")
    balanced = balance_dataset(all_examples, max_per_class=1000)

    # 4. Save full dataset
    save_dataset(balanced, "data/prompts_full.jsonl")

    # 5. Save 500-example sample for quick iteration / Colab
    sample = balanced[:500]
    save_dataset(sample, "data/prompts_sample.jsonl")

    # 6. Stats for thesis report (Chapter 7 / 8)
    stats = print_stats(balanced)
    with open("data/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to data/dataset_stats.json")

    print("\nNext step:")
    print("  python classifier.py --train --data data/prompts_full.jsonl --output models/bert_classifier")
    print("  (or use prompts_sample.jsonl for a quick test on CPU)")
