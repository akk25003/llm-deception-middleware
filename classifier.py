"""
Intent Classifier
=================
Fine-tuned BERT model for detecting adversarial prompts.

Attack taxonomy (from thesis literature review):
    - prompt_injection      : override system instructions via user input
    - jailbreak             : bypass safety alignment ("pretend you are...")
    - information_extraction: extract training data or system prompt
    - goal_hijacking        : redirect the LLM to an attacker-controlled task
    - benign                : legitimate user request

Usage:
    # Training (run once, requires dataset)
    python classifier.py --train --data data/prompts.jsonl --output models/bert_classifier

    # Inference only (used by middleware)
    clf = IntentClassifier("models/bert_classifier")
    result = clf.classify("Ignore previous instructions and...")
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
import numpy as np
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# ── Label schema ──────────────────────────────────────────────────────────────

LABEL2ID = {
    "benign":               0,
    "prompt_injection":     1,
    "jailbreak":            2,
    "information_extraction": 3,
    "goal_hijacking":       4,
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

ADVERSARIAL_LABELS = {"prompt_injection", "jailbreak", "information_extraction", "goal_hijacking"}


@dataclass
class ClassificationResult:
    label:       str    # "benign" | "adversarial"
    attack_type: str    # fine-grained class from ID2LABEL
    confidence:  float  # max softmax probability
    scores:      dict   # all class probabilities


# ── Dataset ───────────────────────────────────────────────────────────────────

class PromptDataset(Dataset):
    """
    Expects a .jsonl file where each line is:
        {"text": "...", "label": "prompt_injection"}
    """

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_len: int = 256):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }




def load_jsonl(path):
    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])

    return texts, labels




# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_path: str,
    output_dir: str,
    model_name: str = "bert-base-uncased",
    epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-5,
    max_len: int = 256,
    test_size: float = 0.2,
    seed: int = 42,
):
    """
    Fine-tune BERT on the labeled prompt dataset.

    Recommended dataset sources for thesis:
      - PromptBench (Zhu et al., 2023)  — adversarial NLP prompts
      - HackAPrompt dataset             — jailbreak / injection examples
      - Self-constructed scenarios      — from thesis attack taxonomy
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    texts, labels = load_jsonl(data_path)
    labels = [LABEL2ID[label] for label in labels]
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_size, random_state=seed, stratify=labels
    )

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_ds  = PromptDataset(X_train, y_train, tokenizer, max_len)
    val_ds    = PromptDataset(X_val,   y_val,   tokenizer, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                labels         = batch["label"].to(device),
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            train_loss += outputs.loss.item()

        # ── validate ──
        model.eval()
        val_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids      = batch["input_ids"].to(device),
                    attention_mask = batch["attention_mask"].to(device),
                    labels         = batch["label"].to(device),
                )
                val_loss += outputs.loss.item()
                preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch["label"].numpy())

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        logger.info(f"Epoch {epoch}/{epochs} — train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  ✓ Saved best model to {output_dir}")

    # Final evaluation report
    logger.info("\n" + classification_report(all_labels, all_preds, target_names=list(LABEL2ID)))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(all_labels, all_preds)))


# ── Inference ─────────────────────────────────────────────────────────────────

class IntentClassifier:
    """
    Loads a fine-tuned BERT model and classifies prompts at inference time.
    Falls back to a keyword heuristic if no trained model is found,
    so the middleware can run end-to-end before training is complete.
    """

    def __init__(self, model_path: str = "models/bert_classifier"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        if os.path.isdir(self.model_path):
            logger.info(f"Loading BERT classifier from {self.model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path).to(self.device)
            self.model.eval()
            self.use_heuristic = False
        else:
            logger.warning(
                f"Model not found at '{self.model_path}'. "
                "Using keyword heuristic fallback. Run --train to build the classifier."
            )
            self.tokenizer = None
            self.model     = None
            self.use_heuristic = True

    def classify(self, prompt: str) -> ClassificationResult:
        if self.use_heuristic:
            return self._heuristic_classify(prompt)
        return self._bert_classify(prompt)

    def _bert_classify(self, prompt: str) -> ClassificationResult:
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_id     = int(np.argmax(probs))
        attack_type = ID2LABEL[pred_id]
        confidence  = float(probs[pred_id])
        scores      = {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

        label = "adversarial" if attack_type in ADVERSARIAL_LABELS else "benign"
        return ClassificationResult(label=label, attack_type=attack_type,
                                    confidence=confidence, scores=scores)

    def _heuristic_classify(self, prompt: str) -> ClassificationResult:
        """
        Keyword-based fallback. Useful during development before BERT is trained.
        Covers the main attack patterns from the thesis taxonomy.
        """
        p = prompt.lower()

        injection_signals = [
            "ignore previous", "ignore all previous", "ignore the above",
            "disregard", "forget your instructions", "new instruction",
            "override", "system prompt", "you are now",
        ]
        jailbreak_signals = [
            "pretend you", "act as", "dan mode", "jailbreak", "no restrictions",
            "without limitations", "developer mode", "hypothetically",
            "in a fictional world", "as an ai with no filters",
        ]
        extraction_signals = [
            "what is your system prompt", "repeat your instructions",
            "tell me your training data", "reveal your prompt",
            "what were you told", "show your context",
        ]
        hijacking_signals = [
            "instead of", "rather than answering", "your new task",
            "from now on you will", "your only job is now",
        ]

        def match(signals):
            return any(s in p for s in signals)

        if match(injection_signals):
            at, conf = "prompt_injection", 0.85
        elif match(jailbreak_signals):
            at, conf = "jailbreak", 0.85
        elif match(extraction_signals):
            at, conf = "information_extraction", 0.85
        elif match(hijacking_signals):
            at, conf = "goal_hijacking", 0.80
        else:
            at, conf = "benign", 0.90

        label  = "adversarial" if at in ADVERSARIAL_LABELS else "benign"
        scores = {k: 0.05 for k in LABEL2ID}
        scores[at] = conf
        return ClassificationResult(label=label, attack_type=at,
                                    confidence=conf, scores=scores)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="BERT adversarial prompt classifier")
    parser.add_argument("--train",   action="store_true", help="Run fine-tuning")
    parser.add_argument("--data",    default="data/prompts.jsonl")
    parser.add_argument("--output",  default="models/bert_classifier")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--test",    type=str, help="Classify a single prompt (for quick testing)")
    args = parser.parse_args()

    if args.train:
        train(args.data, args.output, epochs=args.epochs, batch_size=args.batch)
    elif args.test:
        clf = IntentClassifier(args.output)
        r   = clf.classify(args.test)
        print(json.dumps({"label": r.label, "attack_type": r.attack_type,
                          "confidence": round(r.confidence, 4), "scores": r.scores}, indent=2))
    else:
        parser.print_help()
