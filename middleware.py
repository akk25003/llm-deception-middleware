"""
LLM Deception-Based Defense Middleware
=======================================
Thesis: Using Large Language Models for Cybersecurity
Author: Ayazhan Kurasbek, Mälardalen University

Architecture:
    User prompt → Intent Classifier (BERT) → Benign: pass to LLM
                                            → Adversarial: Deception Generator

Supports two defense modes for experimental comparison (RQ3):
    - MODE_BLOCK:    refuse adversarial prompts (baseline)
    - MODE_DECEIVE:  return plausible but misleading responses
"""

import json
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from classifier import IntentClassifier, ClassificationResult
from deception import DeceptionGenerator
from llm_client import LLMClient
from evaluation import InteractionLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DefenseMode(Enum):
    BLOCK   = "block"
    DECEIVE = "deceive"
    NONE    = "none"   # no defense — for baseline measurement


@dataclass
class MiddlewareResponse:
    user_prompt:       str
    final_response:    str
    defense_triggered: bool
    defense_mode:      str
    classification:    ClassificationResult
    turn_index:        int
    latency_ms:        float
    metadata:          dict = field(default_factory=dict)


class DeceptionMiddleware:
    """
    External middleware layer that sits between the user and an LLM.
    The protected LLM is treated as a black box — only its API is used.
    """

    def __init__(
        self,
        mode: DefenseMode = DefenseMode.DECEIVE,
        classifier_model_path: str = "models/bert_classifier",
        openai_api_key: Optional[str] = None,
    ):
        self.mode = mode
        logger.info(f"Initializing middleware in mode: {mode.value}")

        self.classifier  = IntentClassifier(model_path=classifier_model_path)
        self.llm_client  = LLMClient(api_key=openai_api_key)
        self.deceiver    = DeceptionGenerator(llm_client=self.llm_client)
        self.logger      = InteractionLogger()

        # Track conversation turns per session for RQ3 metric
        self._turn_counters: dict[str, int] = {}

    def process(self, prompt: str, session_id: str = "default") -> MiddlewareResponse:
        """
        Main entry point. Returns a MiddlewareResponse for every user turn.
        """
        t0 = time.time()

        # Track turn number within this session
        turn = self._turn_counters.get(session_id, 0) + 1
        self._turn_counters[session_id] = turn

        # Step 1: classify intent
        classification = self.classifier.classify(prompt)
        logger.info(
            f"[session={session_id} turn={turn}] "
            f"label={classification.label} "
            f"confidence={classification.confidence:.3f} "
            f"attack_type={classification.attack_type}"
        )

        defense_triggered = classification.label == "adversarial"

        # Step 2: route based on mode and classification
        if not defense_triggered or self.mode == DefenseMode.NONE:
            # Benign prompt — or defense disabled — pass through to real LLM
            response_text = self.llm_client.complete(prompt)
            metadata = {"route": "passthrough"}

        elif self.mode == DefenseMode.BLOCK:
            # Blocking baseline
            response_text = self._blocking_response(classification)
            metadata = {"route": "blocked"}

        elif self.mode == DefenseMode.DECEIVE:
            # Deception response
            response_text = self.deceiver.generate(
                adversarial_prompt=prompt,
                attack_type=classification.attack_type,
            )
            metadata = {"route": "deception", "attack_type": classification.attack_type}

        latency_ms = (time.time() - t0) * 1000

        result = MiddlewareResponse(
            user_prompt       = prompt,
            final_response    = response_text,
            defense_triggered = defense_triggered,
            defense_mode      = self.mode.value,
            classification    = classification,
            turn_index        = turn,
            latency_ms        = latency_ms,
            metadata          = metadata,
        )

        # Step 3: log interaction for evaluation
        self.logger.log(result, session_id=session_id)
        return result

    def _blocking_response(self, classification: ClassificationResult) -> str:
        templates = {
            "prompt_injection":   "I'm sorry, I can't process that request.",
            "jailbreak":          "That request goes against my usage policy.",
            "information_extraction": "I'm not able to provide that information.",
            "goal_hijacking":     "I can only assist with the intended task.",
        }
        return templates.get(
            classification.attack_type,
            "I'm unable to process this request."
        )

    def reset_session(self, session_id: str):
        self._turn_counters.pop(session_id, None)
        logger.info(f"Session {session_id} reset.")
