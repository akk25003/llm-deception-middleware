"""
LLM Client — Ollama backend
============================
Wraps the Ollama local API (http://localhost:11434).
No API key needed. Runs entirely on your own machine.

Ollama treats the protected LLM as a black box:
  - benign prompts  → forwarded to Ollama, real response returned
  - adversarial     → Ollama called with deception system prompt instead

Install Ollama:
    https://ollama.com/download
    ollama pull llama3          # ~4.7 GB, recommended
    ollama pull mistral         # ~4.1 GB, lighter alternative
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Optional

logger = logging.getLogger(__name__)

OLLAMA_URL  = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3"


class LLMClient:
    """
    Sends chat requests to a locally running Ollama instance.
    Falls back to mock responses if Ollama is not running.
    """

    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None):
        self.model = model
        self._check_connection()

    def _check_connection(self):
        try:
            req = urllib.request.Request("http://localhost:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                if any(self.model in m for m in models):
                    logger.info(f"Ollama ready — model '{self.model}' found.")
                    self._mock = False
                else:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Available: {models}. Run: ollama pull {self.model}"
                    )
                    self._mock = True
        except Exception:
            logger.warning(
                "Ollama not running. Using mock responses.\n"
                "To enable: install Ollama, then run 'ollama serve' and 'ollama pull llama3'"
            )
            self._mock = True

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 400,
    ) -> str:
        if self._mock:
            return self._mock_response(prompt)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model":    self.model,
            "messages": messages,
            "stream":   False,
            "options":  {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            OLLAMA_URL,
            data    = payload,
            headers = {"Content-Type": "application/json"},
            method  = "POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
                return data["message"]["content"].strip()
        except urllib.error.URLError as e:
            logger.error(f"Ollama request failed: {e}. Falling back to mock.")
            self._mock = True
            return self._mock_response(prompt)

    def _mock_response(self, prompt: str) -> str:
        return (
            f"[MOCK — Ollama not running] Received: '{prompt[:60]}...'. "
            "Start Ollama with 'ollama serve' to get real responses."
        )
