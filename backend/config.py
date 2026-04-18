"""
Config & LLM Client — Centralized configuration and Gemini API wrapper.

Provides:
  • Environment variable loading
  • Singleton Gemini client with retry logic + JSON extraction
  • Global constants (aspect categories, thresholds, etc.)
"""

import os
import json
import asyncio
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# Global Configuration
# ═══════════════════════════════════════════════════════════════════════════════

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Retry settings
MAX_RETRIES: int = 3
RETRY_DELAY_BASE: float = 1.0

# Anomaly detection
ANOMALY_SPIKE_THRESHOLD: float = 0.25  # 25% spike triggers anomaly
SYSTEMIC_ISSUE_MIN_COUNT: int = 3      # 3+ mentions = systemic (vs isolated)
DEDUP_SIMILARITY_THRESHOLD: float = 0.85
SPAM_SCORE_THRESHOLD: float = 0.7

# Aspect categories — the 5 pillars of analysis
ASPECT_CATEGORIES: list[str] = [
    "Battery Life",
    "Taste",
    "Packaging",
    "Price",
    "Customer Support",
]

# Supported languages
SUPPORTED_LANGUAGES: dict[str, str] = {
    "EN": "English",
    "KN": "Kannada",
    "HI": "Hindi",
    "TE": "Telugu",
    "TA": "Tamil",
}

# Engine metadata
ENGINE_VERSION: str = "2.0.0"


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini LLM Client (Singleton)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiClient:
    """
    Lightweight async Gemini wrapper with:
      • Automatic retry with exponential backoff
      • JSON extraction from LLM responses (strips markdown fences)
      • Call counting for stats
      • Graceful fallback when API key is missing
    """

    def __init__(self):
        self._model = None
        self._call_count: int = 0
        self._total_tokens: int = 0

        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self._model = genai.GenerativeModel(GEMINI_MODEL)
            print(f"  [OK] Gemini initialized: {GEMINI_MODEL}")
        else:
            print("  [WARN] GEMINI_API_KEY not set -- LLM calls will return fallbacks.")

    @property
    def is_available(self) -> bool:
        """Check if the LLM client is configured and ready."""
        return self._model is not None

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def model_name(self) -> str:
        return GEMINI_MODEL if self.is_available else "none"

    async def generate_json(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_output_tokens: int = 4096,
    ) -> Optional[dict]:
        """
        Send a prompt to Gemini and parse the response as JSON.

        Args:
            prompt: The full prompt string
            temperature: LLM temperature (lower = more deterministic)
            max_output_tokens: Maximum response length

        Returns:
            Parsed JSON dict, or None on failure
        """
        if not self.is_available:
            return None

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self._model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                self._call_count += 1
                raw_text = response.text.strip()
                return self._extract_json(raw_text)

            except Exception as e:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"  [WARN] LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)

        return None

    async def generate_text(
        self,
        prompt: str,
        temperature: float = 0.4,
        max_output_tokens: int = 4096,
    ) -> Optional[str]:
        """
        Send a prompt to Gemini and return raw text response.
        Used for roadmap generation and other free-form text tasks.
        """
        if not self.is_available:
            return None

        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    self._model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                self._call_count += 1
                return response.text.strip()

            except Exception as e:
                delay = RETRY_DELAY_BASE * (2 ** attempt)
                print(f"  [WARN] LLM text call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)

        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        """
        Extract JSON from LLM response, handling:
          • ```json ... ``` markdown wrappers
          • Leading/trailing whitespace
          • Partial JSON recovery (find first { to last })
        """
        cleaned = text.strip()

        # Strip markdown code fences
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        # Attempt 1: Direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract outermost { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass

        # Attempt 3: Try to find a JSON array
        start = cleaned.find("[")
        end = cleaned.rfind("]") + 1
        if start != -1 and end > start:
            try:
                result = json.loads(cleaned[start:end])
                return {"items": result} if isinstance(result, list) else result
            except json.JSONDecodeError:
                pass

        print(f"  [WARN] JSON parse failed: {cleaned[:200]}...")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Instance
# ═══════════════════════════════════════════════════════════════════════════════

llm = GeminiClient()
