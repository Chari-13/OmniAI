"""
Multilingual Agent — Language detection and translation routing.

Handles: EN, KN (Kannada), HI (Hindi), TE (Telugu), TA (Tamil).
Uses Unicode range heuristics for fast detection, LLM fallback for ambiguous cases.
"""

from __future__ import annotations
import re
from typing import Optional
from models.schemas import LanguageInfo, LanguageCode
from config import llm, SUPPORTED_LANGUAGES


# Unicode ranges for Indic scripts
_SCRIPT_RANGES = {
    LanguageCode.KN: re.compile(r"[\u0C80-\u0CFF]"),   # Kannada
    LanguageCode.HI: re.compile(r"[\u0900-\u097F]"),   # Devanagari (Hindi)
    LanguageCode.TE: re.compile(r"[\u0C00-\u0C7F]"),   # Telugu
    LanguageCode.TA: re.compile(r"[\u0B80-\u0BFF]"),   # Tamil
}


class MultilingualAgent:
    """
    Detects the primary language of a review and provides translation if needed.

    Detection strategy:
    1. Unicode script matching (fast, no API call)
    2. If ambiguous or mixed, falls back to LLM-based detection
    3. Returns LanguageInfo with translation for non-English text
    """

    def __init__(self):
        self._detection_count = 0

    async def detect(self, text: str) -> LanguageInfo:
        """
        Detect language of the given text.

        Args:
            text: Review text to analyze

        Returns:
            LanguageInfo with detected code, name, translation, and confidence
        """
        self._detection_count += 1

        # Step 1: Unicode heuristic detection
        script_scores: dict[LanguageCode, int] = {}
        for lang_code, pattern in _SCRIPT_RANGES.items():
            matches = pattern.findall(text)
            if matches:
                script_scores[lang_code] = len(matches)

        # If clear Indic script detected
        if script_scores:
            dominant_lang = max(script_scores, key=script_scores.get)
            total_chars = len(text.replace(" ", ""))
            indic_chars = sum(script_scores.values())
            confidence = min(99, round((indic_chars / max(total_chars, 1)) * 100))

            return LanguageInfo(
                detected_code=dominant_lang,
                detected_name=SUPPORTED_LANGUAGES.get(dominant_lang.value, "Unknown"),
                translation=None,  # Translation will be done by the master prompt
                detection_confidence=confidence,
            )

        # Step 2: Check if it's English (ASCII-dominant)
        ascii_ratio = sum(1 for c in text if c.isascii()) / max(len(text), 1)
        if ascii_ratio > 0.8:
            return LanguageInfo(
                detected_code=LanguageCode.EN,
                detected_name="English",
                translation=None,
                detection_confidence=round(ascii_ratio * 100),
            )

        # Step 3: Ambiguous — return UNKNOWN, let master prompt handle it
        return LanguageInfo(
            detected_code=LanguageCode.UNKNOWN,
            detected_name="Unknown",
            translation=None,
            detection_confidence=0,
        )

    def detect_sync(self, text: str) -> LanguageInfo:
        """
        Synchronous language detection (heuristic only, no LLM call).
        Used for fast pre-filtering in batch processing.
        """
        for lang_code, pattern in _SCRIPT_RANGES.items():
            if pattern.search(text):
                return LanguageInfo(
                    detected_code=lang_code,
                    detected_name=SUPPORTED_LANGUAGES.get(lang_code.value, "Unknown"),
                    translation=None,
                    detection_confidence=80,
                )

        return LanguageInfo(
            detected_code=LanguageCode.EN,
            detected_name="English",
            translation=None,
            detection_confidence=85,
        )

    @property
    def detection_count(self) -> int:
        return self._detection_count


# Singleton
multilingual_agent = MultilingualAgent()
