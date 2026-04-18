"""
Sarcasm Agent — Dedicated sarcasm and irony detection.

Detects:
  • Verbal irony ("Great, it broke instantly")
  • Situational sarcasm (praising obviously negative experiences)
  • Hyperbole (extreme exaggeration for effect)

When sarcasm is detected, provides the 'true_sentiment' — what the
reviewer actually means, which may differ from the surface-level sentiment.

NOTE: The master analysis prompt already includes sarcasm detection.
This agent provides additional post-processing and validation on top of
the LLM results, plus standalone sarcasm analysis for edge cases.
"""

from __future__ import annotations
import re
from typing import Optional
from models.schemas import SarcasmResult, SarcasmType, SentimentLabel
from config import llm


# Sarcasm indicator patterns
_SARCASM_PATTERNS = [
    # Quoted positive words (suggesting insincerity)
    re.compile(r"['\"](?:amazing|great|wonderful|excellent|fantastic|best|love)['\"]", re.IGNORECASE),
    # Positive word followed by negative context
    re.compile(r"\b(?:great|amazing|love|wonderful)\b.*\b(?:broke|died|failed|terrible|worst|useless|waste)\b", re.IGNORECASE),
    # Exaggerated positive + negative outcome
    re.compile(r"\b(?:SO|SOOO|soooo)\s+(?:good|great|amazing|premium|fast)\b", re.IGNORECASE),
    # Eye-roll or sarcastic emojis
    re.compile(r"🙄|😒|🤡|💀|😂.*(?:bad|terrible|worst|broke)"),
    # "If you consider X" pattern
    re.compile(r"if you (?:consider|call|think|believe)", re.IGNORECASE),
    # "Only took X" (sarcastic time reference)
    re.compile(r"only took (?:them |us |me )?\d+\s*(?:weeks?|months?|days?|hours?)", re.IGNORECASE),
    # Ellipsis followed by negative
    re.compile(r"\.{3,}.*\b(?:not|never|terrible|worst|broke|died)\b", re.IGNORECASE),
    # "Sure" or "Right" at the start (dismissive)
    re.compile(r"^(?:Sure|Right|Yeah right|Oh wow|Oh great),?\s", re.IGNORECASE),
]

# Words that amplify sarcasm detection
_SARCASM_AMPLIFIERS = {"obviously", "clearly", "totally", "absolutely", "definitely", "surely"}


class SarcasmAgent:
    """
    Post-processes and validates sarcasm detection from the master analysis.
    Also provides standalone sarcasm analysis for edge cases.
    """

    def __init__(self):
        self._detection_count = 0
        self._sarcasm_found = 0

    def validate_sarcasm(
        self,
        llm_sarcasm: dict,
        original_text: str,
        surface_sentiment: str,
    ) -> SarcasmResult:
        """
        Validate and potentially correct LLM sarcasm detection using
        pattern-based heuristics.

        Args:
            llm_sarcasm: Sarcasm data from the master LLM analysis
            original_text: The original review text
            surface_sentiment: The surface-level sentiment detected by ABSA

        Returns:
            Validated SarcasmResult
        """
        self._detection_count += 1

        # Start with LLM result
        is_sarcastic = llm_sarcasm.get("is_sarcastic", False)
        sarcasm_confidence = llm_sarcasm.get("sarcasm_confidence", 0)
        sarcasm_type = llm_sarcasm.get("sarcasm_type", "none")
        true_sentiment = llm_sarcasm.get("true_sentiment", surface_sentiment)
        explanation = llm_sarcasm.get("explanation", "")

        # Run pattern-based detection as a secondary check
        pattern_score, pattern_type = self._pattern_detect(original_text)

        # Case 1: LLM says sarcastic, patterns confirm — high confidence
        if is_sarcastic and pattern_score > 0:
            sarcasm_confidence = min(100, sarcasm_confidence + 15)

        # Case 2: LLM says NOT sarcastic, but patterns detect it — flag for review
        elif not is_sarcastic and pattern_score >= 2:
            is_sarcastic = True
            sarcasm_confidence = max(60, pattern_score * 25)
            sarcasm_type = pattern_type
            true_sentiment = self._flip_sentiment(surface_sentiment)
            explanation = "Pattern-based detection overrode LLM (multiple sarcasm indicators found)"

        # Case 3: LLM says sarcastic but low confidence, no pattern support — reduce
        elif is_sarcastic and pattern_score == 0 and sarcasm_confidence < 50:
            is_sarcastic = False
            sarcasm_confidence = max(0, sarcasm_confidence - 20)
            true_sentiment = surface_sentiment

        if is_sarcastic:
            self._sarcasm_found += 1

        # Map sarcasm_type string to enum
        try:
            stype = SarcasmType(sarcasm_type)
        except ValueError:
            stype = SarcasmType.NONE if not is_sarcastic else SarcasmType.VERBAL_IRONY

        # Map true_sentiment to enum
        try:
            tsentiment = SentimentLabel(true_sentiment)
        except ValueError:
            tsentiment = SentimentLabel.NEUTRAL

        return SarcasmResult(
            is_sarcastic=is_sarcastic,
            sarcasm_confidence=sarcasm_confidence,
            sarcasm_type=stype,
            true_sentiment=tsentiment,
            explanation=explanation,
        )

    def _pattern_detect(self, text: str) -> tuple[int, str]:
        """
        Run regex pattern matching for sarcasm indicators.

        Returns:
            Tuple of (match_count, detected_type)
        """
        match_count = 0
        detected_type = "none"

        for pattern in _SARCASM_PATTERNS:
            if pattern.search(text):
                match_count += 1
                # Classify type based on which patterns matched
                if "great" in text.lower() and any(w in text.lower() for w in ["broke", "died", "failed"]):
                    detected_type = "verbal_irony"
                elif "SO " in text or "SOOO" in text:
                    detected_type = "hyperbole"
                else:
                    detected_type = "verbal_irony"

        # Check for amplifiers
        text_lower = text.lower()
        amplifier_count = sum(1 for w in _SARCASM_AMPLIFIERS if w in text_lower)
        if amplifier_count >= 2:
            match_count += 1

        return match_count, detected_type

    @staticmethod
    def _flip_sentiment(sentiment: str) -> str:
        """Flip sentiment polarity (sarcasm inverts meaning)."""
        flips = {
            "positive": "negative",
            "negative": "positive",
            "neutral": "neutral",
            "mixed": "negative",
        }
        return flips.get(sentiment, "neutral")

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @property
    def sarcasm_rate(self) -> float:
        if self._detection_count == 0:
            return 0.0
        return round(self._sarcasm_found / self._detection_count, 3)


# Singleton
sarcasm_agent = SarcasmAgent()
