"""
Sentiment Agent — Aspect-Based Sentiment Analysis (ABSA).

Performs granular sentiment extraction for 5 aspect categories:
  Battery Life, Taste, Packaging, Price, Customer Support

Each aspect gets: sentiment polarity, confidence score (0-100), and evidence quote.
Uses the master analysis prompt for LLM-based analysis, with keyword fallback.
"""

from __future__ import annotations
from typing import Optional
from models.schemas import (
    AspectSentiment,
    SentimentLabel,
    ProcessedReview,
    LanguageInfo,
    SarcasmResult,
    ReviewMeta,
    SarcasmType,
    LanguageCode,
)
from config import llm, ASPECT_CATEGORIES
from prompts.master_analysis import build_master_analysis_prompt


# Keyword banks for fallback analysis
_POSITIVE_KEYWORDS = {
    "good", "great", "amazing", "love", "excellent", "best", "awesome",
    "fantastic", "happy", "perfect", "wonderful", "superb", "brilliant",
    "outstanding", "delicious", "fresh", "reliable", "fast", "worth",
}
_NEGATIVE_KEYWORDS = {
    "bad", "terrible", "worst", "hate", "awful", "horrible", "poor",
    "broken", "waste", "disappointed", "useless", "slow", "expensive",
    "overpriced", "stale", "damaged", "crushed", "leak", "defective",
}

# Aspect keyword mapping for fallback
_ASPECT_KEYWORDS = {
    "Battery Life": {"battery", "charge", "charging", "power", "dies", "lasts", "drain"},
    "Taste": {"taste", "flavor", "delicious", "stale", "fresh", "yummy", "bland"},
    "Packaging": {"packaging", "package", "box", "wrap", "crushed", "damaged", "arrived"},
    "Price": {"price", "expensive", "cheap", "affordable", "cost", "worth", "money", "value"},
    "Customer Support": {"support", "service", "reply", "response", "help", "contact", "email", "call"},
}


class SentimentAgent:
    """
    Performs ABSA using Gemini LLM with keyword-based fallback.

    The agent uses the master analysis prompt to get all intelligence
    in a single LLM call, then extracts and validates the sentiment data.
    """

    def __init__(self):
        self._analysis_count = 0

    async def analyze(
        self,
        text: str,
        category: str = "general",
        review_id: str = "",
    ) -> dict:
        """
        Run full sentiment analysis on a review.

        Returns a dict with keys: language, sentiment, aspects, sarcasm, etc.
        This is the primary analysis entry point used by the dispatcher.
        """
        self._analysis_count += 1

        # Try LLM-based analysis
        if llm.is_available:
            prompt = build_master_analysis_prompt(text, category)
            result = await llm.generate_json(prompt)
            if result:
                return self._validate_llm_result(result, text, category)

        # Fallback to keyword analysis
        return self._fallback_analysis(text, category)

    def _validate_llm_result(self, raw: dict, text: str, category: str) -> dict:
        """Validate and normalize LLM response to ensure consistent schema."""
        result = {}

        # Language
        lang_data = raw.get("language", {})
        result["language"] = {
            "detected_code": lang_data.get("detected_code", "EN"),
            "detected_name": lang_data.get("detected_name", "English"),
            "translation": lang_data.get("translation"),
            "detection_confidence": lang_data.get("detection_confidence", 80),
        }

        # Sentiment
        sent_data = raw.get("sentiment", {})
        result["overall_sentiment"] = sent_data.get("overall_sentiment", "neutral")
        result["overall_confidence"] = min(100, max(0, sent_data.get("overall_confidence", 50)))
        result["verdict"] = sent_data.get("verdict", "")

        # Aspects — validate each one
        raw_aspects = sent_data.get("aspects", [])
        validated_aspects = []
        for a in raw_aspects:
            aspect_name = a.get("aspect", "")
            # Only keep aspects from our defined list
            if aspect_name in ASPECT_CATEGORIES:
                validated_aspects.append({
                    "aspect": aspect_name,
                    "sentiment": a.get("sentiment", "neutral"),
                    "confidence": min(100, max(0, a.get("confidence", 50))),
                    "evidence": a.get("evidence", ""),
                })
        result["aspects"] = validated_aspects

        # Sarcasm
        sarc_data = raw.get("sarcasm", {})
        result["sarcasm"] = {
            "is_sarcastic": sarc_data.get("is_sarcastic", False),
            "sarcasm_confidence": min(100, max(0, sarc_data.get("sarcasm_confidence", 0))),
            "sarcasm_type": sarc_data.get("sarcasm_type", "none"),
            "true_sentiment": sarc_data.get("true_sentiment", result["overall_sentiment"]),
            "explanation": sarc_data.get("explanation", ""),
        }

        return result

    def _fallback_analysis(self, text: str, category: str) -> dict:
        """Keyword-based fallback when LLM is unavailable."""
        import re
        text_lower = text.lower()
        words = set(re.findall(r'\w+', text_lower))

        # Overall sentiment via keyword counting
        pos_count = len(words & _POSITIVE_KEYWORDS)
        neg_count = len(words & _NEGATIVE_KEYWORDS)

        if pos_count > neg_count:
            overall = "positive"
        elif neg_count > pos_count:
            overall = "negative"
        else:
            overall = "neutral"

        # Aspect extraction via keyword matching
        aspects = []
        for aspect, keywords in _ASPECT_KEYWORDS.items():
            if words & keywords:
                # Determine aspect sentiment from surrounding context
                aspect_sentiment = overall  # simplified: use overall as proxy
                aspects.append({
                    "aspect": aspect,
                    "sentiment": aspect_sentiment,
                    "confidence": 35,
                    "evidence": f"Keyword match: {words & keywords}",
                })

        # Language detection (basic)
        kannada = bool(re.search(r'[\u0C80-\u0CFF]', text))
        hindi = bool(re.search(r'[\u0900-\u097F]', text))
        telugu = bool(re.search(r'[\u0C00-\u0C7F]', text))
        tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))

        if kannada:
            lang_code, lang_name = "KN", "Kannada"
        elif hindi:
            lang_code, lang_name = "HI", "Hindi"
        elif telugu:
            lang_code, lang_name = "TE", "Telugu"
        elif tamil:
            lang_code, lang_name = "TA", "Tamil"
        else:
            lang_code, lang_name = "EN", "English"

        return {
            "language": {
                "detected_code": lang_code,
                "detected_name": lang_name,
                "translation": None,
                "detection_confidence": 40,
            },
            "overall_sentiment": overall,
            "overall_confidence": 35,
            "verdict": f"Fallback analysis: {overall} (LLM unavailable)",
            "aspects": aspects,
            "sarcasm": {
                "is_sarcastic": False,
                "sarcasm_confidence": 0,
                "sarcasm_type": "none",
                "true_sentiment": overall,
                "explanation": "Sarcasm detection requires LLM",
            },
        }

    @property
    def analysis_count(self) -> int:
        return self._analysis_count


# Singleton
sentiment_agent = SentimentAgent()
