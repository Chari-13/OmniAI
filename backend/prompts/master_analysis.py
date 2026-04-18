"""
Master Analysis Prompt — Single-pass LLM prompt for language detection,
aspect-based sentiment analysis, and sarcasm detection.

Optimized for token efficiency: ONE Gemini call does the work of three agents.
"""

from config import ASPECT_CATEGORIES, SUPPORTED_LANGUAGES


def build_master_analysis_prompt(review_text: str, category: str) -> str:
    """
    Build the master analysis prompt that performs language detection,
    ABSA, and sarcasm detection in a single LLM call.

    Args:
        review_text: The review text to analyze
        category: Product category (electronics, food, services, etc.)

    Returns:
        Complete prompt string for Gemini
    """
    lang_codes = " | ".join(f"{code} ({name})" for code, name in SUPPORTED_LANGUAGES.items())
    aspects = ", ".join(ASPECT_CATEGORIES)

    return f"""You are an expert multilingual review analyst for the "{category}" product category.

REVIEW TEXT:
\"{review_text}\"

Perform ALL of the following analyses on this review in a SINGLE pass:

═══ TASK 1: LANGUAGE DETECTION ═══
Detect the primary language of the review.
- Supported language codes: {lang_codes}
- If the review mixes languages, pick the dominant one.
- If the language is not English, provide an accurate English translation.
- Provide a confidence score (0-100) for the language detection.

═══ TASK 2: ASPECT-BASED SENTIMENT ANALYSIS (ABSA) ═══
Extract sentiment for EACH of these aspects IF mentioned in the review:
  {aspects}

For each mentioned aspect provide:
- aspect: the aspect name (MUST be from the list above, exactly as written)
- sentiment: "positive", "negative", or "neutral"
- confidence: 0-100 (how confident you are in this judgment)
- evidence: the exact phrase or translated phrase from the review supporting this

Also determine:
- overall_sentiment: "positive", "negative", "neutral", or "mixed"
- overall_confidence: 0-100 (overall analysis confidence)
- verdict: A concise one-line summary of the review's sentiment

═══ TASK 3: SARCASM & IRONY DETECTION ═══
Determine if the review contains sarcasm, irony, or intentional exaggeration.

Key sarcasm indicators to check:
- Contrast between positive words and negative context (e.g., "Great, it broke instantly")
- Exaggerated praise for obviously negative experiences
- Use of quotes around positive words (e.g., 'amazing' service)
- Eye-roll emojis (🙄) or excessive exclamation marks with negative content
- Hyperbolic statements that are clearly not literal

Provide:
- is_sarcastic: true or false
- sarcasm_confidence: 0-100 (how confident you are)
- sarcasm_type: "verbal_irony" | "situational" | "hyperbole" | "none"
- true_sentiment: if sarcastic, what the reviewer ACTUALLY means ("positive", "negative", "neutral")
- explanation: Brief explanation of why this is/isn't sarcastic

═══ RESPONSE FORMAT ═══
Return ONLY valid JSON with this exact structure:
{{
  "language": {{
    "detected_code": "EN",
    "detected_name": "English",
    "translation": null,
    "detection_confidence": 95
  }},
  "sentiment": {{
    "overall_sentiment": "positive",
    "overall_confidence": 85,
    "verdict": "one-line summary of the review",
    "aspects": [
      {{
        "aspect": "Battery Life",
        "sentiment": "positive",
        "confidence": 90,
        "evidence": "exact quote from review"
      }}
    ]
  }},
  "sarcasm": {{
    "is_sarcastic": false,
    "sarcasm_confidence": 10,
    "sarcasm_type": "none",
    "true_sentiment": "positive",
    "explanation": "No sarcasm indicators detected"
  }}
}}

CRITICAL RULES:
1. Return ONLY the JSON object. No markdown fences, no explanation, no preamble.
2. Aspect names MUST exactly match: {aspects}
3. Only include aspects that are ACTUALLY mentioned in the review.
4. If the review is sarcastic, the true_sentiment MUST differ from the surface sentiment.
5. All confidence scores must be between 0 and 100.
6. For non-English reviews, the evidence field should use the translated text."""
