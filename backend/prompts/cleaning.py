"""
Cleaning Prompt — LLM-based text cleaning and spam/bot detection.

Handles: typos, emoji translation, HTML stripping, bot pattern detection.
"""


def build_cleaning_prompt(review_text: str) -> str:
    """
    Build prompt for LLM-based review text cleaning.

    The LLM will:
    1. Fix obvious typos and grammatical errors
    2. Translate emojis to descriptive text
    3. Strip HTML tags or markdown formatting
    4. Detect spam/bot indicators
    5. Assess review authenticity

    Args:
        review_text: Raw review text to clean

    Returns:
        Prompt string for Gemini
    """
    return f"""You are a text cleaning specialist. Clean the following review text and assess its authenticity.

RAW REVIEW TEXT:
\"{review_text}\"

TASKS:
1. FIX TYPOS: Correct obvious spelling and grammar errors while preserving the reviewer's voice.
2. EMOJI TRANSLATION: Convert emojis to descriptive text in brackets (e.g., 🙄 → [eye roll], 😊 → [smiling]).
3. STRIP NOISE: Remove HTML tags, excessive punctuation (!!!!! → !), and formatting artifacts.
4. PRESERVE MEANING: Do NOT change the meaning, tone, or intent of the review.
5. SPAM DETECTION: Assess if this review shows signs of being spam or bot-generated.

Spam/bot indicators to check:
- Generic, overly promotional language
- Unnatural repetition of product names or keywords
- Contains suspicious links or promotional codes
- Grammatically perfect but semantically empty
- Copy-paste patterns or template-like structure

Return ONLY valid JSON:
{{
  "cleaned_text": "the cleaned review text",
  "typos_fixed": 2,
  "emojis_translated": 1,
  "spam_score": 0.15,
  "is_bot": false,
  "spam_indicators": ["none detected"]
}}

RULES:
- spam_score: 0.0 (definitely human) to 1.0 (definitely spam/bot)
- is_bot: true only if spam_score > 0.7
- Return ONLY the JSON. No markdown, no explanation."""
