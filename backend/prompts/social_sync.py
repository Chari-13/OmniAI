"""
Social Sync Prompt — Extract marketing claims from social media transcripts
and compare them against actual customer review sentiment.
"""


def build_social_sync_prompt(social_posts: list[dict], aspect_summary: dict) -> str:
    """
    Build prompt for Social Sync analysis (Marketing Hype vs Actual Sentiment).

    Args:
        social_posts: List of social media posts with text and metadata
        aspect_summary: Aggregated aspect sentiment data from actual reviews

    Returns:
        Prompt string for Gemini
    """
    posts_text = "\n".join(
        f"  - [{p.get('platform', 'social')}] {p.get('text', '')}"
        for p in social_posts
    )

    aspect_text = "\n".join(
        f"  - {aspect}: {data.get('positive', 0)} positive, {data.get('negative', 0)} negative out of {data.get('total', 0)} mentions"
        for aspect, data in aspect_summary.items()
    )

    return f"""You are a marketing honesty analyst. Compare marketing claims from social media against actual customer review data.

═══ MARKETING POSTS (Social Media) ═══
{posts_text}

═══ ACTUAL CUSTOMER SENTIMENT DATA ═══
{aspect_text}

TASKS:
1. Extract each distinct marketing claim from the social media posts.
2. Map each claim to the relevant aspect category (Battery Life, Taste, Packaging, Price, Customer Support).
3. Compare the marketing sentiment vs actual customer sentiment for each aspect.
4. Calculate a "Hype Score" per aspect: 0 = fully honest, 100 = pure hype/misleading.
5. Provide an overall verdict.

Return ONLY valid JSON:
{{
  "claims": [
    {{
      "claim_text": "the marketing claim",
      "aspect": "Battery Life",
      "marketing_sentiment": "positive",
      "actual_sentiment": "negative",
      "hype_score": 85,
      "gap_description": "Marketing claims 2-day battery life, but 60% of reviews report poor battery"
    }}
  ],
  "overall_hype_score": 45,
  "verdict": "Overall assessment of marketing honesty",
  "recommendations": ["suggestion 1", "suggestion 2"]
}}

RULES:
- hype_score: 0 (honest) to 100 (misleading)
- Only include aspects that appear in BOTH marketing claims and review data
- Be objective and data-driven in your assessment
- Return ONLY the JSON. No markdown, no explanation."""
