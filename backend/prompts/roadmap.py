"""
Roadmap Prompt — Generate a 3-month strategic business roadmap
from anomaly data, trend analysis, and issue classifications.
"""


def build_roadmap_prompt(
    anomaly_summary: dict,
    issue_classifications: list[dict],
    sentiment_stats: dict,
    categories: list[str],
) -> str:
    """
    Build prompt for strategic 3-month roadmap generation.

    Args:
        anomaly_summary: Summary of detected anomalies (counts by severity)
        issue_classifications: Isolated vs systemic issue data
        sentiment_stats: Overall sentiment distribution
        categories: Product categories in the dataset

    Returns:
        Prompt string for Gemini
    """
    anomaly_text = "\n".join(
        f"  - {k}: {v}" for k, v in anomaly_summary.items()
    )

    issue_text = "\n".join(
        f"  - {ic.get('aspect', 'Unknown')}: {ic.get('issue_type', 'unknown')} "
        f"({ic.get('mention_count', 0)} negative out of {ic.get('total_mentions', 0)} total, "
        f"trend: {ic.get('trend', 'stable')})"
        for ic in issue_classifications
    )

    sentiment_text = "\n".join(
        f"  - {k}: {v}" for k, v in sentiment_stats.items()
    )

    categories_text = ", ".join(categories) if categories else "general"

    return f"""You are a senior business strategy consultant analyzing customer review intelligence data.
Based on the following analysis results, generate a detailed 3-month strategic business roadmap.

═══ ANOMALY SUMMARY ═══
{anomaly_text}

═══ ISSUE CLASSIFICATIONS ═══
{issue_text}

═══ SENTIMENT DISTRIBUTION ═══
{sentiment_text}

═══ PRODUCT CATEGORIES ═══
{categories_text}

GENERATE A 3-MONTH STRATEGIC ROADMAP:

Month 1 — "Immediate Fixes" (Crisis Response)
- Address CRITICAL and HIGH severity issues
- Quick wins to stop sentiment bleeding
- Focus on the most complained-about aspects

Month 2 — "Medium-term Improvements" (Building Back Trust)
- Tackle systemic issues identified in the data
- Process improvements based on aspect-level trends
- Customer communication strategy

Month 3 — "Strategic Initiatives" (Long-term Growth)
- Proactive measures to prevent future issues
- Competitive differentiation based on strengths
- Innovation roadmap informed by customer voice

For EACH action item provide:
- title: Clear, actionable title
- description: 2-3 sentence description of what to do
- priority: CRITICAL, HIGH, MEDIUM, or LOW
- category: Which product category this applies to
- affected_aspect: Which of the 5 aspects this addresses (if applicable)
- estimated_impact: Expected improvement description
- kpi: Specific KPI to track success
- owner_suggestion: Suggested team/role to own this

Return ONLY valid JSON:
{{
  "analysis_summary": "Executive summary of the current situation in 2-3 sentences",
  "month_1_theme": "Crisis Response: Immediate Fixes",
  "month_2_theme": "Building Trust: Process Improvements",
  "month_3_theme": "Growth: Strategic Innovation",
  "actions": [
    {{
      "month": 1,
      "title": "Action title",
      "description": "What to do and why",
      "priority": "CRITICAL",
      "category": "electronics",
      "affected_aspect": "Battery Life",
      "estimated_impact": "Expected to reduce negative battery mentions by 40%",
      "kpi": "Battery satisfaction score > 75%",
      "owner_suggestion": "Product Engineering Team"
    }}
  ]
}}

RULES:
- Generate 3-5 actions per month (9-15 total)
- Each action must be specific and actionable, not generic platitudes
- Priorities should match the severity of the underlying issue
- KPIs must be measurable and specific
- Return ONLY the JSON. No markdown, no explanation."""
