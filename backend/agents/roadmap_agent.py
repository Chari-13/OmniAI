"""
Roadmap Agent — Converts anomaly data and trend analysis into a
3-month strategic business roadmap using Gemini LLM.

Output structure:
  Month 1: Immediate Fixes (crisis response)
  Month 2: Medium-term Improvements (trust building)
  Month 3: Strategic Initiatives (long-term growth)

Each action includes: priority, impact estimate, KPI, and owner suggestion.
"""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Optional

from models.schemas import (
    TrendReport,
    StrategicRoadmap,
    RoadmapAction,
    SeverityLevel,
    IssueType,
)
from config import llm
from prompts.roadmap import build_roadmap_prompt


class RoadmapAgent:
    """
    Generates a strategic 3-month roadmap from intelligence data.

    Uses Gemini LLM with structured prompt to produce actionable,
    specific recommendations tied to detected anomalies and trends.
    Falls back to a template-based roadmap if LLM is unavailable.
    """

    def __init__(self):
        self._generation_count = 0

    async def generate(
        self,
        trend_report: TrendReport,
        sentiment_stats: dict,
        categories: list[str],
    ) -> StrategicRoadmap:
        """
        Generate a 3-month strategic roadmap.

        Args:
            trend_report: Complete trend analysis report
            sentiment_stats: Sentiment distribution dict
            categories: Product categories in the dataset

        Returns:
            StrategicRoadmap with prioritized actions
        """
        self._generation_count += 1

        # Prepare data for the prompt
        anomaly_summary = trend_report.summary if trend_report.summary else {}
        issue_classifications = [
            {
                "aspect": ic.aspect,
                "issue_type": ic.issue_type.value,
                "mention_count": ic.mention_count,
                "total_mentions": ic.total_mentions,
                "trend": ic.trend.value,
            }
            for ic in trend_report.issue_classifications
        ]

        # Try LLM-based generation
        if llm.is_available:
            prompt = build_roadmap_prompt(
                anomaly_summary=anomaly_summary,
                issue_classifications=issue_classifications,
                sentiment_stats=sentiment_stats,
                categories=categories,
            )

            result = await llm.generate_json(prompt, temperature=0.4)
            if result:
                return self._parse_llm_roadmap(result, trend_report)

        # Fallback: template-based roadmap
        return self._fallback_roadmap(trend_report, sentiment_stats)

    def _parse_llm_roadmap(
        self,
        raw: dict,
        trend_report: TrendReport,
    ) -> StrategicRoadmap:
        """Parse LLM output into a structured StrategicRoadmap."""
        actions = []
        for i, action_data in enumerate(raw.get("actions", [])):
            # Map priority string to SeverityLevel
            priority_str = action_data.get("priority", "MEDIUM").upper()
            try:
                priority = SeverityLevel(priority_str)
            except ValueError:
                priority = SeverityLevel.MEDIUM

            actions.append(RoadmapAction(
                id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
                month=action_data.get("month", 1),
                phase=self._get_phase_name(action_data.get("month", 1)),
                title=action_data.get("title", f"Action {i + 1}"),
                description=action_data.get("description", ""),
                priority=priority,
                category=action_data.get("category", "general"),
                affected_aspect=action_data.get("affected_aspect"),
                estimated_impact=action_data.get("estimated_impact", ""),
                kpi=action_data.get("kpi", ""),
                owner_suggestion=action_data.get("owner_suggestion", ""),
            ))

        return StrategicRoadmap(
            generated_at=datetime.now(),
            analysis_summary=raw.get("analysis_summary", ""),
            total_anomalies_addressed=len(trend_report.anomalies),
            actions=actions,
            month_1_theme=raw.get("month_1_theme", "Immediate Fixes"),
            month_2_theme=raw.get("month_2_theme", "Building Trust"),
            month_3_theme=raw.get("month_3_theme", "Strategic Growth"),
        )

    def _fallback_roadmap(
        self,
        trend_report: TrendReport,
        sentiment_stats: dict,
    ) -> StrategicRoadmap:
        """Generate a template-based roadmap when LLM is unavailable."""
        actions = []

        # Month 1: Address critical and high-severity anomalies
        critical_anomalies = [
            a for a in trend_report.anomalies
            if a.severity in (SeverityLevel.CRITICAL, SeverityLevel.HIGH)
        ]
        systemic_issues = [
            ic for ic in trend_report.issue_classifications
            if ic.issue_type == IssueType.SYSTEMIC
        ]

        # Generate actions from systemic issues
        for ic in systemic_issues:
            actions.append(RoadmapAction(
                id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
                month=1,
                phase="Immediate Fixes",
                title=f"Address {ic.aspect} complaints",
                description=(
                    f"Systemic issue detected: {ic.mention_count} negative mentions "
                    f"out of {ic.total_mentions} total for {ic.aspect}. "
                    f"Investigate root cause and implement fix."
                ),
                priority=SeverityLevel.HIGH,
                affected_aspect=ic.aspect,
                estimated_impact=f"Reduce negative {ic.aspect} mentions by 50%",
                kpi=f"{ic.aspect} satisfaction > 70%",
                owner_suggestion="Product Team",
            ))

        # Month 2: Process improvements
        degrading_issues = [
            ic for ic in trend_report.issue_classifications
            if ic.trend.value == "degrading"
        ]
        for ic in degrading_issues:
            actions.append(RoadmapAction(
                id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
                month=2,
                phase="Medium-term Improvements",
                title=f"Reverse {ic.aspect} degradation trend",
                description=(
                    f"{ic.aspect} sentiment is trending downward. "
                    f"Implement process improvements and quality checks."
                ),
                priority=SeverityLevel.MEDIUM,
                affected_aspect=ic.aspect,
                estimated_impact="Stabilize and begin reversing negative trend",
                kpi=f"{ic.aspect} trend direction: stable or improving",
                owner_suggestion="Quality Assurance Team",
            ))

        # Month 2: Customer communication
        neg_rate = sentiment_stats.get("negative", 0)
        total = sum(sentiment_stats.values()) if sentiment_stats else 1
        if neg_rate / max(total, 1) > 0.3:
            actions.append(RoadmapAction(
                id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
                month=2,
                phase="Medium-term Improvements",
                title="Launch customer satisfaction recovery program",
                description=(
                    f"Overall negative sentiment at {neg_rate/max(total,1):.0%}. "
                    f"Proactive outreach to affected customers with remediation."
                ),
                priority=SeverityLevel.HIGH,
                estimated_impact="Improve overall sentiment by 15-20%",
                kpi="Net Promoter Score increase by 10 points",
                owner_suggestion="Customer Success Team",
            ))

        # Month 3: Strategic initiatives
        actions.append(RoadmapAction(
            id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
            month=3,
            phase="Strategic Initiatives",
            title="Implement continuous review monitoring system",
            description=(
                "Deploy automated sentiment monitoring with real-time anomaly "
                "alerts to prevent future issues from escalating undetected."
            ),
            priority=SeverityLevel.MEDIUM,
            estimated_impact="Reduce mean time to detect issues from weeks to hours",
            kpi="Anomaly detection latency < 24 hours",
            owner_suggestion="Data Engineering Team",
        ))

        actions.append(RoadmapAction(
            id=f"ACT-{uuid.uuid4().hex[:8].upper()}",
            month=3,
            phase="Strategic Initiatives",
            title="Build competitive advantage from review insights",
            description=(
                "Use positive aspect trends to inform marketing strategy. "
                "Double down on strengths identified in customer reviews."
            ),
            priority=SeverityLevel.LOW,
            estimated_impact="Increase positive mention rate by 25%",
            kpi="Positive sentiment share > 60%",
            owner_suggestion="Marketing & Product Strategy",
        ))

        return StrategicRoadmap(
            generated_at=datetime.now(),
            analysis_summary=(
                f"Analysis of {trend_report.total_reviews} reviews found "
                f"{len(trend_report.anomalies)} anomalies and "
                f"{len(systemic_issues)} systemic issues requiring attention."
            ),
            total_anomalies_addressed=len(trend_report.anomalies),
            actions=actions,
            month_1_theme="Crisis Response: Immediate Fixes",
            month_2_theme="Building Trust: Process Improvements",
            month_3_theme="Growth: Strategic Innovation",
        )

    @staticmethod
    def _get_phase_name(month: int) -> str:
        phases = {
            1: "Immediate Fixes",
            2: "Medium-term Improvements",
            3: "Strategic Initiatives",
        }
        return phases.get(month, f"Month {month}")


# Singleton
roadmap_agent = RoadmapAgent()
