"""
In-Memory Data Store — Central state management for processed reviews.

Provides:
  • Thread-safe storage for ProcessedReview objects
  • Aggregate statistics (sentiment distribution, language counts, sarcasm rate)
  • Query methods for dashboard data payloads
  • Anomaly and roadmap result caching
"""

from __future__ import annotations
import uuid
from datetime import datetime
from collections import defaultdict
from typing import Optional

from models.schemas import (
    ProcessedReview,
    TrendReport,
    StrategicRoadmap,
    HypeVsRealityReport,
    DashboardCustomerView,
    DashboardCompanyView,
    SentimentLabel,
)


class ReviewStore:
    """
    In-memory store for all processed reviews and derived intelligence.
    Acts as the single source of truth for the dashboard and API endpoints.
    """

    def __init__(self):
        self._reviews: dict[str, ProcessedReview] = {}
        self._trend_report: Optional[TrendReport] = None
        self._roadmap: Optional[StrategicRoadmap] = None
        self._hype_report: Optional[HypeVsRealityReport] = None
        self._start_time: datetime = datetime.now()

    # ── Write Operations ──────────────────────────────────────────────────

    def add_review(self, review: ProcessedReview) -> str:
        """Add a processed review to the store. Returns the review ID."""
        self._reviews[review.id] = review
        return review.id

    def add_reviews(self, reviews: list[ProcessedReview]) -> int:
        """Add multiple reviews. Returns count added."""
        for r in reviews:
            self._reviews[r.id] = r
        return len(reviews)

    def set_trend_report(self, report: TrendReport) -> None:
        self._trend_report = report

    def set_roadmap(self, roadmap: StrategicRoadmap) -> None:
        self._roadmap = roadmap

    def set_hype_report(self, report: HypeVsRealityReport) -> None:
        self._hype_report = report

    def clear(self) -> None:
        """Reset all stored data."""
        self._reviews.clear()
        self._trend_report = None
        self._roadmap = None
        self._hype_report = None

    # ── Read Operations ───────────────────────────────────────────────────

    def get_review(self, review_id: str) -> Optional[ProcessedReview]:
        return self._reviews.get(review_id)

    def get_all_reviews(self) -> list[ProcessedReview]:
        return list(self._reviews.values())

    def get_reviews_by_category(self, category: str) -> list[ProcessedReview]:
        return [r for r in self._reviews.values() if r.meta.category == category]

    @property
    def total_count(self) -> int:
        return len(self._reviews)

    @property
    def trend_report(self) -> Optional[TrendReport]:
        return self._trend_report

    @property
    def roadmap(self) -> Optional[StrategicRoadmap]:
        return self._roadmap

    # ── Aggregate Statistics ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return aggregate processing statistics."""
        if not self._reviews:
            return {
                "total_processed": 0,
                "sentiments": {},
                "languages": {},
                "sarcasm_rate": 0.0,
                "categories": {},
                "uptime_seconds": (datetime.now() - self._start_time).total_seconds(),
            }

        sentiments: dict[str, int] = defaultdict(int)
        languages: dict[str, int] = defaultdict(int)
        categories: dict[str, int] = defaultdict(int)
        sarcasm_count = 0
        total_confidence = 0.0

        for r in self._reviews.values():
            sentiments[r.sentiment.value] += 1
            languages[r.language.detected_code.value] += 1
            categories[r.meta.category] += 1
            total_confidence += r.overall_confidence
            if r.sarcasm.is_sarcastic:
                sarcasm_count += 1

        total = len(self._reviews)
        return {
            "total_processed": total,
            "sentiments": dict(sentiments),
            "languages": dict(languages),
            "categories": dict(categories),
            "sarcasm_rate": round(sarcasm_count / total, 3) if total else 0,
            "avg_confidence": round(total_confidence / total, 1) if total else 0,
            "uptime_seconds": round((datetime.now() - self._start_time).total_seconds(), 1),
        }

    # ── Dashboard Payloads ────────────────────────────────────────────────

    def get_customer_view(self) -> DashboardCustomerView:
        """Build the Customer View ('The Real Story') data payload."""
        reviews = list(self._reviews.values())
        if not reviews:
            return DashboardCustomerView()

        # Sentiment summary
        sentiments = defaultdict(int)
        for r in reviews:
            sentiments[r.sentiment.value] += 1

        # Aspect aggregation
        aspect_scores: dict[str, dict] = defaultdict(lambda: {"pos": 0, "neg": 0, "total": 0})
        for r in reviews:
            for a in r.aspects:
                key = a.aspect
                aspect_scores[key]["total"] += 1
                if a.sentiment == SentimentLabel.POSITIVE:
                    aspect_scores[key]["pos"] += 1
                elif a.sentiment == SentimentLabel.NEGATIVE:
                    aspect_scores[key]["neg"] += 1

        # Top strengths (highest positive ratio)
        strengths = sorted(
            [
                {"aspect": k, "score": round(v["pos"] / v["total"] * 100, 1), "mentions": v["total"]}
                for k, v in aspect_scores.items() if v["total"] > 0
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:3]

        # Top weaknesses (highest negative ratio)
        weaknesses = sorted(
            [
                {"aspect": k, "score": round(v["neg"] / v["total"] * 100, 1), "mentions": v["total"]}
                for k, v in aspect_scores.items() if v["total"] > 0
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:3]

        # Language distribution
        lang_dist = defaultdict(int)
        for r in reviews:
            lang_dist[r.language.detected_name] += 1

        # Honesty score: weighted average confidence adjusted for sarcasm detection
        sarcasm_count = sum(1 for r in reviews if r.sarcasm.is_sarcastic)
        avg_conf = sum(r.overall_confidence for r in reviews) / len(reviews)
        honesty = min(100, avg_conf + (sarcasm_count / len(reviews)) * 10)

        # Curated highlights
        highlights = []
        for r in sorted(reviews, key=lambda x: x.overall_confidence, reverse=True)[:5]:
            highlights.append({
                "text": r.meta.original_text[:200],
                "sentiment": r.sentiment.value,
                "confidence": r.overall_confidence,
                "language": r.language.detected_name,
                "is_sarcastic": r.sarcasm.is_sarcastic,
            })

        return DashboardCustomerView(
            honesty_score=round(honesty, 1),
            total_reviews=len(reviews),
            top_strengths=strengths,
            top_weaknesses=weaknesses,
            sarcasm_alert_count=sarcasm_count,
            language_distribution=dict(lang_dist),
            curated_highlights=highlights,
            sentiment_summary=dict(sentiments),
        )

    def get_company_view(self) -> DashboardCompanyView:
        """Build the Company View (Technical Deep-Dive) data payload."""
        reviews = list(self._reviews.values())
        if not reviews:
            return DashboardCompanyView()

        # Sentiment distribution
        sentiments = defaultdict(int)
        for r in reviews:
            sentiments[r.sentiment.value] += 1

        # Aspect heatmap
        aspect_data: dict[str, dict] = defaultdict(
            lambda: {"positive": 0, "negative": 0, "neutral": 0, "total": 0, "avg_confidence": 0.0}
        )
        for r in reviews:
            for a in r.aspects:
                aspect_data[a.aspect][a.sentiment.value] += 1
                aspect_data[a.aspect]["total"] += 1
                aspect_data[a.aspect]["avg_confidence"] += a.confidence

        heatmap = []
        for aspect, data in aspect_data.items():
            if data["total"] > 0:
                data["avg_confidence"] = round(data["avg_confidence"] / data["total"], 1)
            heatmap.append({"aspect": aspect, **data})

        # Processing stats
        stats = self.get_stats()

        return DashboardCompanyView(
            total_reviews=len(reviews),
            sentiment_distribution=dict(sentiments),
            aspect_heatmap=heatmap,
            anomaly_timeline=self._trend_report.anomalies if self._trend_report else [],
            issue_classifications=self._trend_report.issue_classifications if self._trend_report else [],
            hype_vs_reality=self._hype_report,
            strategic_roadmap=self._roadmap,
            processing_stats=stats,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Instance
# ═══════════════════════════════════════════════════════════════════════════════

store = ReviewStore()
