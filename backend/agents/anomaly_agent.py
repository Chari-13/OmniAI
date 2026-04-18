"""
Anomaly Agent — Time-series spike detection and issue classification.

Provides:
  • Rolling window negative sentiment spike detection (>25% threshold)
  • Per-aspect anomaly tracking
  • Isolated vs Systemic issue classification
  • Severity grading: CRITICAL / HIGH / MEDIUM / LOW
"""

from __future__ import annotations
import uuid
from datetime import datetime
from collections import defaultdict
from typing import Optional

from models.schemas import (
    ProcessedReview,
    AnomalyAlert,
    TrendWindow,
    TrendReport,
    IssueClassification,
    SentimentLabel,
    SeverityLevel,
    IssueType,
    TrendDirection,
)
from config import ANOMALY_SPIKE_THRESHOLD, SYSTEMIC_ISSUE_MIN_COUNT, ASPECT_CATEGORIES


class AnomalyAgent:
    """
    Detects anomalies in review sentiment using rolling window analysis.

    Key capabilities:
    1. Sentiment spike detection: flags windows where negative sentiment
       increases by >25% compared to the previous window
    2. Aspect-level anomalies: tracks per-aspect negative mention spikes
    3. Issue classification: separates isolated complaints (1-2 mentions)
       from systemic issues (3+ clustered mentions)
    4. Trend direction: classifies each aspect as improving/stable/degrading
    """

    def __init__(self):
        self._analysis_count = 0

    def analyze(
        self,
        reviews: list[ProcessedReview],
        window_size: int = 5,
    ) -> TrendReport:
        """
        Run full anomaly and trend analysis on a batch of processed reviews.

        Args:
            reviews: List of ProcessedReview objects
            window_size: Number of reviews per rolling window

        Returns:
            Complete TrendReport with anomalies, trends, and classifications
        """
        self._analysis_count += 1

        if len(reviews) < window_size:
            return TrendReport(
                total_reviews=len(reviews),
                summary={"message": f"Need at least {window_size} reviews for analysis"},
            )

        # 1. Overall sentiment spike detection
        anomalies = self._detect_sentiment_spikes(reviews, window_size)

        # 2. Per-aspect anomaly detection
        aspect_anomalies = self._detect_aspect_spikes(reviews, window_size)
        anomalies.extend(aspect_anomalies)

        # 3. Build trend windows
        trend_windows = self._build_trend_windows(reviews, window_size)

        # 4. Classify issues (isolated vs systemic)
        classifications = self._classify_issues(reviews)

        # 5. Build summary
        summary = {
            "total_anomalies": len(anomalies),
            "critical": sum(1 for a in anomalies if a.severity == SeverityLevel.CRITICAL),
            "high": sum(1 for a in anomalies if a.severity == SeverityLevel.HIGH),
            "medium": sum(1 for a in anomalies if a.severity == SeverityLevel.MEDIUM),
            "low": sum(1 for a in anomalies if a.severity == SeverityLevel.LOW),
            "systemic_issues": sum(1 for c in classifications if c.issue_type == IssueType.SYSTEMIC),
            "isolated_issues": sum(1 for c in classifications if c.issue_type == IssueType.ISOLATED),
        }

        # Determine analysis period
        timestamps = [r.meta.processed_at for r in reviews if r.meta.processed_at]
        period = ""
        if timestamps:
            start = min(timestamps).strftime("%b %Y")
            end = max(timestamps).strftime("%b %Y")
            period = f"{start} - {end}" if start != end else start

        return TrendReport(
            total_reviews=len(reviews),
            analysis_period=period,
            anomalies=anomalies,
            trend_windows=trend_windows,
            issue_classifications=classifications,
            summary=summary,
        )

    # ── Sentiment Spike Detection ─────────────────────────────────────────

    def _detect_sentiment_spikes(
        self,
        reviews: list[ProcessedReview],
        window_size: int,
    ) -> list[AnomalyAlert]:
        """Detect windows where negative sentiment spikes >25%."""
        anomalies = []

        # Build negative sentiment flags
        neg_flags = [
            1.0 if r.sentiment == SentimentLabel.NEGATIVE else 0.0
            for r in reviews
        ]

        # Calculate rolling window negative rates
        windows = []
        for i in range(len(neg_flags) - window_size + 1):
            window = neg_flags[i: i + window_size]
            neg_rate = sum(window) / len(window)
            windows.append((i, neg_rate))

        # Compare consecutive windows for spikes
        for i in range(1, len(windows)):
            prev_idx, prev_rate = windows[i - 1]
            curr_idx, curr_rate = windows[i]

            if prev_rate > 0:
                change = (curr_rate - prev_rate) / prev_rate
            elif curr_rate > 0:
                change = 1.0  # Spike from zero
            else:
                change = 0.0

            if change > ANOMALY_SPIKE_THRESHOLD:
                spike_pct = round(change * 100, 1)
                severity = self._grade_severity(spike_pct)

                # Collect affected review IDs
                affected = [
                    reviews[j].id
                    for j in range(curr_idx, min(curr_idx + window_size, len(reviews)))
                ]

                anomalies.append(AnomalyAlert(
                    id=f"ANO-{uuid.uuid4().hex[:8].upper()}",
                    type="sentiment_spike",
                    severity=severity,
                    spike_percentage=spike_pct,
                    previous_rate=round(prev_rate, 3),
                    current_rate=round(curr_rate, 3),
                    window_start=curr_idx,
                    window_end=curr_idx + window_size - 1,
                    description=(
                        f"Negative sentiment spiked by {spike_pct}% "
                        f"(from {prev_rate:.0%} to {curr_rate:.0%}) "
                        f"at review window [{curr_idx}-{curr_idx + window_size - 1}]."
                    ),
                    affected_reviews=affected,
                ))

        return anomalies

    # ── Aspect-Level Spike Detection ──────────────────────────────────────

    def _detect_aspect_spikes(
        self,
        reviews: list[ProcessedReview],
        window_size: int,
    ) -> list[AnomalyAlert]:
        """Detect per-aspect negative mention spikes."""
        anomalies = []

        # Build per-aspect negative timelines
        aspect_timeline: dict[str, list[float]] = defaultdict(list)
        for r in reviews:
            for a in r.aspects:
                is_neg = 1.0 if a.sentiment == SentimentLabel.NEGATIVE else 0.0
                aspect_timeline[a.aspect].append(is_neg)

        # Analyze each aspect
        for aspect, neg_flags in aspect_timeline.items():
            if len(neg_flags) < window_size:
                continue

            for i in range(1, len(neg_flags) - window_size + 1):
                prev_window = neg_flags[i - 1: i - 1 + window_size]
                curr_window = neg_flags[i: i + window_size]

                prev_rate = sum(prev_window) / len(prev_window)
                curr_rate = sum(curr_window) / len(curr_window)

                if prev_rate > 0:
                    change = (curr_rate - prev_rate) / prev_rate
                elif curr_rate > 0:
                    change = 1.0
                else:
                    change = 0.0

                if change > ANOMALY_SPIKE_THRESHOLD:
                    spike_pct = round(change * 100, 1)
                    anomalies.append(AnomalyAlert(
                        id=f"ANO-{uuid.uuid4().hex[:8].upper()}",
                        type="aspect_spike",
                        aspect=aspect,
                        severity=SeverityLevel.HIGH if change > 0.5 else SeverityLevel.MEDIUM,
                        spike_percentage=spike_pct,
                        previous_rate=round(prev_rate, 3),
                        current_rate=round(curr_rate, 3),
                        description=(
                            f"Aspect '{aspect}' saw a {spike_pct}% spike in negative mentions "
                            f"(from {prev_rate:.0%} to {curr_rate:.0%})."
                        ),
                    ))
                    break  # One alert per aspect is sufficient

        return anomalies

    # ── Trend Windows ─────────────────────────────────────────────────────

    def _build_trend_windows(
        self,
        reviews: list[ProcessedReview],
        window_size: int,
    ) -> list[TrendWindow]:
        """Build rolling window trend data for visualization."""
        windows = []

        for i in range(len(reviews) - window_size + 1):
            window_reviews = reviews[i: i + window_size]
            neg_count = sum(1 for r in window_reviews if r.sentiment == SentimentLabel.NEGATIVE)
            pos_count = sum(1 for r in window_reviews if r.sentiment == SentimentLabel.POSITIVE)

            windows.append(TrendWindow(
                window_index=i,
                window_start=i,
                window_end=i + window_size - 1,
                negative_rate=round(neg_count / window_size, 3),
                positive_rate=round(pos_count / window_size, 3),
                review_count=window_size,
            ))

        return windows

    # ── Issue Classification ──────────────────────────────────────────────

    def _classify_issues(self, reviews: list[ProcessedReview]) -> list[IssueClassification]:
        """
        Classify each aspect's issues as isolated (1-2 mentions)
        or systemic (3+ clustered mentions).
        """
        classifications = []

        # Aggregate per-aspect data
        aspect_data: dict[str, dict] = defaultdict(
            lambda: {"negative": 0, "positive": 0, "neutral": 0, "total": 0, "review_ids": []}
        )

        for r in reviews:
            for a in r.aspects:
                data = aspect_data[a.aspect]
                data["total"] += 1
                data[a.sentiment.value] += 1
                if a.sentiment == SentimentLabel.NEGATIVE:
                    data["review_ids"].append(r.id)

        for aspect in ASPECT_CATEGORIES:
            data = aspect_data.get(aspect)
            if not data or data["total"] == 0:
                continue

            neg_count = data["negative"]
            total = data["total"]
            neg_ratio = neg_count / total if total > 0 else 0

            # Classify: isolated (1-2 negative) vs systemic (3+)
            if neg_count >= SYSTEMIC_ISSUE_MIN_COUNT:
                issue_type = IssueType.SYSTEMIC
            elif neg_count > 0:
                issue_type = IssueType.ISOLATED
            else:
                continue  # No issues for this aspect

            # Determine trend direction
            # Simple heuristic: look at the last half vs first half
            review_ids = data["review_ids"]
            if len(review_ids) >= 4:
                mid = len(review_ids) // 2
                first_half = mid
                second_half = len(review_ids) - mid
                if second_half > first_half * 1.5:
                    trend = TrendDirection.DEGRADING
                elif second_half < first_half * 0.5:
                    trend = TrendDirection.IMPROVING
                else:
                    trend = TrendDirection.STABLE
            else:
                trend = TrendDirection.STABLE

            classifications.append(IssueClassification(
                aspect=aspect,
                issue_type=issue_type,
                mention_count=neg_count,
                total_mentions=total,
                negative_ratio=round(neg_ratio, 3),
                trend=trend,
                sample_reviews=review_ids[:5],  # Cap at 5 samples
                description=(
                    f"{aspect}: {neg_count} negative out of {total} mentions "
                    f"({neg_ratio:.0%}) — classified as {issue_type.value}, "
                    f"trend: {trend.value}"
                ),
            ))

        return classifications

    # ── Utility ───────────────────────────────────────────────────────────

    @staticmethod
    def _grade_severity(spike_pct: float) -> SeverityLevel:
        """Grade anomaly severity based on spike percentage."""
        if spike_pct > 100:
            return SeverityLevel.CRITICAL
        elif spike_pct > 50:
            return SeverityLevel.HIGH
        elif spike_pct > 25:
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW


# Singleton
anomaly_agent = AnomalyAgent()
