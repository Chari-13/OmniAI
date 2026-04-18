"""
Pydantic Schemas — Complete data models for the Review Intelligence Platform.

Covers:
  • Ingestion inputs (ReviewInput, CleanedReview)
  • Intelligence outputs (AspectSentiment, SarcasmResult, ProcessedReview)
  • Anomaly & Trend models (AnomalyAlert, TrendReport, IssueClassification)
  • Strategic Roadmap (RoadmapAction, StrategicRoadmap)
  • Social Sync (SocialClaim, HypeVsRealityReport)
  • Dashboard views (DashboardCustomerView, DashboardCompanyView)
"""

from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class SentimentLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class SeverityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class SarcasmType(str, Enum):
    VERBAL_IRONY = "verbal_irony"
    SITUATIONAL = "situational"
    HYPERBOLE = "hyperbole"
    NONE = "none"


class LanguageCode(str, Enum):
    EN = "EN"
    KN = "KN"
    HI = "HI"
    TE = "TE"
    TA = "TA"
    UNKNOWN = "UNKNOWN"


class IssueType(str, Enum):
    ISOLATED = "isolated"
    SYSTEMIC = "systemic"


class TrendDirection(str, Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Ingestion Models
# ═══════════════════════════════════════════════════════════════════════════════

class ReviewInput(BaseModel):
    """Raw review input from any ingestion source."""
    text: str = Field(..., min_length=1, description="Review text content")
    category: str = Field(default="general", description="Product category")
    source: str = Field(default="csv", description="Ingestion source: csv, json, api, voice, social")
    timestamp: Optional[datetime] = Field(default=None, description="Review timestamp")
    rating: Optional[float] = Field(default=None, ge=1, le=5, description="Star rating if available")
    reviewer_id: Optional[str] = Field(default=None, description="Reviewer identifier for dedup")


class CleanedReview(BaseModel):
    """Review after LLM-based cleaning and dedup."""
    original_text: str = Field(..., description="Original raw text")
    cleaned_text: str = Field(..., description="Cleaned and normalized text")
    category: str = Field(default="general")
    source: str = Field(default="csv")
    timestamp: Optional[datetime] = None
    rating: Optional[float] = None

    # Cleaning metadata
    typos_fixed: int = Field(default=0, description="Number of typos corrected")
    emojis_translated: int = Field(default=0, description="Number of emojis converted to text")

    # Spam/Bot detection
    spam_score: float = Field(default=0.0, ge=0, le=1, description="Spam probability (0-1)")
    is_bot: bool = Field(default=False, description="Flagged as bot-generated")
    is_duplicate: bool = Field(default=False, description="Flagged as near-duplicate")
    duplicate_of: Optional[str] = Field(default=None, description="ID of the original review if duplicate")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Intelligence Models
# ═══════════════════════════════════════════════════════════════════════════════

class LanguageInfo(BaseModel):
    """Language detection result."""
    detected_code: LanguageCode = Field(default=LanguageCode.EN, description="ISO-style language code")
    detected_name: str = Field(default="English", description="Human-readable language name")
    translation: Optional[str] = Field(default=None, description="English translation if non-English")
    detection_confidence: float = Field(default=0.0, ge=0, le=100, description="Detection confidence %")


class AspectSentiment(BaseModel):
    """Sentiment analysis for a single aspect."""
    aspect: str = Field(..., description="Aspect name (Battery Life, Taste, etc.)")
    sentiment: SentimentLabel = Field(..., description="Sentiment polarity")
    confidence: float = Field(default=0.0, ge=0, le=100, description="Confidence percentage")
    evidence: str = Field(default="", description="Exact phrase from review supporting this")


class SarcasmResult(BaseModel):
    """Sarcasm detection output."""
    is_sarcastic: bool = Field(default=False, description="Whether sarcasm was detected")
    sarcasm_confidence: float = Field(default=0.0, ge=0, le=100, description="Confidence %")
    sarcasm_type: SarcasmType = Field(default=SarcasmType.NONE, description="Type of sarcasm")
    true_sentiment: SentimentLabel = Field(
        default=SentimentLabel.NEUTRAL,
        description="What the reviewer ACTUALLY means (after sarcasm correction)"
    )
    explanation: str = Field(default="", description="Why this was flagged as sarcastic")


class ReviewMeta(BaseModel):
    """Processing metadata for a review."""
    original_text: str
    cleaned_text: Optional[str] = None
    category: str = "general"
    source: str = "csv"
    processed_at: datetime = Field(default_factory=datetime.now)
    engine_version: str = "2.0.0"
    llm_model: str = "gemini-2.0-flash"
    processing_time_ms: float = 0.0
    spam_score: float = 0.0
    is_bot: bool = False
    is_duplicate: bool = False


class ProcessedReview(BaseModel):
    """Complete intelligence output for a single review."""
    id: str = Field(..., description="Unique review identifier")
    language: LanguageInfo
    sentiment: SentimentLabel = Field(default=SentimentLabel.NEUTRAL, description="Overall sentiment")
    overall_confidence: float = Field(default=0.0, ge=0, le=100)
    verdict: str = Field(default="", description="One-line sentiment summary")
    aspects: list[AspectSentiment] = Field(default_factory=list)
    sarcasm: SarcasmResult = Field(default_factory=SarcasmResult)
    meta: ReviewMeta
    status: str = Field(default="success")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Anomaly & Trend Models
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyAlert(BaseModel):
    """A detected anomaly in review sentiment."""
    id: str = Field(..., description="Unique anomaly identifier")
    type: str = Field(..., description="sentiment_spike | aspect_spike | volume_spike")
    aspect: Optional[str] = Field(default=None, description="Affected aspect if aspect_spike")
    severity: SeverityLevel = Field(default=SeverityLevel.MEDIUM)
    spike_percentage: float = Field(default=0.0, description="Percentage increase that triggered alert")
    previous_rate: float = Field(default=0.0, description="Negative rate in previous window")
    current_rate: float = Field(default=0.0, description="Negative rate in current window")
    window_start: int = Field(default=0)
    window_end: int = Field(default=0)
    description: str = Field(default="")
    detected_at: datetime = Field(default_factory=datetime.now)
    affected_reviews: list[str] = Field(default_factory=list, description="IDs of reviews in this anomaly window")


class TrendWindow(BaseModel):
    """A single window in the trend analysis."""
    window_index: int
    window_start: int
    window_end: int
    negative_rate: float
    positive_rate: float
    review_count: int


class IssueClassification(BaseModel):
    """Classifies an issue as isolated or systemic."""
    aspect: str = Field(..., description="The aspect with the issue")
    issue_type: IssueType = Field(..., description="isolated (1-2 reviews) or systemic (3+ cluster)")
    mention_count: int = Field(default=0, description="Number of negative mentions")
    total_mentions: int = Field(default=0, description="Total mentions of this aspect")
    negative_ratio: float = Field(default=0.0, description="Ratio of negative to total")
    trend: TrendDirection = Field(default=TrendDirection.STABLE)
    sample_reviews: list[str] = Field(default_factory=list, description="Sample review IDs for context")
    description: str = Field(default="")


class TrendReport(BaseModel):
    """Complete trend analysis report."""
    total_reviews: int
    analysis_period: str = Field(default="", description="e.g. 'Jan 2026 - Mar 2026'")
    anomalies: list[AnomalyAlert] = Field(default_factory=list)
    trend_windows: list[TrendWindow] = Field(default_factory=list)
    issue_classifications: list[IssueClassification] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Strategic Roadmap Models
# ═══════════════════════════════════════════════════════════════════════════════

class RoadmapAction(BaseModel):
    """A single action item in the strategic roadmap."""
    id: str = Field(..., description="Action identifier")
    month: int = Field(..., ge=1, le=3, description="Target month (1-3)")
    phase: str = Field(..., description="Month 1: Immediate Fixes, Month 2: Improvements, Month 3: Strategic")
    title: str = Field(..., description="Action title")
    description: str = Field(default="", description="Detailed description")
    priority: SeverityLevel = Field(default=SeverityLevel.MEDIUM)
    category: str = Field(default="general", description="Related product category")
    affected_aspect: Optional[str] = Field(default=None, description="Related aspect")
    estimated_impact: str = Field(default="", description="Expected impact description")
    kpi: str = Field(default="", description="Key performance indicator to track")
    owner_suggestion: str = Field(default="", description="Suggested team/role to own this")
    triggered_by: Optional[str] = Field(default=None, description="Anomaly ID that triggered this action")


class StrategicRoadmap(BaseModel):
    """3-month strategic business roadmap."""
    generated_at: datetime = Field(default_factory=datetime.now)
    analysis_summary: str = Field(default="", description="Executive summary of findings")
    total_anomalies_addressed: int = Field(default=0)
    actions: list[RoadmapAction] = Field(default_factory=list)
    month_1_theme: str = Field(default="Immediate Fixes", description="Month 1 focus area")
    month_2_theme: str = Field(default="Medium-term Improvements", description="Month 2 focus area")
    month_3_theme: str = Field(default="Strategic Initiatives", description="Month 3 focus area")


# ═══════════════════════════════════════════════════════════════════════════════
# Social Sync Models
# ═══════════════════════════════════════════════════════════════════════════════

class SocialClaim(BaseModel):
    """A marketing claim extracted from social media."""
    platform: str = Field(default="instagram", description="Social media platform")
    claim_text: str = Field(..., description="The marketing claim")
    aspect: str = Field(..., description="Related aspect category")
    sentiment_implied: SentimentLabel = Field(default=SentimentLabel.POSITIVE)
    post_date: Optional[datetime] = None


class HypeVsRealityReport(BaseModel):
    """Comparison of marketing hype vs actual customer sentiment."""
    platform: str = Field(default="instagram")
    claims_analyzed: int = Field(default=0)
    reviews_compared: int = Field(default=0)
    overall_hype_score: float = Field(
        default=0.0, ge=0, le=100,
        description="0 = honest, 100 = pure hype"
    )
    aspect_gaps: list[dict] = Field(
        default_factory=list,
        description="Per-aspect gap between marketing claim and reality"
    )
    verdict: str = Field(default="", description="Summary of hype vs reality")


# ═══════════════════════════════════════════════════════════════════════════════
# Batch & Dashboard Models
# ═══════════════════════════════════════════════════════════════════════════════

class BatchResult(BaseModel):
    """Result of batch processing."""
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    spam_filtered: int = 0
    duplicates_found: int = 0
    results: list[ProcessedReview] = Field(default_factory=list)
    anomaly_report: Optional[TrendReport] = None
    processing_time_ms: float = 0.0


class DashboardCustomerView(BaseModel):
    """Data payload for the Customer View ('The Real Story')."""
    honesty_score: float = Field(default=0.0, ge=0, le=100, description="Overall honesty metric")
    total_reviews: int = 0
    top_strengths: list[dict] = Field(default_factory=list, description="Top 3 positive aspects")
    top_weaknesses: list[dict] = Field(default_factory=list, description="Top 3 negative aspects")
    sarcasm_alert_count: int = 0
    language_distribution: dict = Field(default_factory=dict)
    curated_highlights: list[dict] = Field(default_factory=list, description="Curated review highlights")
    sentiment_summary: dict = Field(default_factory=dict)


class DashboardCompanyView(BaseModel):
    """Data payload for the Company View (Technical Deep-Dive)."""
    total_reviews: int = 0
    sentiment_distribution: dict = Field(default_factory=dict)
    aspect_heatmap: list[dict] = Field(default_factory=list)
    anomaly_timeline: list[AnomalyAlert] = Field(default_factory=list)
    issue_classifications: list[IssueClassification] = Field(default_factory=list)
    hype_vs_reality: Optional[HypeVsRealityReport] = None
    strategic_roadmap: Optional[StrategicRoadmap] = None
    processing_stats: dict = Field(default_factory=dict)
