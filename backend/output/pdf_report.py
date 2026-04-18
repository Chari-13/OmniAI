"""
PDF Report Generator -- Professional document export with charts.

Generates a multi-section PDF report containing:
  - Executive Summary
  - Sentiment Breakdown (pie chart)
  - Aspect Analysis (bar chart)
  - Anomaly Alerts (table)
  - Trend Visualization (line chart)
  - 3-Month Strategic Roadmap
  - Issue Classifications
"""

from __future__ import annotations
import io
import os
from datetime import datetime
from typing import Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from models.schemas import (
    ProcessedReview,
    TrendReport,
    StrategicRoadmap,
    SentimentLabel,
    SeverityLevel,
)


# Color palette
_COLORS = {
    "primary": colors.HexColor("#6366F1"),      # Indigo
    "positive": colors.HexColor("#10B981"),      # Emerald
    "negative": colors.HexColor("#EF4444"),      # Red
    "neutral": colors.HexColor("#8B5CF6"),       # Violet
    "mixed": colors.HexColor("#F59E0B"),         # Amber
    "bg_dark": colors.HexColor("#1E1B4B"),       # Dark indigo
    "bg_light": colors.HexColor("#F8FAFC"),      # Slate 50
    "text": colors.HexColor("#1E293B"),          # Slate 800
    "text_light": colors.HexColor("#64748B"),    # Slate 500
    "critical": colors.HexColor("#DC2626"),
    "high": colors.HexColor("#EA580C"),
    "medium": colors.HexColor("#D97706"),
    "low": colors.HexColor("#65A30D"),
}

_SENTIMENT_COLORS_MPL = {
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral": "#8B5CF6",
    "mixed": "#F59E0B",
}


class PDFReportGenerator:
    """Generates professional PDF reports from intelligence data."""

    def __init__(self):
        self._styles = getSampleStyleSheet()
        self._setup_custom_styles()

    def _setup_custom_styles(self):
        """Define custom paragraph styles."""
        self._styles.add(ParagraphStyle(
            name="ReportTitle",
            parent=self._styles["Title"],
            fontSize=24,
            textColor=_COLORS["primary"],
            spaceAfter=6,
        ))
        self._styles.add(ParagraphStyle(
            name="SectionHeader",
            parent=self._styles["Heading1"],
            fontSize=16,
            textColor=_COLORS["bg_dark"],
            spaceBefore=16,
            spaceAfter=8,
        ))
        self._styles.add(ParagraphStyle(
            name="SubHeader",
            parent=self._styles["Heading2"],
            fontSize=12,
            textColor=_COLORS["primary"],
            spaceBefore=10,
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            name="BodyText2",
            parent=self._styles["BodyText"],
            fontSize=10,
            textColor=_COLORS["text"],
            spaceAfter=4,
        ))
        self._styles.add(ParagraphStyle(
            name="Caption",
            parent=self._styles["BodyText"],
            fontSize=8,
            textColor=_COLORS["text_light"],
            alignment=TA_CENTER,
        ))

    def generate(
        self,
        reviews: list[ProcessedReview],
        trend_report: Optional[TrendReport] = None,
        roadmap: Optional[StrategicRoadmap] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Generate the full PDF report.

        Args:
            reviews: List of processed reviews
            trend_report: Anomaly/trend analysis (optional)
            roadmap: Strategic roadmap (optional)
            output_path: Custom output path (optional)

        Returns:
            Path to the generated PDF file
        """
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"intelligence_report_{timestamp}.pdf")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=20*mm,
            leftMargin=20*mm,
            topMargin=20*mm,
            bottomMargin=20*mm,
        )

        story = []

        # --- Title Page ---
        story.append(Spacer(1, 60))
        story.append(Paragraph("Customer Review Intelligence Report", self._styles["ReportTitle"]))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
            self._styles["Caption"],
        ))
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=2, color=_COLORS["primary"]))
        story.append(Spacer(1, 20))

        # --- Executive Summary ---
        story.extend(self._build_executive_summary(reviews, trend_report))

        # --- Sentiment Breakdown ---
        story.append(PageBreak())
        story.extend(self._build_sentiment_section(reviews))

        # --- Aspect Analysis ---
        story.extend(self._build_aspect_section(reviews))

        # --- Anomaly Alerts ---
        if trend_report and trend_report.anomalies:
            story.append(PageBreak())
            story.extend(self._build_anomaly_section(trend_report))

        # --- Issue Classifications ---
        if trend_report and trend_report.issue_classifications:
            story.extend(self._build_issue_section(trend_report))

        # --- Strategic Roadmap ---
        if roadmap:
            story.append(PageBreak())
            story.extend(self._build_roadmap_section(roadmap))

        # Build PDF
        doc.build(story)
        return output_path

    # ── Section Builders ──────────────────────────────────────────────────

    def _build_executive_summary(
        self,
        reviews: list[ProcessedReview],
        trend_report: Optional[TrendReport],
    ) -> list:
        """Build the executive summary section."""
        elements = []
        elements.append(Paragraph("Executive Summary", self._styles["SectionHeader"]))

        total = len(reviews)
        pos = sum(1 for r in reviews if r.sentiment == SentimentLabel.POSITIVE)
        neg = sum(1 for r in reviews if r.sentiment == SentimentLabel.NEGATIVE)
        sarc = sum(1 for r in reviews if r.sarcasm.is_sarcastic)
        avg_conf = sum(r.overall_confidence for r in reviews) / max(total, 1)
        anomaly_count = len(trend_report.anomalies) if trend_report else 0

        summary_data = [
            ["Metric", "Value"],
            ["Total Reviews Analyzed", str(total)],
            ["Positive Sentiment", f"{pos} ({pos/max(total,1)*100:.0f}%)"],
            ["Negative Sentiment", f"{neg} ({neg/max(total,1)*100:.0f}%)"],
            ["Sarcastic Reviews Detected", str(sarc)],
            ["Average Confidence Score", f"{avg_conf:.1f}%"],
            ["Anomalies Detected", str(anomaly_count)],
        ]

        table = Table(summary_data, colWidths=[200, 200])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _COLORS["primary"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_COLORS["bg_light"], colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 16))

        return elements

    def _build_sentiment_section(self, reviews: list[ProcessedReview]) -> list:
        """Build sentiment distribution section with pie chart."""
        elements = []
        elements.append(Paragraph("Sentiment Distribution", self._styles["SectionHeader"]))

        # Generate pie chart
        sentiment_counts = {}
        for r in reviews:
            key = r.sentiment.value
            sentiment_counts[key] = sentiment_counts.get(key, 0) + 1

        if sentiment_counts:
            chart_path = self._generate_pie_chart(sentiment_counts, "Sentiment Distribution")
            if chart_path:
                elements.append(Image(chart_path, width=350, height=250))
                elements.append(Spacer(1, 8))

        return elements

    def _build_aspect_section(self, reviews: list[ProcessedReview]) -> list:
        """Build aspect analysis section with bar chart."""
        elements = []
        elements.append(Paragraph("Aspect-Based Analysis", self._styles["SectionHeader"]))

        # Aggregate aspect data
        aspect_data = {}
        for r in reviews:
            for a in r.aspects:
                if a.aspect not in aspect_data:
                    aspect_data[a.aspect] = {"positive": 0, "negative": 0, "neutral": 0}
                aspect_data[a.aspect][a.sentiment.value] += 1

        if aspect_data:
            chart_path = self._generate_aspect_bar_chart(aspect_data)
            if chart_path:
                elements.append(Image(chart_path, width=450, height=280))
                elements.append(Spacer(1, 8))

            # Aspect table
            table_data = [["Aspect", "Positive", "Negative", "Neutral", "Total"]]
            for aspect, counts in sorted(aspect_data.items()):
                total = sum(counts.values())
                table_data.append([
                    aspect,
                    str(counts.get("positive", 0)),
                    str(counts.get("negative", 0)),
                    str(counts.get("neutral", 0)),
                    str(total),
                ])

            table = Table(table_data, colWidths=[120, 70, 70, 70, 70])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _COLORS["primary"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_COLORS["bg_light"], colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(table)

        elements.append(Spacer(1, 16))
        return elements

    def _build_anomaly_section(self, trend_report: TrendReport) -> list:
        """Build anomaly alerts section."""
        elements = []
        elements.append(Paragraph("Anomaly Alerts", self._styles["SectionHeader"]))

        table_data = [["ID", "Type", "Severity", "Spike %", "Description"]]
        for a in trend_report.anomalies:
            table_data.append([
                a.id[:12],
                a.type.replace("_", " ").title(),
                a.severity.value,
                f"{a.spike_percentage:.1f}%",
                a.description[:80],
            ])

        table = Table(table_data, colWidths=[70, 80, 60, 50, 240])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _COLORS["critical"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FEF2F2"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 16))

        return elements

    def _build_issue_section(self, trend_report: TrendReport) -> list:
        """Build issue classification section."""
        elements = []
        elements.append(Paragraph("Issue Classifications", self._styles["SubHeader"]))

        table_data = [["Aspect", "Type", "Neg / Total", "Ratio", "Trend"]]
        for ic in trend_report.issue_classifications:
            table_data.append([
                ic.aspect,
                ic.issue_type.value.upper(),
                f"{ic.mention_count} / {ic.total_mentions}",
                f"{ic.negative_ratio:.0%}",
                ic.trend.value,
            ])

        table = Table(table_data, colWidths=[100, 80, 80, 60, 80])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _COLORS["bg_dark"]),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_COLORS["bg_light"], colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(table)
        elements.append(Spacer(1, 16))

        return elements

    def _build_roadmap_section(self, roadmap: StrategicRoadmap) -> list:
        """Build strategic roadmap section."""
        elements = []
        elements.append(Paragraph("3-Month Strategic Roadmap", self._styles["SectionHeader"]))

        if roadmap.analysis_summary:
            elements.append(Paragraph(roadmap.analysis_summary, self._styles["BodyText2"]))
            elements.append(Spacer(1, 10))

        for month_num, theme_attr in [(1, "month_1_theme"), (2, "month_2_theme"), (3, "month_3_theme")]:
            theme = getattr(roadmap, theme_attr, f"Month {month_num}")
            elements.append(Paragraph(f"Month {month_num}: {theme}", self._styles["SubHeader"]))

            month_actions = [a for a in roadmap.actions if a.month == month_num]
            if not month_actions:
                elements.append(Paragraph("No specific actions for this month.", self._styles["BodyText2"]))
                continue

            table_data = [["Priority", "Action", "Impact", "KPI"]]
            for action in month_actions:
                table_data.append([
                    action.priority.value,
                    f"{action.title}\n{action.description[:100]}",
                    action.estimated_impact[:60] if action.estimated_impact else "-",
                    action.kpi[:50] if action.kpi else "-",
                ])

            table = Table(table_data, colWidths=[55, 180, 130, 130])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _COLORS["primary"]),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_COLORS["bg_light"], colors.white]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 10))

        return elements

    # ── Chart Generators ──────────────────────────────────────────────────

    def _generate_pie_chart(self, data: dict, title: str) -> Optional[str]:
        """Generate a sentiment pie chart and return temp file path."""
        try:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#FAFAFA")

            labels = list(data.keys())
            sizes = list(data.values())
            chart_colors = [_SENTIMENT_COLORS_MPL.get(l, "#94A3B8") for l in labels]

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=[l.capitalize() for l in labels],
                colors=chart_colors,
                autopct="%1.1f%%",
                startangle=140,
                pctdistance=0.75,
                textprops={"fontsize": 9},
            )
            for t in autotexts:
                t.set_fontsize(8)
                t.set_color("white")
                t.set_fontweight("bold")

            ax.set_title(title, fontsize=12, fontweight="bold", color="#1E293B", pad=12)

            path = os.path.join(os.path.dirname(__file__), "..", "data", "_chart_pie.png")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
            plt.close()
            return path
        except Exception as e:
            print(f"  [WARN] Pie chart generation failed: {e}")
            return None

    def _generate_aspect_bar_chart(self, data: dict) -> Optional[str]:
        """Generate an aspect sentiment bar chart."""
        try:
            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor("#FAFAFA")
            ax.set_facecolor("#FAFAFA")

            aspects = list(data.keys())
            pos_vals = [data[a].get("positive", 0) for a in aspects]
            neg_vals = [data[a].get("negative", 0) for a in aspects]
            neu_vals = [data[a].get("neutral", 0) for a in aspects]

            x = range(len(aspects))
            width = 0.25

            bars1 = ax.bar([i - width for i in x], pos_vals, width, label="Positive", color="#10B981", alpha=0.85)
            bars2 = ax.bar(x, neg_vals, width, label="Negative", color="#EF4444", alpha=0.85)
            bars3 = ax.bar([i + width for i in x], neu_vals, width, label="Neutral", color="#8B5CF6", alpha=0.85)

            ax.set_xlabel("Aspects", fontsize=10, color="#475569")
            ax.set_ylabel("Mention Count", fontsize=10, color="#475569")
            ax.set_title("Aspect-Based Sentiment Analysis", fontsize=12, fontweight="bold", color="#1E293B")
            ax.set_xticks(x)
            ax.set_xticklabels(aspects, fontsize=8, rotation=15, ha="right")
            ax.legend(fontsize=8)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            path = os.path.join(os.path.dirname(__file__), "..", "data", "_chart_aspects.png")
            plt.tight_layout()
            plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
            plt.close()
            return path
        except Exception as e:
            print(f"  [WARN] Aspect chart generation failed: {e}")
            return None


# Singleton
pdf_generator = PDFReportGenerator()
