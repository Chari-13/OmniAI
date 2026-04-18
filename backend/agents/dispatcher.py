"""
Pipeline Dispatcher — Lightweight async orchestrator for the review intelligence pipeline.

No external framework needed (no LangGraph, no CrewAI). Just clean async Python.

Pipeline stages:
  1. CLEAN   → LLM-based text cleaning + spam detection
  2. DETECT  → Language detection (heuristic)
  3. ANALYZE → ABSA + sarcasm detection (single Gemini call)
  4. STORE   → Persist to in-memory store
  5. BATCH   → Anomaly detection + trend analysis (post-batch)
  6. ROADMAP → Strategic roadmap generation (post-batch)
"""

from __future__ import annotations
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional, Callable

from models.schemas import (
    ReviewInput,
    CleanedReview,
    ProcessedReview,
    AspectSentiment,
    SarcasmResult,
    LanguageInfo,
    ReviewMeta,
    BatchResult,
    TrendReport,
    StrategicRoadmap,
    SentimentLabel,
    SarcasmType,
    LanguageCode,
)
from config import llm, ENGINE_VERSION, GEMINI_MODEL
from prompts.cleaning import build_cleaning_prompt
from agents.multilingual_agent import multilingual_agent
from agents.sentiment_agent import sentiment_agent
from agents.sarcasm_agent import sarcasm_agent
from agents.anomaly_agent import anomaly_agent
from agents.roadmap_agent import roadmap_agent
from services.store import store


class ReviewPipeline:
    """
    Lightweight async pipeline orchestrator.

    Each review flows through: clean → detect → analyze → validate → store.
    Batch operations add: anomaly detection → roadmap generation.

    Features:
    - Async processing with configurable concurrency
    - Progress callbacks for real-time UI updates
    - Per-review timing and metadata
    - Graceful error handling (no single review failure breaks the batch)
    """

    def __init__(self, max_concurrency: int = 5):
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._processed_count = 0

    # ══════════════════════════════════════════════════════════════════════
    # Single Review Processing
    # ══════════════════════════════════════════════════════════════════════

    async def process_single(
        self,
        review: ReviewInput,
        skip_cleaning: bool = False,
    ) -> ProcessedReview:
        """
        Process a single review through the full pipeline.

        Args:
            review: Raw review input
            skip_cleaning: Skip LLM cleaning step (for speed)

        Returns:
            Fully processed review with all intelligence data
        """
        start_time = time.time()
        review_id = f"REV-{uuid.uuid4().hex[:8].upper()}"

        # Stage 1: Clean (optional LLM call)
        cleaned_text = review.text
        spam_score = 0.0
        is_bot = False
        is_duplicate = False
        typos_fixed = 0
        emojis_translated = 0

        if not skip_cleaning and llm.is_available:
            clean_result = await self._clean_text(review.text)
            if clean_result:
                cleaned_text = clean_result.get("cleaned_text", review.text)
                spam_score = clean_result.get("spam_score", 0.0)
                is_bot = clean_result.get("is_bot", False)
                typos_fixed = clean_result.get("typos_fixed", 0)
                emojis_translated = clean_result.get("emojis_translated", 0)

        # Stage 2: Language detection (fast heuristic)
        lang_info = await multilingual_agent.detect(cleaned_text)

        # Stage 3: Sentiment + Sarcasm analysis (single LLM call)
        analysis = await sentiment_agent.analyze(
            text=cleaned_text,
            category=review.category,
            review_id=review_id,
        )

        # Stage 4: Sarcasm validation (post-processing on LLM result)
        raw_sarcasm = analysis.get("sarcasm", {})
        surface_sentiment = analysis.get("overall_sentiment", "neutral")
        validated_sarcasm = sarcasm_agent.validate_sarcasm(
            llm_sarcasm=raw_sarcasm,
            original_text=review.text,
            surface_sentiment=surface_sentiment,
        )

        # Update language info from LLM if it provided better data
        llm_lang = analysis.get("language", {})
        if llm_lang.get("detection_confidence", 0) > lang_info.detection_confidence:
            try:
                detected_code = LanguageCode(llm_lang.get("detected_code", "EN"))
            except ValueError:
                detected_code = lang_info.detected_code

            lang_info = LanguageInfo(
                detected_code=detected_code,
                detected_name=llm_lang.get("detected_name", lang_info.detected_name),
                translation=llm_lang.get("translation", lang_info.translation),
                detection_confidence=llm_lang.get("detection_confidence", lang_info.detection_confidence),
            )

        # Build aspect list
        aspects = []
        for a in analysis.get("aspects", []):
            try:
                sent = SentimentLabel(a.get("sentiment", "neutral"))
            except ValueError:
                sent = SentimentLabel.NEUTRAL

            aspects.append(AspectSentiment(
                aspect=a.get("aspect", "Unknown"),
                sentiment=sent,
                confidence=a.get("confidence", 50),
                evidence=a.get("evidence", ""),
            ))

        # Determine final sentiment (considering sarcasm correction)
        if validated_sarcasm.is_sarcastic:
            final_sentiment = validated_sarcasm.true_sentiment
        else:
            try:
                final_sentiment = SentimentLabel(analysis.get("overall_sentiment", "neutral"))
            except ValueError:
                final_sentiment = SentimentLabel.NEUTRAL

        processing_time = round((time.time() - start_time) * 1000, 1)

        # Build the ProcessedReview
        processed = ProcessedReview(
            id=review_id,
            language=lang_info,
            sentiment=final_sentiment,
            overall_confidence=analysis.get("overall_confidence", 50),
            verdict=analysis.get("verdict", ""),
            aspects=aspects,
            sarcasm=validated_sarcasm,
            meta=ReviewMeta(
                original_text=review.text,
                cleaned_text=cleaned_text if cleaned_text != review.text else None,
                category=review.category,
                source=review.source,
                processed_at=datetime.now(),
                engine_version=ENGINE_VERSION,
                llm_model=GEMINI_MODEL if llm.is_available else "fallback",
                processing_time_ms=processing_time,
                spam_score=spam_score,
                is_bot=is_bot,
                is_duplicate=is_duplicate,
            ),
            status="success" if llm.is_available else "success_fallback",
        )

        # Stage 5: Store
        store.add_review(processed)
        self._processed_count += 1

        return processed

    # ══════════════════════════════════════════════════════════════════════
    # Batch Processing
    # ══════════════════════════════════════════════════════════════════════

    async def process_batch(
        self,
        reviews: list[ReviewInput],
        run_anomaly_detection: bool = True,
        generate_roadmap: bool = True,
        skip_cleaning: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchResult:
        """
        Process a batch of reviews with optional anomaly detection and roadmap.

        Args:
            reviews: List of raw review inputs
            run_anomaly_detection: Run anomaly detection after processing
            generate_roadmap: Generate strategic roadmap
            skip_cleaning: Skip LLM cleaning for speed
            progress_callback: Optional callback(processed, total) for progress

        Returns:
            BatchResult with all processed reviews and analytics
        """
        start_time = time.time()
        results: list[ProcessedReview] = []
        failed = 0
        spam_filtered = 0

        # Process reviews with concurrency control
        async def process_with_semaphore(review: ReviewInput, index: int):
            async with self._semaphore:
                try:
                    result = await self.process_single(review, skip_cleaning=skip_cleaning)
                    if progress_callback:
                        progress_callback(index + 1, len(reviews))
                    return result
                except Exception as e:
                    print(f"  [FAIL] Failed to process review {index}: {e}")
                    return None

        # Run all reviews concurrently (bounded by semaphore)
        tasks = [
            process_with_semaphore(review, i)
            for i, review in enumerate(reviews)
        ]
        processed_results = await asyncio.gather(*tasks)

        for result in processed_results:
            if result is not None:
                results.append(result)
                if result.meta.spam_score > 0.7:
                    spam_filtered += 1
            else:
                failed += 1

        # Post-batch analytics
        anomaly_report: Optional[TrendReport] = None
        roadmap: Optional[StrategicRoadmap] = None

        if run_anomaly_detection and len(results) >= 5:
            anomaly_report = anomaly_agent.analyze(results)
            store.set_trend_report(anomaly_report)

            # Generate roadmap from anomaly data
            if generate_roadmap and anomaly_report:
                stats = store.get_stats()
                categories = list(set(r.meta.category for r in results))
                roadmap = await roadmap_agent.generate(
                    trend_report=anomaly_report,
                    sentiment_stats=stats.get("sentiments", {}),
                    categories=categories,
                )
                store.set_roadmap(roadmap)

        processing_time = round((time.time() - start_time) * 1000, 1)

        return BatchResult(
            total_processed=len(results) + failed,
            successful=len(results),
            failed=failed,
            spam_filtered=spam_filtered,
            duplicates_found=0,  # TODO: implement cross-review dedup
            results=results,
            anomaly_report=anomaly_report,
            processing_time_ms=processing_time,
        )

    # ══════════════════════════════════════════════════════════════════════
    # Internal Pipeline Stages
    # ══════════════════════════════════════════════════════════════════════

    async def _clean_text(self, text: str) -> Optional[dict]:
        """Stage 1: LLM-based text cleaning."""
        prompt = build_cleaning_prompt(text)
        return await llm.generate_json(prompt, temperature=0.1)

    @property
    def processed_count(self) -> int:
        return self._processed_count


# ═══════════════════════════════════════════════════════════════════════════════
# Singleton Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

pipeline = ReviewPipeline(max_concurrency=5)
