"""
API Gateway -- FastAPI application for the Customer Review Intelligence Platform.

Endpoints:
  POST /ingest           -> Process a single review
  POST /ingest/batch     -> Process multiple reviews with anomaly detection
  POST /ingest/csv       -> Upload and process a CSV file
  POST /ingest/json      -> Upload and process a JSON file
  POST /ingest/voice     -> Upload audio for Whisper transcription
  POST /ingest/social    -> Ingest social media transcripts
  GET  /status           -> System health and aggregate stats
  GET  /trends           -> Anomaly + trend report
  GET  /roadmap          -> Strategic 3-month roadmap
  GET  /report/pdf       -> Download PDF intelligence report
  GET  /dashboard/customer -> Customer View data payload
  GET  /dashboard/company  -> Company View data payload
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import json
import os

from config import llm, ENGINE_VERSION, GEMINI_MODEL
from models.schemas import ReviewInput, BatchResult, SentimentLabel
from agents.dispatcher import pipeline
from agents.anomaly_agent import anomaly_agent
from agents.roadmap_agent import roadmap_agent
from services.store import store
from ingestion.csv_json_loader import csv_json_loader
from ingestion.voice_to_text import voice_to_text
from output.pdf_report import pdf_generator


# -- Pydantic Request Models --------------------------------------------------

class SingleReviewRequest(BaseModel):
    """Single review submission."""
    text: str = Field(..., min_length=1, description="Review text content")
    category: str = Field(default="general", description="Product category")
    source: str = Field(default="api", description="Ingestion source identifier")


class BatchReviewRequest(BaseModel):
    """Batch review submission."""
    reviews: list[SingleReviewRequest] = Field(..., min_length=1)
    run_anomaly_detection: bool = Field(default=True)
    generate_roadmap: bool = Field(default=True)
    skip_cleaning: bool = Field(default=False, description="Skip LLM cleaning for speed")


class SocialSyncRequest(BaseModel):
    """Social media transcript submission."""
    platform: str = Field(default="instagram")
    posts: list[dict] = Field(..., min_length=1)


# -- FastAPI App ---------------------------------------------------------------

app = FastAPI(
    title="Omni-AI Review Intelligence Platform",
    description=(
        "AI-powered Customer Review Intelligence Platform for Hack Malenadu '26. "
        "Multi-modal ingestion, multilingual ABSA, sarcasm detection, "
        "anomaly detection, and strategic roadmap generation."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Health & Status -----------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint -- API welcome message."""
    return {
        "service": "Omni-AI Review Intelligence Platform",
        "version": ENGINE_VERSION,
        "engine": GEMINI_MODEL,
        "llm_available": llm.is_available,
        "docs": "/docs",
    }


@app.get("/status", tags=["System"])
async def get_status():
    """Return aggregate statistics and system health."""
    stats = store.get_stats()
    return {
        "service": "Omni-AI Review Intelligence Platform",
        "health": "operational",
        "llm_available": llm.is_available,
        "llm_model": GEMINI_MODEL,
        "llm_calls": llm.call_count,
        **stats,
    }


# -- Single Review Ingestion ---------------------------------------------------

@app.post("/ingest", tags=["Intelligence"])
async def ingest_review(request: SingleReviewRequest):
    """Process a single review through the full intelligence pipeline."""
    try:
        review_input = ReviewInput(
            text=request.text,
            category=request.category,
            source=request.source,
        )
        result = await pipeline.process_single(review_input)
        return result.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# -- Batch Ingestion -----------------------------------------------------------

@app.post("/ingest/batch", tags=["Intelligence"])
async def ingest_batch(request: BatchReviewRequest):
    """Process multiple reviews with anomaly detection and roadmap generation."""
    try:
        review_inputs = [
            ReviewInput(text=r.text, category=r.category, source=r.source)
            for r in request.reviews
        ]

        result = await pipeline.process_batch(
            reviews=review_inputs,
            run_anomaly_detection=request.run_anomaly_detection,
            generate_roadmap=request.generate_roadmap,
            skip_cleaning=request.skip_cleaning,
        )

        return result.model_dump()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")


# -- CSV Upload ----------------------------------------------------------------

@app.post("/ingest/csv", tags=["Ingestion"])
async def ingest_csv(
    file: UploadFile = File(..., description="CSV file with review data"),
    skip_cleaning: bool = Form(default=False),
    run_anomaly_detection: bool = Form(default=True),
    generate_roadmap: bool = Form(default=True),
):
    """Upload and process a CSV file of reviews."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    try:
        content = await file.read()
        reviews = csv_json_loader.load_csv(content, source="csv")

        if not reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found in CSV")

        result = await pipeline.process_batch(
            reviews=reviews,
            run_anomaly_detection=run_anomaly_detection,
            generate_roadmap=generate_roadmap,
            skip_cleaning=skip_cleaning,
        )

        return {
            "file": file.filename,
            "reviews_parsed": len(reviews),
            **result.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")


# -- JSON Upload ---------------------------------------------------------------

@app.post("/ingest/json", tags=["Ingestion"])
async def ingest_json(
    file: UploadFile = File(..., description="JSON file with review data"),
    skip_cleaning: bool = Form(default=False),
    run_anomaly_detection: bool = Form(default=True),
    generate_roadmap: bool = Form(default=True),
):
    """Upload and process a JSON file of reviews."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="File must be a .json")

    try:
        content = await file.read()
        reviews = csv_json_loader.load_json(content, source="json")

        if not reviews:
            raise HTTPException(status_code=400, detail="No valid reviews found in JSON")

        result = await pipeline.process_batch(
            reviews=reviews,
            run_anomaly_detection=run_anomaly_detection,
            generate_roadmap=generate_roadmap,
            skip_cleaning=skip_cleaning,
        )

        return {
            "file": file.filename,
            "reviews_parsed": len(reviews),
            **result.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"JSON processing failed: {str(e)}")


# -- Voice Upload --------------------------------------------------------------

@app.post("/ingest/voice", tags=["Ingestion"])
async def ingest_voice(
    file: UploadFile = File(..., description="Audio file (.wav, .mp3, .m4a)"),
    category: str = Form(default="general"),
):
    """Upload an audio file for Whisper transcription and analysis."""
    if not voice_to_text.is_available:
        raise HTTPException(
            status_code=503,
            detail="Whisper is not installed. Install with: pip install openai-whisper",
        )

    try:
        audio_bytes = await file.read()
        review = await voice_to_text.transcribe(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            category=category,
        )

        if not review:
            raise HTTPException(status_code=400, detail="Could not transcribe audio")

        result = await pipeline.process_single(review)

        return {
            "transcribed_text": review.text,
            "analysis": result.model_dump(),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Voice processing failed: {str(e)}")


# -- Social Sync ---------------------------------------------------------------

@app.post("/ingest/social", tags=["Intelligence"])
async def ingest_social(request: SocialSyncRequest):
    """Ingest social media transcripts for hype-vs-reality analysis."""
    try:
        from prompts.social_sync import build_social_sync_prompt
        from config import llm as llm_client

        # Get current aspect summary from store
        company_view = store.get_company_view()
        aspect_summary = {}
        for item in company_view.aspect_heatmap:
            aspect_summary[item["aspect"]] = {
                "positive": item.get("positive", 0),
                "negative": item.get("negative", 0),
                "neutral": item.get("neutral", 0),
                "total": item.get("total", 0),
            }

        if not aspect_summary:
            raise HTTPException(
                status_code=400,
                detail="No review data available. Process reviews first before social sync.",
            )

        prompt = build_social_sync_prompt(request.posts, aspect_summary)
        result = await llm_client.generate_json(prompt, temperature=0.3)

        if not result:
            raise HTTPException(status_code=500, detail="Social sync analysis failed")

        return {
            "platform": request.platform,
            "claims_analyzed": len(result.get("claims", [])),
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Social sync failed: {str(e)}")


# -- Trends & Anomalies -------------------------------------------------------

@app.get("/trends", tags=["Analytics"])
async def get_trends():
    """Get the latest anomaly and trend report."""
    report = store.trend_report
    if not report:
        # Try to generate from stored reviews
        reviews = store.get_all_reviews()
        if len(reviews) >= 5:
            report = anomaly_agent.analyze(reviews)
            store.set_trend_report(report)
        else:
            return {"message": "Not enough data. Process at least 5 reviews first."}

    return report.model_dump()


# -- Strategic Roadmap ---------------------------------------------------------

@app.get("/roadmap", tags=["Analytics"])
async def get_roadmap():
    """Get the strategic 3-month roadmap."""
    existing = store.roadmap
    if existing:
        return existing.model_dump()

    # Generate fresh roadmap
    report = store.trend_report
    if not report:
        reviews = store.get_all_reviews()
        if len(reviews) < 5:
            return {"message": "Not enough data. Process at least 5 reviews first."}
        report = anomaly_agent.analyze(reviews)
        store.set_trend_report(report)

    stats = store.get_stats()
    categories = list(stats.get("categories", {}).keys())

    roadmap_result = await roadmap_agent.generate(
        trend_report=report,
        sentiment_stats=stats.get("sentiments", {}),
        categories=categories,
    )
    store.set_roadmap(roadmap_result)

    return roadmap_result.model_dump()


# -- PDF Report ----------------------------------------------------------------

@app.get("/report/pdf", tags=["Export"])
async def download_pdf_report():
    """Generate and download the full PDF intelligence report."""
    reviews = store.get_all_reviews()
    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews processed yet")

    try:
        output_path = pdf_generator.generate(
            reviews=reviews,
            trend_report=store.trend_report,
            roadmap=store.roadmap,
        )

        return FileResponse(
            path=output_path,
            media_type="application/pdf",
            filename=os.path.basename(output_path),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


# -- Dashboard Endpoints -------------------------------------------------------

@app.get("/dashboard/customer", tags=["Dashboard"])
async def dashboard_customer():
    """Get the Customer View ('The Real Story') data payload."""
    view = store.get_customer_view()
    return view.model_dump()


@app.get("/dashboard/company", tags=["Dashboard"])
async def dashboard_company():
    """Get the Company View (Technical Deep-Dive) data payload."""
    view = store.get_company_view()
    return view.model_dump()


# -- Data Management -----------------------------------------------------------

@app.delete("/data/reset", tags=["System"])
async def reset_data():
    """Clear all stored data and start fresh."""
    store.clear()
    return {"message": "All data cleared", "status": "ok"}


# -- Run Server ----------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
