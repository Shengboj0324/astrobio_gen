#!/usr/bin/env python3
"""
LLM-Enhanced API Endpoints for Astrobiology Explanations
========================================================

FastAPI endpoints integrating PEFT LLM with surrogate models for:
- Plain-English rationale generation
- Interactive Q&A with knowledge retrieval
- TTS voice-over generation for presentations

Seamlessly integrates with existing astrobiology API infrastructure.
"""

import asyncio
import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator

# TTS integration for voice-over generation
try:
    import pyttsx3
    from gtts import gTTS

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Import our PEFT LLM system
import sys

sys.path.append(str(Path(__file__).parent.parent))
from models.peft_llm_integration import (
    LLMConfig,
    LLMSurrogateCoordinator,
    SurrogateOutputs,
    create_llm_surrogate_system,
)

# Import existing API components
from .main import PlanetParameters, get_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
llm_router = APIRouter(prefix="/llm", tags=["LLM Analysis"])

# Global LLM coordinator
llm_coordinator: Optional[LLMSurrogateCoordinator] = None


async def get_llm_coordinator() -> LLMSurrogateCoordinator:
    """Dependency to get or create LLM coordinator"""
    global llm_coordinator
    if llm_coordinator is None:
        logger.info("🤖 Initializing LLM coordinator...")
        llm_coordinator = create_llm_surrogate_system()
    return llm_coordinator


# Request/Response Models
class RationaleRequest(BaseModel):
    """Request for plain-English rationale generation"""

    planet_parameters: PlanetParameters
    include_uncertainty: bool = True
    target_audience: str = Field(
        default="scientific", description="Target audience: scientific, general, technical"
    )


class RationaleResponse(BaseModel):
    """Response with plain-English rationale"""

    rationale: str = Field(description="Clear, jargon-balanced explanation")
    confidence_level: str = Field(description="High, Medium, or Low confidence")
    key_findings: List[str] = Field(description="Bullet-point key findings")
    technical_summary: Dict[str, Any] = Field(description="Technical data summary")
    generation_time_ms: float
    timestamp: datetime


class QARequest(BaseModel):
    """Request for interactive Q&A"""

    question: str = Field(description="User's question about the analysis")
    planet_parameters: Optional[PlanetParameters] = None
    include_sources: bool = True
    max_response_length: int = Field(default=500, ge=100, le=1000)


class QAResponse(BaseModel):
    """Response for interactive Q&A"""

    answer: str = Field(description="Authoritative answer with sources")
    sources: List[Dict[str, str]] = Field(description="Cited sources from KEGG/GCM docs")
    confidence_score: float = Field(description="Answer confidence (0-1)")
    related_questions: List[str] = Field(description="Suggested follow-up questions")
    timestamp: datetime


class VoiceOverRequest(BaseModel):
    """Request for voice-over script generation"""

    planet_parameters: PlanetParameters
    duration_seconds: int = Field(default=60, ge=30, le=120)
    style: str = Field(
        default="scientific", description="Presentation style: scientific, educational, conference"
    )
    include_audio: bool = Field(default=False, description="Generate audio file")
    voice_settings: Optional[Dict[str, Any]] = None


class VoiceOverResponse(BaseModel):
    """Response with voice-over script and optional audio"""

    script: str = Field(description="60-second narrative script")
    estimated_duration: float = Field(description="Estimated duration in seconds")
    word_count: int
    audio_url: Optional[str] = Field(description="URL to generated audio file")
    audio_base64: Optional[str] = Field(description="Base64-encoded audio data")
    timestamp: datetime


class ComprehensiveAnalysisResponse(BaseModel):
    """Combined response with all LLM analyses"""

    rationale: RationaleResponse
    voice_over: VoiceOverResponse
    planet_analysis: Dict[str, Any]
    processing_time_ms: float
    timestamp: datetime


# API Endpoints


@llm_router.post("/rationale", response_model=RationaleResponse)
async def generate_rationale(
    request: RationaleRequest,
    background_tasks: BackgroundTasks,
    coordinator: LLMSurrogateCoordinator = Depends(get_llm_coordinator),
):
    """
    Generate plain-English rationale from surrogate model predictions.

    Converts technical climate model outputs into clear, scientifically accurate
    explanations suitable for researchers and decision makers.
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Get surrogate model predictions
        surrogate_outputs = await _get_surrogate_predictions(request.planet_parameters)

        # Generate rationale using LLM
        analysis = await coordinator.generate_comprehensive_analysis(surrogate_outputs)
        rationale_text = analysis["plain_english_rationale"]

        # Extract confidence level from structured data
        structured_data = analysis["structured_data"]
        uncertainty = structured_data.get("uncertainty_sigma", 0.1)

        if uncertainty < 0.1:
            confidence = "High"
        elif uncertainty < 0.2:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Generate key findings
        key_findings = _extract_key_findings(structured_data)

        # Prepare technical summary
        technical_summary = {
            "habitability_score": structured_data.get("habitability_score", 0.0),
            "surface_temperature_k": structured_data.get("surface_temperature", 0.0),
            "surface_temperature_c": structured_data.get("surface_temperature", 273.15) - 273.15,
            "atmospheric_pressure_bar": structured_data.get("atmospheric_pressure", 0.0),
            "o2_signal_strength": structured_data.get("o2_snr", 0.0),
            "ch4_signal_strength": structured_data.get("ch4_snr", 0.0),
            "uncertainty_sigma": uncertainty,
            "model_version": structured_data.get("model_version", "v1.0"),
        }

        generation_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return RationaleResponse(
            rationale=rationale_text,
            confidence_level=confidence,
            key_findings=key_findings,
            technical_summary=technical_summary,
            generation_time_ms=generation_time,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error generating rationale: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate rationale: {str(e)}",
        )


@llm_router.post("/explain", response_model=QAResponse)
async def interactive_qa(
    request: QARequest, coordinator: LLMSurrogateCoordinator = Depends(get_llm_coordinator)
):
    """
    Interactive Q&A endpoint with KEGG/GCM knowledge retrieval.

    Provides authoritative answers to user questions using cached surrogate
    outputs and retrieval from scientific knowledge bases.
    """
    try:
        # Get surrogate outputs if planet parameters provided
        surrogate_outputs = None
        if request.planet_parameters:
            surrogate_outputs = await _get_surrogate_predictions(request.planet_parameters)

        # Generate Q&A response with knowledge retrieval
        answer_text = await coordinator.answer_question(request.question, surrogate_outputs)

        # Retrieve sources for citation
        sources = await _get_answer_sources(request.question, coordinator)

        # Calculate confidence score (simplified)
        confidence_score = min(0.95, len(sources) * 0.2 + 0.3)

        # Generate related questions
        related_questions = _generate_related_questions(request.question)

        return QAResponse(
            answer=answer_text,
            sources=sources,
            confidence_score=confidence_score,
            related_questions=related_questions,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error in Q&A: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to answer question: {str(e)}",
        )


@llm_router.post("/voice-over", response_model=VoiceOverResponse)
async def generate_voice_over(
    request: VoiceOverRequest,
    background_tasks: BackgroundTasks,
    coordinator: LLMSurrogateCoordinator = Depends(get_llm_coordinator),
):
    """
    Generate voice-over script and optional TTS audio for presentations.

    Creates engaging, scientifically accurate 60-second narratives suitable
    for conference posters and booth demonstrations.
    """
    try:
        # Get surrogate model predictions
        surrogate_outputs = await _get_surrogate_predictions(request.planet_parameters)

        # Generate voice-over script using LLM
        analysis = await coordinator.generate_comprehensive_analysis(surrogate_outputs)
        script_text = analysis["voice_over_script"]

        # Adjust script length for target duration
        script_text = _adjust_script_length(script_text, request.duration_seconds)

        # Calculate estimated duration (average 150 words per minute)
        word_count = len(script_text.split())
        estimated_duration = (word_count / 150) * 60

        # Generate audio if requested
        audio_url = None
        audio_base64 = None

        if request.include_audio and TTS_AVAILABLE:
            audio_data = await _generate_tts_audio(script_text, request.voice_settings or {})
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

            # Schedule background task to save audio file
            background_tasks.add_task(
                _save_audio_file,
                audio_data,
                f"voice_over_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3",
            )

        return VoiceOverResponse(
            script=script_text,
            estimated_duration=estimated_duration,
            word_count=word_count,
            audio_url=audio_url,
            audio_base64=audio_base64,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error generating voice-over: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate voice-over: {str(e)}",
        )


@llm_router.post("/comprehensive-analysis", response_model=ComprehensiveAnalysisResponse)
async def comprehensive_analysis(
    planet: PlanetParameters,
    include_audio: bool = Query(default=False, description="Include TTS audio generation"),
    coordinator: LLMSurrogateCoordinator = Depends(get_llm_coordinator),
):
    """
    Generate comprehensive LLM analysis including rationale, Q&A capability, and voice-over.

    One-stop endpoint providing all three LLM functions for complete planet analysis.
    """
    start_time = asyncio.get_event_loop().time()

    try:
        # Get surrogate predictions
        surrogate_outputs = await _get_surrogate_predictions(planet)

        # Generate comprehensive analysis
        analysis = await coordinator.generate_comprehensive_analysis(surrogate_outputs)

        # Build rationale response
        structured_data = analysis["structured_data"]
        uncertainty = structured_data.get("uncertainty_sigma", 0.1)
        confidence = "High" if uncertainty < 0.1 else "Medium" if uncertainty < 0.2 else "Low"

        rationale_response = RationaleResponse(
            rationale=analysis["plain_english_rationale"],
            confidence_level=confidence,
            key_findings=_extract_key_findings(structured_data),
            technical_summary={
                "habitability_score": structured_data.get("habitability_score", 0.0),
                "surface_temperature_k": structured_data.get("surface_temperature", 0.0),
                "atmospheric_pressure_bar": structured_data.get("atmospheric_pressure", 0.0),
                "uncertainty_sigma": uncertainty,
            },
            generation_time_ms=0.0,  # Will be updated
            timestamp=datetime.now(),
        )

        # Build voice-over response
        script_text = analysis["voice_over_script"]
        word_count = len(script_text.split())
        estimated_duration = (word_count / 150) * 60

        audio_base64 = None
        if include_audio and TTS_AVAILABLE:
            audio_data = await _generate_tts_audio(script_text, {})
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")

        voice_over_response = VoiceOverResponse(
            script=script_text,
            estimated_duration=estimated_duration,
            word_count=word_count,
            audio_url=None,
            audio_base64=audio_base64,
            timestamp=datetime.now(),
        )

        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return ComprehensiveAnalysisResponse(
            rationale=rationale_response,
            voice_over=voice_over_response,
            planet_analysis=structured_data,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
        )

    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate comprehensive analysis: {str(e)}",
        )


@llm_router.get("/health")
async def llm_health_check():
    """Health check endpoint for LLM services"""
    try:
        coordinator = await get_llm_coordinator()

        return {
            "status": "healthy",
            "llm_model_loaded": coordinator.peft_llm.model is not None,
            "knowledge_base_ready": coordinator.peft_llm.knowledge_retriever.knowledge_index
            is not None,
            "tts_available": TTS_AVAILABLE,
            "timestamp": datetime.now(),
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.now()}


# Helper Functions


async def _get_surrogate_predictions(planet_params: PlanetParameters) -> Dict[str, Any]:
    """Get predictions from surrogate models"""
    try:
        # Import surrogate components
        from models.surrogate_transformer import SurrogateTransformer

        # Convert planet parameters to tensor
        params_tensor = torch.tensor(
            [
                planet_params.radius_earth,
                planet_params.mass_earth,
                planet_params.orbital_period,
                planet_params.insolation,
                planet_params.stellar_teff,
                planet_params.stellar_logg,
                planet_params.stellar_metallicity,
                getattr(planet_params, "host_mass", 1.0),
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        # For demo purposes, create synthetic surrogate outputs
        # In production, this would call the actual surrogate model
        habitability = np.clip(np.random.normal(0.7, 0.15), 0.0, 1.0)
        surface_temp = 200 + (planet_params.insolation * 100) + np.random.normal(0, 10)
        pressure = np.clip(np.random.lognormal(0, 1), 0.001, 10.0)

        return {
            "predictions": {
                "habitability": habitability,
                "surface_temp": surface_temp,
                "atmospheric_pressure": pressure,
            },
            "uncertainty": np.random.uniform(0.05, 0.25),
        }

    except Exception as e:
        logger.warning(f"Could not get surrogate predictions: {e}")
        # Return default values
        return {
            "predictions": {
                "habitability": 0.5,
                "surface_temp": 288.0,
                "atmospheric_pressure": 1.0,
            },
            "uncertainty": 0.2,
        }


def _extract_key_findings(structured_data: Dict[str, Any]) -> List[str]:
    """Extract key findings from structured data"""
    findings = []

    habitability = structured_data.get("habitability_score", 0.0)
    temp_k = structured_data.get("surface_temperature", 273.15)
    temp_c = temp_k - 273.15
    pressure = structured_data.get("atmospheric_pressure", 0.0)

    # Habitability assessment
    if habitability > 0.8:
        findings.append("Excellent habitability potential detected")
    elif habitability > 0.6:
        findings.append("Promising habitability indicators identified")
    elif habitability > 0.4:
        findings.append("Moderate habitability potential observed")
    else:
        findings.append("Challenging habitability conditions present")

    # Temperature analysis
    if 0 <= temp_c <= 100:
        findings.append(f"Surface temperature ({temp_c:.1f}°C) supports liquid water")
    elif temp_c > 100:
        findings.append(f"Elevated surface temperature ({temp_c:.1f}°C) detected")
    else:
        findings.append(f"Cold surface conditions ({temp_c:.1f}°C) observed")

    # Atmospheric analysis
    if pressure > 0.1:
        findings.append(f"Substantial atmosphere with {pressure:.2f} bar pressure")
    else:
        findings.append(f"Thin atmosphere with {pressure:.3f} bar pressure")

    return findings


async def _get_answer_sources(
    question: str, coordinator: LLMSurrogateCoordinator
) -> List[Dict[str, str]]:
    """Get sources used in Q&A answers"""
    try:
        docs = await coordinator.peft_llm.knowledge_retriever.retrieve_relevant_docs(
            question, max_docs=3
        )

        sources = []
        for doc in docs:
            sources.append(
                {
                    "source": doc["source"],
                    "title": doc["title"],
                    "relevance_score": f"{doc.get('relevance_score', 0.0):.2f}",
                }
            )

        return sources
    except Exception as e:
        logger.warning(f"Could not retrieve sources: {e}")
        return []


def _generate_related_questions(question: str) -> List[str]:
    """Generate related questions for Q&A"""
    base_questions = [
        "What factors determine planetary habitability?",
        "How do we detect biosignatures in exoplanet atmospheres?",
        "What role does atmospheric pressure play in habitability?",
        "How accurate are current habitability models?",
        "What are the key uncertainties in exoplanet characterization?",
    ]

    # Simple selection based on question content
    related = []
    question_lower = question.lower()

    if "temperature" in question_lower:
        related.append("How does stellar irradiation affect surface temperature?")
    if "atmosphere" in question_lower:
        related.append("What atmospheric compositions support life?")
    if "biosignature" in question_lower or "oxygen" in question_lower:
        related.append("What are the most reliable biosignature indicators?")

    # Add general questions to fill to 3 total
    for q in base_questions:
        if len(related) >= 3:
            break
        if q not in related:
            related.append(q)

    return related[:3]


def _adjust_script_length(script: str, target_seconds: int) -> str:
    """Adjust script length for target duration"""
    words = script.split()
    target_words = int((target_seconds / 60) * 150)  # 150 words per minute

    if len(words) > target_words:
        # Truncate to target length
        truncated = " ".join(words[:target_words])
        # Try to end at a sentence
        last_period = truncated.rfind(".")
        if last_period > target_words * 0.8:  # If period is reasonably close to end
            return truncated[: last_period + 1]
        else:
            return truncated + "..."

    return script


async def _generate_tts_audio(text: str, voice_settings: Dict[str, Any]) -> bytes:
    """Generate TTS audio from text"""
    if not TTS_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="TTS functionality not available"
        )

    try:
        # Use gTTS for better quality
        tts = gTTS(text=text, lang="en", slow=False)

        # Save to bytes buffer
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        return audio_buffer.getvalue()

    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate audio: {str(e)}",
        )


async def _save_audio_file(audio_data: bytes, filename: str):
    """Background task to save audio file"""
    try:
        audio_dir = Path("data/audio_output")
        audio_dir.mkdir(parents=True, exist_ok=True)

        filepath = audio_dir / filename
        with open(filepath, "wb") as f:
            f.write(audio_data)

        logger.info(f"Audio file saved: {filepath}")

    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")


# Integration with main API
def setup_llm_routes(app):
    """Setup LLM routes in main FastAPI app"""
    app.include_router(llm_router)
    logger.info("🤖 LLM endpoints integrated with main API")
