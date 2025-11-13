#!/usr/bin/env python3
"""
Feedback Collection API
=======================

REST API endpoints for collecting user feedback in production.

Endpoints:
- POST /api/feedback/submit - Submit new feedback
- GET /api/feedback/{id} - Get feedback by ID
- GET /api/feedback/pending - Get pending review items
- POST /api/feedback/review - Submit review decision
- GET /api/feedback/stats - Get feedback statistics

Author: Astrobiology AI Platform Team
Date: 2025-11-13
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from .user_feedback_system import (
    UserFeedback,
    FeedbackType,
    FeedbackQuality,
    ReviewStatus,
    FeedbackValidator,
    FeedbackDatabase,
    FeedbackStatistics
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Astrobiology AI Feedback API",
    description="API for collecting user feedback to enable continuous self-improvement",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
feedback_validator = FeedbackValidator()
feedback_db = FeedbackDatabase()


# Pydantic models for API
class FeedbackSubmission(BaseModel):
    """Model for feedback submission"""
    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(default="", description="Session identifier")
    feedback_type: str = Field(..., description="Type of feedback")
    original_input: Dict[str, Any] = Field(..., description="Original input to model")
    model_output: Dict[str, Any] = Field(..., description="Model's output")
    user_correction: Dict[str, Any] = Field(default_factory=dict, description="User's correction")
    user_rating: Optional[float] = Field(None, ge=0.0, le=1.0, description="User rating (0-1)")
    user_comment: str = Field(default="", description="User comment")
    model_version: str = Field(default="", description="Model version")
    confidence_score: float = Field(default=0.0, description="Model confidence")
    uncertainty_score: float = Field(default=0.0, description="Model uncertainty")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""
    feedback_id: str
    status: str
    quality_score: float
    feedback_quality: str
    validation_messages: List[str]


class ReviewDecision(BaseModel):
    """Model for review decision"""
    feedback_id: str
    decision: str  # approved, rejected, needs_revision
    reviewer_id: str
    review_notes: str = ""


@app.post("/api/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(
    submission: FeedbackSubmission,
    background_tasks: BackgroundTasks
):
    """
    Submit user feedback
    
    This endpoint accepts user feedback, validates it, and stores it in the database.
    High-quality feedback is automatically queued for training integration.
    """
    try:
        # Create UserFeedback object
        feedback = UserFeedback(
            user_id=submission.user_id,
            session_id=submission.session_id,
            feedback_type=FeedbackType(submission.feedback_type),
            original_input=submission.original_input,
            model_output=submission.model_output,
            user_correction=submission.user_correction,
            user_rating=submission.user_rating,
            user_comment=submission.user_comment,
            model_version=submission.model_version,
            confidence_score=submission.confidence_score,
            uncertainty_score=submission.uncertainty_score,
            metadata=submission.metadata
        )
        
        # Validate feedback
        is_valid, validation_messages = feedback_validator.validate_feedback(feedback)
        
        if not is_valid:
            logger.warning(f"Invalid feedback from user {submission.user_id}: {validation_messages}")
            return FeedbackResponse(
                feedback_id=feedback.feedback_id,
                status="rejected",
                quality_score=feedback.quality_score,
                feedback_quality=feedback.feedback_quality.value,
                validation_messages=validation_messages
            )
        
        # Store in database
        success = feedback_db.store_feedback(feedback)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to store feedback")
        
        # Queue for processing if high quality
        if feedback.feedback_quality == FeedbackQuality.HIGH:
            background_tasks.add_task(process_high_quality_feedback, feedback.feedback_id)
        
        # Queue for human review if medium quality or high uncertainty
        if (feedback.feedback_quality == FeedbackQuality.MEDIUM or 
            feedback.uncertainty_score > 0.7):
            background_tasks.add_task(queue_for_review, feedback.feedback_id)
        
        logger.info(f"✅ Feedback submitted: {feedback.feedback_id} (quality: {feedback.feedback_quality.value})")
        
        return FeedbackResponse(
            feedback_id=feedback.feedback_id,
            status="accepted",
            quality_score=feedback.quality_score,
            feedback_quality=feedback.feedback_quality.value,
            validation_messages=validation_messages
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid feedback type: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/{feedback_id}")
async def get_feedback(feedback_id: str):
    """Get feedback by ID"""
    feedback = feedback_db.get_feedback(feedback_id)
    
    if not feedback:
        raise HTTPException(status_code=404, detail="Feedback not found")
    
    return feedback


@app.get("/api/feedback/pending")
async def get_pending_review(limit: int = 100):
    """Get feedback items pending human review"""
    pending_items = feedback_db.get_pending_review(limit=limit)
    
    return {
        "count": len(pending_items),
        "items": pending_items
    }


@app.get("/api/feedback/high-uncertainty")
async def get_high_uncertainty(threshold: float = 0.7, limit: int = 100):
    """Get high uncertainty samples for active learning"""
    samples = feedback_db.get_high_uncertainty_samples(threshold=threshold, limit=limit)
    
    return {
        "count": len(samples),
        "threshold": threshold,
        "samples": samples
    }


@app.post("/api/feedback/review")
async def submit_review(review: ReviewDecision, background_tasks: BackgroundTasks):
    """
    Submit review decision for feedback
    
    This endpoint allows human reviewers to approve, reject, or request revisions
    for user feedback before it's integrated into training.
    """
    try:
        # Get feedback
        feedback = feedback_db.get_feedback(review.feedback_id)
        
        if not feedback:
            raise HTTPException(status_code=404, detail="Feedback not found")
        
        # Update review status
        if review.decision == "approved":
            feedback.review_status = ReviewStatus.APPROVED
        elif review.decision == "rejected":
            feedback.review_status = ReviewStatus.REJECTED
        elif review.decision == "needs_revision":
            feedback.review_status = ReviewStatus.NEEDS_REVISION
        else:
            raise HTTPException(status_code=400, detail="Invalid decision")
        
        feedback.reviewed_by = review.reviewer_id
        feedback.reviewed_at = datetime.now()
        feedback.review_notes = review.review_notes
        
        # Update in database
        # Note: Need to implement update method in FeedbackDatabase
        # For now, we'll re-store (this would overwrite in production)
        feedback_db.store_feedback(feedback)
        
        # If approved, queue for training integration
        if feedback.review_status == ReviewStatus.APPROVED:
            background_tasks.add_task(integrate_into_training, feedback.feedback_id)
        
        logger.info(f"✅ Review submitted for {review.feedback_id}: {review.decision}")
        
        return {
            "feedback_id": review.feedback_id,
            "status": "reviewed",
            "decision": review.decision
        }
        
    except Exception as e:
        logger.error(f"❌ Error submitting review: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/feedback/stats")
async def get_statistics():
    """Get feedback collection statistics"""
    # This would query the database for statistics
    # Simplified implementation for now
    
    stats = FeedbackStatistics()
    
    # TODO: Implement actual statistics calculation from database
    
    return {
        "total_feedback": stats.total_feedback,
        "feedback_by_type": dict(stats.feedback_by_type),
        "feedback_by_quality": dict(stats.feedback_by_quality),
        "feedback_by_status": dict(stats.feedback_by_status),
        "average_quality_score": stats.average_quality_score,
        "average_user_rating": stats.average_user_rating,
        "high_uncertainty_samples": stats.high_uncertainty_samples,
        "approved_for_training": stats.approved_for_training,
        "integrated_into_training": stats.integrated_into_training
    }


# Background task functions
async def process_high_quality_feedback(feedback_id: str):
    """Process high-quality feedback for immediate integration"""
    logger.info(f"🚀 Processing high-quality feedback: {feedback_id}")
    
    # Auto-approve high-quality feedback
    feedback = feedback_db.get_feedback(feedback_id)
    if feedback and feedback.feedback_quality == FeedbackQuality.HIGH:
        feedback.review_status = ReviewStatus.APPROVED
        feedback.reviewed_by = "auto_reviewer"
        feedback.reviewed_at = datetime.now()
        feedback.review_notes = "Auto-approved based on high quality score"
        
        feedback_db.store_feedback(feedback)
        
        # Queue for training integration
        await integrate_into_training(feedback_id)


async def queue_for_review(feedback_id: str):
    """Queue feedback for human review"""
    logger.info(f"📋 Queued for review: {feedback_id}")
    # In production, this would notify reviewers or add to review queue
    pass


async def integrate_into_training(feedback_id: str):
    """Integrate approved feedback into training pipeline"""
    logger.info(f"🎯 Integrating into training: {feedback_id}")
    
    try:
        # Import integration layer
        from .feedback_integration_layer import FeedbackIntegrationLayer
        
        integration_layer = FeedbackIntegrationLayer()
        success = await integration_layer.integrate_feedback(feedback_id)
        
        if success:
            logger.info(f"✅ Successfully integrated feedback: {feedback_id}")
        else:
            logger.error(f"❌ Failed to integrate feedback: {feedback_id}")
            
    except Exception as e:
        logger.error(f"❌ Error integrating feedback: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Feedback Collection API starting up...")
    logger.info("✅ Feedback validator initialized")
    logger.info("✅ Feedback database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("🛑 Feedback Collection API shutting down...")
    feedback_db.close()
    logger.info("✅ Database connections closed")


def run_api(host: str = "0.0.0.0", port: int = 8001):
    """Run the feedback collection API"""
    logger.info(f"🌐 Starting Feedback Collection API on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    run_api()

