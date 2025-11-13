#!/usr/bin/env python3
"""
User Feedback Capture System
============================

Production-ready system for capturing, validating, and storing user feedback
to enable continuous self-improvement of AI models.

This system implements:
- REST API endpoints for feedback collection
- Feedback validation and quality scoring
- Database storage with versioning
- Integration with data pipeline
- Human-in-the-loop controls
- Active learning selection
- Uncertainty sampling

Author: Astrobiology AI Platform Team
Date: 2025-11-13
"""

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    CORRECTION = "correction"  # User corrects model output
    RATING = "rating"  # User rates prediction quality
    ANNOTATION = "annotation"  # User adds annotations
    VALIDATION = "validation"  # User validates/rejects prediction
    EXPLANATION = "explanation"  # User provides explanation
    FEATURE_REQUEST = "feature_request"  # User requests new capability


class FeedbackQuality(Enum):
    """Quality levels for feedback"""
    HIGH = "high"  # Expert-level, highly reliable
    MEDIUM = "medium"  # Good quality, generally reliable
    LOW = "low"  # Uncertain quality, needs review
    REJECTED = "rejected"  # Failed validation


class ReviewStatus(Enum):
    """Review status for human-in-the-loop"""
    PENDING = "pending"  # Awaiting review
    APPROVED = "approved"  # Approved for training
    REJECTED = "rejected"  # Rejected, not used
    NEEDS_REVISION = "needs_revision"  # Needs correction


@dataclass
class UserFeedback:
    """User feedback data structure"""
    
    # Identifiers
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: str = ""
    
    # Feedback content
    feedback_type: FeedbackType = FeedbackType.CORRECTION
    original_input: Dict[str, Any] = field(default_factory=dict)
    model_output: Dict[str, Any] = field(default_factory=dict)
    user_correction: Dict[str, Any] = field(default_factory=dict)
    user_rating: Optional[float] = None  # 0.0 to 1.0
    user_comment: str = ""
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_version: str = ""
    confidence_score: float = 0.0  # Model's confidence
    uncertainty_score: float = 0.0  # Model's uncertainty
    
    # Quality assessment
    feedback_quality: FeedbackQuality = FeedbackQuality.MEDIUM
    quality_score: float = 0.5  # 0.0 to 1.0
    validation_checks_passed: int = 0
    validation_checks_failed: int = 0
    
    # Human-in-the-loop
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str = ""
    reviewed_at: Optional[datetime] = None
    review_notes: str = ""
    
    # Integration
    integrated_into_training: bool = False
    training_batch_id: str = ""
    data_version_id: str = ""
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackStatistics:
    """Statistics for feedback collection"""
    
    total_feedback: int = 0
    feedback_by_type: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    feedback_by_quality: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    feedback_by_status: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    average_quality_score: float = 0.0
    average_user_rating: float = 0.0
    
    high_uncertainty_samples: int = 0
    approved_for_training: int = 0
    integrated_into_training: int = 0
    
    collection_start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackValidator:
    """Validates user feedback for quality and consistency"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_quality_score = self.config.get('min_quality_score', 0.3)
        self.max_correction_distance = self.config.get('max_correction_distance', 0.8)
        
        logger.info("🔍 Feedback Validator initialized")
    
    def validate_feedback(self, feedback: UserFeedback) -> Tuple[bool, List[str]]:
        """
        Validate feedback for quality and consistency
        
        Returns:
            (is_valid, validation_messages)
        """
        validation_messages = []
        checks_passed = 0
        checks_failed = 0
        
        # Check 1: Required fields present
        if not feedback.user_id:
            validation_messages.append("Missing user_id")
            checks_failed += 1
        else:
            checks_passed += 1
        
        if not feedback.original_input:
            validation_messages.append("Missing original_input")
            checks_failed += 1
        else:
            checks_passed += 1
        
        # Check 2: Feedback type specific validation
        if feedback.feedback_type == FeedbackType.CORRECTION:
            if not feedback.user_correction:
                validation_messages.append("Correction feedback missing user_correction")
                checks_failed += 1
            else:
                checks_passed += 1
        
        elif feedback.feedback_type == FeedbackType.RATING:
            if feedback.user_rating is None:
                validation_messages.append("Rating feedback missing user_rating")
                checks_failed += 1
            elif not (0.0 <= feedback.user_rating <= 1.0):
                validation_messages.append(f"Invalid rating: {feedback.user_rating}")
                checks_failed += 1
            else:
                checks_passed += 1
        
        # Check 3: Consistency check (if correction provided)
        if feedback.user_correction and feedback.model_output:
            consistency_score = self._check_consistency(
                feedback.model_output,
                feedback.user_correction
            )
            
            if consistency_score < 0.1:  # Too similar to model output
                validation_messages.append("Correction too similar to model output")
                checks_failed += 1
            elif consistency_score > self.max_correction_distance:  # Too different
                validation_messages.append("Correction too different from model output")
                checks_failed += 1
            else:
                checks_passed += 1
        
        # Check 4: Quality score calculation
        quality_score = self._calculate_quality_score(feedback, checks_passed, checks_failed)
        feedback.quality_score = quality_score
        feedback.validation_checks_passed = checks_passed
        feedback.validation_checks_failed = checks_failed
        
        # Determine quality level
        if quality_score >= 0.7:
            feedback.feedback_quality = FeedbackQuality.HIGH
        elif quality_score >= 0.4:
            feedback.feedback_quality = FeedbackQuality.MEDIUM
        elif quality_score >= self.min_quality_score:
            feedback.feedback_quality = FeedbackQuality.LOW
        else:
            feedback.feedback_quality = FeedbackQuality.REJECTED
        
        is_valid = quality_score >= self.min_quality_score
        
        return is_valid, validation_messages
    
    def _check_consistency(self, output1: Dict, output2: Dict) -> float:
        """Check consistency between two outputs"""
        # Simple implementation - can be enhanced
        if not output1 or not output2:
            return 0.5
        
        # Convert to strings and compare
        str1 = json.dumps(output1, sort_keys=True)
        str2 = json.dumps(output2, sort_keys=True)
        
        if str1 == str2:
            return 0.0  # Identical
        
        # Simple edit distance approximation
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 0.0
        
        # Count different characters
        diff_count = sum(c1 != c2 for c1, c2 in zip(str1, str2))
        diff_count += abs(len(str1) - len(str2))
        
        return min(diff_count / max_len, 1.0)
    
    def _calculate_quality_score(
        self,
        feedback: UserFeedback,
        checks_passed: int,
        checks_failed: int
    ) -> float:
        """Calculate overall quality score for feedback"""
        
        # Base score from validation checks
        total_checks = checks_passed + checks_failed
        if total_checks > 0:
            base_score = checks_passed / total_checks
        else:
            base_score = 0.5
        
        # Adjust based on user rating (if provided)
        if feedback.user_rating is not None:
            # High ratings indicate user is confident
            confidence_boost = (feedback.user_rating - 0.5) * 0.2
            base_score += confidence_boost
        
        # Adjust based on model uncertainty
        if feedback.uncertainty_score > 0.7:
            # High uncertainty samples are valuable
            base_score += 0.1
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_score))


class FeedbackDatabase:
    """SQLite database for storing user feedback"""
    
    def __init__(self, db_path: str = "data/feedback/user_feedback.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = None
        self._initialize_database()
        
        logger.info(f"💾 Feedback Database initialized: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        cursor = self.conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                feedback_type TEXT NOT NULL,
                original_input TEXT,
                model_output TEXT,
                user_correction TEXT,
                user_rating REAL,
                user_comment TEXT,
                timestamp TEXT NOT NULL,
                model_version TEXT,
                confidence_score REAL,
                uncertainty_score REAL,
                feedback_quality TEXT,
                quality_score REAL,
                validation_checks_passed INTEGER,
                validation_checks_failed INTEGER,
                review_status TEXT,
                reviewed_by TEXT,
                reviewed_at TEXT,
                review_notes TEXT,
                integrated_into_training INTEGER,
                training_batch_id TEXT,
                data_version_id TEXT,
                metadata TEXT
            )
        ''')
        
        # Create indices for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON feedback(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON feedback(feedback_quality)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_status ON feedback(review_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_uncertainty ON feedback(uncertainty_score)')
        
        self.conn.commit()
        logger.info("✅ Database schema initialized")
    
    def store_feedback(self, feedback: UserFeedback) -> bool:
        """Store feedback in database"""
        try:
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO feedback VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            ''', (
                feedback.feedback_id,
                feedback.user_id,
                feedback.session_id,
                feedback.feedback_type.value,
                json.dumps(feedback.original_input),
                json.dumps(feedback.model_output),
                json.dumps(feedback.user_correction),
                feedback.user_rating,
                feedback.user_comment,
                feedback.timestamp.isoformat(),
                feedback.model_version,
                feedback.confidence_score,
                feedback.uncertainty_score,
                feedback.feedback_quality.value,
                feedback.quality_score,
                feedback.validation_checks_passed,
                feedback.validation_checks_failed,
                feedback.review_status.value,
                feedback.reviewed_by,
                feedback.reviewed_at.isoformat() if feedback.reviewed_at else None,
                feedback.review_notes,
                1 if feedback.integrated_into_training else 0,
                feedback.training_batch_id,
                feedback.data_version_id,
                json.dumps(feedback.metadata)
            ))
            
            self.conn.commit()
            logger.info(f"✅ Feedback stored: {feedback.feedback_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store feedback: {e}")
            return False
    
    def get_feedback(self, feedback_id: str) -> Optional[UserFeedback]:
        """Retrieve feedback by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM feedback WHERE feedback_id = ?', (feedback_id,))
        row = cursor.fetchone()
        
        if row:
            return self._row_to_feedback(row)
        return None
    
    def get_pending_review(self, limit: int = 100) -> List[UserFeedback]:
        """Get feedback pending human review"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE review_status = ? 
            ORDER BY uncertainty_score DESC, timestamp DESC
            LIMIT ?
        ''', (ReviewStatus.PENDING.value, limit))
        
        return [self._row_to_feedback(row) for row in cursor.fetchall()]
    
    def get_high_uncertainty_samples(self, threshold: float = 0.7, limit: int = 100) -> List[UserFeedback]:
        """Get high uncertainty samples for active learning"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE uncertainty_score >= ? AND review_status = ?
            ORDER BY uncertainty_score DESC
            LIMIT ?
        ''', (threshold, ReviewStatus.PENDING.value, limit))
        
        return [self._row_to_feedback(row) for row in cursor.fetchall()]
    
    def get_approved_for_training(self, limit: int = 1000) -> List[UserFeedback]:
        """Get approved feedback for training"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM feedback 
            WHERE review_status = ? AND integrated_into_training = 0
            ORDER BY quality_score DESC, timestamp DESC
            LIMIT ?
        ''', (ReviewStatus.APPROVED.value, limit))
        
        return [self._row_to_feedback(row) for row in cursor.fetchall()]
    
    def _row_to_feedback(self, row) -> UserFeedback:
        """Convert database row to UserFeedback object"""
        return UserFeedback(
            feedback_id=row[0],
            user_id=row[1],
            session_id=row[2],
            feedback_type=FeedbackType(row[3]),
            original_input=json.loads(row[4]) if row[4] else {},
            model_output=json.loads(row[5]) if row[5] else {},
            user_correction=json.loads(row[6]) if row[6] else {},
            user_rating=row[7],
            user_comment=row[8],
            timestamp=datetime.fromisoformat(row[9]),
            model_version=row[10],
            confidence_score=row[11],
            uncertainty_score=row[12],
            feedback_quality=FeedbackQuality(row[13]),
            quality_score=row[14],
            validation_checks_passed=row[15],
            validation_checks_failed=row[16],
            review_status=ReviewStatus(row[17]),
            reviewed_by=row[18],
            reviewed_at=datetime.fromisoformat(row[19]) if row[19] else None,
            review_notes=row[20],
            integrated_into_training=bool(row[21]),
            training_batch_id=row[22],
            data_version_id=row[23],
            metadata=json.loads(row[24]) if row[24] else {}
        )
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("💾 Database connection closed")


# Continue in next file due to 300 line limit

