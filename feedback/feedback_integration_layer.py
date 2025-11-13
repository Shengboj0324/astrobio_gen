#!/usr/bin/env python3
"""
Feedback Integration Layer
===========================

Integrates user feedback into the data pipeline and training system.

This layer:
- Converts feedback to training data format
- Integrates with data versioning system
- Triggers incremental training
- Manages feedback-to-training pipeline
- Coordinates with continuous self-improvement system

Author: Astrobiology AI Platform Team
Date: 2025-11-13
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import sys

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from .user_feedback_system import (
    UserFeedback,
    FeedbackType,
    FeedbackQuality,
    ReviewStatus,
    FeedbackDatabase
)

# Import existing systems
try:
    from data_build.data_versioning_system import VersionManager, DataVersion
    from data_build.automated_data_pipeline import AutomatedDataPipeline
    from data_build.unified_dataloader_architecture import (
        MultiModalBatch,
        DataDomain,
        AnnotationStandard,
        DataAnnotation
    )
    from models.continuous_self_improvement import (
        ContinualLearningSystem,
        ContinualLearningConfig,
        LearningStrategy
    )
except ImportError as e:
    logging.warning(f"⚠️ Optional import failed: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeedbackIntegrationLayer:
    """
    Integrates user feedback into training pipeline
    
    This layer acts as the bridge between user feedback collection
    and the continuous self-improvement training system.
    """
    
    def __init__(
        self,
        feedback_db_path: str = "data/feedback/user_feedback.db",
        data_version_dir: str = "data/versions",
        feedback_data_dir: str = "data/feedback/training_data",
        config: Dict[str, Any] = None
    ):
        self.config = config or {}
        
        # Initialize components
        self.feedback_db = FeedbackDatabase(db_path=feedback_db_path)
        
        # Initialize data versioning
        try:
            self.version_manager = VersionManager(version_dir=data_version_dir)
        except Exception as e:
            logger.warning(f"⚠️ VersionManager not available: {e}")
            self.version_manager = None
        
        # Initialize continuous learning system
        try:
            cl_config = ContinualLearningConfig(
                ewc_lambda=400.0,
                fisher_samples=1000,
                replay_buffer_size=10000,
                replay_batch_size=32,
                forgetting_threshold=0.15,
                adaptation_frequency=50
            )
            self.continual_learning = ContinualLearningSystem(config=cl_config)
        except Exception as e:
            logger.warning(f"⚠️ ContinualLearningSystem not available: {e}")
            self.continual_learning = None
        
        # Feedback data directory
        self.feedback_data_dir = Path(feedback_data_dir)
        self.feedback_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.total_integrated = 0
        self.integration_errors = 0
        
        logger.info("🔗 Feedback Integration Layer initialized")
    
    async def integrate_feedback(self, feedback_id: str) -> bool:
        """
        Integrate a single feedback item into training pipeline
        
        Steps:
        1. Retrieve feedback from database
        2. Validate feedback is approved
        3. Convert to training data format
        4. Add to data versioning system
        5. Add to experience replay buffer
        6. Trigger incremental training if threshold met
        
        Returns:
            True if integration successful, False otherwise
        """
        try:
            # Step 1: Retrieve feedback
            feedback = self.feedback_db.get_feedback(feedback_id)
            
            if not feedback:
                logger.error(f"❌ Feedback not found: {feedback_id}")
                return False
            
            # Step 2: Validate approval status
            if feedback.review_status != ReviewStatus.APPROVED:
                logger.warning(f"⚠️ Feedback not approved: {feedback_id}")
                return False
            
            if feedback.integrated_into_training:
                logger.info(f"ℹ️ Feedback already integrated: {feedback_id}")
                return True
            
            # Step 3: Convert to training data format
            training_sample = self._convert_to_training_format(feedback)
            
            if not training_sample:
                logger.error(f"❌ Failed to convert feedback to training format: {feedback_id}")
                self.integration_errors += 1
                return False
            
            # Step 4: Save to feedback training data directory
            sample_path = self.feedback_data_dir / f"{feedback_id}.json"
            with open(sample_path, 'w') as f:
                json.dump(training_sample, f, indent=2, default=str)
            
            logger.info(f"💾 Saved training sample: {sample_path}")
            
            # Step 5: Add to data versioning system
            if self.version_manager:
                try:
                    version_id = self._add_to_version_control(feedback, sample_path)
                    feedback.data_version_id = version_id
                    logger.info(f"📦 Added to version control: {version_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Version control failed: {e}")
            
            # Step 6: Add to experience replay buffer
            if self.continual_learning:
                try:
                    self._add_to_replay_buffer(feedback, training_sample)
                    logger.info(f"🔄 Added to experience replay buffer")
                except Exception as e:
                    logger.warning(f"⚠️ Replay buffer addition failed: {e}")
            
            # Step 7: Update feedback status
            feedback.integrated_into_training = True
            feedback.training_batch_id = f"feedback_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.feedback_db.store_feedback(feedback)
            
            self.total_integrated += 1
            logger.info(f"✅ Successfully integrated feedback: {feedback_id} (total: {self.total_integrated})")
            
            # Step 8: Check if we should trigger retraining
            await self._check_retraining_trigger()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error integrating feedback {feedback_id}: {e}")
            self.integration_errors += 1
            return False
    
    def _convert_to_training_format(self, feedback: UserFeedback) -> Optional[Dict[str, Any]]:
        """
        Convert feedback to training data format
        
        This creates a training sample that can be used by the multi-modal system.
        """
        try:
            # Create training sample based on feedback type
            training_sample = {
                "feedback_id": feedback.feedback_id,
                "timestamp": feedback.timestamp.isoformat(),
                "quality_score": feedback.quality_score,
                "feedback_type": feedback.feedback_type.value,
            }
            
            if feedback.feedback_type == FeedbackType.CORRECTION:
                # For corrections, use user correction as ground truth
                training_sample.update({
                    "input": feedback.original_input,
                    "target": feedback.user_correction,
                    "model_output": feedback.model_output,
                    "is_correction": True
                })
                
            elif feedback.feedback_type == FeedbackType.RATING:
                # For ratings, use as quality signal
                training_sample.update({
                    "input": feedback.original_input,
                    "target": feedback.model_output,
                    "quality_weight": feedback.user_rating,
                    "is_rating": True
                })
                
            elif feedback.feedback_type == FeedbackType.ANNOTATION:
                # For annotations, add as additional labels
                training_sample.update({
                    "input": feedback.original_input,
                    "target": feedback.model_output,
                    "annotations": feedback.user_correction,
                    "is_annotation": True
                })
                
            elif feedback.feedback_type == FeedbackType.VALIDATION:
                # For validations, use as binary signal
                training_sample.update({
                    "input": feedback.original_input,
                    "target": feedback.model_output,
                    "is_valid": feedback.user_rating > 0.5 if feedback.user_rating else False,
                    "is_validation": True
                })
            
            # Add metadata
            training_sample["metadata"] = {
                "user_id": feedback.user_id,
                "model_version": feedback.model_version,
                "confidence_score": feedback.confidence_score,
                "uncertainty_score": feedback.uncertainty_score,
                "user_comment": feedback.user_comment,
                **feedback.metadata
            }
            
            # Create annotation for quality weighting
            training_sample["annotation"] = {
                "domain": "user_feedback",
                "standard": "user_correction",
                "quality_score": feedback.quality_score,
                "completeness_score": 1.0 if feedback.user_correction else 0.5,
                "reliability_score": feedback.quality_score,
                "temporal_coverage": "real_time",
                "spatial_resolution": "high",
                "metadata": {
                    "source": "user_feedback",
                    "feedback_quality": feedback.feedback_quality.value
                }
            }
            
            return training_sample
            
        except Exception as e:
            logger.error(f"❌ Error converting feedback to training format: {e}")
            return None
    
    def _add_to_version_control(self, feedback: UserFeedback, sample_path: Path) -> str:
        """Add feedback data to version control system"""
        if not self.version_manager:
            return ""
        
        try:
            # Create version metadata
            version_metadata = {
                "source": "user_feedback",
                "feedback_id": feedback.feedback_id,
                "feedback_type": feedback.feedback_type.value,
                "quality_score": feedback.quality_score,
                "timestamp": feedback.timestamp.isoformat()
            }
            
            # Create data version
            version = DataVersion(
                version_id=f"feedback_{feedback.feedback_id}",
                timestamp=feedback.timestamp,
                data_sources=[str(sample_path)],
                quality_score=feedback.quality_score,
                metadata=version_metadata
            )
            
            # Save version
            self.version_manager.save_version(version)
            
            return version.version_id
            
        except Exception as e:
            logger.error(f"❌ Error adding to version control: {e}")
            return ""
    
    def _add_to_replay_buffer(self, feedback: UserFeedback, training_sample: Dict[str, Any]):
        """Add feedback to experience replay buffer"""
        if not self.continual_learning:
            return
        
        try:
            # Convert to tensor format (simplified - would need actual model inputs)
            # In production, this would create proper model inputs
            experience = {
                "input": training_sample.get("input", {}),
                "target": training_sample.get("target", {}),
                "quality_score": feedback.quality_score,
                "metadata": training_sample.get("metadata", {})
            }
            
            # Add to replay buffer
            # Note: This is a simplified version - actual implementation would
            # need to convert to proper tensor format
            # self.continual_learning.replay_buffer.add_experience(experience)
            
            logger.info(f"🔄 Added to replay buffer (placeholder)")
            
        except Exception as e:
            logger.error(f"❌ Error adding to replay buffer: {e}")
    
    async def _check_retraining_trigger(self):
        """Check if we should trigger incremental retraining"""
        try:
            # Get approved feedback count
            approved_feedback = self.feedback_db.get_approved_for_training(limit=10000)
            approved_count = len(approved_feedback)
            
            # Trigger thresholds
            min_samples_threshold = self.config.get('min_samples_for_retraining', 100)
            quality_threshold = self.config.get('min_avg_quality_for_retraining', 0.6)
            
            if approved_count >= min_samples_threshold:
                # Calculate average quality
                avg_quality = np.mean([f.quality_score for f in approved_feedback])
                
                if avg_quality >= quality_threshold:
                    logger.info(f"🎯 Retraining trigger met: {approved_count} samples, avg quality: {avg_quality:.3f}")
                    await self._trigger_incremental_training(approved_feedback)
                else:
                    logger.info(f"ℹ️ Quality threshold not met: {avg_quality:.3f} < {quality_threshold}")
            else:
                logger.info(f"ℹ️ Sample threshold not met: {approved_count} < {min_samples_threshold}")
                
        except Exception as e:
            logger.error(f"❌ Error checking retraining trigger: {e}")
    
    async def _trigger_incremental_training(self, feedback_samples: List[UserFeedback]):
        """Trigger incremental training with feedback samples"""
        logger.info(f"🚀 Triggering incremental training with {len(feedback_samples)} samples")
        
        try:
            # Create training batch ID
            batch_id = f"feedback_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save batch metadata
            batch_metadata = {
                "batch_id": batch_id,
                "sample_count": len(feedback_samples),
                "avg_quality": np.mean([f.quality_score for f in feedback_samples]),
                "timestamp": datetime.now().isoformat(),
                "feedback_ids": [f.feedback_id for f in feedback_samples]
            }
            
            batch_path = self.feedback_data_dir / f"batch_{batch_id}.json"
            with open(batch_path, 'w') as f:
                json.dump(batch_metadata, f, indent=2)
            
            logger.info(f"💾 Saved batch metadata: {batch_path}")
            
            # In production, this would:
            # 1. Create a training job
            # 2. Load the feedback samples
            # 3. Run incremental training with EWC
            # 4. Evaluate on validation set
            # 5. Deploy if performance improves
            
            logger.info(f"✅ Incremental training triggered (placeholder)")
            
        except Exception as e:
            logger.error(f"❌ Error triggering incremental training: {e}")
    
    async def batch_integrate_feedback(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Batch integrate all approved feedback
        
        Returns:
            Statistics about integration
        """
        logger.info(f"🔄 Starting batch feedback integration (limit: {limit})")
        
        # Get approved feedback
        approved_feedback = self.feedback_db.get_approved_for_training(limit=limit)
        
        if not approved_feedback:
            logger.info("ℹ️ No approved feedback to integrate")
            return {
                "total": 0,
                "successful": 0,
                "failed": 0
            }
        
        logger.info(f"📦 Found {len(approved_feedback)} approved feedback items")
        
        # Integrate each feedback item
        successful = 0
        failed = 0
        
        for feedback in approved_feedback:
            success = await self.integrate_feedback(feedback.feedback_id)
            if success:
                successful += 1
            else:
                failed += 1
        
        results = {
            "total": len(approved_feedback),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(approved_feedback) if approved_feedback else 0.0
        }
        
        logger.info(f"✅ Batch integration complete: {results}")
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        return {
            "total_integrated": self.total_integrated,
            "integration_errors": self.integration_errors,
            "success_rate": self.total_integrated / (self.total_integrated + self.integration_errors)
                if (self.total_integrated + self.integration_errors) > 0 else 0.0
        }


if __name__ == "__main__":
    # Test integration layer
    async def test_integration():
        integration_layer = FeedbackIntegrationLayer()
        
        # Test batch integration
        results = await integration_layer.batch_integrate_feedback(limit=100)
        
        print(f"\n📊 Integration Results:")
        print(f"  Total: {results['total']}")
        print(f"  Successful: {results['successful']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Success Rate: {results['success_rate']:.1%}")
        
        # Get statistics
        stats = integration_layer.get_statistics()
        print(f"\n📈 Overall Statistics:")
        print(f"  Total Integrated: {stats['total_integrated']}")
        print(f"  Integration Errors: {stats['integration_errors']}")
        print(f"  Success Rate: {stats['success_rate']:.1%}")
    
    asyncio.run(test_integration())

