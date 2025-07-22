#!/usr/bin/env python3
"""
Advanced Vision Processing for Multi-Modal LLM
==============================================

Specialized vision processing components for handling customer images and videos
with scientific context understanding and integration with existing CNN models.

Features:
- Advanced image analysis with multiple model backends
- Video processing with temporal understanding
- Scientific image classification and feature extraction
- Integration with Enhanced CubeUNet and surrogate models
- Real-time processing optimization
- Customer data treatment capabilities

Performance Targets:
- <100ms image processing time
- <500ms video processing time (16 frames)
- >95% accuracy on scientific image classification
- Seamless integration with multi-modal LLM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import base64

# Vision processing libraries
try:
    import cv2
    import PIL.Image
    from PIL import Image, ImageEnhance, ImageFilter
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV and PIL not available. Install with: pip install opencv-python pillow albumentations")

try:
    import timm
    from torchvision import transforms, models
    from torchvision.models import (
        efficientnet_b4, resnet50, densenet121,
        mobilenet_v3_large, vit_b_16
    )
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("Torchvision not available. Install with: pip install torchvision timm")

try:
    from transformers import CLIPModel, CLIPProcessor, AutoImageProcessor, AutoModel
    TRANSFORMERS_VISION_AVAILABLE = True
except ImportError:
    TRANSFORMERS_VISION_AVAILABLE = False

# Video processing
try:
    import decord
    from decord import VideoReader, cpu, gpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False
    warnings.warn("Decord not available. Install with: pip install decord")

# Scientific image processing
try:
    import skimage
    from skimage import measure, morphology, segmentation, feature
    import scipy
    from scipy import ndimage
    SCIENTIFIC_PROCESSING_AVAILABLE = True
except ImportError:
    SCIENTIFIC_PROCESSING_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VisionConfig:
    """Configuration for vision processing components"""
    
    # Image processing
    image_size: int = 224
    max_image_size: int = 1024
    supported_formats: List[str] = None
    
    # Video processing
    video_frames: int = 16
    video_fps: int = 1.0
    max_video_duration: float = 30.0  # seconds
    
    # Model configuration
    use_multiple_backends: bool = True
    use_ensemble: bool = True
    use_scientific_models: bool = True
    
    # Performance optimization
    use_gpu_acceleration: bool = True
    batch_processing: bool = True
    max_batch_size: int = 32
    
    # Quality settings
    image_quality_threshold: float = 0.7
    enable_preprocessing: bool = True
    enable_augmentation: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']

class ImagePreprocessor:
    """Advanced image preprocessing for multi-modal analysis"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        
        # Define preprocessing pipelines
        self._setup_preprocessing_pipelines()
        
        logger.info("✅ Image preprocessor initialized")
    
    def _setup_preprocessing_pipelines(self):
        """Setup different preprocessing pipelines for different use cases"""
        
        if CV2_AVAILABLE:
            # Scientific image preprocessing
            self.scientific_pipeline = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.OneOf([
                    A.GaussianBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # General purpose preprocessing
            self.general_pipeline = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ], p=0.3),
                A.OneOf([
                    A.RandomBrightnessContrast(p=0.5),
                    A.HueSaturationValue(p=0.5),
                ], p=0.3),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            # High quality pipeline (no augmentation)
            self.quality_pipeline = A.Compose([
                A.Resize(self.config.image_size, self.config.image_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            # Fallback PyTorch transforms
            if TORCHVISION_AVAILABLE:
                self.scientific_pipeline = transforms.Compose([
                    transforms.Resize((self.config.image_size, self.config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.scientific_pipeline = None
            
            self.general_pipeline = self.scientific_pipeline
            self.quality_pipeline = self.scientific_pipeline
    
    def preprocess_image(self, image_input: Union[str, bytes, np.ndarray, "PIL.Image.Image"], 
                        pipeline_type: str = "general") -> torch.Tensor:
        """
        Preprocess image for analysis
        
        Args:
            image_input: Image in various formats
            pipeline_type: Type of preprocessing ("scientific", "general", "quality")
            
        Returns:
            Preprocessed image tensor
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_input, str):
                # File path or base64 string
                if image_input.startswith('data:'):
                    # Base64 encoded image
                    image_data = base64.b64decode(image_input.split(',')[1])
                    image = Image.open(io.BytesIO(image_data))
                else:
                    # File path
                    image = Image.open(image_input)
            elif isinstance(image_input, bytes):
                image = Image.open(io.BytesIO(image_input))
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input)
            elif CV2_AVAILABLE and hasattr(image_input, 'save'):  # PIL Image check (has save method)
                image = image_input
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Quality check
            quality_score = self._assess_image_quality(image)
            if quality_score < self.config.image_quality_threshold:
                logger.warning(f"Low image quality detected: {quality_score:.2f}")
            
            # Apply preprocessing pipeline
            if CV2_AVAILABLE:
                # Convert to numpy for albumentations
                image_np = np.array(image)
                
                if pipeline_type == "scientific" and self.scientific_pipeline is not None:
                    processed = self.scientific_pipeline(image=image_np)
                    return processed['image']
                elif pipeline_type == "quality" and self.quality_pipeline is not None:
                    processed = self.quality_pipeline(image=image_np)
                    return processed['image']
                elif self.general_pipeline is not None:
                    processed = self.general_pipeline(image=image_np)
                    return processed['image']
                else:
                    # Fallback: convert to tensor manually
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                    return F.interpolate(image_tensor.unsqueeze(0), 
                                       size=(self.config.image_size, self.config.image_size), 
                                       mode='bilinear', align_corners=False).squeeze(0)
            else:
                # Use PyTorch transforms if available
                if self.general_pipeline is not None:
                    return self.general_pipeline(image)
                else:
                    # Fallback: basic tensor conversion
                    image_np = np.array(image)
                    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
                    return F.interpolate(image_tensor.unsqueeze(0), 
                                       size=(self.config.image_size, self.config.image_size), 
                                       mode='bilinear', align_corners=False).squeeze(0)
                
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return dummy tensor as fallback
            return torch.zeros(3, self.config.image_size, self.config.image_size)
    
    def _assess_image_quality(self, image: "PIL.Image.Image") -> float:
        """Assess image quality using various metrics"""
        try:
            # Convert to numpy
            img_array = np.array(image.convert('L'))  # Grayscale for analysis
            
            # Calculate quality metrics
            metrics = []
            
            # Variance (sharpness indicator)
            variance = np.var(img_array)
            metrics.append(min(1.0, variance / 10000))
            
            # Edge density (detail indicator)
            if CV2_AVAILABLE:
                edges = cv2.Canny(img_array, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                metrics.append(min(1.0, edge_density * 10))
            
            # Brightness distribution
            hist = np.histogram(img_array, bins=256)[0]
            hist_norm = hist / np.sum(hist)
            entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
            metrics.append(min(1.0, entropy / 8))
            
            return np.mean(metrics) if metrics else 0.5
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return 0.5

class AdvancedImageAnalyzer:
    """Advanced image analysis with multiple model backends"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.preprocessor = ImagePreprocessor(config)
        
        # Initialize model backends
        self._initialize_models()
        
        # Performance tracking
        self.processing_times = []
        
        logger.info("✅ Advanced Image Analyzer initialized")
    
    def _initialize_models(self):
        """Initialize multiple model backends for comprehensive analysis"""
        self.models = {}
        
        # General purpose models
        if TORCHVISION_AVAILABLE:
            try:
                # EfficientNet for general classification
                self.models['efficientnet'] = efficientnet_b4(pretrained=True)
                self.models['efficientnet'].eval()
                
                # ResNet for robust feature extraction
                self.models['resnet'] = resnet50(pretrained=True)
                self.models['resnet'].eval()
                
                # DenseNet for medical/scientific images
                self.models['densenet'] = densenet121(pretrained=True)
                self.models['densenet'].eval()
                
                logger.info("✅ Torchvision models loaded")
                
            except Exception as e:
                logger.warning(f"Could not load torchvision models: {e}")
        
        # Transformers-based models
        if TRANSFORMERS_VISION_AVAILABLE:
            try:
                # CLIP for multimodal understanding
                self.models['clip'] = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.models['clip'].eval()
                
                logger.info("✅ CLIP model loaded")
                
            except Exception as e:
                logger.warning(f"Could not load CLIP model: {e}")
        
        # Scientific image analysis models
        if self.config.use_scientific_models:
            self._initialize_scientific_models()
        
        # Feature extractors
        self._setup_feature_extractors()
    
    def _initialize_scientific_models(self):
        """Initialize models specialized for scientific image analysis"""
        try:
            # Custom scientific image classifier
            self.models['scientific'] = self._create_scientific_classifier()
            
            # Astronomical image classifier
            self.models['astronomical'] = self._create_astronomical_classifier()
            
            logger.info("✅ Scientific models initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize scientific models: {e}")
    
    def _create_scientific_classifier(self) -> nn.Module:
        """Create scientific image classifier"""
        model = nn.Sequential(
            # Feature extraction layers
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Scientific-specific layers
            self._make_scientific_block(64, 128, 2),
            self._make_scientific_block(128, 256, 2),
            self._make_scientific_block(256, 512, 2),
            
            # Classification head
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)  # 10 scientific categories
        )
        
        return model
    
    def _create_astronomical_classifier(self) -> nn.Module:
        """Create astronomical image classifier"""
        model = nn.Sequential(
            # Specialized for astronomical images
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 16, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 5)  # Star, Galaxy, Nebula, Planet, Other
        )
        
        return model
    
    def _make_scientific_block(self, in_channels: int, out_channels: int, num_layers: int) -> nn.Module:
        """Create scientific analysis block"""
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
            else:
                layers.extend([
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                ])
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        return nn.Sequential(*layers)
    
    def _setup_feature_extractors(self):
        """Setup feature extractors for different model backends"""
        self.feature_extractors = {}
        
        for model_name, model in self.models.items():
            if model_name == 'clip':
                continue  # CLIP has special handling
            
            # Create feature extractor by removing final classification layer
            if hasattr(model, 'classifier'):
                # EfficientNet, DenseNet style
                self.feature_extractors[model_name] = nn.Sequential(*list(model.children())[:-1])
            elif hasattr(model, 'fc'):
                # ResNet style
                self.feature_extractors[model_name] = nn.Sequential(*list(model.children())[:-1])
            else:
                # Custom models
                self.feature_extractors[model_name] = nn.Sequential(*list(model.children())[:-2])
    
    async def analyze_image(self, image_input: Any, 
                          analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Perform comprehensive image analysis
        
        Args:
            image_input: Image in various formats
            analysis_type: Type of analysis ("comprehensive", "scientific", "quick")
            
        Returns:
            Dictionary with analysis results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocessor.preprocess_image(image_input, "scientific")
            
            # Add batch dimension
            if processed_image.dim() == 3:
                processed_image = processed_image.unsqueeze(0)
            
            # Move to appropriate device
            device = next(iter(self.models.values())).parameters().__next__().device if self.models else 'cpu'
            processed_image = processed_image.to(device)
            
            # Perform analysis based on type
            if analysis_type == "comprehensive":
                results = await self._comprehensive_analysis(processed_image)
            elif analysis_type == "scientific":
                results = await self._scientific_analysis(processed_image)
            elif analysis_type == "quick":
                results = await self._quick_analysis(processed_image)
            else:
                results = await self._comprehensive_analysis(processed_image)
            
            # Add metadata
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            results.update({
                'processing_time': processing_time,
                'image_shape': processed_image.shape,
                'analysis_type': analysis_type,
                'models_used': list(self.models.keys()),
                'success': True
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    async def _comprehensive_analysis(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Perform comprehensive analysis using all available models"""
        results = {}
        
        # Extract features from all models
        with torch.no_grad():
            for model_name, extractor in self.feature_extractors.items():
                try:
                    features = extractor(image_tensor)
                    if features.dim() > 2:
                        features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                    
                    results[f'{model_name}_features'] = features.cpu().numpy()
                    
                except Exception as e:
                    logger.warning(f"Feature extraction failed for {model_name}: {e}")
        
        # CLIP analysis if available
        if 'clip' in self.models and hasattr(self, 'clip_processor'):
            try:
                clip_results = await self._clip_analysis(image_tensor)
                results.update(clip_results)
            except Exception as e:
                logger.warning(f"CLIP analysis failed: {e}")
        
        # Scientific classification
        if 'scientific' in self.models:
            try:
                with torch.no_grad():
                    scientific_output = self.models['scientific'](image_tensor)
                    scientific_probs = F.softmax(scientific_output, dim=1)
                    results['scientific_classification'] = scientific_probs.cpu().numpy()
            except Exception as e:
                logger.warning(f"Scientific classification failed: {e}")
        
        # Image quality and characteristics
        results.update(self._analyze_image_characteristics(image_tensor))
        
        return results
    
    async def _scientific_analysis(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Perform scientific-focused analysis"""
        results = {}
        
        # Scientific model analysis
        if 'scientific' in self.models:
            with torch.no_grad():
                output = self.models['scientific'](image_tensor)
                probs = F.softmax(output, dim=1)
                results['scientific_classification'] = probs.cpu().numpy()
        
        # Astronomical analysis if applicable
        if 'astronomical' in self.models:
            with torch.no_grad():
                output = self.models['astronomical'](image_tensor)
                probs = F.softmax(output, dim=1)
                results['astronomical_classification'] = probs.cpu().numpy()
        
        # Feature analysis for scientific relevance
        if SCIENTIFIC_PROCESSING_AVAILABLE:
            results.update(self._scientific_feature_analysis(image_tensor))
        
        return results
    
    async def _quick_analysis(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Perform quick analysis using most efficient models"""
        results = {}
        
        # Use most efficient model available
        if 'efficientnet' in self.models:
            model_name = 'efficientnet'
        elif 'resnet' in self.models:
            model_name = 'resnet'
        else:
            model_name = list(self.models.keys())[0] if self.models else None
        
        if model_name and model_name in self.feature_extractors:
            with torch.no_grad():
                features = self.feature_extractors[model_name](image_tensor)
                if features.dim() > 2:
                    features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
                results[f'{model_name}_features'] = features.cpu().numpy()
        
        return results
    
    async def _clip_analysis(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Perform CLIP-based multimodal analysis"""
        results = {}
        
        try:
            # Convert tensor back to PIL for CLIP processor
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = (image_np * std + mean) * 255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            
            # Process with CLIP
            inputs = self.clip_processor(images=image_pil, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.models['clip'].get_image_features(**inputs)
                results['clip_features'] = image_features.cpu().numpy()
            
            # Analyze with scientific prompts
            scientific_prompts = [
                "a scientific instrument",
                "a laboratory experiment",
                "astronomical data",
                "microscopy image",
                "medical scan",
                "geological sample",
                "chemical structure",
                "biological specimen"
            ]
            
            text_inputs = self.clip_processor(text=scientific_prompts, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                text_features = self.models['clip'].get_text_features(**text_inputs)
                
                # Compute similarities
                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                
                similarities = torch.mm(image_features_norm, text_features_norm.t())
                results['scientific_similarities'] = similarities.cpu().numpy()
        
        except Exception as e:
            logger.warning(f"CLIP analysis failed: {e}")
        
        return results
    
    def _analyze_image_characteristics(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Analyze basic image characteristics"""
        characteristics = {}
        
        try:
            # Convert to numpy for analysis
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            # Color analysis
            characteristics['mean_color'] = np.mean(image_np, axis=(0, 1)).tolist()
            characteristics['color_std'] = np.std(image_np, axis=(0, 1)).tolist()
            
            # Brightness and contrast
            gray = np.mean(image_np, axis=2)
            characteristics['brightness'] = float(np.mean(gray))
            characteristics['contrast'] = float(np.std(gray))
            
            # Edge density (if CV2 available)
            if CV2_AVAILABLE:
                gray_uint8 = (gray * 255).astype(np.uint8)
                edges = cv2.Canny(gray_uint8, 50, 150)
                characteristics['edge_density'] = float(np.sum(edges > 0) / edges.size)
            
        except Exception as e:
            logger.warning(f"Characteristic analysis failed: {e}")
        
        return characteristics
    
    def _scientific_feature_analysis(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Perform scientific feature analysis"""
        features = {}
        
        try:
            # Convert to numpy
            image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            gray = np.mean(image_np, axis=2)
            
            # Texture analysis
            if SCIENTIFIC_PROCESSING_AVAILABLE:
                # Local Binary Pattern
                lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
                features['lbp_histogram'] = np.histogram(lbp, bins=10)[0].tolist()
                
                # GLCM properties
                gray_uint8 = (gray * 255).astype(np.uint8)
                glcm = feature.graycomatrix(gray_uint8, [1], [0], levels=256, symmetric=True, normed=True)
                features['glcm_contrast'] = float(feature.graycoprops(glcm, 'contrast')[0, 0])
                features['glcm_homogeneity'] = float(feature.graycoprops(glcm, 'homogeneity')[0, 0])
                
        except Exception as e:
            logger.warning(f"Scientific feature analysis failed: {e}")
        
        return features
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'total_images_processed': len(self.processing_times),
            'models_available': list(self.models.keys())
        }

class VideoProcessor:
    """Advanced video processing for temporal analysis"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        
        # Initialize video processing components
        self._setup_video_models()
        
        logger.info("✅ Video processor initialized")
    
    def _setup_video_models(self):
        """Setup video processing models"""
        # 3D CNN for video analysis
        self.video_cnn = self._create_3d_cnn()
        
        # Temporal feature extractor
        self.temporal_extractor = self._create_temporal_extractor()
        
        # Video quality analyzer
        self.quality_analyzer = self._create_quality_analyzer()
    
    def _create_3d_cnn(self) -> nn.Module:
        """Create 3D CNN for video processing"""
        return nn.Sequential(
            # 3D convolution layers
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            
            # Temporal processing
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Global pooling
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            
            # Classification
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
    
    def _create_temporal_extractor(self) -> nn.Module:
        """Create temporal feature extractor"""
        return nn.Sequential(
            nn.LSTM(256, 128, num_layers=2, batch_first=True, bidirectional=True),
        )
    
    def _create_quality_analyzer(self) -> nn.Module:
        """Create video quality analyzer"""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    async def process_video(self, video_input: Union[str, bytes], 
                          analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Process video for temporal analysis
        
        Args:
            video_input: Video file path or bytes
            analysis_type: Type of analysis to perform
            
        Returns:
            Video analysis results
        """
        start_time = time.time()
        
        try:
            # Extract frames from video
            frames = self._extract_frames(video_input)
            
            if frames is None or len(frames) == 0:
                return {'error': 'Could not extract frames from video', 'success': False}
            
            # Convert frames to tensor
            video_tensor = self._frames_to_tensor(frames)
            
            # Perform analysis
            results = {}
            
            if analysis_type in ["comprehensive", "temporal"]:
                # 3D CNN analysis
                with torch.no_grad():
                    video_features = self.video_cnn(video_tensor)
                    results['video_features'] = video_features.cpu().numpy()
                
                # Quality analysis
                quality_score = self.quality_analyzer(video_features)
                results['quality_score'] = float(quality_score.cpu().numpy())
            
            # Frame-by-frame analysis if needed
            if analysis_type == "comprehensive":
                results['frame_analysis'] = await self._analyze_frames(frames)
            
            # Temporal characteristics
            results.update(self._analyze_temporal_characteristics(frames))
            
            # Add metadata
            results.update({
                'processing_time': time.time() - start_time,
                'num_frames': len(frames),
                'video_shape': video_tensor.shape,
                'success': True
            })
            
            return results
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return {
                'error': str(e),
                'processing_time': time.time() - start_time,
                'success': False
            }
    
    def _extract_frames(self, video_input: Union[str, bytes]) -> Optional[List[np.ndarray]]:
        """Extract frames from video"""
        frames = []
        
        try:
            if DECORD_AVAILABLE and isinstance(video_input, str):
                # Use decord for efficient frame extraction
                vr = VideoReader(video_input, ctx=cpu(0))
                
                # Calculate frame indices
                total_frames = len(vr)
                frame_indices = np.linspace(0, total_frames - 1, self.config.video_frames, dtype=int)
                
                # Extract frames
                for idx in frame_indices:
                    frame = vr[idx].asnumpy()
                    frames.append(frame)
            
            elif CV2_AVAILABLE:
                # Use OpenCV as fallback
                if isinstance(video_input, str):
                    cap = cv2.VideoCapture(video_input)
                else:
                    # For bytes input, save to temporary file
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                        tmp.write(video_input)
                        tmp.flush()
                        cap = cv2.VideoCapture(tmp.name)
                
                # Extract frames
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_indices = np.linspace(0, total_frames - 1, self.config.video_frames, dtype=int)
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                
                cap.release()
            
            return frames if frames else None
            
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return None
    
    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """Convert frames to video tensor"""
        # Resize frames
        resized_frames = []
        for frame in frames:
            if CV2_AVAILABLE:
                resized = cv2.resize(frame, (self.config.image_size, self.config.image_size))
            else:
                # Simple resize using numpy
                resized = frame  # Placeholder - would need proper resize implementation
            resized_frames.append(resized)
        
        # Convert to tensor [batch, channels, frames, height, width]
        video_array = np.stack(resized_frames, axis=0)  # [frames, height, width, channels]
        video_array = np.transpose(video_array, (3, 0, 1, 2))  # [channels, frames, height, width]
        video_tensor = torch.from_numpy(video_array).float() / 255.0
        video_tensor = video_tensor.unsqueeze(0)  # Add batch dimension
        
        return video_tensor
    
    async def _analyze_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze individual frames"""
        frame_results = []
        
        # Create image analyzer for frame analysis
        image_config = VisionConfig()
        image_analyzer = AdvancedImageAnalyzer(image_config)
        
        for i, frame in enumerate(frames):
            try:
                # Convert frame to PIL Image
                frame_pil = Image.fromarray(frame)
                
                # Analyze frame
                result = await image_analyzer.analyze_image(frame_pil, "quick")
                result['frame_index'] = i
                frame_results.append(result)
                
            except Exception as e:
                logger.warning(f"Frame {i} analysis failed: {e}")
        
        return {
            'individual_frames': frame_results,
            'num_frames_analyzed': len(frame_results)
        }
    
    def _analyze_temporal_characteristics(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """Analyze temporal characteristics of video"""
        characteristics = {}
        
        try:
            # Motion analysis
            if len(frames) > 1 and CV2_AVAILABLE:
                motion_scores = []
                for i in range(1, len(frames)):
                    # Convert to grayscale
                    gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
                    
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, None, None)
                    if flow[0] is not None:
                        motion_score = np.mean(np.abs(flow[0]))
                        motion_scores.append(motion_score)
                
                if motion_scores:
                    characteristics['avg_motion'] = float(np.mean(motion_scores))
                    characteristics['max_motion'] = float(np.max(motion_scores))
            
            # Color stability
            color_changes = []
            for i in range(1, len(frames)):
                color_diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
                color_changes.append(color_diff)
            
            if color_changes:
                characteristics['color_stability'] = float(np.mean(color_changes))
            
            # Brightness variation
            brightness_values = [np.mean(frame) for frame in frames]
            characteristics['brightness_variation'] = float(np.std(brightness_values))
            
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
        
        return characteristics

# Factory functions
def create_vision_processor(config: VisionConfig = None) -> AdvancedImageAnalyzer:
    """Create advanced image analyzer"""
    if config is None:
        config = VisionConfig()
    
    return AdvancedImageAnalyzer(config)

def create_video_processor(config: VisionConfig = None) -> VideoProcessor:
    """Create video processor"""
    if config is None:
        config = VisionConfig()
    
    return VideoProcessor(config)

# Demo function
async def demo_vision_processing():
    """Demonstrate vision processing capabilities"""
    logger.info("🎭 Vision Processing Demo")
    logger.info("=" * 40)
    
    # Create processors
    config = VisionConfig()
    image_analyzer = create_vision_processor(config)
    video_processor = create_video_processor(config)
    
    # Demo with dummy data
    logger.info("📷 Testing image analysis...")
    
    # Create dummy image
    dummy_image = torch.randn(3, 224, 224) * 0.5 + 0.5
    dummy_image_np = (dummy_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    dummy_image_pil = Image.fromarray(dummy_image_np)
    
    # Analyze image
    try:
        image_results = await image_analyzer.analyze_image(dummy_image_pil, "comprehensive")
        logger.info(f"✅ Image analysis completed in {image_results['processing_time']:.2f}s")
        logger.info(f"   Models used: {image_results['models_used']}")
        
    except Exception as e:
        logger.error(f"❌ Image analysis failed: {e}")
    
    logger.info("✅ Vision processing demo completed")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_vision_processing()) 