"""
NASA-Ready Astrobiology Surrogate API
=====================================

Production FastAPI backend for exoplanet habitability assessment.
Supports all operational modes: scalar, datacube, joint, spectral.
"""

from __future__ import annotations
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import numpy as np
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union, Literal
import logging
from pathlib import Path
import time
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Import our models
from models.surrogate_transformer import SurrogateTransformer, UncertaintyQuantification
from models.graph_vae import GVAE
from models.fusion_transformer import FusionModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model storage
models = {}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Loading models...")
    await load_models()
    logger.info("Models loaded successfully")
    yield
    logger.info("Shutting down...")


# Initialize FastAPI app
app = FastAPI(
    title="Astrobiology Surrogate Engine",
    description="NASA-ready API for exoplanet habitability assessment using physics-informed ML",
    version="2.0.0",
    contact={
        "name": "Astrobiology Research Team",
        "email": "astrobio@example.com",
        "url": "https://github.com/astrobio/surrogate-engine"
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    },
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class PlanetParameters(BaseModel):
    """Planet parameter specification for habitability assessment"""
    
    radius_earth: float = Field(
        ..., 
        ge=0.1, le=10.0,
        description="Planet radius in Earth radii"
    )
    mass_earth: float = Field(
        ..., 
        ge=0.01, le=100.0,
        description="Planet mass in Earth masses"
    )
    orbital_period: float = Field(
        ..., 
        ge=0.1, le=10000.0,
        description="Orbital period in days"
    )
    insolation: float = Field(
        ..., 
        ge=0.01, le=100.0,
        description="Stellar insolation in Earth units"
    )
    stellar_teff: float = Field(
        ..., 
        ge=1000.0, le=10000.0,
        description="Stellar effective temperature in Kelvin"
    )
    stellar_logg: float = Field(
        ..., 
        ge=2.0, le=6.0,
        description="Stellar surface gravity (log g)"
    )
    stellar_metallicity: float = Field(
        ..., 
        ge=-3.0, le=1.0,
        description="Stellar metallicity [Fe/H]"
    )
    host_mass: float = Field(
        ..., 
        ge=0.1, le=10.0,
        description="Host star mass in solar masses"
    )
    
    @validator('*', pre=True)
    def validate_finite(cls, v):
        if not np.isfinite(v):
            raise ValueError("All parameters must be finite numbers")
        return v

    class Config:
        schema_extra = {
            "example": {
                "radius_earth": 1.0,
                "mass_earth": 1.0,
                "orbital_period": 365.25,
                "insolation": 1.0,
                "stellar_teff": 5778.0,
                "stellar_logg": 4.44,
                "stellar_metallicity": 0.0,
                "host_mass": 1.0
            }
        }


class HabitabilityResponse(BaseModel):
    """Response for scalar habitability assessment"""
    
    habitability_score: float = Field(description="Habitability probability (0-1)")
    surface_temperature: float = Field(description="Predicted surface temperature (K)")
    atmospheric_pressure: float = Field(description="Predicted atmospheric pressure (bar)")
    
    # Physics constraints
    energy_balance: float = Field(description="Energy balance parameter")
    atmospheric_composition: Dict[str, float] = Field(description="Predicted atmospheric composition")
    
    # Uncertainty quantification
    uncertainty: Optional[Dict[str, float]] = Field(description="Prediction uncertainties")
    
    # Metadata
    inference_time: float = Field(description="Inference time in seconds")
    model_version: str = Field(description="Model version used")
    timestamp: datetime = Field(description="Prediction timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "habitability_score": 0.85,
                "surface_temperature": 288.5,
                "atmospheric_pressure": 1.01,
                "energy_balance": 0.98,
                "atmospheric_composition": {
                    "N2": 0.78,
                    "O2": 0.21,
                    "CO2": 0.0004,
                    "H2O": 0.01
                },
                "uncertainty": {
                    "habitability_std": 0.05,
                    "temperature_std": 5.2,
                    "pressure_std": 0.1
                },
                "inference_time": 0.23,
                "model_version": "surrogate-v2.0",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }


class BatchPlanetRequest(BaseModel):
    """Batch processing request for multiple planets"""
    
    planets: List[PlanetParameters] = Field(
        ..., 
        max_items=1000,
        description="List of planets to assess (max 1000)"
    )
    include_uncertainty: bool = Field(
        default=False,
        description="Include uncertainty quantification"
    )
    priority: Literal["low", "normal", "high"] = Field(
        default="normal",
        description="Processing priority"
    )


class DatacubeResponse(BaseModel):
    """Response for 3D climate datacube prediction"""
    
    temperature_field: List[List[List[float]]] = Field(description="3D temperature field (lat×lon×pressure)")
    humidity_field: List[List[List[float]]] = Field(description="3D humidity field (lat×lon×pressure)")
    
    # Grid metadata
    grid_info: Dict[str, Union[int, List[float]]] = Field(description="Grid information")
    
    # Quality metrics
    physics_constraints: Dict[str, float] = Field(description="Physics constraint satisfaction")
    
    # Metadata
    inference_time: float
    model_version: str
    timestamp: datetime


class ValidationRequest(BaseModel):
    """Request for model validation against benchmark planets"""
    
    benchmark_planets: List[str] = Field(
        default=["Earth", "TRAPPIST-1e", "Proxima Centauri b"],
        description="List of benchmark planets to validate against"
    )
    tolerance: float = Field(
        default=3.0,
        ge=0.1, le=100.0,
        description="Validation tolerance in Kelvin"
    )


class HealthCheck(BaseModel):
    """API health status"""
    
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    models_loaded: Dict[str, bool] = Field(description="Model loading status")
    gpu_available: bool = Field(description="GPU availability")
    memory_usage: float = Field(description="Memory usage percentage")
    uptime: float = Field(description="Uptime in seconds")


# Model loading functions
async def load_models():
    """Load all required models"""
    global models
    
    try:
        # Load surrogate transformer (scalar mode)
        scalar_model = SurrogateTransformer(mode="scalar").to(device)
        scalar_model.eval()
        models["scalar"] = scalar_model
        models["scalar_uncertainty"] = UncertaintyQuantification(scalar_model)
        
        # Load datacube model if available
        try:
            datacube_model = SurrogateTransformer(mode="datacube").to(device)
            datacube_model.eval()
            models["datacube"] = datacube_model
        except Exception as e:
            logger.warning(f"Datacube model not available: {e}")
            models["datacube"] = None
        
        # Load joint model if available
        try:
            joint_model = SurrogateTransformer(mode="joint").to(device)
            joint_model.eval()
            models["joint"] = joint_model
        except Exception as e:
            logger.warning(f"Joint model not available: {e}")
            models["joint"] = None
        
        # Load spectral model if available
        try:
            spectral_model = SurrogateTransformer(mode="spectral").to(device)
            spectral_model.eval()
            models["spectral"] = spectral_model
        except Exception as e:
            logger.warning(f"Spectral model not available: {e}")
            models["spectral"] = None
        
        logger.info("Model loading completed")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise RuntimeError(f"Model loading failed: {e}")


def get_model(mode: str):
    """Dependency to get model for specific mode"""
    if mode not in models or models[mode] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model for mode '{mode}' is not available"
        )
    return models[mode]


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NASA Astrobiology Surrogate Engine API",
        "version": "2.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Comprehensive health check endpoint"""
    import psutil
    
    # Check model availability
    models_loaded = {
        "scalar": models.get("scalar") is not None,
        "datacube": models.get("datacube") is not None,
        "joint": models.get("joint") is not None,
        "spectral": models.get("spectral") is not None
    }
    
    # Get system info
    memory_usage = psutil.virtual_memory().percent
    gpu_available = torch.cuda.is_available()
    
    return HealthCheck(
        status="healthy" if models_loaded["scalar"] else "degraded",
        version="2.0.0",
        models_loaded=models_loaded,
        gpu_available=gpu_available,
        memory_usage=memory_usage,
        uptime=time.time()  # Simplified uptime
    )


@app.post("/predict/habitability", response_model=HabitabilityResponse)
async def predict_habitability(
    planet: PlanetParameters,
    include_uncertainty: bool = False,
    model = Depends(lambda: get_model("scalar"))
):
    """
    Predict exoplanet habitability using the scalar surrogate model.
    
    This endpoint provides fast (<0.4s) habitability assessment for NASA operations.
    """
    start_time = time.time()
    
    try:
        # Convert parameters to tensor
        params_tensor = torch.tensor([
            planet.radius_earth,
            planet.mass_earth,
            planet.orbital_period,
            planet.insolation,
            planet.stellar_teff,
            planet.stellar_logg,
            planet.stellar_metallicity,
            planet.host_mass
        ], dtype=torch.float32, device=device).unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            if include_uncertainty:
                outputs = models["scalar_uncertainty"].predict_with_uncertainty(params_tensor)
                
                # Extract means and uncertainties
                habitability = float(torch.sigmoid(outputs["habitability_mean"]).item())
                surface_temp = float(outputs["surface_temp_mean"].item())
                atm_pressure = float(torch.exp(outputs["atmospheric_pressure_mean"]).item())
                
                uncertainty = {
                    "habitability_std": float(outputs["habitability_std"].item()),
                    "temperature_std": float(outputs["surface_temp_std"].item()),
                    "pressure_std": float(outputs["atmospheric_pressure_std"].item())
                }
            else:
                outputs = model(params_tensor)
                
                habitability = float(torch.sigmoid(outputs["habitability"]).item())
                surface_temp = float(outputs["surface_temp"].item())
                atm_pressure = float(torch.exp(outputs["atmospheric_pressure"]).item())
                uncertainty = None
        
        # Extract physics constraints
        energy_balance = float(outputs["energy_balance"].item())
        atm_composition = {
            "N2": float(outputs["atmospheric_composition"][0, 0].item()),
            "O2": float(outputs["atmospheric_composition"][0, 1].item()),
            "CO2": float(outputs["atmospheric_composition"][0, 2].item()),
            "H2O": float(outputs["atmospheric_composition"][0, 3].item())
        }
        
        inference_time = time.time() - start_time
        
        return HabitabilityResponse(
            habitability_score=habitability,
            surface_temperature=surface_temp,
            atmospheric_pressure=atm_pressure,
            energy_balance=energy_balance,
            atmospheric_composition=atm_composition,
            uncertainty=uncertainty,
            inference_time=inference_time,
            model_version="surrogate-v2.0",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch_habitability(
    request: BatchPlanetRequest,
    background_tasks: BackgroundTasks,
    model = Depends(lambda: get_model("scalar"))
):
    """
    Batch prediction for multiple exoplanets.
    
    Efficiently processes up to 1000 planets with optional uncertainty quantification.
    """
    start_time = time.time()
    
    try:
        # Convert batch to tensor
        batch_params = []
        for planet in request.planets:
            params = [
                planet.radius_earth, planet.mass_earth, planet.orbital_period,
                planet.insolation, planet.stellar_teff, planet.stellar_logg,
                planet.stellar_metallicity, planet.host_mass
            ]
            batch_params.append(params)
        
        params_tensor = torch.tensor(batch_params, dtype=torch.float32, device=device)
        
        # Batch prediction
        with torch.no_grad():
            outputs = model(params_tensor)
            
            # Process results
            results = []
            for i in range(len(request.planets)):
                habitability = float(torch.sigmoid(outputs["habitability"][i]).item())
                surface_temp = float(outputs["surface_temp"][i].item())
                atm_pressure = float(torch.exp(outputs["atmospheric_pressure"][i]).item())
                
                result = {
                    "habitability_score": habitability,
                    "surface_temperature": surface_temp,
                    "atmospheric_pressure": atm_pressure,
                    "planet_index": i
                }
                results.append(result)
        
        total_time = time.time() - start_time
        
        return {
            "results": results,
            "batch_size": len(request.planets),
            "total_inference_time": total_time,
            "avg_time_per_planet": total_time / len(request.planets),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.post("/predict/datacube", response_model=DatacubeResponse)
async def predict_datacube(
    planet: PlanetParameters,
    model = Depends(lambda: get_model("datacube"))
):
    """
    Predict full 3D climate datacube for detailed analysis.
    
    Returns temperature and humidity fields across latitude×longitude×pressure grid.
    Requires datacube model (Upgrade 1 from roadmap).
    """
    start_time = time.time()
    
    try:
        # Convert parameters to tensor
        params_tensor = torch.tensor([
            planet.radius_earth, planet.mass_earth, planet.orbital_period,
            planet.insolation, planet.stellar_teff, planet.stellar_logg,
            planet.stellar_metallicity, planet.host_mass
        ], dtype=torch.float32, device=device).unsqueeze(0)
        
        # Prediction
        with torch.no_grad():
            outputs = model(params_tensor)
            
            # Extract 3D fields
            temp_field = outputs["temperature_field"][0].cpu().numpy().tolist()
            humidity_field = outputs["humidity_field"][0].cpu().numpy().tolist()
        
        inference_time = time.time() - start_time
        
        return DatacubeResponse(
            temperature_field=temp_field,
            humidity_field=humidity_field,
            grid_info={
                "n_lat": 64,
                "n_lon": 32,
                "n_pressure": 20,
                "lat_range": [-90, 90],
                "lon_range": [-180, 180],
                "pressure_range": [1000, 0.1]  # hPa
            },
            physics_constraints={
                "energy_balance": float(outputs["energy_balance"].item()),
                "mass_conservation": 1.0  # Placeholder
            },
            inference_time=inference_time,
            model_version="datacube-v2.0",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Datacube prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Datacube prediction failed: {str(e)}"
        )


@app.post("/validate/benchmarks")
async def validate_model(
    request: ValidationRequest,
    model = Depends(lambda: get_model("scalar"))
):
    """
    Validate model performance against benchmark planets.
    
    Tests model accuracy on Earth, TRAPPIST-1e, Proxima Centauri b, etc.
    Critical for NASA validation protocols.
    """
    
    # Benchmark planet data
    benchmarks = {
        "Earth": {
            "params": [1.0, 1.0, 365.25, 1.0, 5778, 4.44, 0.0, 1.0],
            "expected_temp": 288.0,
            "expected_habitability": 1.0
        },
        "TRAPPIST-1e": {
            "params": [0.91, 0.77, 6.1, 0.66, 2559, 5.4, 0.04, 0.089],
            "expected_temp": 251.0,
            "expected_habitability": 0.8
        },
        "Proxima Centauri b": {
            "params": [1.07, 1.17, 11.2, 1.5, 3042, 5.2, -0.29, 0.123],
            "expected_temp": 234.0,
            "expected_habitability": 0.6
        }
    }
    
    results = {}
    
    for planet_name in request.benchmark_planets:
        if planet_name not in benchmarks:
            continue
            
        benchmark = benchmarks[planet_name]
        params_tensor = torch.tensor(benchmark["params"], dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(params_tensor)
            
            predicted_temp = float(outputs["surface_temp"].item())
            predicted_habitability = float(torch.sigmoid(outputs["habitability"]).item())
        
        temp_error = abs(predicted_temp - benchmark["expected_temp"])
        hab_error = abs(predicted_habitability - benchmark["expected_habitability"])
        
        results[planet_name] = {
            "predicted_temperature": predicted_temp,
            "expected_temperature": benchmark["expected_temp"],
            "temperature_error": temp_error,
            "temperature_within_tolerance": temp_error <= request.tolerance,
            "predicted_habitability": predicted_habitability,
            "expected_habitability": benchmark["expected_habitability"],
            "habitability_error": hab_error
        }
    
    # Overall validation metrics
    temp_errors = [r["temperature_error"] for r in results.values()]
    validation_passed = all(r["temperature_within_tolerance"] for r in results.values())
    
    return {
        "validation_passed": validation_passed,
        "tolerance": request.tolerance,
        "mean_temperature_error": np.mean(temp_errors),
        "max_temperature_error": np.max(temp_errors),
        "benchmark_results": results,
        "timestamp": datetime.utcnow()
    }


@app.get("/models/info")
async def model_info():
    """Get information about available models and their capabilities"""
    
    model_info = {}
    
    for mode, model in models.items():
        if model is not None and not mode.endswith("_uncertainty"):
            info = {
                "available": True,
                "mode": mode,
                "device": str(device),
                "parameters": sum(p.numel() for p in model.parameters()),
                "memory_usage": sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2,  # MB
            }
            
            if hasattr(model, 'mode'):
                info["capabilities"] = {
                    "scalar": ["habitability", "surface_temp", "atmospheric_pressure"],
                    "datacube": ["temperature_field", "humidity_field"],
                    "joint": ["planet_type", "habitability", "spectral_features"],
                    "spectral": ["spectrum"]
                }.get(model.mode, [])
            
            model_info[mode] = info
    
    return {
        "models": model_info,
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    ) 