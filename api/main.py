"""
NASA-Ready Astrobiology Surrogate API
=====================================

Production FastAPI backend for exoplanet habitability assessment.
Supports all operational modes: scalar, datacube, joint, spectral.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from models.fusion_transformer import FusionModel
from models.graph_vae import GVAE

# Import our models
from models.surrogate_transformer import SurrogateTransformer, UncertaintyQuantification

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
        "url": "https://github.com/astrobio/surrogate-engine",
    },
    license_info={"name": "MIT License", "url": "https://opensource.org/licenses/MIT"},
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and setup LLM endpoints
try:
    from .llm_endpoints import setup_llm_routes

    setup_llm_routes(app)
    logger.info("✅ LLM endpoints integrated successfully")
except ImportError as e:
    logger.warning(f"⚠️ LLM endpoints not available: {e}")
except Exception as e:
    logger.error(f"❌ Failed to setup LLM endpoints: {e}")


# Pydantic models for request/response validation
class PlanetParameters(BaseModel):
    """Planet parameter specification for habitability assessment"""

    radius_earth: float = Field(..., ge=0.1, le=10.0, description="Planet radius in Earth radii")
    mass_earth: float = Field(..., ge=0.01, le=100.0, description="Planet mass in Earth masses")
    orbital_period: float = Field(..., ge=0.1, le=10000.0, description="Orbital period in days")
    insolation: float = Field(
        ..., ge=0.01, le=100.0, description="Stellar insolation in Earth units"
    )
    stellar_teff: float = Field(
        ..., ge=1000.0, le=10000.0, description="Stellar effective temperature in Kelvin"
    )
    stellar_logg: float = Field(..., ge=2.0, le=6.0, description="Stellar surface gravity (log g)")
    stellar_metallicity: float = Field(
        ..., ge=-3.0, le=1.0, description="Stellar metallicity [Fe/H]"
    )
    host_mass: float = Field(..., ge=0.1, le=10.0, description="Host star mass in solar masses")

    @validator("*", pre=True)
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
                "host_mass": 1.0,
            }
        }


class HabitabilityResponse(BaseModel):
    """Response for scalar habitability assessment"""

    habitability_score: float = Field(description="Habitability probability (0-1)")
    surface_temperature: float = Field(description="Predicted surface temperature (K)")
    atmospheric_pressure: float = Field(description="Predicted atmospheric pressure (bar)")

    # Physics constraints
    energy_balance: float = Field(description="Energy balance parameter")
    atmospheric_composition: Dict[str, float] = Field(
        description="Predicted atmospheric composition"
    )

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
                "atmospheric_composition": {"N2": 0.78, "O2": 0.21, "CO2": 0.0004, "H2O": 0.01},
                "uncertainty": {
                    "habitability_std": 0.05,
                    "temperature_std": 5.2,
                    "pressure_std": 0.1,
                },
                "inference_time": 0.23,
                "model_version": "surrogate-v2.0",
                "timestamp": "2025-01-15T10:30:00Z",
            }
        }


class BatchPlanetRequest(BaseModel):
    """Batch processing request for multiple planets"""

    planets: List[PlanetParameters] = Field(
        ..., max_items=1000, description="List of planets to assess (max 1000)"
    )
    include_uncertainty: bool = Field(
        default=False, description="Include uncertainty quantification"
    )
    priority: Literal["low", "normal", "high"] = Field(
        default="normal", description="Processing priority"
    )


class DatacubeResponse(BaseModel):
    """Response for 3D climate datacube prediction"""

    temperature_field: List[List[List[float]]] = Field(
        description="3D temperature field (lat×lon×pressure)"
    )
    humidity_field: List[List[List[float]]] = Field(
        description="3D humidity field (lat×lon×pressure)"
    )

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
        description="List of benchmark planets to validate against",
    )
    tolerance: float = Field(
        default=3.0, ge=0.1, le=100.0, description="Validation tolerance in Kelvin"
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
            detail=f"Model for mode '{mode}' is not available",
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
        "health": "/health",
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
        "spectral": models.get("spectral") is not None,
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
        uptime=time.time(),  # Simplified uptime
    )


@app.post("/predict/habitability", response_model=HabitabilityResponse)
async def predict_habitability(
    planet: PlanetParameters,
    include_uncertainty: bool = False,
    model=Depends(lambda: get_model("scalar")),
):
    """
    Predict exoplanet habitability using the scalar surrogate model.

    This endpoint provides fast (<0.4s) habitability assessment for NASA operations.
    """
    start_time = time.time()

    try:
        # Convert parameters to tensor
        params_tensor = torch.tensor(
            [
                planet.radius_earth,
                planet.mass_earth,
                planet.orbital_period,
                planet.insolation,
                planet.stellar_teff,
                planet.stellar_logg,
                planet.stellar_metallicity,
                planet.host_mass,
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

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
                    "pressure_std": float(outputs["atmospheric_pressure_std"].item()),
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
            "H2O": float(outputs["atmospheric_composition"][0, 3].item()),
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
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch_habitability(
    request: BatchPlanetRequest,
    background_tasks: BackgroundTasks,
    model=Depends(lambda: get_model("scalar")),
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
                planet.radius_earth,
                planet.mass_earth,
                planet.orbital_period,
                planet.insolation,
                planet.stellar_teff,
                planet.stellar_logg,
                planet.stellar_metallicity,
                planet.host_mass,
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
                    "planet_index": i,
                }
                results.append(result)

        total_time = time.time() - start_time

        return {
            "results": results,
            "batch_size": len(request.planets),
            "total_inference_time": total_time,
            "avg_time_per_planet": total_time / len(request.planets),
            "timestamp": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.post("/predict/datacube", response_model=DatacubeResponse)
async def predict_datacube(
    planet: PlanetParameters,
    resolution: str = "medium",
    include_physics_validation: bool = True,
    return_format: str = "json",
    model=Depends(lambda: get_model("datacube")),
):
    """
    Predict full 3D climate datacube for detailed analysis.

    Enhanced datacube prediction with multiple resolution options and physics validation.

    Args:
        planet: Planet parameters
        resolution: Grid resolution (low, medium, high)
        include_physics_validation: Include physics constraint validation
        return_format: Response format (json, zarr, netcdf)
        model: Datacube model
    """
    start_time = time.time()

    try:
        # Import enhanced components
        from surrogate import SurrogateMode, get_surrogate_manager
        from validation.eval_cube import EvaluationConfig, PhysicsValidator

        # Get surrogate manager
        surrogate_manager = get_surrogate_manager()
        datacube_model = surrogate_manager.get_model(SurrogateMode.DATACUBE)

        if not datacube_model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Datacube model not available",
            )

        # Convert parameters to tensor
        params_tensor = torch.tensor(
            [
                planet.radius_earth,
                planet.mass_earth,
                planet.orbital_period,
                planet.insolation,
                planet.stellar_teff,
                planet.stellar_logg,
                planet.stellar_metallicity,
                planet.host_mass,
            ],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        # Prediction
        with torch.no_grad():
            outputs = datacube_model.predict(params_tensor)

            # Convert to numpy for processing
            if isinstance(outputs, torch.Tensor):
                outputs = outputs.cpu().numpy()

        # Create enhanced datacube response
        if outputs.ndim == 5:  # (batch, vars, lat, lon, pressure)
            outputs = outputs[0]  # Remove batch dimension

        # Resolution-dependent grid sizes
        grid_sizes = {
            "low": {"lat": 32, "lon": 64, "pressure": 10},
            "medium": {"lat": 64, "lon": 128, "pressure": 20},
            "high": {"lat": 128, "lon": 256, "pressure": 40},
        }

        grid_size = grid_sizes.get(resolution, grid_sizes["medium"])

        # Extract fields with proper dimensions
        temp_field = outputs[0][: grid_size["lat"], : grid_size["lon"], : grid_size["pressure"]]
        humidity_field = (
            outputs[1][: grid_size["lat"], : grid_size["lon"], : grid_size["pressure"]]
            if outputs.shape[0] > 1
            else np.zeros_like(temp_field)
        )
        cloud_field = (
            outputs[2][: grid_size["lat"], : grid_size["lon"], : grid_size["pressure"]]
            if outputs.shape[0] > 2
            else np.zeros_like(temp_field)
        )
        pressure_field = (
            outputs[3][: grid_size["lat"], : grid_size["lon"], : grid_size["pressure"]]
            if outputs.shape[0] > 3
            else np.ones_like(temp_field)
        )

        # Physics validation
        physics_results = {}
        if include_physics_validation:
            try:
                # Create mock xarray dataset for validation
                import xarray as xr

                lat = np.linspace(-90, 90, grid_size["lat"])
                lon = np.linspace(-180, 180, grid_size["lon"])
                pressure = np.logspace(2, -2, grid_size["pressure"])

                cube = xr.Dataset(
                    {
                        "T_surf": (["lat", "lon", "pressure"], temp_field),
                        "q_H2O": (["lat", "lon", "pressure"], humidity_field),
                        "cldfrac": (["lat", "lon", "pressure"], cloud_field),
                        "psurf": (["lat", "lon"], pressure_field[:, :, 0]),
                    },
                    coords={"lat": lat, "lon": lon, "pressure": pressure},
                )

                cube.attrs["insolation"] = planet.insolation

                # Validate physics
                validator = PhysicsValidator(
                    EvaluationConfig(model_path=Path("dummy"), test_data_path=Path("dummy"))
                )

                physics_results = validator.validate_cube(cube)
            except Exception as e:
                physics_results = {"error": str(e)}

        inference_time = time.time() - start_time

        # Build response
        response = DatacubeResponse(
            temperature_field=temp_field.tolist(),
            humidity_field=humidity_field.tolist(),
            grid_info={
                "n_lat": grid_size["lat"],
                "n_lon": grid_size["lon"],
                "n_pressure": grid_size["pressure"],
                "resolution": resolution,
                "lat_range": [-90, 90],
                "lon_range": [-180, 180],
                "pressure_range": [100, 0.01],
                "units": {"temperature": "K", "humidity": "kg/kg", "pressure": "bar"},
            },
            physics_constraints=physics_results,
            inference_time=inference_time,
            model_version="datacube_v1.0",
            timestamp=datetime.now(),
        )

        return response

    except Exception as e:
        logger.error(f"Datacube prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Datacube prediction failed: {str(e)}",
        )


@app.post("/predict/datacube/streaming")
async def predict_datacube_streaming(
    planet: PlanetParameters,
    resolution: str = "medium",
    chunk_size: int = 1000,
    model=Depends(lambda: get_model("datacube")),
):
    """
    Stream datacube prediction for large datasets.

    Returns data in chunks to handle large datacubes efficiently.
    """
    import json

    from fastapi.responses import StreamingResponse

    async def generate_datacube_chunks():
        try:
            # Get datacube prediction
            from surrogate import SurrogateMode, get_surrogate_manager

            surrogate_manager = get_surrogate_manager()
            datacube_model = surrogate_manager.get_model(SurrogateMode.DATACUBE)

            if not datacube_model:
                yield json.dumps({"error": "Datacube model not available"})
                return

            # Convert parameters to tensor
            params_tensor = torch.tensor(
                [
                    planet.radius_earth,
                    planet.mass_earth,
                    planet.orbital_period,
                    planet.insolation,
                    planet.stellar_teff,
                    planet.stellar_logg,
                    planet.stellar_metallicity,
                    planet.host_mass,
                ],
                dtype=torch.float32,
            ).unsqueeze(0)

            # Prediction
            with torch.no_grad():
                outputs = datacube_model.predict(params_tensor)

                if isinstance(outputs, torch.Tensor):
                    outputs = outputs.cpu().numpy()

            # Stream in chunks
            if outputs.ndim == 5:
                outputs = outputs[0]  # Remove batch dimension

            total_elements = outputs.size if hasattr(outputs, "size") else len(outputs.flatten())

            # Send metadata first
            metadata = {
                "type": "metadata",
                "shape": outputs.shape,
                "total_elements": total_elements,
                "chunk_size": chunk_size,
                "resolution": resolution,
            }
            yield json.dumps(metadata) + "\n"

            # Send data chunks
            flattened = outputs.flatten()
            for i in range(0, len(flattened), chunk_size):
                chunk = flattened[i : i + chunk_size]
                chunk_data = {
                    "type": "data",
                    "chunk_index": i // chunk_size,
                    "data": chunk.tolist(),
                }
                yield json.dumps(chunk_data) + "\n"

            # Send completion signal
            completion = {
                "type": "complete",
                "total_chunks": (len(flattened) + chunk_size - 1) // chunk_size,
            }
            yield json.dumps(completion) + "\n"

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            yield json.dumps(error_data) + "\n"

    return StreamingResponse(
        generate_datacube_chunks(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Encoding": "identity",
        },
    )


@app.post("/predict/datacube/batch")
async def predict_datacube_batch(
    planets: List[PlanetParameters],
    resolution: str = "medium",
    parallel: bool = True,
    background_tasks: BackgroundTasks = None,
    model=Depends(lambda: get_model("datacube")),
):
    """
    Batch datacube prediction for multiple planets.

    Efficiently process multiple planets with optional parallel processing.
    """
    if len(planets) > 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Maximum 50 planets per batch request"
        )

    start_time = time.time()

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from surrogate import SurrogateMode, get_surrogate_manager

        surrogate_manager = get_surrogate_manager()
        datacube_model = surrogate_manager.get_model(SurrogateMode.DATACUBE)

        if not datacube_model:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Datacube model not available",
            )

        def predict_single_planet(planet: PlanetParameters):
            """Process single planet"""
            try:
                # Convert parameters to tensor
                params_tensor = torch.tensor(
                    [
                        planet.radius_earth,
                        planet.mass_earth,
                        planet.orbital_period,
                        planet.insolation,
                        planet.stellar_teff,
                        planet.stellar_logg,
                        planet.stellar_metallicity,
                        planet.host_mass,
                    ],
                    dtype=torch.float32,
                ).unsqueeze(0)

                # Prediction
                with torch.no_grad():
                    outputs = datacube_model.predict(params_tensor)

                    if isinstance(outputs, torch.Tensor):
                        outputs = outputs.cpu().numpy()

                # Extract key metrics instead of full cube for batch processing
                if outputs.ndim == 5:
                    outputs = outputs[0]  # Remove batch dimension

                # Calculate summary statistics
                temp_field = outputs[0] if outputs.shape[0] > 0 else np.zeros((64, 64, 20))
                humidity_field = outputs[1] if outputs.shape[0] > 1 else np.zeros_like(temp_field)

                summary = {
                    "planet_params": planet.dict(),
                    "climate_summary": {
                        "global_mean_temperature": float(temp_field.mean()),
                        "temperature_range": [float(temp_field.min()), float(temp_field.max())],
                        "global_mean_humidity": float(humidity_field.mean()),
                        "humidity_range": [
                            float(humidity_field.min()),
                            float(humidity_field.max()),
                        ],
                        "temperature_std": float(temp_field.std()),
                        "humidity_std": float(humidity_field.std()),
                    },
                    "habitability_indicators": {
                        "temperature_habitable": 250.0 <= temp_field.mean() <= 320.0,
                        "water_present": humidity_field.mean() > 0.001,
                        "temperature_stable": temp_field.std() < 50.0,
                    },
                }

                return summary

            except Exception as e:
                return {"planet_params": planet.dict(), "error": str(e)}

        # Process planets
        results = []

        if parallel and len(planets) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(len(planets), 8)) as executor:
                futures = {
                    executor.submit(predict_single_planet, planet): planet for planet in planets
                }

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
        else:
            # Sequential processing
            for planet in planets:
                result = predict_single_planet(planet)
                results.append(result)

        # Sort results by habitability score
        def get_habitability_score(result):
            if "error" in result:
                return 0.0

            indicators = result.get("habitability_indicators", {})
            score = 0.0

            if indicators.get("temperature_habitable", False):
                score += 0.4
            if indicators.get("water_present", False):
                score += 0.3
            if indicators.get("temperature_stable", False):
                score += 0.3

            return score

        results.sort(key=get_habitability_score, reverse=True)

        processing_time = time.time() - start_time

        return {
            "batch_results": results,
            "batch_summary": {
                "total_planets": len(planets),
                "successful_predictions": len([r for r in results if "error" not in r]),
                "failed_predictions": len([r for r in results if "error" in r]),
                "processing_time": processing_time,
                "average_time_per_planet": processing_time / len(planets),
            },
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Batch datacube prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/models/datacube/performance")
async def get_datacube_performance():
    """
    Get performance statistics for datacube models.
    """
    try:
        from surrogate import get_surrogate_manager

        surrogate_manager = get_surrogate_manager()
        performance_stats = surrogate_manager.get_performance_stats()

        # Filter for datacube models
        datacube_stats = {
            name: stats
            for name, stats in performance_stats.items()
            if "cube" in name.lower() or "datacube" in name.lower()
        }

        return {"datacube_models": datacube_stats, "timestamp": datetime.now()}

    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance stats: {str(e)}",
        )


@app.get("/models/datacube/health")
async def get_datacube_health():
    """
    Get health status of datacube models.
    """
    try:
        from surrogate import get_surrogate_manager

        surrogate_manager = get_surrogate_manager()
        health_status = surrogate_manager.health_check()

        # Filter for datacube models
        datacube_health = {
            name: healthy
            for name, healthy in health_status.items()
            if "cube" in name.lower() or "datacube" in name.lower()
        }

        overall_health = all(datacube_health.values()) if datacube_health else False

        return {
            "overall_health": overall_health,
            "datacube_models": datacube_health,
            "timestamp": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Failed to get health status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get health status: {str(e)}",
        )


@app.post("/validate/benchmarks")
async def validate_model(request: ValidationRequest, model=Depends(lambda: get_model("scalar"))):
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
            "expected_habitability": 1.0,
        },
        "TRAPPIST-1e": {
            "params": [0.91, 0.77, 6.1, 0.66, 2559, 5.4, 0.04, 0.089],
            "expected_temp": 251.0,
            "expected_habitability": 0.8,
        },
        "Proxima Centauri b": {
            "params": [1.07, 1.17, 11.2, 1.5, 3042, 5.2, -0.29, 0.123],
            "expected_temp": 234.0,
            "expected_habitability": 0.6,
        },
    }

    results = {}

    for planet_name in request.benchmark_planets:
        if planet_name not in benchmarks:
            continue

        benchmark = benchmarks[planet_name]
        params_tensor = torch.tensor(
            benchmark["params"], dtype=torch.float32, device=device
        ).unsqueeze(0)

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
            "habitability_error": hab_error,
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
        "timestamp": datetime.utcnow(),
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
                "memory_usage": sum(p.numel() * p.element_size() for p in model.parameters())
                / 1024**2,  # MB
            }

            if hasattr(model, "mode"):
                info["capabilities"] = {
                    "scalar": ["habitability", "surface_temp", "atmospheric_pressure"],
                    "datacube": ["temperature_field", "humidity_field"],
                    "joint": ["planet_type", "habitability", "spectral_features"],
                    "spectral": ["spectrum"],
                }.get(model.mode, [])

            model_info[mode] = info

    return {
        "models": model_info,
        "device": str(device),
        "gpu_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }


# Add SHAP explanation endpoints


@app.post("/explain")
async def explain_prediction(
    request: PredictionRequest,
    domain: str = Query(
        ..., description="Scientific domain (astronomical, exoplanet, environmental, etc.)"
    ),
    include_plots: bool = Query(False, description="Include explanation plots"),
    feature_names: List[str] = Query(None, description="Feature names for explanation"),
):
    """
    Generate SHAP explanations for model predictions

    Provides scientific interpretability including:
    - Feature importance analysis
    - Pathway-level explanations
    - Physics-informed insights
    """

    try:
        # Validate domain
        valid_domains = [
            "astronomical",
            "exoplanet",
            "environmental",
            "physics",
            "optical",
            "physiological",
            "biosignature",
            "metabolomics",
        ]
        if domain not in valid_domains:
            raise HTTPException(
                status_code=400, detail=f"Invalid domain. Must be one of: {valid_domains}"
            )

        # Get surrogate manager
        surrogate_manager = get_surrogate_manager()

        # Initialize SHAP explainer if not already done
        if surrogate_manager.shap_manager is None:
            metadata_manager = MetadataManager()
            surrogate_manager.initialize_shap_explainer(metadata_manager)

        # Prepare input data
        input_data = np.array(request.input_data)

        # Generate explanations
        explanations = surrogate_manager.explain_prediction(
            input_data, domain, feature_names=feature_names
        )

        # Format response
        response = {
            "domain": domain,
            "feature_importance": explanations["feature_importance"],
            "pathway_importance": explanations["pathway_importance"],
            "explanation_time": explanations["explanation_time"],
            "timestamp": explanations["timestamp"],
        }

        # Add plots if requested
        if include_plots:
            try:
                from data_build.metadata_db import DataDomain
                from surrogate.shap_explainer import SHAPExplainer

                # Create temporary explainer for plotting
                domain_enum = DataDomain(domain)
                model = surrogate_manager.get_model()
                explainer = SHAPExplainer(model, domain_enum)
                explainer.fit(input_data, feature_names)

                # Generate plots (base64 encoded)
                import base64
                from io import BytesIO

                # Feature importance plot
                fig = explainer.plot_feature_importance(explanations)
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png")
                img_buffer.seek(0)
                feature_plot = base64.b64encode(img_buffer.read()).decode()

                # Pathway importance plot
                fig = explainer.plot_pathway_importance(explanations)
                img_buffer = BytesIO()
                fig.savefig(img_buffer, format="png")
                img_buffer.seek(0)
                pathway_plot = base64.b64encode(img_buffer.read()).decode()

                response["plots"] = {
                    "feature_importance": feature_plot,
                    "pathway_importance": pathway_plot,
                }

            except Exception as e:
                logger.warning(f"Failed to generate plots: {e}")
                response["plot_error"] = str(e)

        return response

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/predict_with_explanation")
async def predict_with_explanation(
    request: PredictionRequest,
    domain: str = Query(..., description="Scientific domain"),
    resolution: str = Query("128x64", description="Output resolution"),
    include_explanation: bool = Query(True, description="Include SHAP explanation"),
    feature_names: List[str] = Query(None, description="Feature names"),
):
    """
    Make prediction with integrated SHAP explanation

    Combines model prediction with scientific interpretability
    """

    try:
        # Get surrogate manager
        surrogate_manager = get_surrogate_manager()

        # Initialize SHAP explainer if needed
        if include_explanation and surrogate_manager.shap_manager is None:
            metadata_manager = MetadataManager()
            surrogate_manager.initialize_shap_explainer(metadata_manager)

        # Prepare input data
        input_data = np.array(request.input_data)

        # Make prediction with explanation
        result = surrogate_manager.predict_with_explanation(
            input_data, domain, feature_names=feature_names, include_explanation=include_explanation
        )

        # Format prediction output
        prediction = result["prediction"]
        if isinstance(prediction, torch.Tensor):
            prediction = prediction.cpu().numpy()

        response = {
            "prediction": prediction.tolist(),
            "domain": domain,
            "resolution": resolution,
            "timestamp": result["timestamp"],
        }

        # Add explanation if included
        if "explanation" in result:
            explanation = result["explanation"]
            response["explanation"] = {
                "top_features": dict(list(explanation["feature_importance"].items())[:10]),
                "pathway_importance": explanation["pathway_importance"],
                "explanation_time": explanation["explanation_time"],
            }

        if "explanation_error" in result:
            response["explanation_error"] = result["explanation_error"]

        return response

    except Exception as e:
        logger.error(f"Prediction with explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/explanation_stats")
async def get_explanation_stats():
    """Get SHAP explanation statistics"""

    try:
        surrogate_manager = get_surrogate_manager()
        stats = surrogate_manager.get_explanation_stats()

        # Add system info
        stats["shap_available"] = surrogate_manager.shap_manager is not None
        stats["supported_domains"] = [
            "astronomical",
            "exoplanet",
            "environmental",
            "physics",
            "optical",
            "physiological",
            "biosignature",
            "metabolomics",
        ]

        return stats

    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")


@app.post("/batch_explain")
async def batch_explain(
    requests: List[PredictionRequest],
    domains: List[str] = Query(..., description="Domains for each request"),
    feature_names_list: List[List[str]] = Query(None, description="Feature names for each request"),
):
    """
    Generate explanations for multiple predictions in batch

    Efficient batch processing for multiple domains
    """

    if len(requests) != len(domains):
        raise HTTPException(
            status_code=400, detail="Number of requests must match number of domains"
        )

    try:
        # Get surrogate manager
        surrogate_manager = get_surrogate_manager()

        # Initialize SHAP explainer if needed
        if surrogate_manager.shap_manager is None:
            metadata_manager = MetadataManager()
            surrogate_manager.initialize_shap_explainer(metadata_manager)

        # Prepare batch data
        batch_data = {}
        batch_features = {}

        for i, (request, domain) in enumerate(zip(requests, domains)):
            input_data = np.array(request.input_data)

            if domain not in batch_data:
                batch_data[domain] = []
                batch_features[domain] = []

            batch_data[domain].append(input_data)

            # Add feature names if provided
            if feature_names_list and i < len(feature_names_list):
                batch_features[domain] = feature_names_list[i]

        # Convert to numpy arrays
        for domain in batch_data:
            batch_data[domain] = np.vstack(batch_data[domain])

        # Generate batch explanations
        from data_build.metadata_db import DataDomain

        domain_enums = [DataDomain(domain) for domain in batch_data.keys()]

        explanations = surrogate_manager.shap_manager.batch_explain(
            domain_enums, batch_data, batch_features
        )

        # Format response
        response = {
            "batch_size": len(requests),
            "domains_processed": len(batch_data),
            "explanations": explanations,
            "timestamp": datetime.now().isoformat(),
        }

        return response

    except Exception as e:
        logger.error(f"Batch explanation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch explanation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app", host="0.0.0.0", port=8000, reload=False, workers=1, log_level="info"
    )
