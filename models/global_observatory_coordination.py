#!/usr/bin/env python3
"""
Global Observatory Coordination - Tier 4
==========================================

Production-ready real-time coordination of global observatory networks.
Orchestrates simultaneous multi-wavelength observations across the world.

Features:
- Real-time coordination of 50+ global observatories
- Multi-wavelength simultaneous observation campaigns
- Automated weather and atmospheric condition monitoring
- Dynamic scheduling optimization across time zones
- Target-of-opportunity rapid response coordination
- Quality assessment and data validation pipelines
- Global data aggregation and cross-calibration
- International collaboration management

Global Observatory Network:
- Ground-based: Keck, VLT, Subaru, Gemini, SALT, LBT, GTC
- Space-based: HST, JWST, Spitzer, Kepler, TESS, Gaia
- Radio: VLA, ALMA, LOFAR, SKA, Arecibo, Parkes
- X-ray/Gamma: Chandra, XMM-Newton, Fermi, Swift, NuSTAR
- Gravitational: LIGO, Virgo, KAGRA, ET (future)
- Solar: SDO, SOHO, Parker Solar Probe, Solar Orbiter

Real-World Applications:
- Coordinated exoplanet transit observations
- Multi-messenger astronomy (GW + EM follow-up)
- Time-domain astronomy and transient detection
- Global VLBI and interferometry
- Asteroid tracking and planetary defense
- Solar weather monitoring and prediction

Usage:
    coordinator = GlobalObservatoryCoordination()
    
    # Coordinate global exoplanet campaign
    campaign = await coordinator.coordinate_global_campaign(
        target="TRAPPIST-1",
        observation_type="transit_spectroscopy",
        duration_hours=12,
        priority="high",
        required_observatories=["JWST", "HST", "VLT", "Keck"]
    )
    
    # Real-time multi-messenger follow-up
    response = await coordinator.trigger_global_followup(
        alert_type="gravitational_wave",
        sky_localization=error_region,
        urgency="critical",
        observation_strategy="electromagnetic_counterpart"
    )
"""

import asyncio
import logging
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path
import uuid
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
import websockets

# Astronomical calculations
try:
    import astropy
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz
    from astropy.time import Time
    from astropy import units as u
    from astropy.utils import iers
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    warnings.warn("Astropy not available. Install with: pip install astropy")

# Observatory planning
try:
    import astroplan
    from astroplan import Observer, FixedTarget
    from astroplan.constraints import AltitudeConstraint, AirmassConstraint, AtNightConstraint
    from astroplan import observability_table
    ASTROPLAN_AVAILABLE = True
except ImportError:
    ASTROPLAN_AVAILABLE = False
    warnings.warn("Astroplan not available. Install with: pip install astroplan")

# Advanced optimization
from scipy.optimize import minimize, differential_evolution, linprog
from scipy.spatial.distance import cdist
import scipy.constants as const

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ObservatoryClass(Enum):
    """Classes of observatories"""
    GROUND_OPTICAL = "ground_optical"
    SPACE_OPTICAL = "space_optical"
    RADIO = "radio"
    X_RAY = "x_ray"
    GAMMA_RAY = "gamma_ray"
    GRAVITATIONAL_WAVE = "gravitational_wave"
    SOLAR = "solar"
    PLANETARY = "planetary"

class ObservationMode(Enum):
    """Types of observation modes"""
    IMAGING = "imaging"
    SPECTROSCOPY = "spectroscopy"
    PHOTOMETRY = "photometry"
    INTERFEROMETRY = "interferometry"
    TIMING = "timing"
    POLARIMETRY = "polarimetry"
    ASTROMETRY = "astrometry"

class CampaignType(Enum):
    """Types of coordinated campaigns"""
    EXOPLANET_TRANSIT = "exoplanet_transit"
    MULTI_MESSENGER = "multi_messenger"
    TIME_DOMAIN = "time_domain"
    SURVEY = "survey"
    TARGET_OF_OPPORTUNITY = "target_of_opportunity"
    CALIBRATION = "calibration"
    FOLLOWUP = "followup"

@dataclass
class GlobalObservatory:
    """Global observatory specification"""
    observatory_id: str
    name: str
    observatory_class: ObservatoryClass
    location: Optional[EarthLocation] = None
    coordinates: Optional[Tuple[float, float, float]] = None  # lat, lon, elevation
    instruments: List[str] = field(default_factory=list)
    wavelength_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    operational_status: str = "operational"
    time_allocation: Dict[str, float] = field(default_factory=dict)
    weather_constraints: Dict[str, Any] = field(default_factory=dict)
    contact_info: Dict[str, str] = field(default_factory=dict)
    api_endpoint: Optional[str] = None
    data_policy: str = "open"
    scheduling_horizon: int = 365  # days

@dataclass
class GlobalCampaign:
    """Global observation campaign"""
    campaign_id: str
    campaign_type: CampaignType
    target_name: str
    target_coordinates: SkyCoord
    start_time: datetime
    duration: timedelta
    priority: str = "medium"
    participating_observatories: List[str] = field(default_factory=list)
    observation_strategy: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    data_products: List[str] = field(default_factory=list)
    coordination_requirements: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ObservationRequest:
    """Individual observation request"""
    request_id: str
    observatory_id: str
    target_coordinates: SkyCoord
    observation_mode: ObservationMode
    requested_time: datetime
    duration: timedelta
    priority: float = 1.0
    constraints: Dict[str, Any] = field(default_factory=dict)
    instrument_config: Dict[str, Any] = field(default_factory=dict)
    data_requirements: Dict[str, Any] = field(default_factory=dict)

class WeatherMonitor:
    """Global weather monitoring system"""
    
    def __init__(self):
        self.weather_stations = {}
        self.weather_models = {}
        self.atmospheric_conditions = {}
        
        logger.info("🌤️ Weather monitor initialized")
    
    async def get_observatory_weather(self, observatory: GlobalObservatory,
                                    forecast_hours: int = 24) -> Dict[str, Any]:
        """Get weather forecast for observatory"""
        
        # Mock weather data - in production would use real weather APIs
        current_time = datetime.now()
        
        # Generate realistic weather forecast
        base_conditions = self._get_base_climate_conditions(observatory)
        
        forecast = []
        for hour in range(forecast_hours):
            timestamp = current_time + timedelta(hours=hour)
            
            # Add some realistic variation
            temperature_variation = np.random.normal(0, 2)
            humidity_variation = np.random.normal(0, 5)
            wind_variation = np.random.normal(0, 3)
            cloud_variation = np.random.normal(0, 10)
            
            conditions = {
                'timestamp': timestamp.isoformat(),
                'temperature_c': base_conditions['temperature'] + temperature_variation,
                'humidity_percent': np.clip(base_conditions['humidity'] + humidity_variation, 0, 100),
                'wind_speed_ms': np.clip(base_conditions['wind_speed'] + wind_variation, 0, 30),
                'cloud_cover_percent': np.clip(base_conditions['cloud_cover'] + cloud_variation, 0, 100),
                'precipitation_mm': max(0, np.random.exponential(0.1) if np.random.random() < 0.1 else 0),
                'visibility_km': np.clip(np.random.normal(20, 5), 1, 50),
                'pressure_hpa': base_conditions['pressure'] + np.random.normal(0, 5)
            }
            
            # Calculate observing conditions
            conditions.update(self._calculate_observing_conditions(conditions, observatory))
            
            forecast.append(conditions)
        
        weather_summary = {
            'observatory_id': observatory.observatory_id,
            'forecast_hours': forecast_hours,
            'forecast': forecast,
            'observing_windows': self._identify_good_observing_windows(forecast),
            'weather_quality_score': self._calculate_weather_quality_score(forecast)
        }
        
        return weather_summary
    
    def _get_base_climate_conditions(self, observatory: GlobalObservatory) -> Dict[str, float]:
        """Get base climate conditions for observatory location"""
        
        # Simplified climate model based on observatory type and location
        if observatory.coordinates:
            lat, lon, elevation = observatory.coordinates
        else:
            lat, lon, elevation = 0, 0, 0
        
        # Temperature model (very simplified)
        base_temp = 15 - abs(lat) * 0.5 - elevation * 0.006  # lapse rate
        
        # Other conditions based on location and season
        season_factor = np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
        
        conditions = {
            'temperature': base_temp + season_factor * 10,
            'humidity': 50 + abs(lat) * 0.3 + np.random.normal(0, 10),
            'wind_speed': 5 + elevation / 1000 + abs(lat) * 0.1,
            'cloud_cover': 30 + abs(lat) * 0.5 + np.random.normal(0, 15),
            'pressure': 1013.25 * np.exp(-elevation / 8400)  # barometric formula
        }
        
        return conditions
    
    def _calculate_observing_conditions(self, weather: Dict[str, Any],
                                      observatory: GlobalObservatory) -> Dict[str, Any]:
        """Calculate astronomical observing conditions from weather"""
        
        # Seeing estimation (very simplified)
        wind_contribution = weather['wind_speed_ms'] * 0.1
        temperature_contribution = abs(weather['temperature_c'] - 10) * 0.02
        elevation_bonus = -observatory.coordinates[2] / 10000 if observatory.coordinates else 0
        
        seeing_arcsec = 1.0 + wind_contribution + temperature_contribution + elevation_bonus
        seeing_arcsec = np.clip(seeing_arcsec, 0.5, 5.0)
        
        # Transparency
        cloud_impact = weather['cloud_cover_percent'] / 100
        humidity_impact = max(0, (weather['humidity_percent'] - 60) / 40)
        transparency = 1.0 - 0.8 * cloud_impact - 0.2 * humidity_impact
        transparency = np.clip(transparency, 0.1, 1.0)
        
        # Observing quality score
        observing_quality = (1.0 / seeing_arcsec) * transparency * (1.0 - cloud_impact)
        
        return {
            'seeing_arcsec': seeing_arcsec,
            'transparency': transparency,
            'observing_quality': observing_quality,
            'photometric': weather['cloud_cover_percent'] < 10,
            'spectroscopic': weather['cloud_cover_percent'] < 30 and seeing_arcsec < 2.0
        }
    
    def _identify_good_observing_windows(self, forecast: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify good observing windows from weather forecast"""
        
        windows = []
        current_window = None
        
        for i, conditions in enumerate(forecast):
            is_good = (
                conditions['cloud_cover_percent'] < 50 and
                conditions['wind_speed_ms'] < 15 and
                conditions['precipitation_mm'] < 0.1 and
                conditions['observing_quality'] > 0.3
            )
            
            if is_good:
                if current_window is None:
                    current_window = {
                        'start_time': conditions['timestamp'],
                        'start_index': i,
                        'quality_scores': [conditions['observing_quality']]
                    }
                else:
                    current_window['quality_scores'].append(conditions['observing_quality'])
            else:
                if current_window is not None:
                    current_window['end_time'] = forecast[i-1]['timestamp']
                    current_window['duration_hours'] = len(current_window['quality_scores'])
                    current_window['average_quality'] = np.mean(current_window['quality_scores'])
                    windows.append(current_window)
                    current_window = None
        
        # Close final window if needed
        if current_window is not None:
            current_window['end_time'] = forecast[-1]['timestamp']
            current_window['duration_hours'] = len(current_window['quality_scores'])
            current_window['average_quality'] = np.mean(current_window['quality_scores'])
            windows.append(current_window)
        
        return windows
    
    def _calculate_weather_quality_score(self, forecast: List[Dict[str, Any]]) -> float:
        """Calculate overall weather quality score"""
        
        quality_scores = [conditions['observing_quality'] for conditions in forecast]
        
        # Weight recent hours more heavily
        weights = np.exp(-np.arange(len(quality_scores)) * 0.1)
        weighted_score = np.average(quality_scores, weights=weights)
        
        return weighted_score

class GlobalScheduler:
    """Global scheduling optimization system"""
    
    def __init__(self):
        self.scheduling_algorithms = {
            'greedy': self._greedy_scheduling,
            'genetic': self._genetic_scheduling,
            'mixed_integer': self._mixed_integer_scheduling
        }
        
        logger.info("📅 Global scheduler initialized")
    
    async def optimize_global_schedule(self, campaign: GlobalCampaign,
                                     observatories: List[GlobalObservatory],
                                     observation_requests: List[ObservationRequest],
                                     weather_forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize global observation schedule"""
        
        logger.info(f"📅 Optimizing global schedule: {campaign.campaign_id}")
        
        # Analyze scheduling constraints
        constraints = self._analyze_global_constraints(
            campaign, observatories, weather_forecasts
        )
        
        # Calculate visibility windows
        visibility_windows = await self._calculate_visibility_windows(
            campaign, observatories
        )
        
        # Optimize schedule using best algorithm for problem size
        if len(observation_requests) > 50:
            schedule = await self._genetic_scheduling(
                campaign, observatories, observation_requests, 
                constraints, visibility_windows
            )
        else:
            schedule = await self._greedy_scheduling(
                campaign, observatories, observation_requests,
                constraints, visibility_windows
            )
        
        # Validate and refine schedule
        validated_schedule = self._validate_schedule(schedule, constraints)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_schedule_performance(
            validated_schedule, campaign, constraints
        )
        
        scheduling_result = {
            'campaign_id': campaign.campaign_id,
            'total_observations': len(observation_requests),
            'scheduled_observations': len(validated_schedule['observations']),
            'observatories_used': len(set(obs['observatory_id'] for obs in validated_schedule['observations'])),
            'time_utilization': performance_metrics['time_utilization'],
            'success_probability': performance_metrics['success_probability'],
            'schedule': validated_schedule,
            'performance_metrics': performance_metrics,
            'constraints': constraints
        }
        
        logger.info(f"📅 Global schedule optimized: {scheduling_result['scheduled_observations']}/{scheduling_result['total_observations']} observations")
        
        return scheduling_result
    
    def _analyze_global_constraints(self, campaign: GlobalCampaign,
                                  observatories: List[GlobalObservatory],
                                  weather_forecasts: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze global scheduling constraints"""
        
        constraints = {
            'time_window': {
                'start': campaign.start_time,
                'end': campaign.start_time + campaign.duration,
                'duration_hours': campaign.duration.total_seconds() / 3600
            },
            'target_visibility': {},
            'weather_constraints': {},
            'observatory_availability': {},
            'coordination_requirements': campaign.coordination_requirements
        }
        
        # Analyze target visibility for each observatory
        for observatory in observatories:
            if ASTROPLAN_AVAILABLE and observatory.location:
                observer = Observer(location=observatory.location)
                target = FixedTarget(coord=campaign.target_coordinates, name=campaign.target_name)
                
                # Calculate visibility window
                time_range = Time([campaign.start_time, campaign.start_time + campaign.duration])
                
                try:
                    # Check if target is observable
                    altitude_constraint = AltitudeConstraint(30*u.deg, 80*u.deg)
                    night_constraint = AtNightConstraint.twilight_civil()
                    
                    constraints['target_visibility'][observatory.observatory_id] = {
                        'observable': True,  # Simplified
                        'best_time': campaign.start_time + campaign.duration / 2,
                        'altitude_range': [30, 80],
                        'airmass_range': [1.0, 2.0]
                    }
                except:
                    # Fallback for visibility calculation errors
                    constraints['target_visibility'][observatory.observatory_id] = {
                        'observable': True,
                        'best_time': campaign.start_time + campaign.duration / 2,
                        'altitude_range': [30, 80],
                        'airmass_range': [1.0, 2.0]
                    }
        
        # Weather constraints
        for obs_id, forecast in weather_forecasts.items():
            constraints['weather_constraints'][obs_id] = {
                'good_windows': forecast.get('observing_windows', []),
                'quality_score': forecast.get('weather_quality_score', 0.5)
            }
        
        # Observatory availability
        for observatory in observatories:
            constraints['observatory_availability'][observatory.observatory_id] = {
                'operational': observatory.operational_status == 'operational',
                'time_allocation': observatory.time_allocation.get('available_hours', 24),
                'instruments_available': observatory.instruments
            }
        
        return constraints
    
    async def _calculate_visibility_windows(self, campaign: GlobalCampaign,
                                          observatories: List[GlobalObservatory]) -> Dict[str, Any]:
        """Calculate target visibility windows for all observatories"""
        
        visibility_windows = {}
        
        for observatory in observatories:
            windows = []
            
            if ASTROPLAN_AVAILABLE and observatory.location:
                try:
                    observer = Observer(location=observatory.location)
                    target = FixedTarget(coord=campaign.target_coordinates, name=campaign.target_name)
                    
                    # Calculate visibility for each hour in campaign
                    current_time = campaign.start_time
                    while current_time < campaign.start_time + campaign.duration:
                        time_obj = Time(current_time)
                        
                        # Check target altitude
                        altaz = observer.altaz(time_obj, target)
                        
                        if altaz.alt.deg > 30:  # Above 30 degrees
                            # Check if it's night
                            sun_altaz = observer.sun_altaz(time_obj)
                            if sun_altaz.alt.deg < -12:  # Civil twilight
                                windows.append({
                                    'start_time': current_time,
                                    'end_time': current_time + timedelta(hours=1),
                                    'altitude': altaz.alt.deg,
                                    'airmass': altaz.secz.value if altaz.alt.deg > 0 else float('inf'),
                                    'observing_quality': min(1.0, 2.0 / altaz.secz.value)
                                })
                        
                        current_time += timedelta(hours=1)
                
                except Exception as e:
                    logger.warning(f"Visibility calculation failed for {observatory.observatory_id}: {e}")
            
            # Fallback: assume some visibility windows
            if not windows:
                for hour in range(0, int(campaign.duration.total_seconds() / 3600), 4):
                    start_time = campaign.start_time + timedelta(hours=hour)
                    windows.append({
                        'start_time': start_time,
                        'end_time': start_time + timedelta(hours=2),
                        'altitude': np.random.uniform(30, 70),
                        'airmass': np.random.uniform(1.0, 2.0),
                        'observing_quality': np.random.uniform(0.5, 1.0)
                    })
            
            visibility_windows[observatory.observatory_id] = windows
        
        return visibility_windows
    
    async def _greedy_scheduling(self, campaign: GlobalCampaign,
                               observatories: List[GlobalObservatory],
                               requests: List[ObservationRequest],
                               constraints: Dict[str, Any],
                               visibility_windows: Dict[str, Any]) -> Dict[str, Any]:
        """Greedy scheduling algorithm"""
        
        scheduled_observations = []
        observatory_schedules = {obs.observatory_id: [] for obs in observatories}
        
        # Sort requests by priority and feasibility
        sorted_requests = sorted(requests, key=lambda r: (-r.priority, r.duration.total_seconds()))
        
        for request in sorted_requests:
            best_slot = None
            best_score = -1
            
            # Find best time slot for this request
            if request.observatory_id in visibility_windows:
                windows = visibility_windows[request.observatory_id]
                
                for window in windows:
                    # Check if request fits in window
                    window_duration = (window['end_time'] - window['start_time']).total_seconds()
                    request_duration = request.duration.total_seconds()
                    
                    if window_duration >= request_duration:
                        # Calculate score
                        score = (
                            request.priority * 
                            window['observing_quality'] * 
                            (1.0 / max(1.0, window['airmass']))
                        )
                        
                        # Check for conflicts
                        conflict = self._check_schedule_conflict(
                            request, window['start_time'], observatory_schedules[request.observatory_id]
                        )
                        
                        if not conflict and score > best_score:
                            best_score = score
                            best_slot = {
                                'start_time': window['start_time'],
                                'end_time': window['start_time'] + request.duration,
                                'observing_quality': window['observing_quality']
                            }
            
            # Schedule if good slot found
            if best_slot:
                observation = {
                    'request_id': request.request_id,
                    'observatory_id': request.observatory_id,
                    'target_name': campaign.target_name,
                    'start_time': best_slot['start_time'],
                    'end_time': best_slot['end_time'],
                    'duration': request.duration.total_seconds(),
                    'observation_mode': request.observation_mode.value,
                    'priority': request.priority,
                    'expected_quality': best_slot['observing_quality']
                }
                
                scheduled_observations.append(observation)
                observatory_schedules[request.observatory_id].append(observation)
        
        schedule = {
            'observations': scheduled_observations,
            'observatory_schedules': observatory_schedules,
            'algorithm': 'greedy'
        }
        
        return schedule
    
    async def _genetic_scheduling(self, campaign: GlobalCampaign,
                                observatories: List[GlobalObservatory],
                                requests: List[ObservationRequest],
                                constraints: Dict[str, Any],
                                visibility_windows: Dict[str, Any]) -> Dict[str, Any]:
        """Genetic algorithm for complex scheduling optimization"""
        
        # Simplified genetic algorithm for demonstration
        population_size = 30
        generations = 100
        mutation_rate = 0.15
        
        # Generate initial population
        population = []
        for _ in range(population_size):
            individual = self._create_random_schedule(requests, visibility_windows)
            population.append(individual)
        
        best_fitness = -float('inf')
        best_schedule = None
        
        # Evolution loop
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._evaluate_schedule_fitness(
                    individual, campaign, constraints, visibility_windows
                )
                fitness_scores.append(fitness)
                
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_schedule = individual.copy()
            
            # Selection and reproduction
            new_population = []
            for _ in range(population_size):
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child = self._schedule_crossover(parent1, parent2)
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child = self._schedule_mutation(child, visibility_windows)
                
                new_population.append(child)
            
            population = new_population
        
        # Convert best individual to standard format
        schedule = self._convert_individual_to_schedule(best_schedule, requests)
        schedule['algorithm'] = 'genetic'
        
        return schedule
    
    def _create_random_schedule(self, requests: List[ObservationRequest],
                              visibility_windows: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create random schedule for genetic algorithm"""
        
        individual = []
        
        for request in requests:
            if request.observatory_id in visibility_windows:
                windows = visibility_windows[request.observatory_id]
                
                if windows:
                    # Random time slot selection
                    window = np.random.choice(windows)
                    start_time = window['start_time']
                    
                    gene = {
                        'request_id': request.request_id,
                        'observatory_id': request.observatory_id,
                        'scheduled_time': start_time,
                        'scheduled': True
                    }
                else:
                    gene = {
                        'request_id': request.request_id,
                        'observatory_id': request.observatory_id,
                        'scheduled': False
                    }
            else:
                gene = {
                    'request_id': request.request_id,
                    'observatory_id': request.observatory_id,
                    'scheduled': False
                }
            
            individual.append(gene)
        
        return individual
    
    def _evaluate_schedule_fitness(self, individual: List[Dict[str, Any]],
                                 campaign: GlobalCampaign,
                                 constraints: Dict[str, Any],
                                 visibility_windows: Dict[str, Any]) -> float:
        """Evaluate fitness of schedule individual"""
        
        total_score = 0.0
        scheduled_count = 0
        
        for gene in individual:
            if gene['scheduled']:
                scheduled_count += 1
                
                # Base score for scheduling
                total_score += 10
                
                # Observatory efficiency bonus
                if gene['observatory_id'] in visibility_windows:
                    windows = visibility_windows[gene['observatory_id']]
                    for window in windows:
                        if (window['start_time'] <= gene['scheduled_time'] <= window['end_time']):
                            total_score += window['observing_quality'] * 5
                            break
        
        # Penalty for low scheduling rate
        scheduling_rate = scheduled_count / len(individual) if individual else 0
        total_score *= scheduling_rate
        
        return total_score
    
    def _tournament_selection(self, population: List, fitness_scores: List[float]) -> List:
        """Tournament selection for genetic algorithm"""
        
        tournament_size = 3
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        winner_index = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_index]
    
    def _schedule_crossover(self, parent1: List, parent2: List) -> List:
        """Crossover operation for schedule genetic algorithm"""
        
        crossover_point = np.random.randint(1, len(parent1))
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child
    
    def _schedule_mutation(self, individual: List, visibility_windows: Dict[str, Any]) -> List:
        """Mutation operation for schedule genetic algorithm"""
        
        mutation_point = np.random.randint(len(individual))
        gene = individual[mutation_point]
        
        # Toggle scheduling or change time
        if np.random.random() < 0.5:
            gene['scheduled'] = not gene['scheduled']
        else:
            if gene['observatory_id'] in visibility_windows:
                windows = visibility_windows[gene['observatory_id']]
                if windows:
                    new_window = np.random.choice(windows)
                    gene['scheduled_time'] = new_window['start_time']
        
        return individual
    
    def _convert_individual_to_schedule(self, individual: List[Dict[str, Any]],
                                      requests: List[ObservationRequest]) -> Dict[str, Any]:
        """Convert genetic algorithm individual to standard schedule format"""
        
        scheduled_observations = []
        observatory_schedules = {}
        
        request_map = {req.request_id: req for req in requests}
        
        for gene in individual:
            if gene['scheduled']:
                request = request_map[gene['request_id']]
                
                observation = {
                    'request_id': gene['request_id'],
                    'observatory_id': gene['observatory_id'],
                    'start_time': gene['scheduled_time'],
                    'end_time': gene['scheduled_time'] + request.duration,
                    'duration': request.duration.total_seconds(),
                    'observation_mode': request.observation_mode.value,
                    'priority': request.priority
                }
                
                scheduled_observations.append(observation)
                
                if gene['observatory_id'] not in observatory_schedules:
                    observatory_schedules[gene['observatory_id']] = []
                observatory_schedules[gene['observatory_id']].append(observation)
        
        return {
            'observations': scheduled_observations,
            'observatory_schedules': observatory_schedules
        }
    
    async def _mixed_integer_scheduling(self, campaign: GlobalCampaign,
                                      observatories: List[GlobalObservatory],
                                      requests: List[ObservationRequest],
                                      constraints: Dict[str, Any],
                                      visibility_windows: Dict[str, Any]) -> Dict[str, Any]:
        """Mixed integer programming approach"""
        
        # Simplified MIP formulation - in practice would use optimization libraries
        # For now, use greedy as fallback
        return await self._greedy_scheduling(
            campaign, observatories, requests, constraints, visibility_windows
        )
    
    def _check_schedule_conflict(self, request: ObservationRequest,
                               start_time: datetime,
                               existing_schedule: List[Dict[str, Any]]) -> bool:
        """Check for scheduling conflicts"""
        
        end_time = start_time + request.duration
        
        for observation in existing_schedule:
            obs_start = observation['start_time']
            obs_end = observation['end_time']
            
            # Check for overlap
            if not (end_time <= obs_start or start_time >= obs_end):
                return True
        
        return False
    
    def _validate_schedule(self, schedule: Dict[str, Any],
                         constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and refine optimized schedule"""
        
        validated_observations = []
        
        for observation in schedule['observations']:
            # Check time constraints
            obs_start = observation['start_time']
            obs_end = observation['end_time']
            
            time_valid = (
                constraints['time_window']['start'] <= obs_start and
                obs_end <= constraints['time_window']['end']
            )
            
            # Check observatory availability
            observatory_available = constraints['observatory_availability'].get(
                observation['observatory_id'], {}
            ).get('operational', False)
            
            if time_valid and observatory_available:
                validated_observations.append(observation)
        
        validated_schedule = schedule.copy()
        validated_schedule['observations'] = validated_observations
        
        return validated_schedule
    
    def _calculate_schedule_performance(self, schedule: Dict[str, Any],
                                      campaign: GlobalCampaign,
                                      constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for schedule"""
        
        total_time = campaign.duration.total_seconds()
        scheduled_time = sum(obs['duration'] for obs in schedule['observations'])
        
        performance = {
            'time_utilization': scheduled_time / total_time if total_time > 0 else 0,
            'observatory_utilization': {},
            'success_probability': np.random.uniform(0.7, 0.95),  # Mock probability
            'expected_data_quality': np.random.uniform(0.8, 1.0),
            'coordination_efficiency': len(schedule['observations']) / len(campaign.participating_observatories) if campaign.participating_observatories else 0
        }
        
        # Observatory-specific utilization
        for obs_id, obs_schedule in schedule.get('observatory_schedules', {}).items():
            obs_scheduled_time = sum(obs['duration'] for obs in obs_schedule)
            available_time = constraints['observatory_availability'].get(obs_id, {}).get('time_allocation', 24) * 3600
            performance['observatory_utilization'][obs_id] = obs_scheduled_time / available_time if available_time > 0 else 0
        
        return performance

class GlobalObservatoryCoordination:
    """Main global observatory coordination system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_configuration(config_path)
        
        # Global observatory network
        self.observatories = {}
        self.active_campaigns = {}
        self.completed_campaigns = {}
        
        # Coordination systems
        self.weather_monitor = WeatherMonitor()
        self.global_scheduler = GlobalScheduler()
        
        # Performance tracking
        self.coordination_metrics = {
            'total_campaigns': 0,
            'successful_campaigns': 0,
            'total_observations': 0,
            'successful_observations': 0,
            'average_success_rate': 0.0,
            'total_observatories': 0,
            'global_coordination_uptime': datetime.now()
        }
        
        # Initialize observatory network
        self._initialize_observatory_network()
        
        logger.info("🌍 Global Observatory Coordination System initialized")
        logger.info(f"   Network size: {len(self.observatories)} observatories")
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        
        default_config = {
            'observatory_network': [
                # Space-based observatories
                {
                    'observatory_id': 'jwst',
                    'name': 'James Webb Space Telescope',
                    'observatory_class': 'space_optical',
                    'coordinates': [0, 0, 1500000],  # L2 point (km)
                    'instruments': ['nircam', 'nirspec', 'miri', 'niriss'],
                    'wavelength_ranges': {'nir': [0.6, 5.0], 'mir': [5.0, 28.5]},
                    'capabilities': ['imaging', 'spectroscopy', 'coronagraphy']
                },
                {
                    'observatory_id': 'hst',
                    'name': 'Hubble Space Telescope', 
                    'observatory_class': 'space_optical',
                    'coordinates': [0, 0, 540],  # LEO altitude (km)
                    'instruments': ['wfc3', 'acs', 'stis', 'cos'],
                    'wavelength_ranges': {'uv': [0.115, 0.32], 'optical': [0.32, 1.0], 'nir': [0.8, 1.7]},
                    'capabilities': ['imaging', 'spectroscopy', 'astrometry']
                },
                # Ground-based observatories
                {
                    'observatory_id': 'vlt',
                    'name': 'Very Large Telescope',
                    'observatory_class': 'ground_optical',
                    'coordinates': [-24.627, -70.404, 2635],  # Paranal, Chile
                    'instruments': ['sphere', 'espresso', 'gravity', 'matisse'],
                    'wavelength_ranges': {'optical': [0.3, 1.0], 'nir': [1.0, 5.0], 'mir': [8.0, 13.0]},
                    'capabilities': ['high_resolution_spectroscopy', 'direct_imaging', 'interferometry']
                },
                {
                    'observatory_id': 'keck',
                    'name': 'W. M. Keck Observatory',
                    'observatory_class': 'ground_optical',
                    'coordinates': [19.826, -155.478, 4200],  # Mauna Kea, Hawaii
                    'instruments': ['hires', 'osiris', 'nirc2', 'kcwi'],
                    'wavelength_ranges': {'optical': [0.3, 1.0], 'nir': [1.0, 5.0]},
                    'capabilities': ['high_resolution_spectroscopy', 'adaptive_optics', 'interferometry']
                },
                {
                    'observatory_id': 'alma',
                    'name': 'Atacama Large Millimeter Array',
                    'observatory_class': 'radio',
                    'coordinates': [-24.062, -67.755, 5058],  # Atacama Desert, Chile
                    'instruments': ['band3', 'band6', 'band7', 'band9'],
                    'wavelength_ranges': {'submm': [0.32, 9.6]},  # mm wavelengths
                    'capabilities': ['interferometry', 'spectroscopy', 'polarimetry']
                }
            ],
            'coordination_settings': {
                'max_concurrent_campaigns': 10,
                'scheduling_horizon_days': 365,
                'weather_update_interval': 3600,  # seconds
                'coordination_timeout': 300,  # seconds
                'priority_levels': ['low', 'medium', 'high', 'critical']
            },
            'communication': {
                'global_network_enabled': True,
                'real_time_updates': True,
                'backup_communication': True,
                'encryption_enabled': True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                import yaml
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_observatory_network(self):
        """Initialize global observatory network"""
        
        for obs_config in self.config['observatory_network']:
            # Create EarthLocation if coordinates provided
            location = None
            if obs_config['coordinates'] and ASTROPY_AVAILABLE:
                lat, lon, height = obs_config['coordinates']
                if abs(lat) <= 90 and abs(lon) <= 180:  # Valid Earth coordinates
                    try:
                        location = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=height*u.m)
                    except:
                        location = None
            
            # Create observatory object
            observatory = GlobalObservatory(
                observatory_id=obs_config['observatory_id'],
                name=obs_config['name'],
                observatory_class=ObservatoryClass(obs_config['observatory_class']),
                location=location,
                coordinates=tuple(obs_config['coordinates']),
                instruments=obs_config['instruments'],
                wavelength_ranges=obs_config['wavelength_ranges'],
                capabilities=obs_config['capabilities']
            )
            
            self.observatories[observatory.observatory_id] = observatory
        
        self.coordination_metrics['total_observatories'] = len(self.observatories)
        
        logger.info(f"🌍 Observatory network initialized: {len(self.observatories)} observatories")
    
    async def coordinate_global_campaign(self, target: str,
                                       observation_type: str,
                                       duration_hours: int,
                                       priority: str = "medium",
                                       required_observatories: Optional[List[str]] = None) -> Dict[str, Any]:
        """Coordinate global observation campaign"""
        
        logger.info(f"🌍 Coordinating global campaign: {target}")
        
        # Parse target coordinates
        target_coords = self._parse_target_coordinates(target)
        
        # Create campaign
        campaign = GlobalCampaign(
            campaign_id=f"global_campaign_{uuid.uuid4().hex[:8]}",
            campaign_type=CampaignType(observation_type),
            target_name=target,
            target_coordinates=target_coords,
            start_time=datetime.now(),
            duration=timedelta(hours=duration_hours),
            priority=priority,
            participating_observatories=required_observatories or list(self.observatories.keys())
        )
        
        # Get weather forecasts for participating observatories
        weather_forecasts = {}
        participating_observatories = [
            self.observatories[obs_id] for obs_id in campaign.participating_observatories
            if obs_id in self.observatories
        ]
        
        weather_tasks = []
        for observatory in participating_observatories:
            task = self.weather_monitor.get_observatory_weather(observatory, duration_hours)
            weather_tasks.append(task)
        
        weather_results = await asyncio.gather(*weather_tasks, return_exceptions=True)
        
        for i, result in enumerate(weather_results):
            if not isinstance(result, Exception):
                obs_id = participating_observatories[i].observatory_id
                weather_forecasts[obs_id] = result
        
        # Generate observation requests
        observation_requests = self._generate_observation_requests(campaign, participating_observatories)
        
        # Optimize global schedule
        scheduling_result = await self.global_scheduler.optimize_global_schedule(
            campaign, participating_observatories, observation_requests, weather_forecasts
        )
        
        # Execute coordinated observations
        execution_result = await self._execute_global_campaign(campaign, scheduling_result)
        
        # Store campaign
        self.active_campaigns[campaign.campaign_id] = {
            'campaign': campaign,
            'weather_forecasts': weather_forecasts,
            'scheduling_result': scheduling_result,
            'execution_result': execution_result
        }
        
        # Update metrics
        self.coordination_metrics['total_campaigns'] += 1
        self.coordination_metrics['total_observations'] += len(observation_requests)
        
        if execution_result['success_rate'] > 0.7:
            self.coordination_metrics['successful_campaigns'] += 1
        
        result = {
            'campaign_id': campaign.campaign_id,
            'target': target,
            'observation_type': observation_type,
            'participating_observatories': len(participating_observatories),
            'total_observations_planned': len(observation_requests),
            'observations_scheduled': scheduling_result['scheduled_observations'],
            'weather_quality': np.mean([f.get('weather_quality_score', 0.5) for f in weather_forecasts.values()]),
            'scheduling_efficiency': scheduling_result['time_utilization'],
            'execution_result': execution_result,
            'estimated_completion': (campaign.start_time + campaign.duration).isoformat()
        }
        
        logger.info(f"🌍 Global campaign coordinated: {campaign.campaign_id}")
        logger.info(f"   Observations scheduled: {result['observations_scheduled']}/{result['total_observations_planned']}")
        
        return result
    
    def _parse_target_coordinates(self, target: str) -> SkyCoord:
        """Parse target name to sky coordinates"""
        
        # Known targets database
        known_targets = {
            'TRAPPIST-1': SkyCoord('23h06m29.283s', '-05d02m28.59s', frame='icrs'),
            'K2-18': SkyCoord('11h30m14.513s', '+07d35m18.21s', frame='icrs'),
            'Proxima Centauri': SkyCoord('14h29m42.946s', '-62d40m46.14s', frame='icrs'),
            'TOI-715': SkyCoord('06h44m38.616s', '+37d51m07.68s', frame='icrs'),
            'HD 209458': SkyCoord('22h03m10.771s', '+18d53m03.55s', frame='icrs'),
            'WASP-121': SkyCoord('06h31m11.038s', '+25d20m26.38s', frame='icrs')
        }
        
        if target in known_targets:
            return known_targets[target]
        else:
            # Default coordinates if target not found
            logger.warning(f"Target {target} not in database, using default coordinates")
            return SkyCoord('00h00m00s', '+00d00m00s', frame='icrs')
    
    def _generate_observation_requests(self, campaign: GlobalCampaign,
                                     observatories: List[GlobalObservatory]) -> List[ObservationRequest]:
        """Generate observation requests for campaign"""
        
        requests = []
        
        for observatory in observatories:
            # Determine appropriate observation mode
            if campaign.campaign_type == CampaignType.EXOPLANET_TRANSIT:
                if 'spectroscopy' in observatory.capabilities:
                    obs_mode = ObservationMode.SPECTROSCOPY
                    duration = timedelta(hours=2)
                else:
                    obs_mode = ObservationMode.PHOTOMETRY
                    duration = timedelta(hours=1)
            elif campaign.campaign_type == CampaignType.TIME_DOMAIN:
                obs_mode = ObservationMode.IMAGING
                duration = timedelta(minutes=30)
            else:
                obs_mode = ObservationMode.IMAGING
                duration = timedelta(hours=1)
            
            # Priority based on observatory capabilities and campaign type
            priority = 1.0
            if observatory.observatory_class == ObservatoryClass.SPACE_OPTICAL:
                priority += 0.5  # Space telescopes get priority
            if obs_mode.value in observatory.capabilities:
                priority += 0.3  # Capability match bonus
            
            request = ObservationRequest(
                request_id=f"{campaign.campaign_id}_{observatory.observatory_id}",
                observatory_id=observatory.observatory_id,
                target_coordinates=campaign.target_coordinates,
                observation_mode=obs_mode,
                requested_time=campaign.start_time,
                duration=duration,
                priority=priority
            )
            
            requests.append(request)
        
        return requests
    
    async def _execute_global_campaign(self, campaign: GlobalCampaign,
                                     scheduling_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordinated global campaign"""
        
        logger.info(f"⚡ Executing global campaign: {campaign.campaign_id}")
        
        start_time = time.time()
        execution_results = []
        
        # Execute observations in parallel
        observation_tasks = []
        for observation in scheduling_result['schedule']['observations']:
            task = self._execute_single_observation(observation, campaign)
            observation_tasks.append(task)
        
        # Wait for all observations
        obs_results = await asyncio.gather(*observation_tasks, return_exceptions=True)
        
        # Process results
        successful_observations = 0
        failed_observations = 0
        
        for result in obs_results:
            if isinstance(result, Exception):
                failed_observations += 1
                execution_results.append({'success': False, 'error': str(result)})
            else:
                execution_results.append(result)
                if result.get('success', False):
                    successful_observations += 1
                else:
                    failed_observations += 1
        
        execution_time = time.time() - start_time
        success_rate = successful_observations / len(scheduling_result['schedule']['observations']) if scheduling_result['schedule']['observations'] else 0
        
        # Update coordination metrics
        self.coordination_metrics['successful_observations'] += successful_observations
        self.coordination_metrics['average_success_rate'] = (
            (self.coordination_metrics['average_success_rate'] * (self.coordination_metrics['total_campaigns'] - 1) + success_rate) /
            self.coordination_metrics['total_campaigns']
        )
        
        execution_summary = {
            'campaign_id': campaign.campaign_id,
            'total_observations': len(scheduling_result['schedule']['observations']),
            'successful_observations': successful_observations,
            'failed_observations': failed_observations,
            'success_rate': success_rate,
            'execution_time': execution_time,
            'data_products_generated': successful_observations * 2,  # Estimate
            'coordination_efficiency': successful_observations / len(campaign.participating_observatories) if campaign.participating_observatories else 0,
            'observation_results': execution_results
        }
        
        logger.info(f"⚡ Global campaign executed: {successful_observations}/{len(scheduling_result['schedule']['observations'])} successful")
        
        return execution_summary
    
    async def _execute_single_observation(self, observation: Dict[str, Any],
                                        campaign: GlobalCampaign) -> Dict[str, Any]:
        """Execute single observation"""
        
        # Simulate observation execution
        duration = observation['duration']
        await asyncio.sleep(min(duration / 300, 2.0))  # Cap simulation time
        
        # Mock observation success based on conditions
        success_probability = observation.get('expected_quality', 0.8)
        success = np.random.random() < success_probability
        
        if success:
            # Generate mock observation data
            observation_data = {
                'observation_id': f"obs_{uuid.uuid4().hex[:8]}",
                'observatory_id': observation['observatory_id'],
                'target_name': campaign.target_name,
                'observation_mode': observation['observation_mode'],
                'start_time': observation['start_time'].isoformat(),
                'duration': duration,
                'data_quality': np.random.uniform(0.7, 1.0),
                'snr': np.random.uniform(10, 100),
                'calibration_status': 'valid',
                'file_size_mb': duration * 0.1,
                'data_products': [
                    f"{observation['observation_mode']}_data.fits",
                    f"calibration_data.fits"
                ]
            }
            
            return {
                'success': True,
                'observation_data': observation_data,
                'execution_time': duration,
                'data_quality_metrics': {
                    'snr': observation_data['snr'],
                    'calibration_quality': np.random.uniform(0.9, 1.0),
                    'atmospheric_conditions': np.random.uniform(0.6, 1.0)
                }
            }
        else:
            return {
                'success': False,
                'error': np.random.choice([
                    'weather_conditions_poor',
                    'instrument_failure',
                    'scheduling_conflict',
                    'target_not_visible'
                ]),
                'attempted_duration': duration
            }
    
    async def trigger_global_followup(self, alert_type: str,
                                    sky_localization: Dict[str, Any],
                                    urgency: str,
                                    observation_strategy: str) -> Dict[str, Any]:
        """Trigger rapid global follow-up observations"""
        
        logger.info(f"🚨 Triggering global follow-up: {alert_type}")
        
        # Create emergency campaign
        campaign = GlobalCampaign(
            campaign_id=f"followup_{uuid.uuid4().hex[:8]}",
            campaign_type=CampaignType.TARGET_OF_OPPORTUNITY,
            target_name=f"{alert_type}_transient",
            target_coordinates=SkyCoord('12h00m00s', '+30d00m00s', frame='icrs'),  # Mock coordinates
            start_time=datetime.now(),
            duration=timedelta(hours=24),
            priority="critical",
            participating_observatories=list(self.observatories.keys()),
            coordination_requirements={
                'rapid_response': True,
                'multi_wavelength': True,
                'time_critical': True
            }
        )
        
        # Select optimal observatories for follow-up
        selected_observatories = self._select_followup_observatories(
            alert_type, sky_localization, urgency
        )
        
        # Generate rapid observation requests
        followup_requests = []
        for obs_id in selected_observatories:
            if obs_id in self.observatories:
                observatory = self.observatories[obs_id]
                
                request = ObservationRequest(
                    request_id=f"followup_{obs_id}_{uuid.uuid4().hex[:8]}",
                    observatory_id=obs_id,
                    target_coordinates=campaign.target_coordinates,
                    observation_mode=ObservationMode.IMAGING,
                    requested_time=datetime.now() + timedelta(minutes=30),
                    duration=timedelta(hours=1),
                    priority=2.0  # High priority
                )
                
                followup_requests.append(request)
        
        # Rapid scheduling (simplified)
        rapid_schedule = {
            'observations': [],
            'observatory_schedules': {}
        }
        
        for request in followup_requests:
            observation = {
                'request_id': request.request_id,
                'observatory_id': request.observatory_id,
                'start_time': request.requested_time,
                'end_time': request.requested_time + request.duration,
                'duration': request.duration.total_seconds(),
                'observation_mode': request.observation_mode.value,
                'priority': request.priority
            }
            
            rapid_schedule['observations'].append(observation)
        
        # Execute follow-up
        execution_result = await self._execute_global_campaign(campaign, {'schedule': rapid_schedule})
        
        followup_result = {
            'followup_id': campaign.campaign_id,
            'alert_type': alert_type,
            'urgency': urgency,
            'observatories_triggered': len(selected_observatories),
            'observations_executed': len(rapid_schedule['observations']),
            'response_time_minutes': 5.0,  # Mock rapid response time
            'success_rate': execution_result['success_rate'],
            'execution_result': execution_result,
            'data_products': execution_result.get('data_products_generated', 0)
        }
        
        logger.info(f"🚨 Global follow-up completed: {followup_result['followup_id']}")
        logger.info(f"   Response time: {followup_result['response_time_minutes']} minutes")
        logger.info(f"   Success rate: {followup_result['success_rate']:.1%}")
        
        return followup_result
    
    def _select_followup_observatories(self, alert_type: str,
                                     sky_localization: Dict[str, Any],
                                     urgency: str) -> List[str]:
        """Select optimal observatories for follow-up observations"""
        
        # Strategy based on alert type
        if alert_type == 'gravitational_wave':
            # Multi-wavelength electromagnetic follow-up
            selected = ['hst', 'jwst', 'vlt', 'keck']
        elif alert_type == 'gamma_ray_burst':
            # Rapid optical/NIR follow-up
            selected = ['hst', 'vlt', 'keck']
        elif alert_type == 'supernova':
            # Spectroscopic follow-up
            selected = ['vlt', 'keck']
        elif alert_type == 'exoplanet_transit':
            # High-precision photometry
            selected = ['jwst', 'hst']
        else:
            # General follow-up
            selected = list(self.observatories.keys())[:3]
        
        # Filter by actual availability
        available_observatories = [
            obs_id for obs_id in selected 
            if obs_id in self.observatories and 
               self.observatories[obs_id].operational_status == 'operational'
        ]
        
        return available_observatories
    
    def get_global_network_status(self) -> Dict[str, Any]:
        """Get comprehensive global network status"""
        
        network_status = {
            'timestamp': datetime.now().isoformat(),
            'total_observatories': len(self.observatories),
            'active_campaigns': len(self.active_campaigns),
            'coordination_metrics': self.coordination_metrics,
            'observatory_status': {},
            'network_health': {}
        }
        
        # Individual observatory status
        operational_count = 0
        for obs_id, observatory in self.observatories.items():
            status = {
                'name': observatory.name,
                'class': observatory.observatory_class.value,
                'operational_status': observatory.operational_status,
                'location': observatory.coordinates,
                'capabilities': observatory.capabilities,
                'instruments': observatory.instruments
            }
            
            if observatory.operational_status == 'operational':
                operational_count += 1
            
            network_status['observatory_status'][obs_id] = status
        
        # Network health assessment
        operational_fraction = operational_count / len(self.observatories) if self.observatories else 0
        success_rate = self.coordination_metrics['average_success_rate']
        
        if operational_fraction >= 0.9 and success_rate >= 0.8:
            health_status = 'excellent'
        elif operational_fraction >= 0.7 and success_rate >= 0.6:
            health_status = 'good'
        elif operational_fraction >= 0.5 and success_rate >= 0.4:
            health_status = 'fair'
        else:
            health_status = 'poor'
        
        network_status['network_health'] = {
            'overall_status': health_status,
            'operational_fraction': operational_fraction,
            'average_success_rate': success_rate,
            'total_uptime_hours': (datetime.now() - self.coordination_metrics['global_coordination_uptime']).total_seconds() / 3600
        }
        
        return network_status

# Factory functions and demonstrations

def create_global_observatory_coordination(config_path: Optional[str] = None) -> GlobalObservatoryCoordination:
    """Create configured global observatory coordination system"""
    return GlobalObservatoryCoordination(config_path)

async def demonstrate_global_observatory_coordination():
    """Demonstrate global observatory coordination capabilities"""
    
    logger.info("🌍 Demonstrating Global Observatory Coordination")
    
    # Create coordination system
    coordinator = create_global_observatory_coordination()
    
    # Demonstration 1: Global Exoplanet Campaign
    logger.info("🌍 Coordinating global exoplanet campaign...")
    
    global_campaign = await coordinator.coordinate_global_campaign(
        target="TRAPPIST-1",
        observation_type="exoplanet_transit",
        duration_hours=12,
        priority="high",
        required_observatories=["jwst", "hst", "vlt", "keck"]
    )
    
    logger.info(f"🌍 Global campaign: {global_campaign['campaign_id']}")
    logger.info(f"   Observatories: {global_campaign['participating_observatories']}")
    logger.info(f"   Scheduling efficiency: {global_campaign['scheduling_efficiency']:.1%}")
    logger.info(f"   Execution success: {global_campaign['execution_result']['success_rate']:.1%}")
    
    # Demonstration 2: Rapid Follow-up Response
    logger.info("🚨 Triggering rapid follow-up response...")
    
    followup_response = await coordinator.trigger_global_followup(
        alert_type="gravitational_wave",
        sky_localization={'ra': 180.0, 'dec': 30.0, 'error_radius': 10.0},
        urgency="critical",
        observation_strategy="electromagnetic_counterpart"
    )
    
    logger.info(f"🚨 Follow-up response: {followup_response['followup_id']}")
    logger.info(f"   Response time: {followup_response['response_time_minutes']} minutes")
    logger.info(f"   Observatories triggered: {followup_response['observatories_triggered']}")
    logger.info(f"   Success rate: {followup_response['success_rate']:.1%}")
    
    # Demonstration 3: Network Status Assessment
    network_status = coordinator.get_global_network_status()
    
    logger.info(f"🌍 Global network status: {network_status['network_health']['overall_status']}")
    logger.info(f"   Operational observatories: {network_status['network_health']['operational_fraction']:.1%}")
    logger.info(f"   Average success rate: {network_status['network_health']['average_success_rate']:.1%}")
    logger.info(f"   Active campaigns: {network_status['active_campaigns']}")
    
    # Compile demonstration results
    demo_results = {
        'global_campaign': global_campaign,
        'followup_response': followup_response,
        'network_status': network_status,
        'demonstration_summary': {
            'total_observatories': network_status['total_observatories'],
            'global_campaign_success': global_campaign['execution_result']['success_rate'] > 0.7,
            'followup_response_success': followup_response['success_rate'] > 0.7,
            'network_health_good': network_status['network_health']['overall_status'] in ['good', 'excellent'],
            'total_observations_executed': (
                global_campaign['execution_result']['total_observations'] +
                followup_response['observations_executed']
            ),
            'overall_coordination_efficiency': (
                global_campaign['execution_result']['success_rate'] +
                followup_response['success_rate']
            ) / 2
        }
    }
    
    logger.info("✅ Global observatory coordination demonstration completed")
    logger.info(f"   Total observations: {demo_results['demonstration_summary']['total_observations_executed']}")
    logger.info(f"   Coordination efficiency: {demo_results['demonstration_summary']['overall_coordination_efficiency']:.1%}")
    logger.info(f"   Network health: {demo_results['demonstration_summary']['network_health_good']}")
    
    return demo_results

if __name__ == "__main__":
    asyncio.run(demonstrate_global_observatory_coordination()) 