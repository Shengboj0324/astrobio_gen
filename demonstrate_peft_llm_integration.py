#!/usr/bin/env python3
"""
PEFT LLM Integration Demonstration for Astrobiology Platform
===========================================================

Comprehensive demonstration of Parameter-Efficient Fine-tuned LLM integration
with surrogate models, CNN datacubes, and multi-modal data sources.

Demonstrates all three required functions:
1. Plain-English rationale generation from technical outputs
2. Interactive Q&A with KEGG/GCM knowledge retrieval  
3. TTS voice-over generation for presentations

Features:
- Seamless integration with existing surrogate transformer
- Multi-modal coordination with enhanced CNN systems
- Enterprise-grade data source integration
- Real-time explanations and interactive assistance
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our systems
from models.peft_llm_integration import (
    create_llm_surrogate_system,
    LLMSurrogateCoordinator,
    SurrogateOutputs,
    LLMConfig
)

# Import API components for testing
try:
    from api.llm_endpoints import (
        RationaleRequest,
        QARequest, 
        VoiceOverRequest,
        PlanetParameters
    )
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False
    # Create basic PlanetParameters for testing
    class PlanetParameters:
        def __init__(self, **kwargs):
            self.radius_earth = kwargs.get('radius_earth', 1.0)
            self.mass_earth = kwargs.get('mass_earth', 1.0)
            self.orbital_period = kwargs.get('orbital_period', 365.25)
            self.insolation = kwargs.get('insolation', 1.0)
            self.stellar_teff = kwargs.get('stellar_teff', 5778)
            self.stellar_logg = kwargs.get('stellar_logg', 4.44)
            self.stellar_metallicity = kwargs.get('stellar_metallicity', 0.0)

# Import existing model components for integration testing
try:
    from models.surrogate_transformer import SurrogateTransformer
    from models.enhanced_surrogate_integration import EnhancedSurrogateIntegration, MultiModalConfig
    from models.enhanced_datacube_unet import EnhancedCubeUNet
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

class PEFTLLMIntegrationDemo:
    """Comprehensive demonstration of PEFT LLM integration"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_coordinator = None
        self.surrogate_model = None
        self.cnn_model = None
        self.results = {}
        
    async def run_comprehensive_demo(self):
        """Run complete PEFT LLM integration demonstration"""
        logger.info("🚀 STARTING PEFT LLM INTEGRATION DEMONSTRATION")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Initialize systems
            await self._initialize_systems()
            
            # Test all three LLM functions
            await self._test_plain_english_rationale()
            await self._test_interactive_qa()
            await self._test_voice_over_generation()
            
            # Test multi-modal integration
            await self._test_multimodal_coordination()
            
            # Performance benchmarking
            await self._benchmark_performance()
            
            # Integration validation
            await self._validate_integration()
            
            total_time = time.time() - start_time
            self.results['total_demo_time'] = total_time
            
            # Save results
            await self._save_results()
            
            logger.info("✅ PEFT LLM INTEGRATION DEMONSTRATION COMPLETED")
            logger.info(f"Total time: {total_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def _initialize_systems(self):
        """Initialize LLM and surrogate model systems"""
        logger.info("\n🤖 INITIALIZING PEFT LLM SYSTEMS")
        logger.info("-" * 40)
        
        # Initialize LLM coordinator
        logger.info("Loading PEFT LLM coordinator...")
        llm_config = LLMConfig(
            base_model_name="microsoft/DialoGPT-medium",
            lora_r=8,  # Smaller for demo
            lora_alpha=16,
            device="cpu",  # Use CPU for broader compatibility
            use_4bit=False  # Disable quantization for CPU
        )
        
        self.llm_coordinator = create_llm_surrogate_system(llm_config)
        logger.info("✅ PEFT LLM coordinator initialized")
        
        # Initialize surrogate models if available
        if MODELS_AVAILABLE:
            try:
                logger.info("Loading surrogate transformer...")
                self.surrogate_model = SurrogateTransformer(
                    dim=128,  # Smaller for demo
                    depth=4,
                    heads=4,
                    n_inputs=8,
                    mode="scalar"
                ).to(self.device)
                logger.info("✅ Surrogate transformer loaded")
                
                logger.info("Loading enhanced CNN...")
                self.cnn_model = EnhancedCubeUNet(
                    n_input_vars=5,
                    n_output_vars=5,
                    base_features=32,
                    depth=3,
                    use_attention=True,
                    use_physics_constraints=True
                ).to(self.device)
                logger.info("✅ Enhanced CNN loaded")
                
            except Exception as e:
                logger.warning(f"Could not load all models: {e}")
                MODELS_AVAILABLE = False
        
        self.results['initialization'] = {
            'llm_coordinator': True,
            'surrogate_model': MODELS_AVAILABLE,
            'cnn_model': MODELS_AVAILABLE,
            'device': str(self.device)
        }
    
    async def _test_plain_english_rationale(self):
        """Test plain-English rationale generation"""
        logger.info("\n📝 TESTING PLAIN-ENGLISH RATIONALE GENERATION")
        logger.info("-" * 50)
        
        # Create test planet parameters
        test_planets = [
            {
                'name': 'Earth-like Exoplanet',
                'params': PlanetParameters(
                    radius_earth=1.1,
                    mass_earth=1.2,
                    orbital_period=372.0,
                    insolation=0.95,
                    stellar_teff=5650,
                    stellar_logg=4.5,
                    stellar_metallicity=0.1
                )
            },
            {
                'name': 'TRAPPIST-1e Analog',
                'params': PlanetParameters(
                    radius_earth=0.92,
                    mass_earth=0.77,
                    orbital_period=6.1,
                    insolation=0.66,
                    stellar_teff=2559,
                    stellar_logg=5.2,
                    stellar_metallicity=-0.04
                )
            },
            {
                'name': 'Hot Super-Earth',
                'params': PlanetParameters(
                    radius_earth=1.8,
                    mass_earth=3.2,
                    orbital_period=8.5,
                    insolation=5.2,
                    stellar_teff=6100,
                    stellar_logg=4.3,
                    stellar_metallicity=0.2
                )
            }
        ]
        
        rationale_results = []
        
        for planet_data in test_planets:
            logger.info(f"\n🌍 Analyzing {planet_data['name']}...")
            
            try:
                # Get surrogate predictions
                surrogate_outputs = await self._get_surrogate_predictions(planet_data['params'])
                
                # Generate plain-English rationale
                start_time = time.time()
                analysis = await self.llm_coordinator.generate_comprehensive_analysis(surrogate_outputs)
                rationale_time = time.time() - start_time
                
                rationale_text = analysis['plain_english_rationale']
                structured_data = analysis['structured_data']
                
                result = {
                    'planet_name': planet_data['name'],
                    'rationale': rationale_text,
                    'generation_time_ms': rationale_time * 1000,
                    'habitability_score': structured_data.get('habitability_score', 0.0),
                    'surface_temperature_c': structured_data.get('surface_temperature', 273.15) - 273.15,
                    'confidence': "High" if structured_data.get('uncertainty_sigma', 0.2) < 0.1 else "Medium"
                }
                
                rationale_results.append(result)
                
                logger.info(f"✅ Generated rationale ({rationale_time*1000:.1f}ms)")
                logger.info(f"📊 Habitability: {result['habitability_score']:.2f}")
                logger.info(f"🌡️ Temperature: {result['surface_temperature_c']:.1f}°C")
                logger.info(f"💬 Rationale: {rationale_text[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Failed to analyze {planet_data['name']}: {e}")
                rationale_results.append({
                    'planet_name': planet_data['name'],
                    'error': str(e)
                })
        
        self.results['plain_english_rationale'] = {
            'test_count': len(test_planets),
            'success_count': len([r for r in rationale_results if 'error' not in r]),
            'results': rationale_results,
            'avg_generation_time_ms': np.mean([r.get('generation_time_ms', 0) for r in rationale_results if 'generation_time_ms' in r])
        }
    
    async def _test_interactive_qa(self):
        """Test interactive Q&A with knowledge retrieval"""
        logger.info("\n❓ TESTING INTERACTIVE Q&A SYSTEM")
        logger.info("-" * 40)
        
        # Test questions spanning different knowledge domains
        test_questions = [
            "What does an oxygen signal-to-noise ratio of 7.5 indicate for exoplanet habitability?",
            "How do KEGG metabolic pathways relate to biosignature detection?",
            "What are the key uncertainties in climate model predictions for M-dwarf planets?",
            "How does atmospheric pressure affect the habitability zone boundaries?",
            "What role does stellar metallicity play in planetary atmosphere retention?"
        ]
        
        qa_results = []
        
        # Create sample surrogate outputs for context
        sample_surrogate_outputs = {
            'predictions': {
                'habitability': 0.75,
                'surface_temp': 292.0,
                'atmospheric_pressure': 1.3
            },
            'uncertainty': 0.12
        }
        
        for question in test_questions:
            logger.info(f"\n🤔 Question: {question}")
            
            try:
                start_time = time.time()
                answer = await self.llm_coordinator.answer_question(
                    question, 
                    sample_surrogate_outputs
                )
                response_time = time.time() - start_time
                
                result = {
                    'question': question,
                    'answer': answer,
                    'response_time_ms': response_time * 1000,
                    'answer_length': len(answer),
                    'includes_sources': 'KEGG' in answer or 'GCM' in answer or 'source' in answer.lower()
                }
                
                qa_results.append(result)
                
                logger.info(f"✅ Generated answer ({response_time*1000:.1f}ms)")
                logger.info(f"💬 Answer: {answer[:150]}...")
                
            except Exception as e:
                logger.error(f"❌ Failed to answer question: {e}")
                qa_results.append({
                    'question': question,
                    'error': str(e)
                })
        
        self.results['interactive_qa'] = {
            'test_count': len(test_questions),
            'success_count': len([r for r in qa_results if 'error' not in r]),
            'results': qa_results,
            'avg_response_time_ms': np.mean([r.get('response_time_ms', 0) for r in qa_results if 'response_time_ms' in r]),
            'avg_answer_length': np.mean([r.get('answer_length', 0) for r in qa_results if 'answer_length' in r])
        }
    
    async def _test_voice_over_generation(self):
        """Test voice-over script generation for presentations"""
        logger.info("\n🎤 TESTING VOICE-OVER GENERATION")
        logger.info("-" * 35)
        
        # Test different presentation scenarios
        scenarios = [
            {
                'name': 'Conference Poster',
                'planet': PlanetParameters(
                    radius_earth=1.0,
                    mass_earth=1.0,
                    insolation=1.0,
                    stellar_teff=5778
                ),
                'target_duration': 60
            },
            {
                'name': 'Research Briefing',
                'planet': PlanetParameters(
                    radius_earth=0.8,
                    mass_earth=0.6,
                    insolation=0.4,
                    stellar_teff=3500
                ),
                'target_duration': 45
            }
        ]
        
        voice_over_results = []
        
        for scenario in scenarios:
            logger.info(f"\n🎯 Scenario: {scenario['name']}")
            
            try:
                # Get surrogate predictions
                surrogate_outputs = await self._get_surrogate_predictions(scenario['planet'])
                
                # Generate voice-over script
                start_time = time.time()
                analysis = await self.llm_coordinator.generate_comprehensive_analysis(surrogate_outputs)
                generation_time = time.time() - start_time
                
                script = analysis['voice_over_script']
                word_count = len(script.split())
                estimated_duration = (word_count / 150) * 60  # 150 words per minute
                
                result = {
                    'scenario': scenario['name'],
                    'script': script,
                    'word_count': word_count,
                    'estimated_duration_seconds': estimated_duration,
                    'target_duration_seconds': scenario['target_duration'],
                    'generation_time_ms': generation_time * 1000,
                    'duration_accuracy': abs(estimated_duration - scenario['target_duration']) / scenario['target_duration']
                }
                
                voice_over_results.append(result)
                
                logger.info(f"✅ Generated script ({generation_time*1000:.1f}ms)")
                logger.info(f"📝 Word count: {word_count}")
                logger.info(f"⏱️ Duration: {estimated_duration:.1f}s (target: {scenario['target_duration']}s)")
                logger.info(f"🎬 Script preview: {script[:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Failed to generate script for {scenario['name']}: {e}")
                voice_over_results.append({
                    'scenario': scenario['name'],
                    'error': str(e)
                })
        
        self.results['voice_over_generation'] = {
            'test_count': len(scenarios),
            'success_count': len([r for r in voice_over_results if 'error' not in r]),
            'results': voice_over_results,
            'avg_generation_time_ms': np.mean([r.get('generation_time_ms', 0) for r in voice_over_results if 'generation_time_ms' in r]),
            'avg_duration_accuracy': np.mean([r.get('duration_accuracy', 1.0) for r in voice_over_results if 'duration_accuracy' in r])
        }
    
    async def _test_multimodal_coordination(self):
        """Test coordination between LLM, surrogate models, and CNN datacubes"""
        logger.info("\n🎯 TESTING MULTI-MODAL COORDINATION")
        logger.info("-" * 42)
        
        coordination_results = []
        
        if MODELS_AVAILABLE and self.surrogate_model and self.cnn_model:
            try:
                # Create test input for multi-modal analysis
                test_input = {
                    'datacube': torch.randn(1, 5, 16, 32, 32).to(self.device),
                    'scalar_params': torch.randn(1, 8).to(self.device)
                }
                
                logger.info("🔬 Running surrogate model inference...")
                with torch.no_grad():
                    surrogate_output = self.surrogate_model(test_input['scalar_params'])
                
                logger.info("🧠 Running CNN datacube processing...")
                with torch.no_grad():
                    cnn_output = self.cnn_model(test_input['datacube'])
                
                # Convert model outputs to structured format
                surrogate_data = {
                    'predictions': {
                        'habitability': float(surrogate_output.get('habitability', torch.tensor([0.7]))[0]),
                        'surface_temp': float(surrogate_output.get('surface_temp', torch.tensor([290.0]))[0]),
                        'atmospheric_pressure': float(surrogate_output.get('atmospheric_pressure', torch.tensor([1.0]))[0])
                    },
                    'uncertainty': 0.1
                }
                
                logger.info("🤖 Generating LLM explanation...")
                analysis = await self.llm_coordinator.generate_comprehensive_analysis(surrogate_data)
                
                coordination_results.append({
                    'surrogate_model_success': True,
                    'cnn_model_success': True,
                    'llm_integration_success': True,
                    'surrogate_output_shape': list(surrogate_output['habitability'].shape) if 'habitability' in surrogate_output else [],
                    'cnn_output_shape': list(cnn_output.shape),
                    'llm_rationale_length': len(analysis['plain_english_rationale']),
                    'coordination_successful': True
                })
                
                logger.info("✅ Multi-modal coordination successful")
                logger.info(f"📊 Surrogate output: {list(surrogate_output.keys())}")
                logger.info(f"🧱 CNN output shape: {cnn_output.shape}")
                logger.info(f"💬 LLM explanation: {analysis['plain_english_rationale'][:100]}...")
                
            except Exception as e:
                logger.error(f"❌ Multi-modal coordination failed: {e}")
                coordination_results.append({
                    'coordination_successful': False,
                    'error': str(e)
                })
        else:
            logger.warning("⚠️ Full models not available, testing LLM-only coordination")
            
            # Test LLM coordination with synthetic data
            synthetic_data = {
                'predictions': {'habitability': 0.8, 'surface_temp': 295.0, 'atmospheric_pressure': 1.2},
                'uncertainty': 0.08
            }
            
            analysis = await self.llm_coordinator.generate_comprehensive_analysis(synthetic_data)
            
            coordination_results.append({
                'llm_only_coordination': True,
                'coordination_successful': True,
                'rationale_generated': len(analysis['plain_english_rationale']) > 0,
                'voice_over_generated': len(analysis['voice_over_script']) > 0
            })
            
            logger.info("✅ LLM-only coordination successful")
        
        self.results['multimodal_coordination'] = {
            'models_available': MODELS_AVAILABLE,
            'coordination_results': coordination_results
        }
    
    async def _benchmark_performance(self):
        """Benchmark LLM performance across different scenarios"""
        logger.info("\n⚡ PERFORMANCE BENCHMARKING")
        logger.info("-" * 30)
        
        benchmark_results = {
            'rationale_generation': [],
            'qa_response': [],
            'voice_over_generation': []
        }
        
        # Benchmark rationale generation
        logger.info("📊 Benchmarking rationale generation...")
        sample_data = {'predictions': {'habitability': 0.7, 'surface_temp': 288.0, 'atmospheric_pressure': 1.0}, 'uncertainty': 0.15}
        
        for i in range(3):  # 3 iterations for average
            start_time = time.time()
            analysis = await self.llm_coordinator.generate_comprehensive_analysis(sample_data)
            elapsed = (time.time() - start_time) * 1000
            benchmark_results['rationale_generation'].append(elapsed)
        
        # Benchmark Q&A
        logger.info("❓ Benchmarking Q&A responses...")
        test_question = "What factors determine planetary habitability?"
        
        for i in range(3):
            start_time = time.time()
            answer = await self.llm_coordinator.answer_question(test_question, sample_data)
            elapsed = (time.time() - start_time) * 1000
            benchmark_results['qa_response'].append(elapsed)
        
        # Calculate statistics
        performance_stats = {}
        for task, times in benchmark_results.items():
            performance_stats[task] = {
                'avg_time_ms': np.mean(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'std_time_ms': np.std(times)
            }
        
        self.results['performance_benchmark'] = performance_stats
        
        logger.info("✅ Performance benchmarking completed")
        for task, stats in performance_stats.items():
            logger.info(f"🎯 {task}: {stats['avg_time_ms']:.1f}ms avg (±{stats['std_time_ms']:.1f}ms)")
    
    async def _validate_integration(self):
        """Validate integration with existing systems"""
        logger.info("\n✅ INTEGRATION VALIDATION")
        logger.info("-" * 25)
        
        validation_results = {}
        
        # Test API endpoint availability
        if API_AVAILABLE:
            validation_results['api_endpoints'] = True
            logger.info("✅ API endpoints available")
        else:
            validation_results['api_endpoints'] = False
            logger.warning("⚠️ API endpoints not available")
        
        # Test knowledge base
        try:
            retriever = self.llm_coordinator.peft_llm.knowledge_retriever
            docs = await retriever.retrieve_relevant_docs("habitability assessment")
            validation_results['knowledge_base'] = len(docs) > 0
            logger.info(f"✅ Knowledge base functional ({len(docs)} docs retrieved)")
        except Exception as e:
            validation_results['knowledge_base'] = False
            logger.warning(f"⚠️ Knowledge base issues: {e}")
        
        # Test model integration
        validation_results['model_integration'] = MODELS_AVAILABLE
        if MODELS_AVAILABLE:
            logger.info("✅ Model integration verified")
        else:
            logger.warning("⚠️ Full model integration not available")
        
        # Test LLM functionality
        try:
            test_data = {'predictions': {'habitability': 0.5}, 'uncertainty': 0.1}
            result = await self.llm_coordinator.generate_comprehensive_analysis(test_data)
            validation_results['llm_functionality'] = len(result['plain_english_rationale']) > 0
            logger.info("✅ LLM functionality verified")
        except Exception as e:
            validation_results['llm_functionality'] = False
            logger.error(f"❌ LLM functionality failed: {e}")
        
        self.results['integration_validation'] = validation_results
    
    async def _get_surrogate_predictions(self, planet_params: PlanetParameters) -> Dict[str, Any]:
        """Get predictions from surrogate models (real or simulated)"""
        if MODELS_AVAILABLE and self.surrogate_model:
            try:
                # Use real surrogate model
                params_tensor = torch.tensor([
                    planet_params.radius_earth,
                    planet_params.mass_earth,
                    planet_params.orbital_period,
                    planet_params.insolation,
                    planet_params.stellar_teff,
                    planet_params.stellar_logg,
                    planet_params.stellar_metallicity,
                    1.0  # host_mass default
                ], dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.surrogate_model(params_tensor)
                
                return {
                    'predictions': outputs,
                    'uncertainty': torch.tensor(0.1)
                }
                
            except Exception as e:
                logger.warning(f"Could not use real surrogate model: {e}")
        
        # Simulate surrogate outputs based on planet parameters
        # More realistic simulation based on actual parameters
        temp_factor = (planet_params.insolation ** 0.25) * (planet_params.stellar_teff / 5778) ** 0.5
        base_temp = 255 * temp_factor
        
        # Habitability decreases with extreme temperatures and masses
        temp_hab = 1.0 - abs(base_temp - 288) / 100
        mass_hab = 1.0 - abs(planet_params.mass_earth - 1.0) / 2
        habitability = np.clip(temp_hab * mass_hab * np.random.uniform(0.8, 1.2), 0.0, 1.0)
        
        return {
            'predictions': {
                'habitability': habitability,
                'surface_temp': base_temp + np.random.normal(0, 10),
                'atmospheric_pressure': np.clip(np.random.lognormal(0, 1), 0.001, 10.0)
            },
            'uncertainty': np.random.uniform(0.05, 0.25)
        }
    
    async def _save_results(self):
        """Save demonstration results"""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"peft_llm_demo_results_{timestamp}.json"
        
        # Make results JSON serializable
        serializable_results = json.loads(json.dumps(self.results, default=str))
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"📁 Results saved to: {results_file}")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print demonstration summary"""
        logger.info("\n🎯 PEFT LLM INTEGRATION DEMONSTRATION SUMMARY")
        logger.info("=" * 55)
        
        # Initialization
        init = self.results.get('initialization', {})
        logger.info(f"🤖 LLM Coordinator: {'✅' if init.get('llm_coordinator') else '❌'}")
        logger.info(f"🔬 Surrogate Model: {'✅' if init.get('surrogate_model') else '⚠️'}")
        logger.info(f"🧠 CNN Model: {'✅' if init.get('cnn_model') else '⚠️'}")
        
        # Function Tests
        rationale = self.results.get('plain_english_rationale', {})
        qa = self.results.get('interactive_qa', {})
        voice_over = self.results.get('voice_over_generation', {})
        
        logger.info(f"\n📝 Plain-English Rationale: {rationale.get('success_count', 0)}/{rationale.get('test_count', 0)} tests passed")
        logger.info(f"❓ Interactive Q&A: {qa.get('success_count', 0)}/{qa.get('test_count', 0)} tests passed")
        logger.info(f"🎤 Voice-Over Generation: {voice_over.get('success_count', 0)}/{voice_over.get('test_count', 0)} tests passed")
        
        # Performance
        perf = self.results.get('performance_benchmark', {})
        if perf:
            logger.info(f"\n⚡ Performance:")
            for task, stats in perf.items():
                logger.info(f"   {task}: {stats.get('avg_time_ms', 0):.1f}ms avg")
        
        # Integration
        validation = self.results.get('integration_validation', {})
        logger.info(f"\n✅ Integration Status:")
        logger.info(f"   API Endpoints: {'✅' if validation.get('api_endpoints') else '⚠️'}")
        logger.info(f"   Knowledge Base: {'✅' if validation.get('knowledge_base') else '⚠️'}")
        logger.info(f"   Model Integration: {'✅' if validation.get('model_integration') else '⚠️'}")
        logger.info(f"   LLM Functionality: {'✅' if validation.get('llm_functionality') else '❌'}")
        
        logger.info(f"\n⏱️ Total Demo Time: {self.results.get('total_demo_time', 0):.2f} seconds")

async def main():
    """Main demonstration function"""
    demo = PEFTLLMIntegrationDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 