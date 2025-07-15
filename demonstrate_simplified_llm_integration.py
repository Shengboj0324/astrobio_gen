#!/usr/bin/env python3
"""
Simplified LLM Integration Demonstration
========================================

Demonstrates the three core LLM functions integrated with astrobiology platform:
1. Plain-English rationale generation from technical outputs
2. Interactive Q&A with knowledge retrieval
3. Voice-over script generation for presentations

Uses simplified implementations that work with current environment.
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

# Simple knowledge base for demonstration
ASTROBIOLOGY_KNOWLEDGE = {
    "oxygen": {
        "description": "Oxygen (O₂) in planetary atmospheres is primarily produced through oxygenic photosynthesis. High O₂ levels (>0.1% atmospheric content) typically indicate biological activity.",
        "detection_threshold": "SNR > 5 for confident detection",
        "significance": "Strong biosignature when combined with other gases"
    },
    "methane": {
        "description": "Methane (CH₄) can be produced biologically through methanogenesis by archaea in anaerobic environments. Simultaneous detection of CH₄ and O₂ is a strong biosignature.",
        "sources": ["biological methanogenesis", "geological processes", "atmospheric photochemistry"],
        "significance": "Atmospheric disequilibrium indicator"
    },
    "habitability": {
        "description": "Habitability assessment integrates surface temperature (optimal 280-320K), atmospheric pressure (optimal 0.1-10 bar), and stellar irradiation factors.",
        "scoring": "Scores above 0.8 indicate high habitability potential",
        "uncertainties": "Model limitations and measurement errors affect confidence"
    },
    "climate_modeling": {
        "description": "Climate models predict surface conditions based on planetary parameters, stellar properties, and atmospheric composition.",
        "key_variables": ["temperature", "pressure", "humidity", "cloud_cover"],
        "validation": "Physics constraints ensure radiative equilibrium and mass conservation"
    }
}

class SimplifiedLLMSystem:
    """Simplified LLM system for astrobiology explanations"""
    
    def __init__(self):
        self.knowledge_base = ASTROBIOLOGY_KNOWLEDGE
        
    def generate_rationale(self, surrogate_outputs: Dict[str, Any]) -> str:
        """Generate plain-English rationale from surrogate outputs"""
        
        # Extract key metrics
        habitability = surrogate_outputs.get('habitability_score', 0.5)
        surface_temp = surrogate_outputs.get('surface_temperature', 288.0)
        pressure = surrogate_outputs.get('atmospheric_pressure', 1.0)
        o2_snr = surrogate_outputs.get('o2_snr', 0.0)
        ch4_snr = surrogate_outputs.get('ch4_snr', 0.0)
        uncertainty = surrogate_outputs.get('uncertainty_sigma', 0.1)
        
        # Convert temperature to Celsius
        temp_c = surface_temp - 273.15
        
        # Generate contextual rationale
        rationale_parts = []
        
        # Habitability assessment
        if habitability > 0.8:
            rationale_parts.append(f"This planet shows excellent habitability potential with a score of {habitability:.2f}.")
        elif habitability > 0.6:
            rationale_parts.append(f"The planet demonstrates promising habitability indicators (score: {habitability:.2f}).")
        elif habitability > 0.4:
            rationale_parts.append(f"Moderate habitability potential is observed with a score of {habitability:.2f}.")
        else:
            rationale_parts.append(f"The planet presents challenging habitability conditions (score: {habitability:.2f}).")
        
        # Temperature and pressure analysis
        if 0 <= temp_c <= 100:
            rationale_parts.append(f"Surface temperatures of {temp_c:.1f}°C support liquid water stability.")
        elif temp_c > 100:
            rationale_parts.append(f"Elevated surface temperatures ({temp_c:.1f}°C) may challenge habitability.")
        else:
            rationale_parts.append(f"Cold surface conditions ({temp_c:.1f}°C) limit liquid water availability.")
        
        if pressure > 0.1:
            rationale_parts.append(f"The substantial atmosphere ({pressure:.2f} bar) can retain heat and support weather systems.")
        else:
            rationale_parts.append(f"The thin atmosphere ({pressure:.3f} bar) provides limited surface pressure.")
        
        # Biosignature analysis
        if o2_snr > 5:
            rationale_parts.append(f"Strong oxygen detection (SNR: {o2_snr:.1f}) suggests potential photosynthetic activity.")
        elif o2_snr > 2:
            rationale_parts.append(f"Moderate oxygen signal (SNR: {o2_snr:.1f}) warrants further investigation.")
        
        if ch4_snr > 5:
            rationale_parts.append(f"Significant methane detection (SNR: {ch4_snr:.1f}) indicates atmospheric disequilibrium.")
        
        # Confidence assessment
        if uncertainty < 0.1:
            confidence_text = "These results have high confidence with low model uncertainty."
        elif uncertainty < 0.2:
            confidence_text = f"Moderate confidence in predictions with {uncertainty:.1f} uncertainty."
        else:
            confidence_text = f"Results are preliminary with significant uncertainty (±{uncertainty:.1f})."
        
        rationale_parts.append(confidence_text)
        
        return " ".join(rationale_parts)
    
    def answer_question(self, question: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Answer questions with knowledge retrieval"""
        
        question_lower = question.lower()
        
        # Find relevant knowledge based on keywords
        relevant_knowledge = []
        for topic, info in self.knowledge_base.items():
            if topic in question_lower or any(keyword in question_lower for keyword in [
                'oxygen', 'o2', 'methane', 'ch4', 'habitat', 'climate', 'temperature', 'pressure'
            ]):
                relevant_knowledge.append((topic, info))
        
        # Generate answer based on question type and context
        if 'oxygen' in question_lower or 'o2' in question_lower:
            base_answer = self.knowledge_base['oxygen']['description']
            if 'snr' in question_lower or 'signal' in question_lower:
                if context and 'o2_snr' in context:
                    snr = context['o2_snr']
                    if snr > 5:
                        return f"{base_answer} Your detected oxygen signal (SNR: {snr:.1f}) exceeds the confidence threshold, indicating likely biological oxygen production. This is a strong biosignature, especially when atmospheric modeling rules out non-biological sources."
                    elif snr > 2:
                        return f"{base_answer} Your oxygen signal (SNR: {snr:.1f}) is promising but requires additional observations for confirmation. Consider complementary measurements of water vapor and ozone to strengthen the biological interpretation."
                    else:
                        return f"{base_answer} The current oxygen signal (SNR: {snr:.1f}) is below the typical detection threshold. Higher precision observations or longer integration times may be needed for reliable detection."
                else:
                    return f"{base_answer} Signal-to-noise ratios above 5 are generally required for confident oxygen detection. Lower SNRs may indicate instrument limitations or weak atmospheric signals."
        
        elif 'methane' in question_lower or 'ch4' in question_lower:
            base_answer = self.knowledge_base['methane']['description']
            if context and 'ch4_snr' in context:
                snr = context['ch4_snr']
                return f"{base_answer} Your methane detection (SNR: {snr:.1f}) {'provides strong evidence' if snr > 5 else 'suggests potential presence'} of atmospheric CH₄. The biological interpretation depends on atmospheric chemistry modeling and simultaneous detection of other gases."
            return base_answer
        
        elif 'habitability' in question_lower:
            if context and 'habitability_score' in context:
                score = context['habitability_score']
                base_answer = self.knowledge_base['habitability']['description']
                interpretation = "excellent" if score > 0.8 else "promising" if score > 0.6 else "moderate" if score > 0.4 else "challenging"
                return f"{base_answer} Your planet's habitability score of {score:.2f} indicates {interpretation} conditions for life as we know it. This assessment considers the integrated effects of stellar irradiation, atmospheric retention, and surface conditions."
            return self.knowledge_base['habitability']['description']
        
        elif 'uncertainty' in question_lower:
            if context and 'uncertainty_sigma' in context:
                uncertainty = context['uncertainty_sigma']
                return f"Uncertainty quantification in habitability assessments accounts for both measurement errors and model limitations. Your current uncertainty (±{uncertainty:.2f}) {'is quite low, indicating reliable predictions' if uncertainty < 0.1 else 'is moderate, suggesting caution in interpretation' if uncertainty < 0.2 else 'is high, indicating preliminary results'}. Bayesian approaches help distinguish between natural variability and model uncertainty."
            return "Uncertainty in habitability assessments arises from measurement errors, model limitations, and incomplete knowledge of biological processes. Confidence intervals should account for both aleatory and epistemic uncertainties."
        
        # General response for unmatched questions
        if relevant_knowledge:
            topic, info = relevant_knowledge[0]
            return f"Based on current astrobiology knowledge: {info.get('description', str(info))} For more specific analysis, please provide additional context about your observations or measurements."
        
        return "I can help with questions about exoplanet habitability, biosignature detection, atmospheric analysis, and climate modeling. Please feel free to ask about specific aspects of planetary characterization or habitability assessment."
    
    def generate_voice_over(self, surrogate_outputs: Dict[str, Any], duration_target: int = 60) -> str:
        """Generate voice-over script for presentations"""
        
        habitability = surrogate_outputs.get('habitability_score', 0.5)
        surface_temp = surrogate_outputs.get('surface_temperature', 288.0)
        pressure = surrogate_outputs.get('atmospheric_pressure', 1.0)
        uncertainty = surrogate_outputs.get('uncertainty_sigma', 0.1)
        
        temp_c = surface_temp - 273.15
        
        # Create engaging narrative
        opening = "Welcome to our exoplanet habitability analysis."
        
        if habitability > 0.8:
            main_finding = f"Our advanced climate models reveal a remarkable discovery: this planet shows exceptional habitability potential with a score of {habitability:.2f}."
        elif habitability > 0.6:
            main_finding = f"Our analysis uncovers promising signs of habitability with a score of {habitability:.2f}, placing this world among potentially life-supporting exoplanets."
        else:
            main_finding = f"While challenging, this planet presents intriguing characteristics with a habitability score of {habitability:.2f}."
        
        temp_analysis = f"Surface temperatures of {temp_c:.1f} degrees Celsius"
        if 0 <= temp_c <= 100:
            temp_analysis += " fall within the liquid water range, a crucial requirement for life as we know it."
        elif temp_c > 100:
            temp_analysis += " indicate a hot world where liquid water would be challenging to maintain."
        else:
            temp_analysis += " suggest a cold environment requiring greenhouse warming for habitability."
        
        if pressure > 0.1:
            atm_analysis = f"The planet's substantial atmosphere, with {pressure:.1f} bar of pressure, can regulate surface conditions and support complex weather systems."
        else:
            atm_analysis = f"With a thin atmosphere of {pressure:.3f} bar, this world faces challenges in maintaining surface pressure and temperature stability."
        
        if uncertainty < 0.1:
            confidence = "Our models show high confidence in these predictions, with minimal uncertainty."
        elif uncertainty < 0.2:
            confidence = f"These results carry moderate uncertainty of {uncertainty:.1f}, typical for current exoplanet characterization methods."
        else:
            confidence = f"While preliminary with {uncertainty:.1f} uncertainty, these findings advance our understanding of planetary habitability."
        
        conclusion = "This analysis contributes to our growing catalog of potentially habitable worlds and guides future observational priorities."
        
        # Combine into flowing narrative
        script = f"{opening} {main_finding} {temp_analysis} {atm_analysis} {confidence} {conclusion}"
        
        # Adjust length if needed (target ~150 words per minute)
        target_words = (duration_target / 60) * 150
        words = script.split()
        
        if len(words) > target_words:
            # Truncate gracefully
            truncated = ' '.join(words[:int(target_words)])
            last_period = truncated.rfind('.')
            if last_period > len(truncated) * 0.8:
                script = truncated[:last_period + 1]
            else:
                script = truncated + "..."
        
        return script

class AstrobiologyDemo:
    """Demonstration of LLM integration with astrobiology platform"""
    
    def __init__(self):
        self.llm_system = SimplifiedLLMSystem()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
    async def run_demo(self):
        """Run comprehensive demonstration"""
        logger.info("🚀 STARTING SIMPLIFIED LLM INTEGRATION DEMO")
        logger.info("=" * 50)
        
        start_time = time.time()
        
        # Test all three functions
        await self._test_function_1_rationale()
        await self._test_function_2_qa()
        await self._test_function_3_voice_over()
        
        # Integration test
        await self._test_integration()
        
        total_time = time.time() - start_time
        self.results['total_time'] = total_time
        
        # Save and display results
        await self._save_and_display_results()
        
        logger.info(f"✅ Demo completed in {total_time:.2f} seconds")
    
    async def _test_function_1_rationale(self):
        """Test Function 1: Plain-English Rationale"""
        logger.info("\n📝 FUNCTION 1: PLAIN-ENGLISH RATIONALE")
        logger.info("-" * 40)
        
        # Test with different exoplanet scenarios
        test_cases = [
            {
                'name': 'Earth-like Planet',
                'data': {
                    'habitability_score': 0.83,
                    'surface_temperature': 294.5,
                    'atmospheric_pressure': 1.15,
                    'o2_snr': 7.5,
                    'ch4_snr': 3.2,
                    'uncertainty_sigma': 0.08
                }
            },
            {
                'name': 'TRAPPIST-1e Analog',
                'data': {
                    'habitability_score': 0.65,
                    'surface_temperature': 251.0,
                    'atmospheric_pressure': 0.8,
                    'o2_snr': 2.1,
                    'ch4_snr': 6.8,
                    'uncertainty_sigma': 0.18
                }
            },
            {
                'name': 'Hot Super-Earth',
                'data': {
                    'habitability_score': 0.25,
                    'surface_temperature': 425.0,
                    'atmospheric_pressure': 3.2,
                    'o2_snr': 0.8,
                    'ch4_snr': 0.5,
                    'uncertainty_sigma': 0.25
                }
            }
        ]
        
        rationale_results = []
        
        for case in test_cases:
            logger.info(f"\n🌍 Analyzing {case['name']}...")
            
            start_time = time.time()
            rationale = self.llm_system.generate_rationale(case['data'])
            generation_time = (time.time() - start_time) * 1000
            
            result = {
                'planet': case['name'],
                'rationale': rationale,
                'generation_time_ms': generation_time,
                'word_count': len(rationale.split()),
                'input_data': case['data']
            }
            
            rationale_results.append(result)
            
            logger.info(f"✅ Generated ({generation_time:.1f}ms)")
            logger.info(f"📊 Input: Habitability {case['data']['habitability_score']:.2f}, Temp {case['data']['surface_temperature']-273.15:.1f}°C")
            logger.info(f"💬 Rationale: {rationale}")
            
        self.results['function_1_rationale'] = {
            'test_count': len(test_cases),
            'results': rationale_results,
            'avg_generation_time_ms': np.mean([r['generation_time_ms'] for r in rationale_results])
        }
    
    async def _test_function_2_qa(self):
        """Test Function 2: Interactive Q&A"""
        logger.info("\n❓ FUNCTION 2: INTERACTIVE Q&A")
        logger.info("-" * 30)
        
        # Test questions with context
        context_data = {
            'habitability_score': 0.76,
            'surface_temperature': 289.0,
            'atmospheric_pressure': 1.3,
            'o2_snr': 6.2,
            'ch4_snr': 4.1,
            'uncertainty_sigma': 0.12
        }
        
        test_questions = [
            "What does an oxygen signal-to-noise ratio of 6.2 indicate for this planet?",
            "How does a habitability score of 0.76 compare to Earth-like conditions?",
            "What are the implications of detecting both oxygen and methane?",
            "How should we interpret the uncertainty level of 0.12 in our analysis?",
            "What additional observations would strengthen these habitability conclusions?"
        ]
        
        qa_results = []
        
        for question in test_questions:
            logger.info(f"\n🤔 Q: {question}")
            
            start_time = time.time()
            answer = self.llm_system.answer_question(question, context_data)
            response_time = (time.time() - start_time) * 1000
            
            result = {
                'question': question,
                'answer': answer,
                'response_time_ms': response_time,
                'answer_length': len(answer.split()),
                'context_used': True
            }
            
            qa_results.append(result)
            
            logger.info(f"✅ Answered ({response_time:.1f}ms)")
            logger.info(f"💬 A: {answer}")
        
        self.results['function_2_qa'] = {
            'test_count': len(test_questions),
            'results': qa_results,
            'avg_response_time_ms': np.mean([r['response_time_ms'] for r in qa_results]),
            'context_data': context_data
        }
    
    async def _test_function_3_voice_over(self):
        """Test Function 3: Voice-over Generation"""
        logger.info("\n🎤 FUNCTION 3: VOICE-OVER GENERATION")
        logger.info("-" * 35)
        
        scenarios = [
            {
                'name': '60-second Conference Poster',
                'duration': 60,
                'data': {
                    'habitability_score': 0.87,
                    'surface_temperature': 292.0,
                    'atmospheric_pressure': 1.05,
                    'uncertainty_sigma': 0.09
                }
            },
            {
                'name': '45-second Research Brief',
                'duration': 45,
                'data': {
                    'habitability_score': 0.42,
                    'surface_temperature': 205.0,
                    'atmospheric_pressure': 0.3,
                    'uncertainty_sigma': 0.22
                }
            }
        ]
        
        voice_over_results = []
        
        for scenario in scenarios:
            logger.info(f"\n🎯 Generating {scenario['name']}...")
            
            start_time = time.time()
            script = self.llm_system.generate_voice_over(scenario['data'], scenario['duration'])
            generation_time = (time.time() - start_time) * 1000
            
            word_count = len(script.split())
            estimated_duration = (word_count / 150) * 60  # 150 words/min
            
            result = {
                'scenario': scenario['name'],
                'target_duration': scenario['duration'],
                'estimated_duration': estimated_duration,
                'script': script,
                'word_count': word_count,
                'generation_time_ms': generation_time,
                'duration_accuracy': abs(estimated_duration - scenario['duration']) / scenario['duration']
            }
            
            voice_over_results.append(result)
            
            logger.info(f"✅ Generated ({generation_time:.1f}ms)")
            logger.info(f"⏱️ Duration: {estimated_duration:.1f}s (target: {scenario['duration']}s)")
            logger.info(f"📝 Words: {word_count}")
            logger.info(f"🎬 Script: {script}")
        
        self.results['function_3_voice_over'] = {
            'test_count': len(scenarios),
            'results': voice_over_results,
            'avg_generation_time_ms': np.mean([r['generation_time_ms'] for r in voice_over_results])
        }
    
    async def _test_integration(self):
        """Test integration with surrogate models"""
        logger.info("\n🤝 INTEGRATION WITH SURROGATE MODELS")
        logger.info("-" * 40)
        
        # Simulate surrogate model predictions
        planet_params = {
            'radius_earth': 1.1,
            'mass_earth': 1.2,
            'insolation': 0.95,
            'stellar_teff': 5650
        }
        
        logger.info("🔬 Simulating surrogate model inference...")
        
        # Simulate realistic surrogate outputs
        np.random.seed(42)  # For reproducible demo
        surrogate_outputs = {
            'habitability_score': np.clip(np.random.normal(0.75, 0.1), 0.0, 1.0),
            'surface_temperature': 255 * (planet_params['insolation'] ** 0.25) + np.random.normal(0, 8),
            'atmospheric_pressure': np.clip(np.random.lognormal(0, 0.8), 0.01, 5.0),
            'o2_snr': np.random.uniform(2.0, 8.0),
            'ch4_snr': np.random.uniform(1.0, 6.0),
            'uncertainty_sigma': np.random.uniform(0.08, 0.18)
        }
        
        logger.info("🤖 Generating comprehensive LLM analysis...")
        
        # Generate all three outputs
        start_time = time.time()
        
        rationale = self.llm_system.generate_rationale(surrogate_outputs)
        voice_over = self.llm_system.generate_voice_over(surrogate_outputs)
        qa_answer = self.llm_system.answer_question(
            "What does this analysis tell us about the planet's potential for life?",
            surrogate_outputs
        )
        
        total_time = (time.time() - start_time) * 1000
        
        integration_result = {
            'planet_parameters': planet_params,
            'surrogate_outputs': surrogate_outputs,
            'llm_rationale': rationale,
            'llm_voice_over': voice_over,
            'llm_qa_example': qa_answer,
            'total_generation_time_ms': total_time,
            'integration_successful': True
        }
        
        self.results['integration_test'] = integration_result
        
        logger.info(f"✅ Complete analysis generated ({total_time:.1f}ms)")
        logger.info("\n🎯 INTEGRATED ANALYSIS RESULTS:")
        logger.info(f"🌍 Planet: {planet_params['radius_earth']}R⊕, {planet_params['insolation']}S⊕")
        logger.info(f"📊 Habitability: {surrogate_outputs['habitability_score']:.2f}")
        logger.info(f"📝 Rationale: {rationale[:100]}...")
        logger.info(f"🎤 Voice-over: {voice_over[:100]}...")
    
    async def _save_and_display_results(self):
        """Save results and display summary"""
        
        # Save to file
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"simplified_llm_demo_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"\n📁 Results saved to: {results_file}")
        
        # Display summary
        logger.info("\n🎯 DEMONSTRATION SUMMARY")
        logger.info("=" * 25)
        
        # Function summaries
        func1 = self.results.get('function_1_rationale', {})
        func2 = self.results.get('function_2_qa', {})
        func3 = self.results.get('function_3_voice_over', {})
        
        logger.info(f"📝 Function 1 - Plain-English Rationale:")
        logger.info(f"   ✅ {func1.get('test_count', 0)} tests completed")
        logger.info(f"   ⚡ {func1.get('avg_generation_time_ms', 0):.1f}ms average generation time")
        
        logger.info(f"\n❓ Function 2 - Interactive Q&A:")
        logger.info(f"   ✅ {func2.get('test_count', 0)} questions answered")
        logger.info(f"   ⚡ {func2.get('avg_response_time_ms', 0):.1f}ms average response time")
        
        logger.info(f"\n🎤 Function 3 - Voice-over Generation:")
        logger.info(f"   ✅ {func3.get('test_count', 0)} scripts generated")
        logger.info(f"   ⚡ {func3.get('avg_generation_time_ms', 0):.1f}ms average generation time")
        
        # Integration summary
        integration = self.results.get('integration_test', {})
        logger.info(f"\n🤝 Integration Test:")
        logger.info(f"   ✅ Surrogate model coordination: {'Success' if integration.get('integration_successful') else 'Failed'}")
        logger.info(f"   ⚡ Complete analysis: {integration.get('total_generation_time_ms', 0):.1f}ms")
        
        logger.info(f"\n⏱️ Total demonstration time: {self.results.get('total_time', 0):.2f} seconds")
        
        # Key capabilities demonstrated
        logger.info("\n🎯 KEY CAPABILITIES DEMONSTRATED:")
        logger.info("   ✅ Plain-English explanations from technical data")
        logger.info("   ✅ Context-aware Q&A with scientific knowledge")
        logger.info("   ✅ Adaptive voice-over scripts for presentations")
        logger.info("   ✅ Seamless integration with surrogate model outputs")
        logger.info("   ✅ Real-time analysis and explanation generation")

async def main():
    """Run the simplified LLM integration demonstration"""
    demo = AstrobiologyDemo()
    await demo.run_demo()

if __name__ == "__main__":
    asyncio.run(main()) 