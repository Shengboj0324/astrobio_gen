#!/usr/bin/env python3
"""
Priority 2: Enhanced Narrative Chat System
==========================================

Builds on Priority 1 evolutionary process modeling to create an intelligent research
companion that helps scientists navigate the boundaries between quantitative and 
qualitative understanding in astrobiology.

Key Capabilities:
1. Recognizes when quantitative analysis reaches fundamental limits
2. Suggests qualitative research directions and methodologies
3. Provides evolutionary storytelling assistance
4. Bridges data-driven insights with philosophical understanding
5. Guides researchers through paradigm transitions
"""

import sys
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import existing chat components
sys.path.append(str(Path(__file__).parent.parent))
from chat.enhanced_tool_router import EnhancedToolRouter
from models.evolutionary_process_tracker import EvolutionaryProcessTracker, EvolutionaryTimeScale

logger = logging.getLogger(__name__)

class ResearchPhase(Enum):
    """Different phases of astrobiology research"""
    QUANTITATIVE_ANALYSIS = "quantitative_analysis"
    BOUNDARY_IDENTIFICATION = "boundary_identification"
    PARADIGM_TRANSITION = "paradigm_transition"
    QUALITATIVE_EXPLORATION = "qualitative_exploration"
    NARRATIVE_CONSTRUCTION = "narrative_construction"
    PHILOSOPHICAL_INTEGRATION = "philosophical_integration"

class QuantitativeLimitType(Enum):
    """Types of limits encountered in quantitative analysis"""
    DATA_INSUFFICIENCY = "data_insufficiency"          # Not enough data
    MODEL_UNCERTAINTY = "model_uncertainty"            # Model can't predict
    EMERGENCE_THRESHOLD = "emergence_threshold"        # New properties appear
    TEMPORAL_CONTINGENCY = "temporal_contingency"      # Path-dependent evolution
    PHILOSOPHICAL_BOUNDARY = "philosophical_boundary"  # Life vs non-life questions
    COMPLEXITY_OVERFLOW = "complexity_overflow"        # Too many interacting factors

@dataclass
class NarrativeContext:
    """Context for narrative construction and philosophical guidance"""
    research_question: str
    current_phase: ResearchPhase
    quantitative_findings: Dict[str, Any]
    uncertainty_sources: List[str]
    limit_types_encountered: List[QuantitativeLimitType]
    evolutionary_timescale: Optional[float] = None  # Gya if relevant
    philosophical_questions: List[str] = field(default_factory=list)
    narrative_threads: List[str] = field(default_factory=list)

class PhilosophicalGuidanceEngine:
    """Provides philosophical guidance for astrobiology research"""
    
    def __init__(self):
        self.philosophical_frameworks = {
            "process_philosophy": {
                "description": "Focus on becoming rather than being, evolution as continuous process",
                "key_concepts": ["emergence", "novelty", "temporal_asymmetry", "creative_evolution"],
                "applications": ["evolutionary_transitions", "life_definition", "complexity_emergence"],
                "guiding_questions": [
                    "How does this process unfold over time?",
                    "What new properties emerge that weren't present before?",
                    "How does history shape current possibilities?"
                ]
            },
            "systems_thinking": {
                "description": "Understanding wholes and relationships rather than isolated parts",
                "key_concepts": ["holism", "networks", "feedback_loops", "self_organization"],
                "applications": ["ecosystem_dynamics", "planetary_systems", "life_environment_coupling"],
                "guiding_questions": [
                    "What relationships are we missing?",
                    "How do feedback loops shape the system?",
                    "What emerges from the network that individual parts can't explain?"
                ]
            },
            "contingency_theory": {
                "description": "Recognition that outcomes depend on specific historical sequences",
                "key_concepts": ["path_dependence", "historical_contingency", "alternative_histories"],
                "applications": ["evolutionary_trajectories", "planetary_formation", "civilization_development"],
                "guiding_questions": [
                    "How might things have been different?",
                    "What critical junctures shaped this outcome?",
                    "Which factors were necessary vs sufficient?"
                ]
            },
            "phenomenology": {
                "description": "Focus on experience and meaning rather than objective measurement",
                "key_concepts": ["lived_experience", "intentionality", "embodiment", "temporality"],
                "applications": ["consciousness_studies", "biological_agency", "environmental_perception"],
                "guiding_questions": [
                    "What is the experience like for this organism?",
                    "How does meaning emerge in biological systems?",
                    "What does 'agency' mean at this level of organization?"
                ]
            }
        }
        
        self.limit_to_framework_mapping = {
            QuantitativeLimitType.EMERGENCE_THRESHOLD: ["process_philosophy", "systems_thinking"],
            QuantitativeLimitType.TEMPORAL_CONTINGENCY: ["contingency_theory", "process_philosophy"],
            QuantitativeLimitType.PHILOSOPHICAL_BOUNDARY: ["phenomenology", "process_philosophy"],
            QuantitativeLimitType.COMPLEXITY_OVERFLOW: ["systems_thinking", "process_philosophy"],
            QuantitativeLimitType.MODEL_UNCERTAINTY: ["contingency_theory", "systems_thinking"],
            QuantitativeLimitType.DATA_INSUFFICIENCY: ["phenomenology", "systems_thinking"]
        }
    
    def suggest_philosophical_framework(self, limit_types: List[QuantitativeLimitType]) -> Dict[str, Any]:
        """Suggest appropriate philosophical frameworks for encountered limits"""
        framework_scores = {}
        
        for limit_type in limit_types:
            frameworks = self.limit_to_framework_mapping.get(limit_type, [])
            for framework in frameworks:
                framework_scores[framework] = framework_scores.get(framework, 0) + 1
        
        # Get top frameworks
        sorted_frameworks = sorted(framework_scores.items(), key=lambda x: x[1], reverse=True)
        
        suggestions = []
        for framework_name, score in sorted_frameworks[:3]:  # Top 3 frameworks
            framework_info = self.philosophical_frameworks[framework_name]
            suggestions.append({
                "framework": framework_name,
                "relevance_score": score,
                "description": framework_info["description"],
                "key_concepts": framework_info["key_concepts"],
                "guiding_questions": framework_info["guiding_questions"][:3]  # Top 3 questions
            })
        
        return {
            "recommended_frameworks": suggestions,
            "integration_advice": self._generate_integration_advice(suggestions),
            "methodological_shifts": self._suggest_methodological_shifts(limit_types)
        }
    
    def _generate_integration_advice(self, frameworks: List[Dict]) -> str:
        """Generate advice for integrating philosophical perspectives"""
        if len(frameworks) == 0:
            return "Continue with quantitative analysis - no major philosophical shifts needed."
        
        primary_framework = frameworks[0]["framework"]
        
        integration_templates = {
            "process_philosophy": "Shift focus from 'what life is' to 'how life becomes.' Track evolutionary processes over time rather than snapshots.",
            "systems_thinking": "Look for emergent properties and network effects. Consider how relationships between components create new behaviors.",
            "contingency_theory": "Explore alternative evolutionary pathways. Ask what made this particular outcome likely vs other possibilities.",
            "phenomenology": "Consider the 'experience' of biological systems. What does environment 'mean' to this organism?"
        }
        
        base_advice = integration_templates.get(primary_framework, "Integrate multiple philosophical perspectives.")
        
        if len(frameworks) > 1:
            base_advice += f" Also consider {frameworks[1]['framework']} to complement this approach."
        
        return base_advice
    
    def _suggest_methodological_shifts(self, limit_types: List[QuantitativeLimitType]) -> List[str]:
        """Suggest specific methodological shifts based on limits encountered"""
        methodological_suggestions = {
            QuantitativeLimitType.DATA_INSUFFICIENCY: [
                "Ethnographic study of research practices",
                "Phenomenological analysis of researcher intuitions",
                "Historical case study analysis"
            ],
            QuantitativeLimitType.MODEL_UNCERTAINTY: [
                "Scenario-based storytelling",
                "Comparative case analysis",
                "Systems mapping and relationship modeling"
            ],
            QuantitativeLimitType.EMERGENCE_THRESHOLD: [
                "Multi-level analysis (micro-macro interactions)",
                "Process tracing over time",
                "Pattern recognition in complex data"
            ],
            QuantitativeLimitType.TEMPORAL_CONTINGENCY: [
                "Historical narrative construction",
                "Counterfactual analysis",
                "Critical juncture identification"
            ],
            QuantitativeLimitType.PHILOSOPHICAL_BOUNDARY: [
                "Conceptual analysis and definition",
                "Thought experiments",
                "Interdisciplinary dialogue"
            ],
            QuantitativeLimitType.COMPLEXITY_OVERFLOW: [
                "Systems thinking workshops",
                "Stakeholder mapping",
                "Network analysis"
            ]
        }
        
        suggestions = []
        for limit_type in limit_types:
            suggestions.extend(methodological_suggestions.get(limit_type, []))
        
        # Remove duplicates and return top 5
        unique_suggestions = list(dict.fromkeys(suggestions))
        return unique_suggestions[:5]

class EvolutionaryNarrativeBuilder:
    """Builds coherent evolutionary narratives from quantitative findings"""
    
    def __init__(self, evolutionary_tracker: Optional[EvolutionaryProcessTracker] = None):
        self.evolutionary_tracker = evolutionary_tracker
        self.time_scales = EvolutionaryTimeScale()
        
        # Narrative templates for different evolutionary phases
        self.narrative_templates = {
            "origin_of_life": {
                "timeframe": [4.6, 3.8],  # Gya
                "key_themes": ["abiogenesis", "early_chemistry", "first_metabolisms"],
                "template": "In the {timeframe}, life emerged through {key_processes}. The transition from {initial_state} to {final_state} involved {critical_steps}.",
                "uncertainty_factors": ["chemical_pathways", "environmental_conditions", "energy_sources"]
            },
            "metabolic_innovation": {
                "timeframe": [3.8, 2.5],
                "key_themes": ["photosynthesis", "metabolic_diversity", "ecosystem_formation"],
                "template": "The period from {timeframe} Gya saw {major_innovations}. This {process_type} fundamentally altered {affected_systems}.",
                "uncertainty_factors": ["innovation_timing", "environmental_feedback", "evolutionary_pressure"]
            },
            "great_oxidation": {
                "timeframe": [2.5, 2.0],
                "key_themes": ["atmospheric_transition", "mass_extinction", "new_possibilities"],
                "template": "The Great Oxidation Event marked a {transition_type}. While {quantitative_data} shows {measurements}, the {qualitative_aspects} reveal {deeper_meaning}.",
                "uncertainty_factors": ["exact_mechanisms", "global_vs_local", "biological_response"]
            },
            "complex_life": {
                "timeframe": [2.0, 0.5],
                "key_themes": ["eukaryotes", "multicellularity", "innovation_acceleration"],
                "template": "The evolution of {complexity_type} during {timeframe} represents {philosophical_significance}. {quantitative_patterns} suggest {process_interpretation}.",
                "uncertainty_factors": ["complexity_drivers", "contingent_events", "alternative_pathways"]
            }
        }
    
    def construct_narrative(
        self, 
        context: NarrativeContext,
        quantitative_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Construct evolutionary narrative integrating quantitative and qualitative insights"""
        
        # Determine relevant narrative phase
        relevant_phase = self._identify_narrative_phase(context)
        
        # Extract key quantitative patterns
        patterns = self._extract_quantitative_patterns(quantitative_results)
        
        # Identify narrative gaps (where numbers can't tell the story)
        gaps = self._identify_narrative_gaps(patterns, context.limit_types_encountered)
        
        # Construct coherent narrative
        narrative = self._build_coherent_narrative(relevant_phase, patterns, gaps, context)
        
        return {
            "narrative_phase": relevant_phase,
            "quantitative_patterns": patterns,
            "narrative_gaps": gaps,
            "constructed_narrative": narrative,
            "philosophical_insights": self._extract_philosophical_insights(narrative, context),
            "research_directions": self._suggest_narrative_research_directions(gaps, context)
        }
    
    def _identify_narrative_phase(self, context: NarrativeContext) -> str:
        """Identify which evolutionary narrative phase is most relevant"""
        if context.evolutionary_timescale is None:
            return "general_evolution"
        
        time_gya = context.evolutionary_timescale
        
        for phase_name, phase_info in self.narrative_templates.items():
            timeframe = phase_info["timeframe"]
            if timeframe[1] <= time_gya <= timeframe[0]:
                return phase_name
        
        return "general_evolution"
    
    def _extract_quantitative_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key patterns from quantitative results"""
        patterns = {
            "temporal_trends": [],
            "spatial_patterns": [],
            "coupling_strengths": [],
            "thresholds_detected": [],
            "uncertainty_levels": []
        }
        
        # Extract patterns from evolutionary results
        if "metabolic_complexity" in results:
            complexity_data = results["metabolic_complexity"]
            patterns["temporal_trends"].append({
                "variable": "metabolic_complexity",
                "trend": "increasing" if complexity_data > 0.5 else "stable",
                "confidence": "high"
            })
        
        if "atmospheric_disequilibrium" in results:
            disequilibrium = results["atmospheric_disequilibrium"]
            patterns["thresholds_detected"].append({
                "threshold_type": "atmospheric_disequilibrium",
                "value": float(disequilibrium) if hasattr(disequilibrium, 'item') else disequilibrium,
                "significance": "biosignature_indicator"
            })
        
        if "biosignature_strength" in results:
            biosig = results["biosignature_strength"]
            patterns["coupling_strengths"].append({
                "coupling_type": "life_atmosphere",
                "strength": "strong" if biosig > 0.5 else "weak",
                "evidence_type": "biosignature"
            })
        
        return patterns
    
    def _identify_narrative_gaps(
        self, 
        patterns: Dict[str, Any], 
        limit_types: List[QuantitativeLimitType]
    ) -> List[Dict[str, Any]]:
        """Identify gaps where quantitative analysis cannot tell the complete story"""
        gaps = []
        
        gap_templates = {
            QuantitativeLimitType.EMERGENCE_THRESHOLD: {
                "gap_type": "emergence_explanation",
                "description": "How new properties emerge from component interactions",
                "questions": ["Why does this particular organization create new behaviors?", "What makes this transition qualitatively different?"]
            },
            QuantitativeLimitType.TEMPORAL_CONTINGENCY: {
                "gap_type": "historical_contingency",
                "description": "Why this particular evolutionary path was taken",
                "questions": ["What made this pathway more likely than alternatives?", "How did historical events constrain future possibilities?"]
            },
            QuantitativeLimitType.PHILOSOPHICAL_BOUNDARY: {
                "gap_type": "conceptual_boundary",
                "description": "Fundamental questions about the nature of life",
                "questions": ["What distinguishes living from non-living processes?", "How do we define biological agency?"]
            },
            QuantitativeLimitType.COMPLEXITY_OVERFLOW: {
                "gap_type": "complexity_understanding",
                "description": "How complex systems maintain coherence and function",
                "questions": ["How does the system maintain stability amid complexity?", "What principles govern large-scale organization?"]
            }
        }
        
        for limit_type in limit_types:
            if limit_type in gap_templates:
                gap_info = gap_templates[limit_type]
                gaps.append({
                    "gap_type": gap_info["gap_type"],
                    "description": gap_info["description"],
                    "research_questions": gap_info["questions"],
                    "limit_source": limit_type.value
                })
        
        return gaps
    
    def _build_coherent_narrative(
        self,
        phase: str,
        patterns: Dict[str, Any],
        gaps: List[Dict[str, Any]],
        context: NarrativeContext
    ) -> Dict[str, Any]:
        """Build a coherent narrative that integrates quantitative and qualitative elements"""
        
        # Get phase template
        template_info = self.narrative_templates.get(phase, self.narrative_templates["complex_life"])
        
        # Construct narrative components
        narrative_components = {
            "quantitative_foundation": self._describe_quantitative_foundation(patterns),
            "qualitative_interpretation": self._interpret_qualitative_meaning(patterns, gaps),
            "synthesis": self._synthesize_quantitative_qualitative(patterns, gaps, context),
            "uncertainty_acknowledgment": self._acknowledge_uncertainties(gaps, template_info),
            "broader_implications": self._explore_broader_implications(context, patterns)
        }
        
        # Create coherent narrative text
        narrative_text = self._compose_narrative_text(narrative_components, template_info, context)
        
        return {
            "narrative_components": narrative_components,
            "narrative_text": narrative_text,
            "phase_template": template_info,
            "integration_quality": self._assess_integration_quality(patterns, gaps)
        }
    
    def _describe_quantitative_foundation(self, patterns: Dict[str, Any]) -> str:
        """Describe what the quantitative data tells us"""
        descriptions = []
        
        for trend in patterns.get("temporal_trends", []):
            descriptions.append(f"{trend['variable']} shows a {trend['trend']} pattern with {trend['confidence']} confidence")
        
        for threshold in patterns.get("thresholds_detected", []):
            descriptions.append(f"Detected {threshold['threshold_type']} threshold with significance for {threshold['significance']}")
        
        if not descriptions:
            return "Quantitative analysis provides baseline measurements and trends."
        
        return "Our quantitative analysis reveals: " + "; ".join(descriptions) + "."
    
    def _interpret_qualitative_meaning(self, patterns: Dict[str, Any], gaps: List[Dict[str, Any]]) -> str:
        """Interpret the qualitative meaning beyond the numbers"""
        interpretations = []
        
        for gap in gaps:
            if gap["gap_type"] == "emergence_explanation":
                interpretations.append("The emergence of new properties suggests qualitative transitions that transcend quantitative thresholds")
            elif gap["gap_type"] == "historical_contingency":
                interpretations.append("The specific evolutionary pathway reflects historical contingency and path-dependent processes")
            elif gap["gap_type"] == "conceptual_boundary":
                interpretations.append("Fundamental conceptual questions about the nature of life emerge")
        
        if not interpretations:
            return "The data suggests deeper processes operating beyond measurable parameters."
        
        return "However, " + "; ".join(interpretations) + "."
    
    def _synthesize_quantitative_qualitative(
        self, 
        patterns: Dict[str, Any], 
        gaps: List[Dict[str, Any]], 
        context: NarrativeContext
    ) -> str:
        """Synthesize quantitative and qualitative insights"""
        return f"The integration of quantitative patterns with qualitative understanding suggests that {context.research_question} involves both measurable processes and emergent phenomena that require narrative interpretation."
    
    def _acknowledge_uncertainties(self, gaps: List[Dict[str, Any]], template_info: Dict[str, Any]) -> str:
        """Acknowledge uncertainties and limitations"""
        uncertainty_factors = template_info.get("uncertainty_factors", ["temporal_processes", "complex_interactions"])
        
        acknowledgment = f"Significant uncertainties remain regarding {', '.join(uncertainty_factors[:2])}."
        
        if gaps:
            gap_types = [gap["gap_type"] for gap in gaps]
            acknowledgment += f" These uncertainties particularly affect our understanding of {', '.join(gap_types)}."
        
        return acknowledgment
    
    def _explore_broader_implications(self, context: NarrativeContext, patterns: Dict[str, Any]) -> str:
        """Explore broader implications for astrobiology"""
        implications = [
            "These findings contribute to our understanding of life as a cosmic phenomenon",
            "The results suggest new research directions for detecting life beyond Earth",
            "This work highlights the importance of temporal perspective in astrobiology"
        ]
        
        return "Broader implications include: " + "; ".join(implications) + "."
    
    def _compose_narrative_text(
        self, 
        components: Dict[str, str], 
        template_info: Dict[str, Any], 
        context: NarrativeContext
    ) -> str:
        """Compose coherent narrative text"""
        narrative_sections = [
            f"# Evolutionary Narrative: {context.research_question}",
            "",
            "## Quantitative Foundation",
            components["quantitative_foundation"],
            "",
            "## Qualitative Interpretation", 
            components["qualitative_interpretation"],
            "",
            "## Synthesis",
            components["synthesis"],
            "",
            "## Uncertainties and Limitations",
            components["uncertainty_acknowledgment"],
            "",
            "## Broader Implications",
            components["broader_implications"]
        ]
        
        return "\n".join(narrative_sections)
    
    def _assess_integration_quality(self, patterns: Dict[str, Any], gaps: List[Dict[str, Any]]) -> float:
        """Assess quality of quantitative-qualitative integration"""
        # Simple scoring based on balance of quantitative patterns and narrative gaps
        pattern_count = sum(len(v) for v in patterns.values() if isinstance(v, list))
        gap_count = len(gaps)
        
        if pattern_count == 0 and gap_count == 0:
            return 0.5  # Neutral
        
        balance_score = min(pattern_count, gap_count) / max(pattern_count, gap_count, 1)
        return balance_score
    
    def _extract_philosophical_insights(self, narrative: Dict[str, Any], context: NarrativeContext) -> List[str]:
        """Extract philosophical insights from the narrative"""
        insights = []
        
        if context.evolutionary_timescale and context.evolutionary_timescale > 3.0:
            insights.append("Deep time perspective reveals the contingent nature of evolutionary outcomes")
        
        if QuantitativeLimitType.EMERGENCE_THRESHOLD in context.limit_types_encountered:
            insights.append("Emergence demonstrates that wholes possess properties not predictable from parts")
        
        if QuantitativeLimitType.TEMPORAL_CONTINGENCY in context.limit_types_encountered:
            insights.append("Historical contingency shows that current life forms represent one of many possible outcomes")
        
        insights.append("The integration of quantitative and qualitative approaches is essential for understanding life")
        
        return insights
    
    def _suggest_narrative_research_directions(self, gaps: List[Dict[str, Any]], context: NarrativeContext) -> List[str]:
        """Suggest research directions based on narrative gaps"""
        directions = []
        
        for gap in gaps:
            if gap["gap_type"] == "emergence_explanation":
                directions.append("Investigate mechanisms of emergence through multi-level analysis")
            elif gap["gap_type"] == "historical_contingency":
                directions.append("Explore alternative evolutionary scenarios through comparative planetology")
            elif gap["gap_type"] == "conceptual_boundary":
                directions.append("Engage in interdisciplinary dialogue on the definition of life")
        
        directions.append("Develop methodologies that integrate quantitative and narrative approaches")
        
        return directions

class EnhancedNarrativeChat:
    """
    Main enhanced narrative chat system that integrates philosophical guidance
    and evolutionary storytelling with existing chat capabilities
    """
    
    def __init__(
        self, 
        base_tool_router: Optional[EnhancedToolRouter] = None,
        evolutionary_tracker: Optional[EvolutionaryProcessTracker] = None
    ):
        self.base_tool_router = base_tool_router
        self.evolutionary_tracker = evolutionary_tracker
        
        # Initialize components
        self.philosophical_engine = PhilosophicalGuidanceEngine()
        self.narrative_builder = EvolutionaryNarrativeBuilder(evolutionary_tracker)
        
        # Conversation memory with narrative context
        self.conversation_memory = []
        self.current_context = None
        
        # Database for conversation and context logging
        self.db_path = Path("chat/enhanced_narrative_conversations.db")
        self._initialize_database()
        
        logger.info("Enhanced Narrative Chat system initialized")
    
    def _initialize_database(self):
        """Initialize database for conversation and context tracking"""
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    user_input TEXT,
                    assistant_response TEXT,
                    research_phase TEXT,
                    limit_types TEXT,
                    philosophical_frameworks TEXT,
                    narrative_phase TEXT
                )
            ''')
            
            # Context transitions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS context_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    from_phase TEXT,
                    to_phase TEXT,
                    transition_reason TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            ''')
            
            conn.commit()
    
    def process_research_query(
        self, 
        user_input: str,
        current_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process research query with enhanced narrative and philosophical guidance
        
        Args:
            user_input: User's research question or input
            current_results: Current quantitative results if available
            
        Returns:
            Enhanced response with philosophical guidance and narrative suggestions
        """
        # Analyze user input to understand research context
        analysis = self._analyze_user_input(user_input, current_results)
        
        # Identify research phase and limits
        research_phase = self._identify_research_phase(analysis)
        limit_types = self._identify_quantitative_limits(analysis, current_results)
        
        # Create narrative context
        context = NarrativeContext(
            research_question=analysis["research_question"],
            current_phase=research_phase,
            quantitative_findings=current_results or {},
            uncertainty_sources=analysis.get("uncertainty_sources", []),
            limit_types_encountered=limit_types,
            evolutionary_timescale=analysis.get("evolutionary_timescale"),
            philosophical_questions=analysis.get("philosophical_questions", [])
        )
        
        # Generate response based on phase
        if research_phase == ResearchPhase.QUANTITATIVE_ANALYSIS:
            response = self._handle_quantitative_phase(context, user_input)
        elif research_phase == ResearchPhase.BOUNDARY_IDENTIFICATION:
            response = self._handle_boundary_identification(context, user_input)
        elif research_phase == ResearchPhase.PARADIGM_TRANSITION:
            response = self._handle_paradigm_transition(context, user_input)
        elif research_phase == ResearchPhase.QUALITATIVE_EXPLORATION:
            response = self._handle_qualitative_exploration(context, user_input)
        elif research_phase == ResearchPhase.NARRATIVE_CONSTRUCTION:
            response = self._handle_narrative_construction(context, user_input)
        else:  # PHILOSOPHICAL_INTEGRATION
            response = self._handle_philosophical_integration(context, user_input)
        
        # Log conversation
        self._log_conversation(user_input, response, context)
        
        # Update conversation memory
        self.conversation_memory.append({
            "user_input": user_input,
            "response": response,
            "context": context,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update current context
        self.current_context = context
        
        return response
    
    def _analyze_user_input(self, user_input: str, current_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user input to understand research context and needs"""
        analysis = {
            "research_question": user_input,
            "uncertainty_sources": [],
            "philosophical_questions": [],
            "evolutionary_timescale": None
        }
        
        # Detect uncertainty indicators
        uncertainty_indicators = ["uncertain", "don't know", "unclear", "ambiguous", "conflicting", "inconsistent"]
        if any(indicator in user_input.lower() for indicator in uncertainty_indicators):
            analysis["uncertainty_sources"].append("user_expressed_uncertainty")
        
        # Detect philosophical questions
        philosophical_indicators = ["what is life", "how do we define", "what does it mean", "why does", "what makes"]
        if any(indicator in user_input.lower() for indicator in philosophical_indicators):
            analysis["philosophical_questions"].append(user_input)
        
        # Detect temporal references
        temporal_indicators = ["billion years", "gya", "early earth", "origin of life", "evolution"]
        if any(indicator in user_input.lower() for indicator in temporal_indicators):
            # Extract approximate timescale (simplified)
            if "billion" in user_input.lower():
                # Try to extract number before "billion"
                words = user_input.lower().split()
                for i, word in enumerate(words):
                    if "billion" in word and i > 0:
                        try:
                            analysis["evolutionary_timescale"] = float(words[i-1])
                        except ValueError:
                            analysis["evolutionary_timescale"] = 3.5  # Default to approximate life origin
        
        return analysis
    
    def _identify_research_phase(self, analysis: Dict[str, Any]) -> ResearchPhase:
        """Identify current research phase based on input analysis"""
        
        # Check for explicit uncertainty or limit expressions
        if analysis["uncertainty_sources"]:
            return ResearchPhase.BOUNDARY_IDENTIFICATION
        
        # Check for philosophical questions
        if analysis["philosophical_questions"]:
            return ResearchPhase.PHILOSOPHICAL_INTEGRATION
        
        # Check for evolutionary/narrative content
        if analysis["evolutionary_timescale"] is not None:
            return ResearchPhase.NARRATIVE_CONSTRUCTION
        
        # Default to quantitative analysis
        return ResearchPhase.QUANTITATIVE_ANALYSIS
    
    def _identify_quantitative_limits(
        self, 
        analysis: Dict[str, Any], 
        current_results: Optional[Dict[str, Any]]
    ) -> List[QuantitativeLimitType]:
        """Identify types of quantitative limits encountered"""
        limits = []
        
        # Check user-expressed uncertainty
        if analysis["uncertainty_sources"]:
            limits.append(QuantitativeLimitType.MODEL_UNCERTAINTY)
        
        # Check philosophical questions
        if analysis["philosophical_questions"]:
            limits.append(QuantitativeLimitType.PHILOSOPHICAL_BOUNDARY)
        
        # Check results for high uncertainty
        if current_results:
            # Check for emergence indicators
            if "complexity" in str(current_results).lower():
                limits.append(QuantitativeLimitType.EMERGENCE_THRESHOLD)
            
            # Check for temporal indicators
            if "time" in str(current_results).lower() or analysis.get("evolutionary_timescale"):
                limits.append(QuantitativeLimitType.TEMPORAL_CONTINGENCY)
        
        return limits
    
    def _handle_quantitative_phase(self, context: NarrativeContext, user_input: str) -> Dict[str, Any]:
        """Handle quantitative analysis phase"""
        return {
            "phase": "quantitative_analysis",
            "response": f"I'll help you analyze the quantitative aspects of: {context.research_question}",
            "tools_suggested": ["simulate_planet", "analyze_atmospheric_composition", "calculate_habitability_metrics"],
            "guidance": "Let's start with data-driven analysis to establish baseline measurements and identify patterns.",
            "next_steps": [
                "Gather relevant quantitative data",
                "Run appropriate models and simulations", 
                "Look for patterns and statistical relationships",
                "Monitor for uncertainty indicators that suggest model limits"
            ],
            "philosophical_readiness": "Continue with quantitative approach - no philosophical transitions needed yet."
        }
    
    def _handle_boundary_identification(self, context: NarrativeContext, user_input: str) -> Dict[str, Any]:
        """Handle identification of quantitative boundaries"""
        limit_descriptions = {
            QuantitativeLimitType.MODEL_UNCERTAINTY: "Your models are reaching their predictive limits",
            QuantitativeLimitType.DATA_INSUFFICIENCY: "Available data is insufficient for reliable conclusions",
            QuantitativeLimitType.EMERGENCE_THRESHOLD: "New properties are emerging that models can't predict",
            QuantitativeLimitType.TEMPORAL_CONTINGENCY: "Historical contingency affects outcomes unpredictably",
            QuantitativeLimitType.PHILOSOPHICAL_BOUNDARY: "Fundamental conceptual questions are arising",
            QuantitativeLimitType.COMPLEXITY_OVERFLOW: "System complexity exceeds analytical capabilities"
        }
        
        detected_limits = [limit_descriptions.get(limit, str(limit)) for limit in context.limit_types_encountered]
        
        return {
            "phase": "boundary_identification",
            "response": f"I've identified that your quantitative analysis is reaching important boundaries: {', '.join(detected_limits)}",
            "limits_detected": context.limit_types_encountered,
            "guidance": "This is actually valuable! Recognizing the limits of quantitative analysis is crucial for scientific progress.",
            "transition_recommendation": "Consider transitioning to qualitative and philosophical approaches to address these limitations.",
            "philosophical_frameworks_suggested": self.philosophical_engine.suggest_philosophical_framework(context.limit_types_encountered),
            "next_steps": [
                "Acknowledge the specific nature of these limits",
                "Explore qualitative research methodologies",
                "Consider philosophical frameworks for interpretation",
                "Integrate quantitative findings with broader understanding"
            ]
        }
    
    def _handle_paradigm_transition(self, context: NarrativeContext, user_input: str) -> Dict[str, Any]:
        """Handle paradigm transition guidance"""
        philosophical_guidance = self.philosophical_engine.suggest_philosophical_framework(context.limit_types_encountered)
        
        return {
            "phase": "paradigm_transition",
            "response": "You're at a crucial juncture where quantitative analysis needs to be complemented by qualitative understanding.",
            "transition_guidance": {
                "from": "Database-driven prediction and environmental parameter analysis",
                "to": "Process-oriented understanding and evolutionary narrative construction",
                "why": "Life emerges from billion-year evolutionary processes that transcend snapshot analysis"
            },
            "philosophical_frameworks": philosophical_guidance["recommended_frameworks"],
            "methodological_shifts": philosophical_guidance["methodological_shifts"],
            "integration_advice": philosophical_guidance["integration_advice"],
            "research_questions_to_explore": [
                "How do processes unfold over deep time?",
                "What role does historical contingency play?", 
                "Where do emergent properties come from?",
                "How do we integrate data with narrative understanding?"
            ],
            "next_steps": [
                "Choose a philosophical framework that resonates with your research",
                "Design qualitative research approaches",
                "Begin constructing evolutionary narratives",
                "Integrate quantitative findings with process understanding"
            ]
        }
    
    def _handle_qualitative_exploration(self, context: NarrativeContext, user_input: str) -> Dict[str, Any]:
        """Handle qualitative research exploration"""
        return {
            "phase": "qualitative_exploration", 
            "response": "Let's explore the qualitative dimensions of your research question using process-oriented thinking.",
            "qualitative_approaches": [
                "Process tracing: How did this system develop over time?",
                "Systems thinking: What relationships and feedback loops are involved?",
                "Phenomenological analysis: What is the 'experience' like for biological systems?",
                "Historical analysis: What contingent events shaped this outcome?"
            ],
            "research_methods": [
                "Comparative case studies across different planetary environments",
                "Historical narrative construction of evolutionary transitions",
                "Systems mapping of life-environment interactions",
                "Thought experiments about alternative evolutionary pathways"
            ],
            "philosophical_questions": [
                "What makes this process qualitatively different from others?",
                "How does meaning emerge in biological systems?",
                "What role does agency play at this level of organization?",
                "How do we understand wholes that transcend their parts?"
            ],
            "integration_with_quantitative": "Use your quantitative findings as the foundation for deeper process understanding.",
            "next_steps": [
                "Select appropriate qualitative methods",
                "Begin process-oriented analysis",
                "Construct preliminary narratives",
                "Look for patterns that transcend quantitative measurements"
            ]
        }
    
    def _handle_narrative_construction(self, context: NarrativeContext, user_input: str) -> Dict[str, Any]:
        """Handle evolutionary narrative construction"""
        # Use the narrative builder to construct evolutionary story
        narrative_result = self.narrative_builder.construct_narrative(context, context.quantitative_findings)
        
        return {
            "phase": "narrative_construction",
            "response": "Let's construct a coherent evolutionary narrative that integrates your quantitative findings with deep time understanding.",
            "narrative_framework": narrative_result,
            "storytelling_guidance": {
                "temporal_perspective": f"Consider how this process unfolds over {context.evolutionary_timescale or 'relevant'} billion years",
                "contingency_factors": "Identify critical junctures where different outcomes were possible",
                "emergence_points": "Highlight where new properties or capabilities emerged",
                "coupling_dynamics": "Show how life and environment co-evolved"
            },
            "narrative_structure": {
                "beginning": "Initial conditions and early processes",
                "development": "Key transitions and innovations", 
                "current_state": "Present-day outcomes and patterns",
                "implications": "Broader significance for astrobiology"
            },
            "research_directions": narrative_result["research_directions"],
            "philosophical_insights": narrative_result["philosophical_insights"],
            "next_steps": [
                "Refine the narrative with additional evidence",
                "Explore alternative evolutionary scenarios",
                "Connect to broader astrobiology questions",
                "Consider implications for detecting life elsewhere"
            ]
        }
    
    def _handle_philosophical_integration(self, context: NarrativeContext, user_input: str) -> Dict[str, Any]:
        """Handle philosophical integration and synthesis"""
        return {
            "phase": "philosophical_integration",
            "response": "Let's integrate your research findings with broader philosophical understanding of life and evolution.",
            "integration_approaches": {
                "data_and_meaning": "How do quantitative patterns relate to biological meaning?",
                "process_and_structure": "How do evolutionary processes create stable structures?",
                "individual_and_system": "How do individual components create system-level properties?",
                "time_and_emergence": "How does deep time enable novel forms of organization?"
            },
            "philosophical_synthesis": {
                "life_as_process": "Life is not a thing but a billion-year process of becoming",
                "contingency_and_necessity": "Outcomes reflect both physical laws and historical accidents",
                "emergence_and_reduction": "Wholes possess properties not predictable from parts alone",
                "meaning_and_mechanism": "Biological systems create meaning through their organization"
            },
            "implications_for_astrobiology": [
                "Life detection requires understanding processes, not just chemistry",
                "Alternative life forms may follow different evolutionary narratives",
                "Habitability emerges from dynamic life-environment coupling",
                "Deep time perspective is essential for understanding biological possibility"
            ],
            "research_methodology": "Combine rigorous data analysis with philosophical reflection and narrative construction",
            "next_steps": [
                "Articulate your philosophical position on life and evolution",
                "Connect your findings to broader questions in astrobiology", 
                "Develop research methodologies that honor both data and process",
                "Contribute to ongoing dialogue about the nature of life"
            ]
        }
    
    def _log_conversation(self, user_input: str, response: Dict[str, Any], context: NarrativeContext):
        """Log conversation to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations 
                (timestamp, user_input, assistant_response, research_phase, limit_types, 
                 philosophical_frameworks, narrative_phase)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                user_input,
                json.dumps(response, default=str),
                context.current_phase.value,
                json.dumps([lt.value for lt in context.limit_types_encountered]),
                json.dumps(response.get("philosophical_frameworks", {}), default=str),
                response.get("phase", "unknown")
            ))
            
            conn.commit()
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation progression"""
        return {
            "total_exchanges": len(self.conversation_memory),
            "current_phase": self.current_context.current_phase.value if self.current_context else "unknown",
            "research_question": self.current_context.research_question if self.current_context else "unknown",
            "limits_encountered": [lt.value for lt in self.current_context.limit_types_encountered] if self.current_context else [],
            "recent_exchanges": self.conversation_memory[-3:] if self.conversation_memory else []
        }

def create_enhanced_narrative_chat(
    evolutionary_tracker: Optional[EvolutionaryProcessTracker] = None
) -> EnhancedNarrativeChat:
    """Factory function to create enhanced narrative chat system"""
    return EnhancedNarrativeChat(evolutionary_tracker=evolutionary_tracker) 