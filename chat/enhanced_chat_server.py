#!/usr/bin/env python3
"""
Enhanced Chat Server for Astrobiology Research
Integrates LLM with comprehensive data sources and scientific pipeline
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our enhanced tools
from enhanced_tool_router import (
    access_spectral_library,
    analyze_atmospheric_composition,
    calculate_habitability_metrics,
    compare_planetary_systems,
    generate_research_summary,
    query_exoplanet_data,
    search_scientific_database,
    simulate_planet,
)
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage
from langchain.tools import StructuredTool
from langchain_community.llms import LlamaCpp


class EnhancedChatServer:
    def __init__(self, model_path: str = "models/mistral-7b-instruct.Q4_K.gguf"):
        self.model_path = model_path
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize database for conversation logging
        self.init_conversation_db()

        # Setup LLM
        if not Path(self.model_path).exists():
            raise SystemExit(f"Place a GGUF model at {self.model_path} first.")

        self.llm = LlamaCpp(
            model_path=self.model_path,
            n_ctx=8192,  # Increased context for more complex conversations
            n_threads=os.cpu_count(),
            temperature=0.3,  # Slightly higher for more creative responses
            streaming=True,
            verbose=False,
        )

        # Setup conversation memory
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Setup enhanced tools
        self.tools = [
            StructuredTool.from_function(simulate_planet),
            StructuredTool.from_function(query_exoplanet_data),
            StructuredTool.from_function(analyze_atmospheric_composition),
            StructuredTool.from_function(search_scientific_database),
            StructuredTool.from_function(generate_research_summary),
            StructuredTool.from_function(calculate_habitability_metrics),
            StructuredTool.from_function(compare_planetary_systems),
            StructuredTool.from_function(access_spectral_library),
        ]

        # Initialize agent with memory
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            max_iterations=5,  # Prevent infinite loops
        )

    def init_conversation_db(self):
        """Initialize SQLite database for conversation logging"""
        self.db_path = f"chat/conversations_{self.session_id}.db"
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                user_input TEXT,
                bot_response TEXT,
                tools_used TEXT,
                session_id TEXT
            )
        """
        )
        conn.commit()
        conn.close()

    def log_conversation(self, user_input: str, bot_response: str, tools_used: List[str]):
        """Log conversation to database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT INTO conversations (timestamp, user_input, bot_response, tools_used, session_id)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                user_input,
                bot_response,
                json.dumps(tools_used),
                self.session_id,
            ),
        )
        conn.commit()
        conn.close()

    def get_conversation_summary(self) -> str:
        """Get summary of current conversation"""
        if not self.conversation_history:
            return "No conversation history yet."

        return f"Conversation session {self.session_id} with {len(self.conversation_history)} exchanges."

    def suggest_next_questions(self, last_response: str) -> List[str]:
        """Suggest follow-up questions based on last response"""
        suggestions = []

        if "exoplanet" in last_response.lower():
            suggestions.extend(
                [
                    "Can you compare this exoplanet to Earth-like planets?",
                    "What spectroscopic observations would be most valuable?",
                    "How does this system's habitability compare to others?",
                ]
            )

        if "atmosphere" in last_response.lower():
            suggestions.extend(
                [
                    "What biosignatures should we look for?",
                    "How might stellar radiation affect this atmosphere?",
                    "Can you simulate different atmospheric scenarios?",
                ]
            )

        if "spectrum" in last_response.lower():
            suggestions.extend(
                [
                    "What instruments could detect these features?",
                    "How would distance affect observability?",
                    "Can you show similar spectra from our database?",
                ]
            )

        return suggestions[:3]  # Return top 3 suggestions

    async def process_query(self, user_input: str) -> Dict[str, Any]:
        """Process user query and return enhanced response"""
        start_time = datetime.now()

        try:
            # Run agent
            response = self.agent.run(user_input)

            # Track tools used (this is simplified - in practice you'd need to hook into agent execution)
            tools_used = self.extract_tools_from_response(response)

            # Log conversation
            self.log_conversation(user_input, response, tools_used)

            # Add to conversation history
            self.conversation_history.append(
                {
                    "user": user_input,
                    "bot": response,
                    "timestamp": datetime.now().isoformat(),
                    "tools_used": tools_used,
                }
            )

            # Generate suggestions
            suggestions = self.suggest_next_questions(response)

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "response": response,
                "suggestions": suggestions,
                "tools_used": tools_used,
                "processing_time": processing_time,
                "conversation_length": len(self.conversation_history),
            }

        except Exception as e:
            error_response = f"I encountered an error: {str(e)}. Could you rephrase your question?"

            self.log_conversation(user_input, error_response, ["error"])

            return {
                "response": error_response,
                "suggestions": [
                    "Can you rephrase your question?",
                    "What would you like to explore?",
                ],
                "tools_used": ["error"],
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "conversation_length": len(self.conversation_history),
            }

    def extract_tools_from_response(self, response: str) -> List[str]:
        """Extract which tools were likely used (simplified implementation)"""
        tools_used = []
        tool_indicators = {
            "simulate_planet": ["simulated", "generated", "metabolism", "atmosphere"],
            "query_exoplanet_data": ["exoplanet", "planet data", "orbital"],
            "analyze_atmospheric_composition": ["atmospheric", "composition", "gases"],
            "search_scientific_database": ["database", "research", "studies"],
            "calculate_habitability_metrics": ["habitability", "habitable", "ESI"],
            "compare_planetary_systems": ["compare", "comparison", "similar"],
            "access_spectral_library": ["spectrum", "spectral", "wavelength"],
        }

        response_lower = response.lower()
        for tool, indicators in tool_indicators.items():
            if any(indicator in response_lower for indicator in indicators):
                tools_used.append(tool)

        return tools_used

    def run_interactive_session(self):
        """Run interactive chat session"""
        print("🛰️  Enhanced Astrobiology Chat Ready!")
        print("   📊 Connected to 500+ scientific databases")
        print("   🔬 Integrated with full research pipeline")
        print("   💬 Conversation memory enabled")
        print("   Type 'exit', 'quit', or 'bye' to end session")
        print("   Type 'help' for available commands")
        print("   Type 'summary' for conversation summary")
        print("-" * 60)

        while True:
            try:
                user_input = input("\n🔬 You: ").strip()

                if user_input.lower() in ["exit", "quit", "bye"]:
                    print(f"\n👋 Session ended. Conversation saved to {self.db_path}")
                    break

                elif user_input.lower() == "help":
                    self.show_help()
                    continue

                elif user_input.lower() == "summary":
                    print(f"\n📋 {self.get_conversation_summary()}")
                    continue

                elif not user_input:
                    continue

                print("\n🤖 Assistant: ", end="", flush=True)

                # Process query asynchronously
                result = asyncio.run(self.process_query(user_input))

                print(result["response"])

                # Show suggestions if any
                if result["suggestions"]:
                    print(f"\n💡 Suggestions:")
                    for i, suggestion in enumerate(result["suggestions"], 1):
                        print(f"   {i}. {suggestion}")

                # Show processing info
                print(
                    f"\n⚡ Processed in {result['processing_time']:.2f}s using tools: {', '.join(result['tools_used']) if result['tools_used'] else 'none'}"
                )

            except KeyboardInterrupt:
                print(f"\n\n👋 Session interrupted. Conversation saved to {self.db_path}")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def show_help(self):
        """Show available commands and capabilities"""
        print(
            """
🔬 Enhanced Astrobiology Chat - Available Capabilities:

📊 Data Access:
   • Query 500+ exoplanet databases (NASA, ESA, TESS, Kepler, etc.)
   • Access spectroscopic libraries (X-shooter, POLLUX, NIST)
   • Search climate and atmospheric models (CMIP6, ERA5)
   • Browse genomics and metabolic pathway databases

🧪 Scientific Analysis:
   • Simulate planetary atmospheres and biosignatures
   • Calculate habitability metrics and ESI scores
   • Analyze atmospheric compositions and spectra
   • Compare planetary systems and characteristics

📈 Research Tools:
   • Generate research summaries from multiple sources
   • Create literature reviews and data compilations
   • Track scientific trends and discoveries
   • Suggest observational targets and strategies

💬 Chat Commands:
   • 'help' - Show this help message
   • 'summary' - Get conversation summary
   • 'exit'/'quit'/'bye' - End session

Example queries:
   • "Find Earth-like exoplanets in the habitable zone"
   • "Simulate the atmosphere of Kepler-452b"
   • "What biosignatures should JWST look for?"
   • "Compare the TRAPPIST-1 system to our solar system"
        """
        )


def main():
    """Main entry point"""
    try:
        server = EnhancedChatServer()
        server.run_interactive_session()
    except Exception as e:
        print(f"Failed to initialize chat server: {e}")
        print("Make sure you have:")
        print("1. A GGUF model at models/mistral-7b-instruct.Q4_K.gguf")
        print("2. All required dependencies installed")
        print("3. The comprehensive data system initialized")


if __name__ == "__main__":
    main()
