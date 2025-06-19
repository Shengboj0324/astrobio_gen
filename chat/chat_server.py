
from pathlib import Path
import os, json

from langchain_community.llms import LlamaCpp
from langchain.tools import StructuredTool
from langchain.agents import initialize_agent, AgentType
from tool_router import simulate_planet

MODEL_PATH = "models/mistral-7b-instruct.Q4_K.gguf"
if not Path(MODEL_PATH).exists():
    raise SystemExit("Place a GGUF model at models/*.gguf first.")

llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=os.cpu_count(),
    temperature=0.2,
    streaming=True,
    verbose=False,
)

tools = [
    StructuredTool.from_function(simulate_planet),
]

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

print("🛰️  Chat ready.  Type 'exit' to quit.")
while True:
    try:
        q = input("\nYou: ")
        if q.strip().lower() == "exit":
            break
        print("Bot:", end=" ", flush=True)
        ans = agent.run(q)
        print(ans)
    except KeyboardInterrupt:
        break