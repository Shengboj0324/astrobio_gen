"""
tool_router.py
==============
Exposes the astrobio pipeline as a callable *tool* for the chat model.

`simulate_planet` accepts JSON:
    { "planet": "Test-Earth",
      "methanogenic_flux": 0.1 }

It returns a short JSON blob that includes the final detectability score.
"""
import json, random
from pathlib import Path
from pipeline.generate_metabolism import generate_metabolism
from pipeline.simulate_atmosphere import simulate_atmosphere
from pipeline.generate_spectrum import generate_spectrum
from pipeline.score_detectability import score_spectrum

BASE_ATM = {"N2": 0.78, "O2": 0.21, "CO2": 0.01}
DUMMY_PLANETS = {
    p["name"]: p
    for p in json.load(open(Path("data/dummy_planets.csv").with_suffix(".json"), "r"))
} if (Path("data/dummy_planets.csv").with_suffix(".json")).exists() else {}

def simulate_planet(planet: str, methanogenic_flux: float = 0.1):
    p = DUMMY_PLANETS.get(planet, {"name": planet})
    env = [random.random() for _ in range(4)]
    net, flux = generate_metabolism(env)
    # overwrite CH4 flux with user value
    flux["CH4"] = methanogenic_flux
    atm = simulate_atmosphere(BASE_ATM, flux)
    w, f = generate_spectrum(atm)
    _, D = score_spectrum(w, f)
    return {
        "planet": planet,
        "flux": flux,
        "detectability": round(D, 3),
    }

# metadata JSON the LLM will see
simulate_planet.openai_tool = {
    "name": "simulate_planet",
    "description": "Run the astrobio pipeline for a named toy planet.",
    "parameters": {
        "type": "object",
        "properties": {
            "planet": {"type": "string", "description": "Planet name"},
            "methanogenic_flux": {
                "type": "number",
                "description": "Surface CH4 flux (0–1 arbitrary units)",
            },
        },
        "required": ["planet"],
    },
}