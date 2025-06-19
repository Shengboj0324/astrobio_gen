"""End-to-end dummy run (uses FAST_MODE by default)."""
import json, random
from utils.data_utils import load_dummy_planets
from pipeline.generate_metabolism import generate_metabolism
from pipeline.simulate_atmosphere import simulate_atmosphere
from pipeline.generate_spectrum import generate_spectrum
from pipeline.score_detectability import score_spectrum
from pipeline.rank_planets import rank

BASE_ATM = {"N2": 0.78, "O2": 0.21, "CO2": 0.01}

def main():
    planets = load_dummy_planets()
    out = []
    for pl in planets:
        env = [random.random() for _ in range(4)]
        net, flux = generate_metabolism(env)
        atm = simulate_atmosphere(BASE_ATM, flux)
        w, f = generate_spectrum(atm, planet=pl)
        _, D = score_spectrum(w, f)
        out.append({**pl, "D": D, "P": 0.8, "F": 0.9})
    print(json.dumps(rank(out), indent=2))

if __name__ == "__main__":
    main()