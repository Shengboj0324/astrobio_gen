"""Step 3 – convert gas flux → steady-state mixing ratios (box model)."""
from typing import Dict


def simulate_atmosphere(base: Dict[str, float], flux: Dict[str, float]) -> Dict[str, float]:
    comp = base.copy()
    for gas, rate in flux.items():
        comp[gas] = comp.get(gas, 0) + min(0.2, rate)  # cap so total ≤1
    tot = sum(comp.values())
    return {g: v / tot for g, v in comp.items()}


if __name__ == "__main__":
    b = {"N2": 0.78, "O2": 0.21, "CO2": 0.01}
    print(simulate_atmosphere(b, {"CH4": 0.02, "O2": 0.05}))