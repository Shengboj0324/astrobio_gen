"""Step 5b – rank by Score = D × P × F."""
def rank(pl_list):
    for p in pl_list:
        p["score"] = p["D"] * p["P"] * p["F"]
    return sorted(pl_list, key=lambda d: d["score"], reverse=True)