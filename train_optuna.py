import json
import pathlib
import subprocess

import optuna
import yaml

BASE = yaml.safe_load(pathlib.Path("config/defaults.yaml").read_text())


def objective(trial):
    cfg = {**BASE}
    cfg["model"]["fusion"]["latent_dim"] = trial.suggest_int("latent", 64, 256, step=32)
    cfg["trainer"]["max_epochs"] = trial.suggest_int("epochs", 100, 400, step=100)
    yml = pathlib.Path("tmp.yml")
    yml.write_text(yaml.dump(cfg))
    proc = subprocess.run(["python", "train.py", "--config", yml], capture_output=True)
    # Lightning saves metrics to last row of val_loss.csv
    loss = float(proc.stdout.strip().split()[-1])
    return loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)
print("best", study.best_params)
