import optuna, subprocess, json, pathlib, random, yaml, shutil, os

def objective(trial):
    latent  = trial.suggest_int("latent",4,32,step=4)
    epochs  = trial.suggest_int("epochs",50,200,step=50)
    limit   = 10000
    cmd = ["python","train.py","--model","vae","--epochs",str(epochs),"--limit",str(limit)]
    env = os.environ.copy(); env["LATENT"]=str(latent)
    p = subprocess.run(cmd, capture_output=True, env=env, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("loss:"):
            loss=float(line.split()[-1]); break
    return loss

study=optuna.create_study(direction="minimize")
study.optimize(objective,n_trials=10)
print("Best", study.best_params)