# Week 3 — Bayesian Optimization for PD Tuning

Scripts:

- `BO_for_pid_tuning.py` — Bayesian optimization (scikit-optimize's `gp_minimize`) over the PD gains of a feedback-linearized joint controller; the objective is the tracking error of a 10-second episode.
- `gp_functions.py` — GP fitting/plotting helpers used by the main script.

## Prerequisites (once per machine)

1. Clone RoboEnv with submodules and install its pixi environment:

   ```bash
   git clone --recurse-submodules https://github.com/VModugno/RoboEnv.git
   cd RoboEnv && pixi install
   pixi run smoke-test   # → prints "simulation_and_control OK"
   ```

`scikit-optimize` is already part of RoboEnv's default environment — nothing extra to install.

## One-time setup for this folder

`configs/` and `models/` are intentionally not part of the lab repo. Copy or link them from your RoboEnv checkout:

**Windows (PowerShell):**

```powershell
$robo = "C:\path\to\RoboEnv"

Copy-Item "$robo\configs\pandaconfig.json" .\configs\ -Force
New-Item -ItemType Junction -Path .\models -Target "$robo\models"
```

**macOS / Linux:**

```bash
robo=/path/to/RoboEnv
mkdir -p configs && cp "$robo"/configs/pandaconfig.json configs/
ln -s "$robo"/models models
```

## Run

From the **RoboEnv root**:

```bash
pixi run python "<full path>/week_3/2026/BO_for_pid_tuning.py"
```

or activate once and work from this folder:

```bash
cd RoboEnv && pixi shell
cd "<full path>/week_3/2026"
python BO_for_pid_tuning.py
```

## What to expect

- The pybullet GUI opens and runs **10 optimization episodes** (10 simulated seconds each), so the whole run takes a few minutes of wall-clock time.
- Each episode prints its `Tracking error` and the candidate `kp`/`kd` vectors.
- At the end the script prints the optimal gains, e.g.:

  ```
  Optimal Kp: [...], Optimal Kd: [...]
  ```

  and shows GP regression plots for `kp0` and `kd0`.
- A scikit-learn `ConvergenceWarning` about the `length_scale` bound is benign — feel free to experiment with the bounds.

## Stopping

The script terminates by itself after `n_calls=10` evaluations. Press `q` inside the pybullet window if you ever need to bail out mid-run.