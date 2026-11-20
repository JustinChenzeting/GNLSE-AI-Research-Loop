# Run the notebooks

The notebooks are grouped by the input being optimized, then by search method. Start with [single-channel launch power, 20–200 kW](../notebooks/Total%20launch%20power/Gradient%20descent/dio_power_optimization_20_200kw.ipynb) to run a classical experiment. It varies LP01 launch power and reports input–output distortion and the fraction of that launch’s LP01 photons recovered at the output.

### Install the environment

These commands assume macOS or Linux with Git, Python 3.12 and Julia 1.12.5 on your path. Run them in a terminal. Keep the Python environment active when starting Jupyter.

```sh
git clone https://github.com/JustinChenzeting/GNLSE-AI-Research-Loop.git
cd GNLSE-AI-Research-Loop
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r notebook_helpers/requirements.txt
python -m ipykernel install --user --name gnlse-python --display-name "GNLSE (Python)"
```

The optics notebooks use Prof. Rivera’s Julia solver. Install the revision used for these experiments inside the checkout. The polynomial demo requires only the Python environment and Codex.

```sh
git clone https://github.com/nrivera494/FastDifferentiableMMNLO.git
git -C FastDifferentiableMMNLO checkout --detach 9c1288712ea9cd7ef858d4e6dff9dea350e9291a
export PYTHON="$(command -v python)"
export JUPYTER="$(command -v jupyter)"
julia --project=FastDifferentiableMMNLO -e 'using Pkg; Pkg.instantiate(); Pkg.build("PyCall"); using PulsePropagation, PyPlot'
julia -e 'using Pkg; Pkg.add("IJulia"); using IJulia; IJulia.installkernel("GNLSE", "--project=$(abspath("FastDifferentiableMMNLO"))"; specname="gnlse-julia", displayname="GNLSE (Julia)")'
```

### Run an experiment

From the repository root, run `jupyter lab`, open a file under `notebooks/`, and run its cells from the top. Use **GNLSE (Julia)** for classical power and modal gradient descent. The research loop, Bayesian optimization and spectral-mask notebooks use **GNLSE (Python)**. Python starts Julia for optical simulations. Set `JULIA` to the Julia executable if it is outside your path.

AI-guided notebooks also need the Codex CLI on your path and an authenticated account. Run `codex login` once and check with `codex login status`. The model settings in the notebooks can be changed for your account. Classical gradient descent and grid sweeps run without Codex.

Each experiment prints its local output directory. The spectral-mask sweeps can write many gigabytes of fields and have default time budgets of 90 minutes (balanced search) and 120 minutes (six-band sweep). A time-limited run saves its measurements. For optional comparisons, supply the output directory of a matching local run. The gradient-descent continuation accepts a two-step pilot directory or creates a pilot first.
