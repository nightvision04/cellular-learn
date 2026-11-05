# CA Lab

Learns the hidden local rule of a spatial process (contagion, SIR epidemics,
terrain-dependent spread) on a 100×100 grid with a genetic algorithm, and
scores the learned rule on held-out and deliberately "tweaked" starting
conditions.

- **Model:** a rule table. For every (current state, terrain type, number of
  infectious Moore neighbours 0–8) it holds the probabilities of the next state.
  Deterministic problems use the most likely next state.
- **Fitting:** evolutionary only (no neural networks). Candidates are scored by
  simulating them against observed data: full grids for deterministic problems,
  district-level counts for random ones.
- **Backends:** a general GPU engine (wgpu: Metal/Vulkan/DX12) that runs every
  simulation of a generation at once, plus a fast multi-core CPU engine. The CPU
  engine is used automatically if no GPU is found or with `--cpu`.

## Run

```bash
cargo build --release
./target/release/ca-lab exp3 site/data --until-converged   # train (exp1, exp2, exp2b, exp3, exp3b, exp4, exp4b)
./target/release/ca-lab ood exp3 site/data                 # tweaked-start generalisation test
./target/release/ca-lab gpubench exp3                      # GPU vs CPU speed + agreement check
python3 -m http.server 8766 --directory site               # live results at http://localhost:8766
```

Run from this folder. Checkpoints go to `checkpoints/`; add `--resume` to continue a run.
