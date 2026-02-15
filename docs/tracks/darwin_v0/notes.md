# Darwin V0 Track Notes

This is the original darwin simulation track, reorganized into the track structure on 2026-01-13.

## Key Components

- **Simulation**: JAX-based 2D grid world with LSTM-brained agents
- **Evolution**: Offspring inherit parent weights + Gaussian mutation (std=0.1)
- **World Types**: Gaussian, bridge, maze, thermotaxis, orbiting blobs, etc.

## Directory Structure

```
src/darwin_v0/           # Python source code
configs/darwin_v0/       # YAML configs (run/ and world/)
scripts/darwin_v0/       # Bash execution scripts
data/darwin_v0/          # Experiment outputs
```

## Completed Experiments

See `docs/research_context.md` for full details on:
- Diffusion experiments (resource-dependent movement)
- Genotype cluster behavioral divergence (V1-V4)
- Thermotaxis experiments
- Linear chemotaxis experiments

## Key Findings

- **The Turning Problem**: Evolution eliminated turning (0% left/right) because turns cost energy with no immediate benefit. Without turning, agents can't do gradient following.
- **Allopatric Speciation**: Geographic separation drove cluster divergence, with Cluster 5 ("circlers") vs Cluster 17 ("explorers") showing opposite turning chirality.

## Current State (2026-01-13)

Track is stable. All experiments completed. No active development - this serves as the baseline for potential future tracks.
