# Diffusion Experiment Setup

## Summary
Set up controlled "neuroscience-style" experiment to test hypothesis: agents have higher dispersion/diffusion when resource levels are lower.

## Experiment Design
- **Agents**: 128 from run_from_pretrain_simple checkpoint (simple agent, evolved in Gaussian environment)
- **Environment**: Uniform constant resources (no spatial variation)
- **Control**: Reproduction disabled (cost_reproduce: 10000)
- **Conditions**: 4 resource levels (15, 20, 25, 30)
- **Duration**: 1k steps each
- **Logging**: detailed_interval=50, checkpoint_interval=100

## Hypothesis
Lower resource availability → higher movement/dispersion as agents search for food.

## Files Created

### New World Type
- `src/worlds/uniform.py` - Constant resource level everywhere, no toxin

### World Configs (`configs/world/diffusion/`)
- `r15.yaml` - resource_level: 15
- `r20.yaml` - resource_level: 20
- `r25.yaml` - resource_level: 25
- `r30.yaml` - resource_level: 30

### Run Configs (`configs/run/diffusion/`)
- `r15.yaml`, `r20.yaml`, `r25.yaml`, `r30.yaml`
- All transfer 128 agents from `ckpt_050001.pkl`

### Scripts (`scripts/diffusion/`)
- `r15.sh`, `r20.sh`, `r25.sh`, `r30.sh` - Individual experiments
- `run_all.sh` - Meta script to run all 4

## Also Created This Session
- `configs/world/default_simple.yaml` - Gaussian world for simple agent (no toxin)
- `configs/run/from_pretrain_simple.yaml` - Transfer pretrain to default-like simple env
- `scripts/run_from_pretrain_simple.sh`

## Usage
```bash
# After run_from_pretrain_simple finishes (creates ckpt_050001.pkl):
bash scripts/diffusion/run_all.sh --overwrite
```

## Output
- `data/diffusion_r15/`
- `data/diffusion_r20/`
- `data/diffusion_r25/`
- `data/diffusion_r30/`

## Analysis TODO
- Compare agent position variance/spread over time across conditions
- Calculate mean squared displacement (MSD) to estimate diffusion coefficient
- Plot dispersion vs resource level
