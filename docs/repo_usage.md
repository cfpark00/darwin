# Repository Usage Guide

## Core Development Principles

### 1. Research Integrity First - Fail Fast Philosophy
- **This is a research repo**: Hidden invisible fallbacks are *criminal* to research integrity
- **Failing is fine**: NEVER make code do something secretly without being obvious
- **Stay explicit**: Implicit behavior corrupts experiments and wastes time
- **No silent failures**: Crash immediately on missing configs or invalid states
- **No fallbacks**: Required parameters must be explicitly provided
- **Loud failures**: Better to crash in 1 second than run for 10 hours with wrong behavior
- **Why**: Silent failures waste compute hours and corrupt research results

### 2. Implementation vs Orchestration
- **Implementation (HOW)**: Lives in `src/<track>/` modules - reusable functions and core logic
- **Orchestration (WHAT/WHEN)**: Lives in `src/<track>/scripts/` - experiment flow and coordination at Python level
- **Bash scripts in `/scripts/<track>/`**: Minimal wrappers that just call Python scripts, e.g., `uv run python src/behavioral/scripts/run_benchmark.py configs/behavioral/benchmark.yaml`

### 3. Abstraction Guidelines
- **Just the right amount**: Do not over-abstract, aim for clarity over cleverness
- **Consult before major changes**: When introducing new structures/abstractions, ask first
- **Watch for spaghetti**: If code is getting tangled, stop and discuss restructuring

### 4. Script Legibility
- Scripts should read like a story - each line is a meaningful step
- Everything happens in `main()` function
- Clear, sequential flow from config loading to execution

## Research Tracks

### Why Tracks?

Research is fundamentally different from product development:

- **Product development**: Linear progression where v1 → v2 → v3, each replacing the last
- **Research**: Parallel exploration where you try approach A, gain an insight, then try completely different approach B, maybe C branches from A's insight but uses B's infrastructure...

Unlike git branches where you eventually merge or abandon, research tracks often:
- Run in parallel indefinitely
- Inform each other without merging
- Have completely different code, not just different parameters
- Need to remain runnable even when you're focused on another track

### What is a Track?

A **track** is a self-contained research direction with its own code, configs, scripts, and outputs:

```
src/<track_name>/           # Track-specific implementation
configs/<track_name>/       # Track-specific configs
scripts/<track_name>/       # Track-specific bash scripts
data/<track_name>/          # Track-specific outputs (gitignored)
docs/tracks/<track_name>/   # Track-specific documentation
```

**Example with two tracks:**
```
src/
├── utils.py                # Cross-track utilities (stable API)
├── behavioral/             # Behavioral analysis track
│   ├── utils.py            # Track-specific utilities
│   ├── dat_scorer.py
│   └── scripts/
│       └── run_benchmark.py
└── mechanistic/            # Mechanistic analysis track
    ├── utils.py            # Track-specific utilities
    ├── circuit_analysis.py
    └── scripts/
        └── run_intervention.py

configs/
├── behavioral/
│   └── benchmark_gemini.yaml
└── mechanistic/
    └── ablation_study.yaml

scripts/
├── behavioral/
│   └── run_benchmark.sh
└── mechanistic/
    └── run_ablation.sh

data/
├── behavioral/
│   └── gemini_benchmark_001/
└── mechanistic/
    └── ablation_run_001/

docs/
├── repo_usage.md           # Cross-track documentation
├── research_context.md     # Overall research context
└── tracks/
    ├── behavioral/         # Track-specific docs
    │   ├── notes.md        # Tacit knowledge, decisions, insights
    │   └── results.md      # Key findings summary
    └── mechanistic/
        └── notes.md
```

### Track Documentation

Each track should have a `docs/tracks/<track_name>/` folder for:
- **Tacit knowledge**: Design decisions, why certain approaches were chosen/abandoned
- **Notes**: Observations, insights, things you'd forget in a month
- **Results summaries**: Key findings without digging through data/
- **Gotchas**: Known issues, edge cases, things that surprised you

This documentation is invaluable when:
- Returning to a track after working on another
- Explaining the track to collaborators
- Deciding whether to build on or abandon a track

### When to Create a New Track

**Create a new track when:**
- You're exploring a fundamentally different approach or question
- The code structure/abstractions would be significantly different
- You want to preserve a working approach while experimenting with another
- Insights from one direction lead to a completely different implementation

**Do NOT create a new track for:**
- Parameter variations (use different configs instead)
- Bug fixes
- Adding new models/providers to existing infrastructure
- Incremental improvements to existing approach

### Cross-Track Rules

1. **Each track is self-contained** - no imports between tracks
2. **Shared utilities only in `src/utils.py`** - and these must have stable APIs (see below)
3. **Tracks can read each other's data** - for comparison/analysis, but not modify
4. **No track is "primary"** - they're parallel, not hierarchical

## Script Organization

### Required Config Structure
- All configs **MUST** have `output_dir` field - this is non-negotiable
- All scripts should *only* modify the `output_dir` specified in config (except system-level cache/temp files)

### Recommended Script Interface
```bash
python src/scripts/<script_name>.py configs/<config_file>.yaml --overwrite --debug
```
- **Only two runtime flags are recommended:**
  - `--overwrite`: Whether to overwrite existing `output_dir`
  - `--debug`: Debug mode for temporary testing
- Everything else should be in the config file

### Bash Scripts
- All .sh scripts must be in `scripts/<track>/`
- Each track has its own scripts folder: `scripts/behavioral/`, `scripts/mechanistic/`, etc.
- All .sh scripts should be as minimal as possible, mostly they simply keep track of what .py scripts should be ran to recreate the experiment
- **IMPORTANT: Always pass through arguments using `"$@"`** to allow flags like `--overwrite` and `--debug` to be passed from bash to Python
- Example:
  ```bash
  #!/bin/bash
  uv run python src/behavioral/scripts/run_benchmark.py configs/behavioral/benchmark.yaml "$@"
  ```
  This allows you to run: `bash scripts/behavioral/run_benchmark.sh --overwrite --debug`

### Config Files
- All configs must be in `configs/<track>/`
- Each track has its own configs folder: `configs/behavioral/`, `configs/mechanistic/`, etc.
- Within a track, you can further organize by experiment type if needed:
  - `configs/behavioral/benchmarks/`
  - `configs/behavioral/ablations/`
- This organization makes it easy to find and manage configs per track

### Python Scripts
- Orchestration scripts (entry points) live in `src/<track>/scripts/`
- Implementation modules live in `src/<track>/` (but not in `scripts/` subdirectory)
- All orchestration scripts *must* read in *exactly* a `config_path` argument (required)
- Optional flags: `--overwrite` (default False) and `--debug` (default False)
- All orchestration scripts *must* write *all* results to the `output_dir` mentioned by config (error out if this is not given by the config)
- All orchestration scripts should validate the output directory and handle overwrite logic appropriately
- **All scripts should copy the config YAML file to the output directory immediately after creating it** (recommended to save as `config.yaml` for consistency) - this ensures reproducibility and keeps a record of exact parameters used for each run

## Practical Guidelines

### Config Validation
```python
# Always validate upfront - fail fast!
def validate_config(config):
    # Check for required output_dir FIRST
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")

    # Check other required fields
    required_fields = ['experiment', 'model']  # Add your requirements
    for field in required_fields:
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")

    # Validate nested fields explicitly
    if 'learning_rate' not in config.get('training', {}):
        raise ValueError("FATAL: 'training.learning_rate' is required")
```

### Error Handling
```python
# ❌ BAD: Silent fallback
if config is None:
    print("Warning: config missing")
    return default_value

# ✅ GOOD: Immediate failure
if config is None:
    raise ValueError("FATAL: config is required")

# ✅ GOOD: Exit immediately on critical failures
import sys
if not Path(data_path).exists():
    print(f"Error: Data path {data_path} does not exist!")
    sys.exit(1)
```

### Output Directory Management
```python
# Use the safe init_directory utility from src/utils.py
# IMPORTANT: Scripts in src/scripts/ need to add project root to path first
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import init_directory

# In your orchestration script
output_dir = init_directory(config['output_dir'], overwrite=args.overwrite)

# Create standard subdirectories
(output_dir / 'figures').mkdir(parents=True, exist_ok=True)
(output_dir / 'results').mkdir(parents=True, exist_ok=True)
(output_dir / 'logs').mkdir(parents=True, exist_ok=True)
```

### Code Organization

**Utilities hierarchy:**
- **`src/utils.py`** (cross-track): Stable, rarely-touched functions used across multiple tracks
  - API must remain stable - changing signatures breaks multiple tracks
  - Examples: `init_directory()`, `load_config()`, generic file I/O helpers
  - Think of these as "infrastructure" - boring but reliable
- **`src/<track>/utils.py`** (track-specific): Utilities specific to one track
  - Can evolve freely without breaking other tracks
  - Examples: track-specific data loaders, formatting functions, domain helpers

**Code placement:**
- **Orchestration scripts**: Put in `src/<track>/scripts/` - entry points that coordinate experiments
- **Implementation modules**: Put in `src/<track>/` - track-specific logic and classes
- Don't over-modularize - functions that do one complete operation are fine
- Trust framework features (e.g., use HuggingFace's TrainingArguments instead of custom schedulers)

### Resources Directory Usage
The `resources/` folder is for reference materials that inform your research:

- **Git repositories**: Clone external repos you're studying or referencing
- **Papers**: Store PDFs of papers relevant to the research
- **Documentation**: Keep external docs, specifications, or references
- **Examples**: Reference implementations or code samples

```bash
# Example structure
resources/
├── repos/
│   └── some-reference-repo/
├── papers/
│   ├── attention-is-all-you-need.pdf
│   └── scaling-laws.pdf
└── docs/
    └── api-spec.md
```

This folder is gitignored - it's for local reference only, not committed to the repo.

### Scratch Directory Usage
**CRITICAL: NEVER place files directly in `scratch/`!**

- **Always create a subfolder** within `scratch/` for your temporary work: `scratch/{subfolder_name}/`
- **Why this matters**: Direct files in scratch/ create clutter and make it impossible to track what belongs to what task
- **Examples of proper usage:**
  - `scratch/test_visualization/` - for testing plotting code
  - `scratch/debug_model/` - for debugging model behavior
  - `scratch/quick_analysis/` - for one-off data checks
- **Everything in these subfolders is gitignored** (except .gitkeep in scratch/ itself)
- **Clean up subfolders when done** - delete entire subdirectories when the temporary work is complete

```bash
# ✅ CORRECT: Create subfolder for temporary work
mkdir scratch/my_test
touch scratch/my_test/temp_script.py
# ... do your work ...
rm -rf scratch/my_test  # Clean up when done

# ❌ WRONG: Never do this!
touch scratch/temp_script.py  # DO NOT put files directly in scratch/
```

### Running Scripts
Always execute from project root:
```bash
# Via bash script (recommended)
bash scripts/behavioral/run_benchmark.sh
bash scripts/behavioral/run_benchmark.sh --overwrite --debug

# Direct Python execution
uv run python src/behavioral/scripts/run_benchmark.py configs/behavioral/benchmark.yaml

# With flags
uv run python src/behavioral/scripts/run_benchmark.py configs/behavioral/benchmark.yaml --overwrite

# From activated environment
python src/mechanistic/scripts/run_ablation.py configs/mechanistic/ablation.yaml
```

## Quick Decision Guide

**Track decisions:**
- "Is this a fundamentally different approach?" → New track in `src/<new_track>/`
- "Is this a parameter/config variation?" → Same track, new config
- "Is this a bug fix or incremental improvement?" → Same track, same code

**Code placement:**
- "Could ALL tracks reuse this with a stable API?" → `src/utils.py`
- "Could experiments within THIS track reuse this?" → `src/<track>/utils.py`
- "Is this specific to one experiment?" → Orchestration script in `src/<track>/scripts/`
- "Is this HOW to do something?" → Implementation in `src/<track>/` modules
- "Is this WHEN/WHETHER to do something?" → Orchestration in `src/<track>/scripts/`

## Standard Script Template

For a script at `src/<track>/scripts/run_experiment.py`:

```python
import argparse
import yaml
from pathlib import Path

# Cross-track utilities (stable API)
from src.utils import init_directory

# Track-specific utilities (can import from parent track module)
# from src.<track>.utils import some_helper

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config - fail fast!
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required in config")

    # Initialize output directory with safety checks
    output_dir = init_directory(config['output_dir'], overwrite=overwrite)

    # Create standard subdirectories
    (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
    (output_dir / 'results').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Debug mode handling
    if debug:
        print(f"DEBUG MODE: Output will be written to {output_dir}")
        # Add any debug-specific behavior

    # Your experiment code here

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode for testing')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)
```

**Corresponding bash script** at `scripts/<track>/run_experiment.sh`:
```bash
#!/bin/bash
uv run python src/<track>/scripts/run_experiment.py configs/<track>/experiment.yaml "$@"
```

## Downstream Scripts Pattern

### What is Downstream Work?

Downstream scripts operate on the output of a previous run. This includes:
- **Analysis**: plotting, statistics, summarization
- **Mech interp**: probing, activation analysis, circuit discovery
- **Fine-tuning**: further training on a checkpoint
- **Data processing**: extracting subsets, reformatting, distribution analysis

The key idea: run something expensive once, then do multiple cheaper downstream operations on it.

### Config Structure for Downstream Scripts

```yaml
upstream_dir: "data/my_experiment/run_001"  # Points to existing run
output_dir: "data/my_experiment/run_001/downstream/mech_interp"  # Subdirectory within upstream
```

### Key Principles

- `upstream_dir`: Points to the **run directory** containing results to work on
- `output_dir`: Points to where **downstream outputs** should be saved (typically `upstream_dir/downstream/<name>/`)
- Downstream scripts **never modify** the upstream data
- `--overwrite` flag only affects the downstream subdirectory, not the upstream
- **Composable**: downstream of downstream is natural - just point `upstream_dir` to a previous downstream output

### Example Downstream Script

```python
def main(config_path, overwrite=False):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    if 'upstream_dir' not in config:
        raise ValueError("FATAL: 'upstream_dir' required for downstream work")
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' required")

    upstream_dir = Path(config['upstream_dir'])
    output_dir = Path(config['output_dir'])

    # Check upstream results exist
    if not upstream_dir.exists():
        raise FileNotFoundError(f"Upstream dir not found: {upstream_dir}")

    # Create downstream output directory (only overwrites this downstream, not upstream!)
    if output_dir.exists() and not overwrite:
        raise ValueError(f"Output {output_dir} exists. Use --overwrite to replace.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load upstream results and do downstream work
    # ... your code here ...

    # Save downstream outputs
    # ...
```

### Example Directory Structure

```
data/my_experiment/run_001/
├── config.yaml              # Original run config (untouched)
├── results/                 # Original results (untouched)
├── logs/                    # Original logs (untouched)
└── downstream/
    ├── mech_interp/         # First downstream
    │   ├── config.yaml
    │   ├── results/
    │   └── downstream/      # Downstream of downstream!
    │       └── attention_heads/
    │           └── ...
    ├── fine_tuning/         # Another downstream of run_001
    │   └── ...
    └── data_distribution/   # Yet another
        └── ...
```

### Why This Pattern

- **Preserves integrity**: upstream data is never modified
- **Multiple downstream**: run many different operations on the same upstream
- **Composable**: downstream of downstream works naturally
- **Re-runnable**: `--overwrite` lets you iterate on downstream without re-running expensive upstream
- **Clear lineage**: directory structure shows what depends on what

## Key Takeaways
- **Implementation is HOW. Orchestration is WHAT/WHEN/WHETHER.**
- **Research integrity demands explicit behavior** - no hidden magic
- **Better to crash loudly than fail silently**
- **When in doubt, be explicit and ask before abstracting**

## IMPORTANT
- NEVER create file variations like _v2, _new, _final, _updated. Use git for versioning.
- Always follow the existing project structure as defined in CLAUDE.md
- Config files must specify `output_dir` - this is non-negotiable
- All data outputs go to the path specified in `output_dir` in the config
