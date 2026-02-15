darwin/
├── CLAUDE.md                    # Development guidelines
├── README.md                    # Project readme
├── pyproject.toml               # Python dependencies (uv)
├── uv.lock                      # Locked dependencies
├── .env                         # Environment config (DATA_DIR)
├── .env.example                 # Environment template
├── .gitignore
│
├── src/                         # Python source code
│   ├── __init__.py
│   ├── utils.py                 # Cross-track utilities (init_directory, make_key, expand_simple_to_full_params)
│   │
│   ├── evolvability_v1/         # Evolvability V1 track (redesigned framework)
│   │   ├── __init__.py
│   │   ├── types.py             # SimState, AgentConfig, PhysicsConfig, RunConfig
│   │   ├── agent.py             # Unified agent with configurable I/O
│   │   ├── environment.py       # Environment ABC + implementations
│   │   ├── physics.py           # Movement, collision, resource mechanics
│   │   ├── simulation.py        # Main engine with handshake validation
│   │   └── scripts/
│   │       └── run.py           # Orchestration with logging/checkpointing
│   │
│   └── darwin_v0/               # Darwin V0 track (legacy)
│       ├── __init__.py
│       ├── agent.py             # Agent neural network (2-layer LSTM, 7 in / 7 out)
│       ├── agent_simple.py      # Simplified agent (6 in / 6 out, no toxin/attack)
│       ├── physics.py           # Physics definitions (actions, movement, conflicts)
│       ├── world.py             # World creation dispatcher
│       ├── simulation.py        # Main simulation engine (full, with toxin/attack, lineage tracking)
│       ├── simulation_simple.py # Simplified simulation (no toxin/attack, lineage tracking)
│       │
│       ├── worlds/              # Modular world generators
│       │   ├── __init__.py      # Registry and create_world dispatch
│       │   ├── base.py          # Shared utilities (Gaussian fields, regeneration)
│       │   ├── default.py       # Gaussian random world
│       │   ├── bridge.py        # Two fertile strips + bridge
│       │   ├── maze.py          # 8x8 maze with toxic walls
│       │   ├── pretrain.py      # Food curriculum (30->12 over 10k steps)
│       │   ├── thermotaxis.py   # Dynamic temperature with phase lag
│       │   ├── uniform.py       # Constant resource level (for controlled experiments)
│       │   ├── linear_gradient.py   # Linear resource gradient along x-axis
│       │   ├── temporal_gaussian.py # Time-varying Gaussian (Fourier phase rotation)
│       │   └── orbiting_gaussian.py # Orbiting high-resource blob
│       │
│       └── scripts/             # Orchestration scripts (entry points)
│           ├── run.py           # Main experiment runner (full simulation)
│           ├── run_simple.py    # Runner for simple simulation (pretrain/thermotaxis)
│           ├── run_maze_dynamic.py  # Dynamic maze with moving goal
│           ├── transfer.py      # Transfer experiment (full→full)
│           ├── transfer_simple.py   # Transfer for simple simulation (simple→simple)
│           ├── transfer_simple_to_full.py  # Transfer with weight expansion (simple→full)
│           ├── transfer_by_cluster.py      # Transfer filtering agents by genotype cluster
│           ├── transfer_competition.py     # Two clusters head-to-head, tracks lineage
│           └── continue_no_regen.py        # Continue simulation with dynamic regeneration
│
├── configs/                     # Configuration files
│   ├── evolvability_v1/         # Evolvability V1 track configs
│   │   └── run/
│   │       ├── demo/            # Demo experiment group
│   │       │   ├── quick.yaml, gaussian.yaml, uniform.yaml, large.yaml
│   │       ├── pretrain/        # Pretraining experiments
│   │       │   └── default.yaml
│   │       └── test_clamped_diffusion/  # Controlled diffusion experiments
│   │           ├── resource_10.yaml, resource_25.yaml
│   │
│   └── darwin_v0/               # Darwin V0 track configs
│       ├── run/                 # Run configs (experiment parameters)
│       │   ├── default.yaml, pretrain.yaml, thermotaxis.yaml, ...
│       │   ├── diffusion/       # Diffusion experiment configs
│       │   ├── divergent_genotype_v1/, v2/, v3/  # Genotype cluster experiments
│       │   └── linear_chemotaxis_v1/
│       └── world/               # World configs (arena parameters)
│           ├── default.yaml, pretrain.yaml, thermotaxis.yaml, ...
│           ├── diffusion/       # Diffusion world configs
│           └── divergent_genotype_v1/, v2/, v3/
│
├── scripts/                     # Bash execution scripts
│   ├── evolvability_v1/         # Evolvability V1 track scripts
│   │   ├── demo/                # Demo experiment group
│   │   │   ├── quick.sh, gaussian.sh, uniform.sh, large.sh, run_all.sh
│   │   ├── pretrain/            # Pretraining scripts
│   │   │   └── default.sh
│   │   └── test_clamped_diffusion/  # Controlled diffusion scripts
│   │       ├── resource_10.sh, resource_25.sh
│   │
│   └── darwin_v0/               # Darwin V0 track scripts
│       ├── run.sh, run_pretrain.sh, run_thermotaxis.sh, ...
│       ├── diffusion/           # Diffusion experiment scripts
│       ├── divergent_genotype_v1/, v2/, v3/
│       └── linear_chemotaxis_v1/
│
├── data/                        # Experiment outputs (gitignored)
│   ├── evolvability_v1/         # Evolvability V1 track outputs
│   │   ├── demo/                # Demo experiment outputs
│   │   ├── pretrain/            # Pretraining outputs
│   │   ├── test_clamped_diffusion/  # Diffusion experiment outputs
│   │   │   ├── resource_10/, resource_25/
│   │   └── <group>/<experiment>/
│   │       ├── config.yaml, figures/, logs/, checkpoints/
│   │
│   └── darwin_v0/               # Darwin V0 track outputs
│       ├── run_default/         # Default simulation outputs
│       ├── run_pretrain/        # Pretrain outputs
│       ├── diffusion_analysis_v1/, v2/  # Diffusion results
│       ├── divergent_genotype_v1/, v2/, v3/  # Genotype experiments
│       ├── linear_chemotaxis_v1/
│       └── <run_name>/
│           ├── config.yaml      # Copy of run config
│           ├── world_config.yaml    # Copy of world config
│           ├── figures/         # Plots
│           ├── logs/            # Logs (JSONL)
│           └── checkpoints/     # Simulation state (pkl)
│
├── docs/                        # Documentation
│   ├── start.md                 # Initial reading list
│   ├── repo_usage.md            # Detailed usage guide
│   ├── closing_tasks.md         # End-of-session tasks
│   ├── research_context.md      # Research goals and state
│   ├── structure.md             # This file
│   ├── maybe_vision.md          # Vision system design discussion
│   ├── computational_compromises.md  # Shared physics approximations
│   ├── concrete_questions/      # Documented research questions
│   ├── tracks/
│   │   ├── evolvability_v1/     # Evolvability V1 track documentation
│   │   │   ├── notes.md         # Design decisions
│   │   │   ├── progress.md      # Current state and next steps
│   │   │   └── computational_compromises.md  # V1-specific choices
│   │   └── darwin_v0/           # Darwin V0 track documentation
│   │       └── notes.md
│   └── logs/
│       └── YYYY-MM-DD/          # Daily development logs
│           └── HHMM_topic.md
│
├── scratch/                     # Temporary work (gitignored)
│   ├── genotype_analysis_v1/    # Genotype clustering analysis + reference data
│   │   ├── load_genotypes.py    # PCA + HDBSCAN clustering
│   │   ├── cluster_assignments.npz  # Cluster assignments (used by divergent_genotype experiments)
│   │   └── cluster_table.csv
│   ├── divergent_genotype_v1_analysis/  # Analysis scripts for v1 experiments
│   ├── v3_analysis/             # Action profile analysis
│   ├── v4_analysis/             # Phylogenetic analysis
│   │   └── output/              # Phylogenetic figures
│   ├── diffusion_analysis/      # Diffusion experiment analysis (darwin_v0)
│   ├── diffusion_analysis_v1/   # Diffusion analysis (evolvability_v1)
│   │   ├── analyze_diffusion.py
│   │   └── output/              # Analysis results and plots
│   ├── trajectory_analysis/     # Agent trajectory and age analysis
│   └── <subfolder>/             # Always use subfolders
│
└── resources/                   # Static resources (gitignored)
