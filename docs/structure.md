darwin/
├── CLAUDE.md                    # Development guidelines
├── README.md                    # Project readme
├── pyproject.toml               # Python dependencies (uv)
├── uv.lock                      # Locked dependencies
├── .env                         # Environment config (DATA_DIR)
├── .env.example                 # Environment template
├── .gitignore
│
├── configs/                     # Configuration files
│   ├── run/
│   │   ├── default.yaml         # Default run config (seed, steps, logging, min_buffer_size)
│   │   ├── default_h24.yaml     # Default with hidden_dim=24
│   │   ├── bridge.yaml          # Transfer experiment config (base)
│   │   ├── bridge_explorers.yaml    # Explorer selection iteration 1
│   │   ├── bridge_explorers2.yaml   # Explorer selection iteration 2
│   │   ├── bridge_explorers3.yaml   # Explorer selection iteration 3
│   │   ├── bridge_explorers4.yaml   # Explorer selection iteration 4
│   │   ├── maze.yaml            # Transfer agents to maze
│   │   ├── maze_solvers.yaml    # Select goal-reaching agents, restart
│   │   ├── maze_solvers2.yaml   # Second iteration of maze solvers
│   │   ├── maze_dynamic.yaml    # Dynamic maze with moving goal
│   │   ├── pretrain.yaml        # Pretrain run (10k steps, food curriculum)
│   │   ├── thermotaxis.yaml     # Thermotaxis run (transfer from pretrain)
│   │   ├── from_pretrain.yaml   # Transfer pretrain agents to full simulation
│   │   ├── from_pretrain_simple.yaml  # Transfer pretrain to simple env (no toxin)
│   │   ├── from_pretrain_temporal.yaml # Transfer pretrain to temporal gaussian env
│   │   ├── maze_from_default.yaml     # Transfer from run_default to maze
│   │   ├── default_e500.yaml          # Default run with energy max 500
│   │   ├── e500_decay_regen.yaml       # Continue E500 with decaying regeneration
│   │   ├── single_agent.yaml          # Single agent observation
│   │   ├── diffusion/           # Diffusion experiment configs (E=100, v1)
│   │   │   ├── r5.yaml, r10.yaml, r15.yaml, r20.yaml
│   │   │   ├── r25.yaml, r30.yaml, r35.yaml, r40.yaml
│   │   ├── diffusion_e30/       # Diffusion experiment configs (E=30, v1)
│   │   │   └── r5.yaml ... r40.yaml
│   │   ├── diffusion_e15/       # Diffusion experiment configs (E=15, v1)
│   │   │   └── r5.yaml ... r40.yaml
│   │   ├── diffusion_v2/        # Diffusion v2 configs (E=100, fast regen)
│   │   │   └── r5.yaml ... r40.yaml
│   │   ├── diffusion_v2_e30/    # Diffusion v2 configs (E=30)
│   │   │   └── r5.yaml ... r40.yaml
│   │   ├── diffusion_v2_e15/    # Diffusion v2 configs (E=15)
│   │   │   └── r5.yaml ... r40.yaml
│   │   ├── linear_chemotaxis_v1/      # Linear gradient chemotaxis (multiple sources)
│   │   │   ├── from_pretrain_simple.yaml
│   │   │   ├── from_temporal.yaml
│   │   │   └── from_orbiting.yaml
│   │   ├── from_pretrain_orbiting.yaml   # Transfer pretrain to orbiting gaussian env
│   │   ├── from_pretrain_orbiting_harsh.yaml  # Harsh orbiting (low base resource)
│   │   ├── corner_expansion.yaml         # Corner spawn, no regen, reproduction on
│   │   ├── divergent_genotype_v1/        # Genotype cluster behavioral comparison
│   │   │   ├── diffusion_c5.yaml         # Diffusion experiment - cluster 5
│   │   │   ├── diffusion_c17.yaml        # Diffusion experiment - cluster 17
│   │   │   ├── chemotaxis_c5.yaml        # Chemotaxis experiment - cluster 5
│   │   │   └── chemotaxis_c17.yaml       # Chemotaxis experiment - cluster 17
│   │   ├── divergent_genotype_v2/        # Competition and reproduction experiments
│   │   │   ├── reproduction_c5.yaml      # Isolated reproduction - cluster 5
│   │   │   ├── reproduction_c17.yaml     # Isolated reproduction - cluster 17
│   │   │   └── competition.yaml          # Head-to-head competition C5 vs C17
│   │   └── divergent_genotype_v3/        # Action profiles at different resource levels
│   │       ├── action_high_c5.yaml       # High resource (r=30) - cluster 5
│   │       ├── action_high_c17.yaml      # High resource (r=30) - cluster 17
│   │       ├── action_low_c5.yaml        # Low resource (r=5) - cluster 5
│   │       └── action_low_c17.yaml       # Low resource (r=5) - cluster 17
│   └── world/
│       ├── default.yaml         # Gaussian world (resources, toxin, temperature)
│       ├── default_h24.yaml     # Gaussian world with hidden_dim=24
│       ├── bridge.yaml          # Bridge world (two strips + connecting bridge)
│       ├── maze.yaml            # 8x8 maze with toxic walls
│       ├── maze_dynamic.yaml    # Maze for dynamic goal experiment
│       ├── pretrain.yaml        # Pretrain world (food curriculum, static temp)
│       ├── thermotaxis.yaml     # Thermotaxis world (dynamic temp, food gradient)
│       ├── default_simple.yaml  # Gaussian world for simple agent (no toxin)
│       ├── default_e500.yaml    # Gaussian world with energy max 500
│       ├── e500_decay_regen.yaml  # Disabled built-in regen (for dynamic regen script)
│       ├── single_agent.yaml    # Single agent observation (uniform r=10)
│       ├── temporal_gaussian_simple.yaml  # Time-varying Gaussian (Fourier phase rotation)
│       ├── orbiting_gaussian_simple.yaml  # Orbiting blob (high resource circles arena)
│       ├── orbiting_gaussian_harsh.yaml   # Harsh orbiting (base_resource=3, blob_max=40)
│       ├── corner_expansion_simple.yaml   # Uniform r=20, no regen (for expansion exp)
│       ├── diffusion/           # Diffusion world configs (E=100, uniform resource, v1)
│       │   └── r5.yaml ... r40.yaml
│       ├── diffusion_e30/       # Diffusion world configs (E=30, v1)
│       │   └── r5.yaml ... r40.yaml
│       ├── diffusion_e15/       # Diffusion world configs (E=15, v1)
│       │   └── r5.yaml ... r40.yaml
│       ├── diffusion_v2/        # Diffusion v2 world configs (fast regen timescale=10)
│       │   └── r5.yaml ... r40.yaml
│       ├── diffusion_v2_e30/    # Diffusion v2 world configs (E=30)
│       │   └── r5.yaml ... r40.yaml
│       ├── diffusion_v2_e15/    # Diffusion v2 world configs (E=15)
│       │   └── r5.yaml ... r40.yaml
│       ├── linear_chemotaxis_v1/      # Linear gradient chemotaxis
│       │   └── default.yaml
│       ├── divergent_genotype_v1/     # Genotype cluster behavioral comparison
│       │   ├── diffusion.yaml         # Uniform resource, energy clamped
│       │   └── chemotaxis.yaml        # Linear gradient, energy clamped
│       ├── divergent_genotype_v2/     # Competition and reproduction worlds
│       │   ├── reproduction.yaml      # Uniform resource, reproduction enabled
│       │   └── competition.yaml       # Uniform resource, both clusters mixed
│       └── divergent_genotype_v3/     # Action profile worlds
│           ├── action_high.yaml       # Uniform resource r=30, energy clamped
│           └── action_low.yaml        # Uniform resource r=5, energy clamped
│
├── scripts/                     # Bash execution scripts
│   ├── run.sh                   # Run default simulation
│   ├── run_h24.sh               # Run with hidden_dim=24
│   ├── run_maze_dynamic.sh      # Run dynamic maze experiment
│   ├── run_pretrain.sh          # Run pretrain (food curriculum)
│   ├── run_thermotaxis.sh       # Run thermotaxis (transfers from pretrain)
│   ├── run_from_pretrain.sh     # Transfer pretrain to full simulation
│   ├── run_from_pretrain_simple.sh  # Transfer pretrain to simple env
│   ├── run_from_pretrain_temporal.sh # Transfer pretrain to temporal gaussian env
│   ├── run_single_agent.sh      # Single agent observation
│   ├── diffusion/               # Diffusion experiment scripts (E=100, v1)
│   │   ├── r5.sh ... r40.sh     # Individual resource levels
│   │   └── run_all.sh           # Run all diffusion experiments
│   ├── diffusion_e30/           # Diffusion experiment scripts (E=30, v1)
│   │   ├── r5.sh ... r40.sh
│   │   └── run_all.sh
│   ├── diffusion_e15/           # Diffusion experiment scripts (E=15, v1)
│   │   ├── r5.sh ... r40.sh
│   │   └── run_all.sh
│   ├── diffusion_v2/            # Diffusion v2 scripts (fast regen)
│   │   ├── r5.sh ... r40.sh, e15_r5.sh ... e30_r40.sh
│   │   ├── run_all.sh
│   │   └── run_remaining.sh
│   ├── linear_chemotaxis_v1/    # Linear gradient chemotaxis (multiple sources)
│   │   ├── from_pretrain_simple.sh
│   │   ├── from_temporal.sh
│   │   ├── from_orbiting.sh
│   │   └── run_all.sh
│   ├── run_from_pretrain_orbiting.sh  # Transfer to orbiting gaussian env
│   ├── run_from_pretrain_orbiting_harsh.sh  # Transfer to harsh orbiting env
│   ├── run_corner_expansion.sh        # Corner expansion experiment
│   ├── divergent_genotype_v1/         # Genotype cluster behavioral experiments
│   │   ├── diffusion_c5.sh, diffusion_c17.sh
│   │   ├── chemotaxis_c5.sh, chemotaxis_c17.sh
│   │   ├── analyze_diffusion.py       # MSD and diffusion analysis
│   │   ├── analyze_actions.py         # Action distribution analysis
│   │   ├── analyze_chemotaxis.py      # Gradient following analysis
│   │   └── create_summary.py          # Summary figure generation
│   ├── divergent_genotype_v2/         # Competition and reproduction experiments
│   │   ├── reproduction_c5.sh         # Isolated reproduction - cluster 5
│   │   ├── reproduction_c17.sh        # Isolated reproduction - cluster 17
│   │   ├── competition.sh             # Head-to-head competition
│   │   └── run_all.sh                 # Run all v2 experiments
│   ├── divergent_genotype_v3/         # Action profile experiments
│   │   ├── action_high_c5.sh, action_high_c17.sh
│   │   ├── action_low_c5.sh, action_low_c17.sh
│   │   └── run_all.sh                 # Run all v3 experiments
│   ├── transfer.sh              # Run transfer experiment (base bridge)
│   ├── transfer_explorers.sh    # Explorer selection iteration 1
│   ├── transfer_explorers2.sh   # Explorer selection iteration 2
│   ├── transfer_explorers3.sh   # Explorer selection iteration 3
│   ├── transfer_explorers4.sh   # Explorer selection iteration 4
│   ├── transfer_maze.sh         # Transfer to maze
│   ├── transfer_maze_solvers.sh     # Maze solver selection
│   ├── transfer_maze_solvers2.sh    # Maze solver iteration 2
│   ├── run_maze_from_default.sh     # Transfer from run_default to maze
│   ├── run_default_e500.sh          # Default run with energy max 500
│   └── run_e500_decay_regen.sh       # Continue E500 with decaying regeneration
│
├── src/                         # Python source code
│   ├── __init__.py
│   ├── agent.py                 # Agent neural network (2-layer LSTM, 7 in / 7 out)
│   ├── agent_simple.py          # Simplified agent (6 in / 6 out, no toxin/attack)
│   ├── physics.py               # Physics definitions (actions, movement, conflicts)
│   ├── world.py                 # World creation dispatcher
│   ├── simulation.py            # Main simulation engine (full, with toxin/attack, lineage tracking)
│   ├── simulation_simple.py     # Simplified simulation (no toxin/attack, lineage tracking)
│   ├── utils.py                 # Utilities (init_directory, make_key, expand_simple_to_full_params)
│   │
│   ├── worlds/                  # Modular world generators
│   │   ├── __init__.py          # Registry and create_world dispatch
│   │   ├── base.py              # Shared utilities (Gaussian fields, regeneration)
│   │   ├── default.py           # Gaussian random world
│   │   ├── bridge.py            # Two fertile strips + bridge
│   │   ├── maze.py              # 8x8 maze with toxic walls
│   │   ├── pretrain.py          # Food curriculum (30->12 over 10k steps)
│   │   ├── thermotaxis.py       # Dynamic temperature with phase lag
│   │   ├── uniform.py           # Constant resource level (for controlled experiments)
│   │   ├── linear_gradient.py   # Linear resource gradient along x-axis
│   │   ├── temporal_gaussian.py # Time-varying Gaussian (Fourier phase rotation)
│   │   └── orbiting_gaussian.py # Orbiting high-resource blob
│   │
│   └── scripts/                 # Orchestration scripts (entry points)
│       ├── run.py               # Main experiment runner (full simulation)
│       ├── run_simple.py        # Runner for simple simulation (pretrain/thermotaxis)
│       ├── run_maze_dynamic.py  # Dynamic maze with moving goal
│       ├── transfer.py          # Transfer experiment (full→full)
│       ├── transfer_simple.py   # Transfer for simple simulation (simple→simple)
│       ├── transfer_simple_to_full.py  # Transfer with weight expansion (simple→full)
│       ├── transfer_by_cluster.py      # Transfer filtering agents by genotype cluster
│       ├── transfer_competition.py     # Two clusters head-to-head, tracks lineage
│       └── continue_no_regen.py # Continue simulation with dynamic regeneration
│
├── data/                        # Experiment outputs (gitignored)
│   ├── reference/               # Static reference data for reproducibility
│   │   └── genotype_clusters_v1/    # Cluster assignments for behavioral comparison
│   │       ├── README.md
│   │       ├── cluster_assignments.npz
│   │       └── cluster_table.csv
│   ├── diffusion_analysis_v1/   # Diffusion v1 results (24 conditions)
│   ├── diffusion_analysis_v2/   # Diffusion v2 results (fast regen)
│   ├── linear_chemotaxis_v1/    # Linear chemotaxis (folder hosting multiple runs)
│   │   ├── from_pretrain_simple/
│   │   ├── from_temporal/
│   │   └── from_orbiting/
│   ├── divergent_genotype_v1/   # Genotype cluster behavioral comparison
│   │   ├── FINDINGS.md          # Results documentation
│   │   ├── summary.png          # Summary figure
│   │   ├── diffusion_c5/, diffusion_c17/
│   │   └── chemotaxis_c5/, chemotaxis_c17/
│   ├── divergent_genotype_v2/   # Competition and reproduction experiments
│   │   ├── RESULTS_SUMMARY.md   # Results documentation
│   │   ├── comparison_*.png     # Comparison plots
│   │   ├── cluster_*.png        # Spatial distribution plots
│   │   ├── reproduction_c5/, reproduction_c17/
│   │   └── competition/         # Head-to-head competition results
│   ├── divergent_genotype_v3/   # Action profile experiments
│   │   ├── figures/action_profile_comparison.png
│   │   ├── action_high_c5/, action_high_c17/
│   │   └── action_low_c5/, action_low_c17/
│   ├── divergent_genotype_v4/   # Phylogenetic analysis (genotype only, no experiments)
│   │   └── figures/
│   │       ├── phylogenetic_clusters.png     # Dendrogram with split ordering
│   │       ├── spatial_2_clades.png ... spatial_8_clades.png  # Progressive clade splits
│   │       └── spatial_split_points.png      # All 35 split locations on arena
│   └── <run_name>/
│       ├── config.yaml          # Copy of run config
│       ├── world_config.yaml    # Copy of world config
│       ├── figures/             # Plots (arena, population, actions, energy, attacks, toxin_deaths)
│       ├── logs/                # Base logs (JSONL), timing, goal_moves (dynamic maze)
│       └── checkpoints/         # Simulation state checkpoints (pkl)
│
├── docs/                        # Documentation
│   ├── start.md                # Initial reading list
│   ├── repo_usage.md            # Detailed usage guide
│   ├── closing_tasks.md         # End-of-session tasks
│   ├── research_context.md      # Research goals and state
│   ├── structure.md            # This file
│   ├── maybe_vision.md          # Vision system design discussion
│   ├── concrete_questions/      # Documented research questions
│   │   └── genotype_cluster_behavioral_divergence.md
│   └── logs/
│       └── YYYY-MM-DD/          # Daily development logs
│           └── HHMM_topic.md
│
├── scratch/                     # Temporary work (gitignored)
│   ├── diffusion_analysis/      # Diffusion experiment analysis
│   │   ├── analyze_diffusion.py    # v1 MSD and diffusion coefficient analysis
│   │   ├── analyze_diffusion_v2.py # v2 analysis (fast regen)
│   │   ├── output/              # v1 plots and results
│   │   └── output_v2/           # v2 plots and results
│   ├── single_agent_analysis/   # Single agent trajectory analysis
│   │   ├── plot_trajectory.py
│   │   └── output/
│   ├── linear_chemotaxis_v1_analysis/  # Linear chemotaxis comparison (all sources)
│   │   ├── plot_avg_x.py
│   │   └── output/
│   ├── corner_expansion_video/         # Video generation for corner expansion
│   │   ├── make_video.py
│   │   ├── frames/
│   │   └── corner_expansion.mp4
│   ├── temporal_video/                  # Video generation for temporal gaussian
│   │   ├── make_video.py               # Checkpoint-based video
│   │   ├── resume_video.py             # Step-by-step resume video
│   │   ├── frames/
│   │   ├── resume_frames/
│   │   ├── temporal_gaussian.mp4
│   │   └── temporal_resume.mp4
│   ├── temporal_changing_field_v1/     # Time-varying field demo
│   │   ├── demo.py
│   │   └── output/
│   ├── genotype_analysis_v1/           # Genotype clustering analysis
│   │   ├── load_genotypes.py           # PCA + HDBSCAN clustering
│   │   ├── plot_centers.py             # Cluster center visualization
│   │   ├── cluster_table.csv
│   │   ├── cluster_assignments.npz
│   │   ├── pca_umap.png
│   │   └── arena_clusters.png
│   ├── v3_analysis/                    # Action profile analysis
│   │   ├── analyze.py                  # Action distribution comparison
│   │   └── plot_results.py             # Visualization
│   ├── v4_analysis/                    # Phylogenetic analysis
│   │   ├── phylogenetic_coarse.py      # Dendrogram with split ordering
│   │   ├── spatial_splits.py           # Progressive clade visualization
│   │   └── spatial_split_points.py     # All split locations
│   ├── trajectory_analysis/            # Agent trajectory and age analysis
│   │   ├── trace_trajectories.py       # Trace agent positions across checkpoints
│   │   ├── trace_with_blob.py          # Trajectory with blob overlay
│   │   ├── age_pyramid.py              # Age distribution plots (temporal)
│   │   ├── age_pyramid_harsh.py        # Age distribution plots (harsh orbiting)
│   │   └── monitor_harsh.py            # Live monitoring of harsh experiment
│   └── <subfolder>/             # Always use subfolders
│
└── resources/                   # Static resources
