# Research Question: Genotype Cluster Behavioral Divergence

## Question
**Can we detect qualitative behavioral differences between genetically distinct clusters of evolved agents?**

## Background
After 50k steps of evolution in `run_from_pretrain_simple`, the population has diverged into distinct genotype clusters visible in PCA space. HDBSCAN identifies 36 clusters with clear spatial separation.

**Key observation**: PC1 explains 53.6% of variance, suggesting a major axis of genetic divergence.

## Data Source (STATIC REFERENCE)

**IMPORTANT**: Use these exact files for reproducibility across all investigations.

```
data/reference/genotype_clusters_v1/
├── README.md                  # Full documentation
├── cluster_assignments.npz    # Cluster labels + alive indices
└── cluster_table.csv          # Cluster centroids in PCA space
```

- **Checkpoint**: `data/run_from_pretrain_simple/checkpoints/ckpt_050001.pkl`
- **Cluster assignments**: `data/reference/genotype_clusters_v1/cluster_assignments.npz`
- **Cluster table**: `data/reference/genotype_clusters_v1/cluster_table.csv`

## Target Clusters for Comparison

| Cluster | N agents | PC1 | PC2 | Notes |
|---------|----------|-----|-----|-------|
| **5** | 1324 | 38.2 | -15.8 | Largest cluster, RIGHT side of PC1 |
| **17** | 1040 | -34.7 | -0.6 | 2nd largest, LEFT side of PC1 |

These clusters are:
- Both large (>1000 agents) → statistically robust
- Maximally separated on PC1 (~73 units apart)
- Represent the two main "superclusters" in the population

See `data/reference/genotype_clusters_v1/README.md` for all available clusters.

## Proposed Experiments

**Note**: These are just examples to get started. Feel free to be creative and design your own experiments! The goal is to find *any* measurable behavioral difference between clusters.

### 1. Diffusion Coefficient
**Setup**: 256 agents in uniform resource environment (r=20), energy clamped (immortal), reproduction disabled, spawn in 16x16 center grid, 1000 steps.

**Metric**: Mean squared displacement (MSD) over time → diffusion coefficient D from MSD = 4Dt

**Hypothesis**: If clusters have different movement strategies, their diffusion coefficients will differ.

### 2. Chemotaxis (Gradient Following)
**Setup**: 256 agents in linear resource gradient (0 at left, 30 at right), energy clamped, reproduction disabled, spawn as vertical line at center, 5000 steps.

**Metric**: Mean x-position over time, final x-distribution

**Hypothesis**: Clusters may differ in gradient-sensing ability.

### 3. Reproduction Rate
**Setup**: 256 agents in uniform resource environment (r=25), reproduction enabled, energy NOT clamped, 5000 steps.

**Metric**: Population growth rate, reproduction frequency per agent

**Hypothesis**: Clusters may have different reproduction strategies (early vs late reproduction, energy thresholds).

### 4. Action Distribution
**Setup**: Same as diffusion experiment.

**Metric**: Fraction of each action (eat, forward, left, right, stay) over time

**Hypothesis**: Clusters may prefer different action mixes (more movement vs more eating).

## Infrastructure
- **Transfer script**: `src/scripts/transfer_by_cluster.py` - loads agents filtered by cluster ID
- **Group structure**: `configs/run/divergent_genotype_<name>/`, `scripts/divergent_genotype_<name>/`, `data/divergent_genotype_<name>/`

### Existing Groups (DO NOT USE)
The following groups are already in use by other teams:
- `divergent_genotype_v1` - Diffusion and chemotaxis experiments
- `divergent_genotype_v2` - Reproduction and competition experiments
- `divergent_genotype_v3` - Action profiles at high/low resource levels (found turning bias!)
- `divergent_genotype_v4` - Phylogenetic tree reconstruction (genotype-only, no behavioral experiments)

**New teams**: Create your own group with a unique name (e.g., `divergent_genotype_v5`, `divergent_genotype_maze`, etc.)

## Success Criteria
The question is answered "YES" if:
- At least one metric shows statistically significant difference between clusters
- The difference is robust across multiple random seeds
- The behavioral difference can be qualitatively described (e.g., "cluster 5 moves more", "cluster 16 eats more frequently")

## Notes for Independent Investigation
- Use same checkpoint and cluster assignments for reproducibility
- Can investigate different cluster pairs (e.g., 5 vs 25, or 16 vs 4)
- Can add new experiment types (e.g., maze navigation, predator avoidance)
- Keep same seed (42) for initial comparison, then vary seeds for robustness
