#!/bin/bash
# Run all divergent_genotype_v2 experiments
# Usage: bash scripts/divergent_genotype_v2/run_all.sh [--overwrite]

set -e

echo "=== Divergent Genotype V2 Experiments ==="
echo ""

echo "[1/3] Running reproduction experiment - Cluster 5..."
bash scripts/divergent_genotype_v2/reproduction_c5.sh "$@"
echo ""

echo "[2/3] Running reproduction experiment - Cluster 17..."
bash scripts/divergent_genotype_v2/reproduction_c17.sh "$@"
echo ""

echo "[3/3] Running competition experiment - C5 vs C17..."
bash scripts/divergent_genotype_v2/competition.sh "$@"
echo ""

echo "=== All experiments complete ==="
