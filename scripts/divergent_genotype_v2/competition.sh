#!/bin/bash
# Competition experiment - Cluster 5 vs Cluster 17 head-to-head
uv run python src/scripts/transfer_competition.py configs/run/divergent_genotype_v2/competition.yaml "$@"
