#!/bin/bash
# Competition experiment - Cluster 5 vs Cluster 17 head-to-head
uv run python src/darwin_v0/scripts/transfer_competition.py configs/darwin_v0/run/divergent_genotype_v2/competition.yaml "$@"
