#!/bin/bash
# Run new diffusion experiments only (r5, r35, r40) with energy=30

set -e  # Exit on error

echo "=== Diffusion E30 Experiment: Resource Level 5 ==="
bash scripts/diffusion_e30/r5.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 35 ==="
bash scripts/diffusion_e30/r35.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 40 ==="
bash scripts/diffusion_e30/r40.sh "$@"

echo "=== New diffusion E30 experiments complete ==="
