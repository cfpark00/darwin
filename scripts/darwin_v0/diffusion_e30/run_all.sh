#!/bin/bash
# Run all diffusion experiments with energy=30 (r5, r10, r15, r20, r25, r30, r35, r40)

set -e  # Exit on error

echo "=== Diffusion E30 Experiment: Resource Level 5 ==="
bash scripts/diffusion_e30/r5.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 10 ==="
bash scripts/diffusion_e30/r10.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 15 ==="
bash scripts/diffusion_e30/r15.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 20 ==="
bash scripts/diffusion_e30/r20.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 25 ==="
bash scripts/diffusion_e30/r25.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 30 ==="
bash scripts/diffusion_e30/r30.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 35 ==="
bash scripts/diffusion_e30/r35.sh "$@"

echo "=== Diffusion E30 Experiment: Resource Level 40 ==="
bash scripts/diffusion_e30/r40.sh "$@"

echo "=== All diffusion E30 experiments complete ==="
