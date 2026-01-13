#!/bin/bash
# Continue E500 with decaying resource regeneration (timescale +10% every 100 steps)
uv run python src/scripts/continue_no_regen.py configs/run/e500_decay_regen.yaml "$@"
