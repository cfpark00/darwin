#!/bin/bash
# Continue E500 with decaying resource regeneration (timescale +10% every 100 steps)
uv run python src/darwin_v0/scripts/continue_no_regen.py configs/darwin_v0/run/e500_decay_regen.yaml "$@"
