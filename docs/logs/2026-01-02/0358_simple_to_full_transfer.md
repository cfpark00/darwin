# Simple to Full Agent Transfer

## Summary
Added infrastructure to transfer pretrained simple agents (no toxin/attack) into full simulation environments. Fixed critical issue where loading mismatched checkpoint dimensions would silently produce garbage instead of failing.

## Problem Identified
- Simple agent: 6 inputs (no toxin), 6 outputs (no attack) → 1238 params
- Full agent: 7 inputs (with toxin), 7 outputs (with attack) → 1279 params
- Loading simple checkpoint into full simulation would silently slice arrays incorrectly
- Violated "fail fast" principle from repo guidelines

## Changes Made

### 1. Fixed pyproject.toml Deprecation Warning
- Changed `[tool.uv] dev-dependencies` to `[dependency-groups] dev`
- Fixed in both darwin and research-template repos

### 2. Added Weight Expansion Function (`src/utils.py`)
- `expand_simple_to_full_params()` - expands 1238→1279 params
- Inserts zero row at index 2 in LSTM1 input weights (for toxin)
- Adds zero column/element for attack output
- Includes validation of input/output dimensions

### 3. Created Transfer Script (`src/scripts/transfer_simple_to_full.py`)
- Loads simple agent checkpoint
- Validates checkpoint has simple agent dimensions
- Expands weights using utility function
- Runs in full simulation with expanded agents

### 4. Added Fail-Fast Validation
- `transfer.py`: Now checks checkpoint param_dim matches expected full agent dim
- `transfer_simple.py`: Now checks checkpoint param_dim matches expected simple agent dim
- Both fail immediately with clear error message pointing to correct script

### 5. Created Config and Script
- `configs/run/from_pretrain.yaml` - transfer from pretrain to default world
- `scripts/run_from_pretrain.sh` - runs the transfer

### 6. Reduced Max Steps
- Changed from 100k to 50k in `default.yaml` and `from_pretrain.yaml`

## Files Modified
- `pyproject.toml` - fixed deprecation warning
- `src/utils.py` - added `expand_simple_to_full_params()`
- `src/scripts/transfer.py` - added param validation
- `src/scripts/transfer_simple.py` - added param validation

## Files Created
- `src/scripts/transfer_simple_to_full.py`
- `configs/run/from_pretrain.yaml`
- `scripts/run_from_pretrain.sh`

## Usage
```bash
bash scripts/run_from_pretrain.sh --overwrite
```

## Design Decisions
- Toxin weights initialized to 0: agents ignore toxin initially, must evolve to use it
- Attack weights initialized to 0: attack logit starts at 0, giving ~14% probability with 7 actions
- This preserves learned behavior for existing actions while allowing evolution of new capabilities
