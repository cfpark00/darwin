# E500 Run Video Generation

## Summary
Created a video from the simulation snapshots of the `run_default_e500` experiment.

## Task Completed
- Assembled all numbered PNG files (`step_NNNNNN.png`) from `data/run_default_e500/figures/` into an MP4 video
- Used ffmpeg with H.264 encoding at 10 fps

## Output
- `data/run_default_e500/figures/simulation.mp4`
- 201 frames (steps 0 to 50000 in increments of 250)
- Duration: ~20 seconds
- Resolution: 1000x1000
- Size: ~21 MB

## Command Used
```bash
ffmpeg -framerate 10 -pattern_type glob -i 'data/run_default_e500/figures/step_*.png' \
  -c:v libx264 -pix_fmt yuv420p -y data/run_default_e500/figures/simulation.mp4
```

## Notes
- ffmpeg was installed via `sudo apt install ffmpeg` during this session
