"""
Thermotaxis Environment Visualization

Environment specs:
- Grid: 256 (y) x 512 (x)
- Food: Gradient (0 at bottom, 20 at top) + Gaussian noise field
- Temperature:
  - Top (y=256): sinusoidal oscillation between 0.0 and 1.0
  - Amplitude and phase decay with depth (soil heat diffusion model)
- Reproduction success: 1.0 if temp <= 0.5, linear decay to 0.0 at temp = 1.0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import gaussian_filter

# Grid dimensions (matches simulation: square grid)
HEIGHT = 512  # y
WIDTH = 512   # x

# Create coordinate grids
y = np.arange(HEIGHT)
x = np.arange(WIDTH)
X, Y = np.meshgrid(x, y)

# --- Food Distribution (gradient + noise) ---
def generate_gaussian_field(shape, length_scale, seed=42):
    """Generate smooth Gaussian random field."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(shape)
    # Smooth with Gaussian filter (sigma ~ length_scale)
    smooth = gaussian_filter(noise, sigma=length_scale)
    # Normalize to [0, 1]
    smooth = (smooth - smooth.min()) / (smooth.max() - smooth.min())
    return smooth

food_gradient = 20 * Y / (HEIGHT - 1)  # 0 at bottom, 20 at top
food_noise = generate_gaussian_field((HEIGHT, WIDTH), length_scale=30, seed=42)
food_noise_amplitude = 8.0  # +/- variation around gradient
food = np.clip(food_gradient + food_noise_amplitude * (food_noise - 0.5), 0, None)

# --- Temperature Model ---
def get_temperature(y_coords, time, period=1000, damping_depth=64.0):
    """
    Soil temperature model with phase lag (from Ramot et al. 2008).

    T(z,t) = Tave + T1/2 * exp(-z/zd) * sin(2πt/p - z/zd)

    Where z = depth from surface = (HEIGHT-1) - y
    - Top (y=max, z=0): oscillates between 0.0 and 1.0
    - Bottom (y=0, z=max): amplitude decays + phase lags
    - damping_depth (zd): depth at which amplitude decays by factor of e
    """
    max_y = HEIGHT - 1
    depth = max_y - y_coords  # depth from surface (0 at top, max at bottom)

    # Amplitude decay and phase lag both scale with depth/zd
    decay = np.exp(-depth / damping_depth)
    phase_lag = depth / damping_depth

    # Oscillation: 0.5 + 0.5*sin() gives range [0, 1] at surface
    temp = 0.5 + 0.5 * decay * np.sin(2 * np.pi * time / period - phase_lag)

    return temp

# --- Reproduction Success ---
def reproduction_success(temp):
    return np.clip(1.0 - 2 * np.maximum(0, temp - 0.5), 0, 1)

# --- Create visualizations ---
output_dir = Path(__file__).parent

# Figure 1: Combined overview
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
period = 1000

ax = axes[0]
im = ax.imshow(food, origin='lower', aspect='auto', cmap='Greens', extent=[0, WIDTH, 0, HEIGHT])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Food Distribution')
plt.colorbar(im, ax=ax, label='Food (0-20)')

ax = axes[1]
temp_peak = get_temperature(Y, period/4, period=period)  # t=T/4 is actual peak for sin
im = ax.imshow(temp_peak, origin='lower', aspect='auto', cmap='coolwarm',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Temperature (at peak, top=1.0)')
plt.colorbar(im, ax=ax, label='Temperature')

ax = axes[2]
temp_trough = get_temperature(Y, 3*period/4, period=period)  # t=3T/4 is actual trough
im = ax.imshow(temp_trough, origin='lower', aspect='auto', cmap='coolwarm',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Temperature (at trough, top=0.0)')
plt.colorbar(im, ax=ax, label='Temperature')

ax = axes[3]
repro_peak = reproduction_success(temp_peak)
im = ax.imshow(repro_peak, origin='lower', aspect='auto', cmap='RdYlGn',
               extent=[0, WIDTH, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Repro Success (at peak temp)')
plt.colorbar(im, ax=ax, label='Success Prob')

plt.suptitle('Thermotaxis Environment Overview (256 y x 512 x)', fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / 'environment_overview.png', dpi=150)
plt.close()

# Figure 2: Temperature heatmap (y vs time)
n_times = 200
times = np.linspace(0, 2 * period, n_times)  # 2 full cycles
y_vals = np.arange(HEIGHT)

# Build heatmap: rows=y, cols=time
temp_heatmap = np.zeros((HEIGHT, n_times))
for i, t in enumerate(times):
    temp_heatmap[:, i] = get_temperature(y_vals, t, period=period)

fig, ax = plt.subplots(figsize=(12, 6))
im = ax.imshow(temp_heatmap, origin='lower', aspect='auto', cmap='coolwarm',
               extent=[0, 2*period, 0, HEIGHT], vmin=0, vmax=1)
ax.set_xlabel('Time (steps)')
ax.set_ylabel('y position')
ax.set_title('Temperature over Time (2 cycles)')
plt.colorbar(im, ax=ax, label='Temperature')
plt.tight_layout()
plt.savefig(output_dir / 'temperature_yt_heatmap.png', dpi=150)
plt.close()

print(f"Saved: environment_overview.png, temperature_yt_heatmap.png")
