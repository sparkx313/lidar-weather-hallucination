# LiDAR Weather Corruption Toolkit

A small command-line utility for creating deterministic, synthetic weather-style perturbations of one LiDAR point cloud at a time. It reads user-supplied NumPy arrays or KITTI Velodyne scans, writes a corrupted scan, and can save a JSON manifest with input/output hashes and point counts.

This repository provides a data-pipeline stress tool. It does not contain a trained neural network, reconstruct clean points, or establish autonomous-driving safety. Its procedural perturbations are not calibrated to real rain, fog, or snow sensors.

## Requirements

- Python 3.10 or newer
- NumPy

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Input formats

- `.npy`: a numeric array shaped `(N, 3+)`, with XYZ in the first three columns. Additional columns such as intensity are preserved for original returns.
- `.bin`: a KITTI Velodyne scan of little-endian float32 `(x, y, z, intensity)` records.

Non-finite values and malformed KITTI records are rejected. One frame is loaded into memory at a time.

## Use with a real scan

Create a rain-style stress input from a KITTI scan and save a reproducibility manifest:

```powershell
python -m src.weather_simulator `
  --input "data/000001.bin" `
  --weather rain `
  --intensity 0.6 `
  --seed 17 `
  --output "data/perturbed/000001-rain.bin" `
  --report "data/perturbed/000001-rain.json"
```

Use `--weather fog` or `--weather snow` for the other perturbations. The same input, weather, intensity, and seed produce the same generated points. Existing output files are protected by default; pass `--overwrite` to replace them.

For an array saved with NumPy:

```powershell
python -m src.weather_simulator --input "scan.npy" --weather fog --output "scan-fog.npy"
```

Run `python -m src.weather_simulator --help` for all options.

## What each scenario changes

- **Rain:** adds synthetic returns sampled within the input scan's XYZ bounds.
- **Fog:** drops more returns as horizontal range increases and adds a small amount of bounded clutter.
- **Snow:** randomly drops a small share of returns and adds bounded clutter.

The returned points are procedural. The parameters do not represent a measured weather intensity, and the generated clouds should not be used as ground truth for sensor performance or vehicle safety claims.

## Manifest

When `--report` is provided, the JSON file records the weather parameters, seed, point counts, and SHA-256 hashes for the input and output files. It also records the simulation limitations so downstream users do not mistake a generated scan for sensor evidence.

## Project files

- `src/weather_simulator.py` — validated corruption functions and CLI.
- `requirements.txt` — NumPy runtime dependency.

There is no training pipeline or pretrained model in this repository.
