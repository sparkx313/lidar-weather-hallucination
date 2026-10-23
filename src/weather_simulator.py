"""Create repeatable procedural weather perturbations for LiDAR point clouds.

The functions preserve any per-point columns after XYZ. These are engineering
stress inputs, not physically calibrated sensor models or clean-point recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal

import numpy as np

Weather = Literal["rain", "fog", "snow"]


def _point_cloud(points: np.ndarray) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise ValueError("point cloud must be a two-dimensional array with at least XYZ columns")
    if not np.isfinite(cloud).all():
        raise ValueError("point cloud contains NaN or infinite values")
    return np.ascontiguousarray(cloud)


def _intensity(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("intensity must be a finite number between 0 and 1")
    return value


def _fraction(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return value


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _sample_artifacts(
    cloud: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample synthetic returns inside the input cloud's XYZ bounding box."""
    if count <= 0 or len(cloud) == 0:
        return np.empty((0, cloud.shape[1]), dtype=np.float32)
    bounds_min = cloud[:, :3].min(axis=0)
    bounds_max = cloud[:, :3].max(axis=0)
    artifacts = np.zeros((count, cloud.shape[1]), dtype=np.float32)
    artifacts[:, :3] = rng.uniform(bounds_min, bounds_max, size=(count, 3))
    return artifacts


def simulate_rain(
    points: np.ndarray,
    intensity: float = 0.5,
    artifact_fraction: float = 0.08,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add bounded synthetic returns; artifact count is relative to input size."""
    cloud = _point_cloud(points)
    strength = _intensity(intensity)
    fraction = _fraction(artifact_fraction, "artifact_fraction")
    generator = _rng(rng)
    count = int(round(len(cloud) * strength * fraction))
    artifacts = _sample_artifacts(cloud, count, generator)
    return np.concatenate((cloud, artifacts), axis=0)


def simulate_fog(
    points: np.ndarray,
    intensity: float = 0.5,
    max_range: float | None = None,
    artifact_fraction: float = 0.02,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Thin farther returns and add sparse bounded clutter as a stress scenario."""
    cloud = _point_cloud(points)
    strength = _intensity(intensity)
    fraction = _fraction(artifact_fraction, "artifact_fraction")
    generator = _rng(rng)
    if len(cloud) == 0:
        return cloud.copy()

    ranges = np.linalg.norm(cloud[:, :2], axis=1)
    reference_range = float(max_range) if max_range is not None else float(np.percentile(ranges, 95))
    if not np.isfinite(reference_range) or reference_range <= 0:
        raise ValueError("max_range must be a finite positive value")
    normalized_range = np.clip(ranges / reference_range, 0.0, 1.0)
    drop_probability = 0.75 * strength * normalized_range
    surviving = cloud[generator.random(len(cloud)) >= drop_probability]

    count = int(round(len(cloud) * strength * fraction))
    artifacts = _sample_artifacts(cloud, count, generator)
    return np.concatenate((surviving, artifacts), axis=0)


def simulate_snow(
    points: np.ndarray,
    intensity: float = 0.5,
    artifact_fraction: float = 0.05,
    dropout_fraction: float = 0.15,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Thin returns slightly and add bounded synthetic clutter."""
    cloud = _point_cloud(points)
    strength = _intensity(intensity)
    artifacts_per_input = _fraction(artifact_fraction, "artifact_fraction")
    dropout = _fraction(dropout_fraction, "dropout_fraction")
    generator = _rng(rng)
    keep = generator.random(len(cloud)) >= (strength * dropout)
    surviving = cloud[keep]

    count = int(round(len(cloud) * strength * artifacts_per_input))
    artifacts = _sample_artifacts(cloud, count, generator)
    return np.concatenate((surviving, artifacts), axis=0)


def corrupt_point_cloud(
    points: np.ndarray,
    weather: Weather,
    intensity: float = 0.5,
    seed: int = 42,
) -> np.ndarray:
    """Apply a seeded procedural stress perturbation to one point cloud."""
    if weather not in ("rain", "fog", "snow"):
        raise ValueError(f"unknown weather: {weather}")
    generator = np.random.default_rng(seed)
    if weather == "rain":
        return simulate_rain(points, intensity, rng=generator)
    if weather == "fog":
        return simulate_fog(points, intensity, rng=generator)
    return simulate_snow(points, intensity, rng=generator)


def _load_points(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        points = np.load(path, allow_pickle=False)
    elif suffix == ".bin":
        values = np.fromfile(path, dtype="<f4")
        if values.size % 4:
            raise ValueError("KITTI .bin input must contain complete float32 x,y,z,intensity records")
        points = values.reshape((-1, 4))
    else:
        raise ValueError("input format must be .npy or KITTI Velodyne .bin")
    points = _point_cloud(points)
    if suffix == ".bin" and points.shape[1] != 4:
        raise ValueError("KITTI .bin input must have exactly four columns: x, y, z, intensity")
    return points


def _save_points(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".npy":
        with path.open("wb") as destination:
            np.save(destination, points, allow_pickle=False)
        return
    if path.suffix.lower() == ".bin":
        if points.shape[1] != 4:
            raise ValueError("KITTI .bin output requires exactly four columns: x, y, z, intensity")
        points.astype("<f4", copy=False).tofile(path)
        return
    raise ValueError("output format must be .npy or KITTI Velodyne .bin")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a seeded synthetic weather perturbation for one LiDAR point cloud."
    )
    parser.add_argument("--input", type=Path, required=True, help=".npy (N x 3+) or KITTI Velodyne .bin (N x 4)")
    parser.add_argument("--weather", choices=("rain", "fog", "snow"), required=True)
    parser.add_argument("--intensity", type=float, default=0.5, help="severity from 0 to 1 (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for repeatable output")
    parser.add_argument("--output", type=Path, required=True, help="destination .npy or .bin file")
    parser.add_argument("--report", type=Path, help="optional JSON manifest with hashes and point counts")
    parser.add_argument("--overwrite", action="store_true", help="replace existing output or report files")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source = args.input.expanduser()
    output = args.output.expanduser()
    report_path = args.report.expanduser() if args.report else None

    try:
        source_resolved = source.resolve()
        output_resolved = output.resolve()
        report_resolved = report_path.resolve() if report_path else None
        if not source.is_file():
            parser.error(f"input file not found: {source}")
        if source_resolved == output_resolved:
            parser.error("output must not overwrite the input file")
        if report_resolved is not None and report_resolved in {source_resolved, output_resolved}:
            parser.error("report path must be different from input and output paths")
        if not args.overwrite:
            existing = [path for path in (output, report_path) if path is not None and path.exists()]
            if existing:
                parser.error("destination already exists (use --overwrite): " + ", ".join(map(str, existing)))

        points = _load_points(source)
        corrupted = corrupt_point_cloud(points, args.weather, args.intensity, args.seed)
        _save_points(output, corrupted)
        summary = {
            "tool": "lidar-weather-corruption-toolkit",
            "weather": args.weather,
            "intensity": _intensity(args.intensity),
            "seed": args.seed,
            "input": str(source),
            "input_sha256": _sha256(source),
            "input_points": int(len(points)),
            "output": str(output),
            "output_sha256": _sha256(output),
            "output_points": int(len(corrupted)),
            "net_point_count_change": int(len(corrupted) - len(points)),
            "limitations": [
                "Procedural stress perturbations are not physically calibrated weather models.",
                "This tool does not reconstruct clean points or produce vehicle safety evidence.",
            ],
        }
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {len(corrupted):,} points to {output}")
        print(f"Scenario: {args.weather}, intensity={summary['intensity']}, seed={args.seed}")
        print("This is a procedural stress input, not a calibrated sensor simulation or safety result.")
        if report_path is not None:
            print(f"Manifest: {report_path}")
        return 0
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
