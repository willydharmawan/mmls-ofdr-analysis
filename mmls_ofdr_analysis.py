"""Reproducible analysis of native Luna ODiSI distributed-strain exports.

The workflow:
  1. parses tab-delimited exports and caches numeric measurements;
  2. selects the active sensing interval and verifies sampling-rate evidence;
  3. detects repeated responses and extracts event-specific profiles;
  4. quantifies repeatability, drift, spatial consistency, and anomalies;
  5. evaluates sequence-dependent profile recurrence;
  6. performs filter, gauge-pitch, missing-section, and threshold sensitivity analyses;
  7. runs a deterministic synthetic benchmark, which is not physical validation.

Data are not included. Provide the three source exports in a local folder, or use
--data-dir to specify another location. Optional manual annotations are read
from CSV files outside this repository and should only be redistributed when
the relevant data custodian permits it. Derived output files can also contain
measurements and should be reviewed before public redistribution.

Requirements: Python 3.10+; numpy, pandas, scipy, and matplotlib.

Example:
    python mmls_ofdr_analysis.py --data-dir ./Dataset_MMLS --output-dir ./results

For manual validation, provide one CSV per configuration in the directory passed
with --manual-labels-dir. Each CSV must contain either a zero-based `sample`
column or a `time_s` column. Do not commit raw exports or restricted annotations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap
from scipy import stats
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter, welch


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "Dataset_MMLS"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"
DEFAULT_MANUAL_LABEL_DIR = SCRIPT_DIR / "manual_pass_labels"

DATA_DIR = DEFAULT_DATA_DIR
OUTPUT_DIR = DEFAULT_OUTPUT_DIR
CACHE_DIR = OUTPUT_DIR / "cache"
MANUAL_LABEL_DIR = DEFAULT_MANUAL_LABEL_DIR

# Expected file names. Raw measurement files are supplied separately and are
# deliberately not bundled with this source code.
INPUT_FILENAMES = {
    "2024-08-20_vertical_steel": "mmls_raw_data_8_20_2024.txt",
    "2024-11-05_vertical_wood_steel": "mmls_raw_data_11_05_2024.txt",
    "2024-12-11_vertical_horizontal_wood": "mmls_raw_data_12_11_2024.txt",
}
INPUT_FILES = {name: DATA_DIR / filename for name, filename in INPUT_FILENAMES.items()}

CONFIGURATION_DESCRIPTIONS = {
    "2024-08-20_vertical_steel": "Steel edge supports; vertical movement restrained",
    "2024-11-05_vertical_wood_steel": "Wood and steel edge supports; vertical movement restrained",
    "2024-12-11_vertical_horizontal_wood": "Wood supports; vertical and horizontal movement restrained",
}

CONFIGURATION_SHORT_LABELS = {
    "2024-08-20_vertical_steel": "V--S",
    "2024-11-05_vertical_wood_steel": "V--WS",
    "2024-12-11_vertical_horizontal_wood": "VH--W",
}

# Model attribution is based on the per-file metadata and the acquisition
# information cited in the manuscript.
INSTRUMENT_MODEL_BY_CONFIGURATION = {
    "2024-08-20_vertical_steel": "ODiSI 6101",
    "2024-11-05_vertical_wood_steel": "ODiSI 6101",
    "2024-12-11_vertical_horizontal_wood": "ODiSI 6101",
}
INSTRUMENT_MODEL_BASIS_BY_CONFIGURATION = {
    "2024-08-20_vertical_steel": "Confirmed from acquisition information; cite the source record in the manuscript.",
    "2024-11-05_vertical_wood_steel": "Dataset Product metadata and raw-file headers.",
    "2024-12-11_vertical_horizontal_wood": "Dataset Product metadata and raw-file headers.",
}

ACQUISITION_RATE_OVERRIDES_HZ = {
    "2024-08-20_vertical_steel": None,
    "2024-11-05_vertical_wood_steel": None,
    "2024-12-11_vertical_horizontal_wood": None,
}
FALLBACK_ACQUISITION_RATE_HZ = 40.0
NOMINAL_WHEEL_LOAD_LB = 600.0
NOMINAL_TIRE_PRESSURE_PSI = 100.0
NOMINAL_MMLS_SPEED_M_S = 2.5

# The report's 7,200 passes/hour = 2 Hz is retained only as a documented
# reference. It is NOT forced into event detection. Each file's recurrence is
# estimated independently from PSD and autocorrelation.
REPORT_NOMINAL_PASS_FREQUENCY_HZ = 2.0
PASS_FREQUENCY_SEARCH_HZ = (0.4, 6.0)
MIN_PEAK_DISTANCE_PERIOD_FRACTION = 0.70
ACTIVITY_SPATIAL_WINSOR_PERCENTILE = 95.0
ALLOW_REPORT_FREQUENCY_FALLBACK = False

# Report-constrained physical sensing intervals verified against the measured
# spatial-activity transitions. Coordinates are absolute ODiSI distances.
SLAB_REGIONS_M = {
    "2024-08-20_vertical_steel": (1.7804, 2.6982),
    "2024-11-05_vertical_wood_steel": (2.8672, 3.7850),
    "2024-12-11_vertical_horizontal_wood": (2.9296, 3.8474),
}

# The report displays an approximately 0.9 m instrumented slab interval. The
# automatic selector slides this window over the full fiber and maximizes the
# robust temporal response. Change if the physical bonded length differs.
EXPECTED_ACTIVE_LENGTH_M = 0.92
ROI_ANALYSIS_MAX_ROWS = 2500

# ROI-boundary perturbations for sensitivity analysis. These are analysis ranges,
# not surveyed installation tolerances; revise them if a sensor map becomes
# available.
ROI_BOUNDARY_OFFSETS_MM = (-20.0, -10.0, 0.0, 10.0, 20.0)

# One-at-a-time threshold sensitivity values. The primary settings remain the
# values declared above; these alternatives quantify dependence on them.
SENSITIVITY_GAUGE_COMPLETENESS = (0.70, 0.80, 0.90)
SENSITIVITY_ACTIVITY_WINSOR_PERCENTILES = (90.0, 95.0, 99.0)
SENSITIVITY_PEAK_SPACING_FRACTIONS = (0.60, 0.70, 0.80)
SENSITIVITY_EVENT_HALF_PERIOD_FRACTIONS = (0.35, 0.40, 0.45)
SENSITIVITY_MATCH_TOLERANCE_FRACTIONS = (0.10, 0.15, 0.20)
SENSITIVITY_ANOMALY_Z_THRESHOLDS = (3.0, 3.5, 4.0)
SENSITIVITY_RECURRENCE_MIN_PAIRS = (2, 3, 4)
SENSITIVITY_RECURRENCE_MIN_CORRELATIONS = (0.70, 0.75, 0.80)
SENSITIVITY_RECURRENCE_MIN_GAINS = (0.10, 0.20, 0.30)

# Deterministic synthetic benchmark settings. The benchmark assesses algorithm
# behavior under declared synthetic assumptions; it is not physical validation.
SYNTHETIC_BENCHMARK_REPLICATES = 100
SYNTHETIC_BENCHMARK_FS_HZ = 25.0
SYNTHETIC_BENCHMARK_DURATION_S = 55.0
SYNTHETIC_BENCHMARK_GAUGES = 120
SYNTHETIC_BENCHMARK_SEED = 20261107

# Event extraction uses +/- this fraction of one pass period. With a 2 Hz pass
# rate, 0.40 gives +/-0.20 s and avoids overlap with neighboring passes.
EVENT_HALF_PERIOD_FRACTION = 0.40
MIN_EVENTS_FOR_ANALYSIS = 8

# Filtering is applied only to the one-dimensional temporal activity signal
# used for event detection. Accepted event profiles and all reported strain
# features are extracted from locally baseline-corrected raw ODiSI data.
PRIMARY_DETECTION_FILTER = "raw"  # raw, gaussian, or savgol
TEMPORAL_GAUSSIAN_SIGMA_S = 0.05
TEMPORAL_SAVGOL_WINDOW_S = 0.25
TEMPORAL_SAVGOL_POLYORDER = 2

ROBUST_Z_THRESHOLD = 3.5
RANDOM_SEED = 42

# A gauge must contain finite measurements for at least this fraction of the
# recording. The fixed 80% rule retains all 354 gauges for V--S and VH--W in
# the supplied complete exports and 169 gauges in 47 runs for fragmented V--WS.
# Missing values are never imputed for reported strain features.
GAUGE_MIN_VALID_FRACTION = 0.80

SPATIAL_PITCH_FACTORS = (1, 2, 4, 8, 16)
MISSING_SEGMENT_LENGTHS_MM = (10, 25, 50, 100)
MISSING_SEGMENT_TRIALS = 20
PEAK_MATCH_TOLERANCE_PERIOD_FRACTION = 0.15

HEATMAP_SECONDS = 30.0
BLOCK_BOOTSTRAP_ITERATIONS = 10000
CACHE_FORMAT_VERSION = 5

# Post hoc sequence-recurrence diagnostic for continuous profile matrices.
# A multi-family result is reported only when the pooled profiles are
# heterogeneous, a non-unit lag has strong correlation, and that lag improves
# materially over adjacent-event (lag-one) correlation.  This prevents an
# already stable single-family record such as V--S from being split merely
# because every lag is highly correlated.
RECURRENCE_MAX_LAG_EVENTS = 8
RECURRENCE_MIN_PAIRS = 3
RECURRENCE_MIN_MEDIAN_CORRELATION = 0.75
RECURRENCE_MIN_GAIN_OVER_LAG_ONE = 0.20
RECURRENCE_SINGLE_FAMILY_CORRELATION = 0.90
RECURRENCE_FUNDAMENTAL_TOLERANCE = 0.02


# =============================================================================
# Data structures and general utilities
# =============================================================================


@dataclass
class ODiSIFile:
    configuration: str
    source_path: str
    source_size: int
    source_mtime_ns: int
    metadata: dict[str, str]
    declared_rate_hz: float | None
    timestamp_rate_hz: float | None
    timestamp_interval_cv_percent: float | None
    used_rate_hz: float
    used_rate_source: str
    units: str
    quantity: str
    distance_m: np.ndarray
    tare: np.ndarray | None
    data_path: str
    timestamps_path: str
    n_scans: int
    n_gages: int

    @property
    def data(self) -> np.memmap:
        return np.load(self.data_path, mmap_mode="r")


@dataclass
class DetectionResult:
    activity: np.ndarray
    detrended_activity: np.ndarray
    time_s: np.ndarray
    psd_frequency_hz: np.ndarray
    psd: np.ndarray
    dominant_frequency_hz: float | None
    autocorrelation_frequency_hz: float | None
    autocorrelation_peak: float | None
    used_pass_frequency_hz: float
    used_frequency_source: str
    peaks: np.ndarray
    peak_prominence_threshold: float
    peak_height_threshold: float
    minimum_peak_distance_samples: int


def clean_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def finite_float(text: str, default: float = math.nan) -> float:
    normalized = str(text or "").strip().upper()
    if not normalized or any(token in normalized for token in ("NAN", "IND", "INF")):
        return default
    try:
        value = float(text)
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text or ""))
        return float(match.group()) if match else default


def robust_mad(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(values - median), axis=axis)
    return mad


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    median = np.nanmedian(values, axis=0)
    scale = robust_mad(values, axis=0)
    fallback = np.nanstd(values, axis=0)
    scale = np.where((scale > 1e-12) & np.isfinite(scale), scale, fallback)
    scale = np.where((scale > 1e-12) & np.isfinite(scale), scale, 1.0)
    return (values - median) / scale


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(json_safe(content), indent=2), encoding="utf-8")


def select_acquisition_rate(
    configuration: str,
    declared_rate_hz: float | None,
    timestamp_rate_hz: float | None,
) -> tuple[float, str]:
    """Resolve the effective saved-data rate while retaining its provenance."""
    override = ACQUISITION_RATE_OVERRIDES_HZ.get(configuration)
    if override is not None:
        if not np.isfinite(override) or override <= 0:
            raise ValueError(
                f"Invalid acquisition-rate override for {configuration}: {override}"
            )
        return float(override), "explicit run-specific override"
    if (
        declared_rate_hz is not None
        and timestamp_rate_hz is not None
        and np.isfinite(declared_rate_hz)
        and np.isfinite(timestamp_rate_hz)
        and math.isclose(declared_rate_hz, timestamp_rate_hz, rel_tol=0.02)
    ):
        return float(declared_rate_hz), "raw-file metadata corroborated by timestamps"
    if timestamp_rate_hz is not None and np.isfinite(timestamp_rate_hz) and timestamp_rate_hz > 0:
        return float(timestamp_rate_hz), "timestamp-inferred saved-data rate"
    if declared_rate_hz is not None and np.isfinite(declared_rate_hz) and declared_rate_hz > 0:
        return float(declared_rate_hz), "raw-file metadata"
    return float(FALLBACK_ACQUISITION_RATE_HZ), "configured fallback (rate evidence unavailable)"


def infer_timestamp_rate(timestamps: Iterable[str]) -> tuple[float | None, float | None]:
    """Infer the saved-data rate from measurement-row timestamps.

    Returns the inverse median positive interval and the robust interval CV.
    Numeric elapsed times and ordinary date/time strings are both supported.
    """
    values = [str(value).strip() for value in timestamps if str(value).strip()]
    if len(values) < 3:
        return None, None
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    if np.isfinite(numeric).mean() >= 0.95:
        seconds = numeric[np.isfinite(numeric)]
    else:
        # Native ODiSI exports commonly use rounded MM:SS.s clock strings.
        # Direct datetime parsing is ambiguous, and the repeated rounded
        # values make the median positive difference incorrectly imply 10 Hz.
        clock_seconds: list[float] = []
        clock_valid = True
        for value in values:
            parts = value.split(":")
            try:
                if len(parts) == 2:
                    clock_seconds.append(60.0 * float(parts[0]) + float(parts[1]))
                elif len(parts) == 3:
                    clock_seconds.append(
                        3600.0 * float(parts[0])
                        + 60.0 * float(parts[1])
                        + float(parts[2])
                    )
                else:
                    clock_valid = False
                    break
            except ValueError:
                clock_valid = False
                break
        if clock_valid:
            seconds = np.asarray(clock_seconds, dtype=float)
            # Unwrap a possible hour/day clock rollover.
            for index in range(1, len(seconds)):
                while seconds[index] < seconds[index - 1] - 30.0:
                    seconds[index:] += 3600.0
        else:
            parsed = pd.to_datetime(pd.Series(values), errors="coerce", utc=True)
            valid = parsed.notna().to_numpy()
            if valid.mean() < 0.95:
                return None, None
            nanoseconds = (
                parsed[parsed.notna()].astype("int64").to_numpy(dtype=np.int64)
            )
            seconds = nanoseconds.astype(float) / 1e9
    if len(seconds) < 3 or seconds[-1] <= seconds[0]:
        return None, None
    sample_index = np.arange(len(seconds), dtype=float)
    slope = float(np.polyfit(sample_index, seconds, 1)[0])
    if not np.isfinite(slope) or slope <= 0:
        return None, None
    rate = float(1.0 / slope)

    # Estimate stability from long blocks so rounded 0.1-s timestamps do not
    # masquerade as zero intervals or a 10-Hz acquisition rate.
    boundaries = np.unique(np.linspace(0, len(seconds) - 1, 21).astype(int))
    block_rates = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        elapsed = seconds[stop] - seconds[start]
        if stop > start and elapsed > 0:
            block_rates.append((stop - start) / elapsed)
    block_rates = np.asarray(block_rates, dtype=float)
    stability_cv = (
        float(100.0 * robust_mad(block_rates) / np.median(block_rates))
        if len(block_rates) >= 3 and np.median(block_rates) > 0
        else None
    )
    return rate, stability_cv


def resolve_inputs() -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    missing: list[Path] = []
    for configuration, path in INPUT_FILES.items():
        if path.exists():
            resolved[configuration] = path
        else:
            missing.append(path)
    if missing:
        expected = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The following complete MMLS files were not found:\n"
            f"{expected}\n\n"
            "Place the three complete source exports in the directory passed with "
            "--data-dir, using the expected file names listed in INPUT_FILENAMES."
        )
    return resolved


# =============================================================================
# Native ODiSI parser and disk cache
# =============================================================================


def inspect_odisi(path: Path) -> dict:
    metadata: dict[str, str] = {}
    distance: np.ndarray | None = None
    tare: np.ndarray | None = None
    data_start: int | None = None
    quantity = "unknown"
    row_width = 0

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row_number, row in enumerate(reader):
            row_width = max(row_width, len(row))
            first = row[0].strip() if row else ""
            if first.endswith(":"):
                metadata[first[:-1]] = row[1].strip() if len(row) > 1 else ""
            elif first == "Tare" and len(row) > 3:
                tare = np.asarray([finite_float(item) for item in row[3:]], dtype=np.float32)
            elif first == "x-axis":
                distance = np.asarray([finite_float(item) for item in row[3:]], dtype=float)
                data_start = row_number + 1
                break

    if distance is None or data_start is None:
        raise ValueError(f"No x-axis row was found in {path}")

    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row_number, row in enumerate(reader):
            if row_number == data_start:
                if len(row) > 2:
                    quantity = row[2].strip().lower()
                break

    valid_distance = np.isfinite(distance)
    if not valid_distance.any():
        raise ValueError(f"The x-axis in {path} contains no finite coordinates")
    return {
        "metadata": metadata,
        "distance_raw": distance,
        "valid_distance": valid_distance,
        "tare_raw": tare,
        "data_start": data_start,
        "quantity": quantity,
        "row_width": row_width,
    }


def count_measurement_rows(path: Path, data_start: int) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row_number, row in enumerate(reader):
            if row_number < data_start:
                continue
            if len(row) > 2 and row[1].strip().lower() == "measurement":
                count += 1
    return count


def cache_key(path: Path) -> str:
    stat = path.stat()
    payload = (
        f"v{CACHE_FORMAT_VERSION}|{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_and_cache_odisi(configuration: str, path: Path) -> ODiSIFile:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(path)
    stem = f"{clean_name(configuration)}_{key}"
    matrix_path = CACHE_DIR / f"{stem}.npy"
    timestamps_path = CACHE_DIR / f"{stem}_timestamps.txt"
    manifest_path = CACHE_DIR / f"{stem}_manifest.json"

    if matrix_path.exists() and timestamps_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        declared_rate = manifest.get("declared_rate_hz")
        timestamp_rate, timestamp_cv = infer_timestamp_rate(
            timestamps_path.read_text(encoding="utf-8").splitlines()
        )
        used_rate, rate_source = select_acquisition_rate(
            configuration, declared_rate, timestamp_rate
        )
        return ODiSIFile(
            configuration=configuration,
            source_path=str(path),
            source_size=path.stat().st_size,
            source_mtime_ns=path.stat().st_mtime_ns,
            metadata=manifest["metadata"],
            declared_rate_hz=declared_rate,
            timestamp_rate_hz=timestamp_rate,
            timestamp_interval_cv_percent=timestamp_cv,
            used_rate_hz=used_rate,
            used_rate_source=rate_source,
            units=manifest["units"],
            quantity=manifest["quantity"],
            distance_m=np.asarray(manifest["distance_m"], dtype=float),
            tare=np.asarray(manifest["tare"], dtype=np.float32)
            if manifest.get("tare") is not None
            else None,
            data_path=str(matrix_path),
            timestamps_path=str(timestamps_path),
            n_scans=int(manifest["n_scans"]),
            n_gages=int(manifest["n_gages"]),
        )

    structure = inspect_odisi(path)
    valid_distance = structure["valid_distance"]
    distance = structure["distance_raw"][valid_distance]
    tare_raw = structure["tare_raw"]
    tare = None
    if tare_raw is not None and len(tare_raw) == len(valid_distance):
        tare = tare_raw[valid_distance]
    n_scans = count_measurement_rows(path, structure["data_start"])
    n_gages = int(valid_distance.sum())
    if n_scans == 0:
        raise ValueError(f"No measurement rows found in {path}")

    metadata = structure["metadata"]
    declared_rate = finite_float(metadata.get("Measurement Rate per Channel", ""))
    declared_rate_or_none = (
        float(declared_rate) if np.isfinite(declared_rate) and declared_rate > 0 else None
    )
    matrix = open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n_scans, n_gages))
    timestamps: list[str] = []
    write_index = 0
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        for row_number, row in enumerate(reader):
            if row_number < structure["data_start"]:
                continue
            if len(row) <= 2 or row[1].strip().lower() != "measurement":
                continue
            raw_values = row[3 : 3 + len(valid_distance)]
            if len(raw_values) < len(valid_distance):
                raw_values += [""] * (len(valid_distance) - len(raw_values))
            values = np.asarray([finite_float(item) for item in raw_values], dtype=np.float32)
            matrix[write_index] = values[valid_distance]
            timestamps.append(row[0].strip())
            write_index += 1
    matrix.flush()
    del matrix
    if write_index != n_scans:
        raise RuntimeError(f"Counted {n_scans} scans but wrote {write_index} for {path}")
    timestamps_path.write_text("\n".join(timestamps) + "\n", encoding="utf-8")
    timestamp_rate, timestamp_cv = infer_timestamp_rate(timestamps)
    used_rate, rate_source = select_acquisition_rate(
        configuration, declared_rate_or_none, timestamp_rate
    )

    safe_metadata_keys = {
        "Test Name",
        "Product",
        "Measurement Rate per Channel",
        "Units",
        "Gauge Pitch",
        "Sensor Length",
    }
    public_metadata = {
        key: value for key, value in metadata.items() if key in safe_metadata_keys
    }
    manifest = {
        "metadata": public_metadata,
        "declared_rate_hz": declared_rate_or_none,
        "timestamp_rate_hz": timestamp_rate,
        "timestamp_interval_cv_percent": timestamp_cv,
        "used_rate_hz": used_rate,
        "used_rate_source": rate_source,
        "units": metadata.get("Units", ""),
        "quantity": structure["quantity"],
        "distance_m": distance.tolist(),
        "tare": tare.tolist() if tare is not None else None,
        "n_scans": n_scans,
        "n_gages": n_gages,
    }
    write_json(manifest_path, manifest)
    return parse_and_cache_odisi(configuration, path)


# =============================================================================
# ROI, filtering, wheel-pass detection, and event features
# =============================================================================


def evenly_spaced_indices(length: int, maximum: int) -> np.ndarray:
    if length <= maximum:
        return np.arange(length)
    return np.unique(np.linspace(0, length - 1, maximum).round().astype(int))


def temporal_response_score(data: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low = np.nanpercentile(data, 5, axis=0)
        high = np.nanpercentile(data, 95, axis=0)
    score = high - low
    score[~np.isfinite(score)] = 0.0
    return score


def automatic_roi(item: ODiSIFile) -> tuple[np.ndarray, dict]:
    data = item.data
    sampled = np.asarray(data[evenly_spaced_indices(item.n_scans, ROI_ANALYSIS_MAX_ROWS)], dtype=float)
    score = temporal_response_score(sampled)
    x = item.distance_m
    pitch = np.nanmedian(np.diff(x))
    width_gages = max(10, int(round(EXPECTED_ACTIVE_LENGTH_M / pitch)))
    width_gages = min(width_gages, len(x))
    kernel = np.ones(width_gages, dtype=float)
    # Median-normalize to reduce the influence of one extremely noisy gauge.
    capped = np.minimum(score, np.nanpercentile(score, 99))
    window_scores = np.convolve(capped, kernel, mode="valid")
    start = int(np.argmax(window_scores))
    stop = start + width_gages
    indices = np.arange(start, stop)
    return indices, {
        "method": "automatic sliding robust-response window",
        "start_m": float(x[start]),
        "stop_m": float(x[stop - 1]),
        "n_gages": int(len(indices)),
        "expected_length_m": EXPECTED_ACTIVE_LENGTH_M,
        "score": score,
    }


def choose_roi(item: ODiSIFile) -> tuple[np.ndarray, dict]:
    requested = SLAB_REGIONS_M.get(item.configuration)
    if requested is None:
        return automatic_roi(item)
    mask = (item.distance_m >= requested[0]) & (item.distance_m <= requested[1])
    indices = np.flatnonzero(mask)
    if len(indices) < 10:
        raise ValueError(
            f"Manual interval {requested} for {item.configuration} contains fewer than 10 gauges"
        )
    sampled = np.asarray(
        item.data[evenly_spaced_indices(item.n_scans, ROI_ANALYSIS_MAX_ROWS)], dtype=float
    )
    return indices, {
        "method": "manual physical interval",
        "start_m": float(item.distance_m[indices[0]]),
        "stop_m": float(item.distance_m[indices[-1]]),
        "n_gages": int(len(indices)),
        "expected_length_m": None,
        "score": temporal_response_score(sampled),
    }


def apply_temporal_activity_filter(activity: np.ndarray, fs: float, method: str) -> np.ndarray:
    """Filter only the temporal detection signal, never the strain profiles."""
    activity = np.asarray(activity, dtype=float)
    if method == "raw":
        return activity.copy()
    if method == "gaussian":
        sigma = max(0.5, TEMPORAL_GAUSSIAN_SIGMA_S * fs)
        return gaussian_filter1d(activity, sigma=sigma, mode="nearest")
    if method == "savgol":
        window = max(3, int(round(TEMPORAL_SAVGOL_WINDOW_S * fs)))
        if window % 2 == 0:
            window += 1
        maximum = len(activity) if len(activity) % 2 == 1 else len(activity) - 1
        window = min(window, maximum)
        if window <= TEMPORAL_SAVGOL_POLYORDER or window < 3:
            return activity.copy()
        return savgol_filter(
            activity,
            window_length=window,
            polyorder=TEMPORAL_SAVGOL_POLYORDER,
            mode="interp",
        )
    raise ValueError(f"Unknown temporal detection filter: {method}")


def audit_gauge_validity(
    profiles: np.ndarray,
    x: np.ndarray,
    minimum_valid_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply the fixed temporal-validity mask and identify spatial runs."""
    profiles = np.asarray(profiles, dtype=np.float32)
    x = np.asarray(x, dtype=float)
    threshold = (
        GAUGE_MIN_VALID_FRACTION
        if minimum_valid_fraction is None
        else float(minimum_valid_fraction)
    )
    if not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1].")
    valid_fraction = np.isfinite(profiles).mean(axis=0)
    keep = valid_fraction >= threshold
    if keep.sum() < 10:
        raise ValueError(
            "Fewer than ten gauges remain after the fixed temporal-validity mask; "
            "do not impute missing values for publication analysis."
        )
    retained = profiles[:, keep]
    retained_x = x[keep]
    nominal_pitch = float(np.nanmedian(np.diff(x)))
    breaks = np.flatnonzero(np.diff(retained_x) > 1.5 * nominal_pitch) + 1
    starts = np.r_[0, breaks]
    stops = np.r_[breaks, len(retained_x)]
    run_ids = np.empty(len(retained_x), dtype=int)
    run_rows = []
    for run_id, (start, stop) in enumerate(zip(starts, stops)):
        run_ids[start:stop] = run_id
        length = float(retained_x[stop - 1] - retained_x[start]) if stop - start > 1 else 0.0
        run_rows.append(
            {
                "run_id": int(run_id),
                "start_m": float(retained_x[start]),
                "stop_m": float(retained_x[stop - 1]),
                "n_gages": int(stop - start),
                "length_m": length,
            }
        )
    audit = {
        "nominal_gages": int(len(x)),
        "valid_gages": int(keep.sum()),
        "minimum_temporal_valid_fraction": threshold,
        "coverage_percent": float(100.0 * keep.mean()),
        "n_contiguous_runs": int(len(run_rows)),
        "longest_run_m": float(max(row["length_m"] for row in run_rows)),
        "continuous_full_interval": bool(len(run_rows) == 1),
        "runs": run_rows,
    }
    return retained, keep, run_ids, audit


def compute_activity(
    profiles: np.ndarray,
    spatial_winsor_percentile: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = np.nanmedian(profiles, axis=0)
    residual = profiles - baseline
    squared = residual**2
    # A single bad ODiSI gauge must not create a wheel-pass candidate. The
    # upper 5% of spatial squared residuals are winsorized per scan, while a
    # spatially distributed pavement response is retained.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        percentile = (
            ACTIVITY_SPATIAL_WINSOR_PERCENTILE
            if spatial_winsor_percentile is None
            else float(spatial_winsor_percentile)
        )
        if not 0.0 < percentile <= 100.0:
            raise ValueError("spatial_winsor_percentile must be in (0, 100].")
        cap = np.nanpercentile(squared, percentile, axis=1, keepdims=True)
    squared = np.minimum(squared, cap)
    activity = np.sqrt(np.nanmean(squared, axis=1))
    return activity, baseline


def estimate_pass_frequency(
    activity: np.ndarray, fs: float
) -> tuple[np.ndarray, np.ndarray, float | None, float | None, float | None]:
    if len(activity) < max(32, int(4 * fs)):
        return np.array([]), np.array([]), None, None, None
    activity = np.asarray(activity, dtype=float)
    finite = np.isfinite(activity)
    if not finite.any():
        return np.array([]), np.array([]), None, None, None
    filled = np.where(finite, activity, np.nanmedian(activity[finite]))
    centered = filled - np.mean(filled)
    nperseg = min(len(activity), max(64, int(16 * fs)))
    frequency, power = welch(centered, fs=fs, nperseg=nperseg)
    mask = (frequency >= PASS_FREQUENCY_SEARCH_HZ[0]) & (
        frequency <= min(PASS_FREQUENCY_SEARCH_HZ[1], 0.45 * fs)
    )
    dominant = float(frequency[mask][np.argmax(power[mask])]) if mask.any() else None

    # FFT autocorrelation supplies an independent period estimate and is less
    # likely than PSD alone to select a strong pulse-train harmonic.
    n = len(centered)
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=nfft)
    autocorrelation = np.fft.irfft(spectrum * np.conj(spectrum), n=nfft)[:n]
    overlap = np.arange(n, 0, -1, dtype=float)
    autocorrelation = autocorrelation / overlap
    if autocorrelation[0] > 0:
        autocorrelation = autocorrelation / autocorrelation[0]
    minimum_lag = max(1, int(math.floor(fs / PASS_FREQUENCY_SEARCH_HZ[1])))
    maximum_lag = min(n - 2, int(math.ceil(fs / PASS_FREQUENCY_SEARCH_HZ[0])))
    autocorrelation_frequency = None
    autocorrelation_peak = None
    if maximum_lag > minimum_lag:
        segment = autocorrelation[minimum_lag : maximum_lag + 1]
        candidate_indices, _ = find_peaks(segment)
        if len(candidate_indices):
            candidate_values = segment[candidate_indices]
            useful = candidate_values > max(0.03, 0.15 * np.nanmax(candidate_values))
            if useful.any():
                candidate_indices = candidate_indices[useful]
                candidate_values = candidate_values[useful]
            candidate_lags = minimum_lag + candidate_indices
            candidate_frequencies = fs / candidate_lags
            if dominant is not None and np.isfinite(dominant):
                # Select the ACF peak nearest the PSD candidate. Choosing the
                # globally largest ACF peak can select two, three, or more
                # complete periods instead of the first recurrence.
                best_candidate = int(
                    np.argmin(np.abs(np.log(candidate_frequencies / dominant)))
                )
            else:
                strong = candidate_values >= 0.50 * np.nanmax(candidate_values)
                strong_indices = np.flatnonzero(strong)
                best_candidate = int(
                    strong_indices[np.argmax(candidate_frequencies[strong])]
                )
            best_local = int(candidate_indices[best_candidate])
        else:
            best_local = int(np.argmax(segment))
        best_lag = minimum_lag + best_local
        autocorrelation_frequency = float(fs / best_lag)
        autocorrelation_peak = float(autocorrelation[best_lag])
    return (
        frequency,
        power,
        dominant,
        autocorrelation_frequency,
        autocorrelation_peak,
    )


def choose_observed_frequency(
    psd_frequency_hz: float | None,
    autocorrelation_frequency_hz: float | None,
    autocorrelation_peak: float | None,
) -> tuple[float, str]:
    valid_psd = psd_frequency_hz is not None and np.isfinite(psd_frequency_hz)
    valid_acf = (
        autocorrelation_frequency_hz is not None
        and np.isfinite(autocorrelation_frequency_hz)
    )
    if valid_psd and valid_acf:
        low = min(float(psd_frequency_hz), float(autocorrelation_frequency_hz))
        high = max(float(psd_frequency_hz), float(autocorrelation_frequency_hz))
        if high / low <= 1.20:
            return 0.5 * (low + high), "PSD-autocorrelation agreement"
        ratio = high / low
        harmonic = int(round(ratio))
        if harmonic >= 2 and abs(ratio - harmonic) <= 0.12:
            return low, "PSD-autocorrelation fundamental/harmonic resolution"
        if autocorrelation_peak is not None and autocorrelation_peak >= 0.15:
            return float(autocorrelation_frequency_hz), "autocorrelation"
        return float(psd_frequency_hz), "PSD"
    if valid_acf:
        return float(autocorrelation_frequency_hz), "autocorrelation"
    if valid_psd:
        return float(psd_frequency_hz), "PSD"
    if ALLOW_REPORT_FREQUENCY_FALLBACK:
        return REPORT_NOMINAL_PASS_FREQUENCY_HZ, (
            "report nominal fallback; insufficient frequency evidence"
        )
    raise ValueError(
        "Pass frequency could not be estimated from PSD or autocorrelation. "
        "The report's nominal 2 Hz value was not forced. Inspect the ROI, "
        "sampling rate, and activity signal before continuing."
    )


def detect_passes(
    profiles: np.ndarray,
    fs: float,
    activity_filter: str = "raw",
    reference: DetectionResult | None = None,
    lock_thresholds: bool = False,
    minimum_peak_distance_fraction: float | None = None,
    activity_spatial_winsor_percentile: float | None = None,
) -> DetectionResult:
    spacing_fraction = (
        MIN_PEAK_DISTANCE_PERIOD_FRACTION
        if minimum_peak_distance_fraction is None
        else float(minimum_peak_distance_fraction)
    )
    if spacing_fraction <= 0.0:
        raise ValueError("minimum_peak_distance_fraction must be positive.")
    activity, _ = compute_activity(
        profiles,
        spatial_winsor_percentile=activity_spatial_winsor_percentile,
    )
    detection_activity = apply_temporal_activity_filter(activity, fs, activity_filter)
    # Remove only slow activity drift; the pass-frequency band is retained.
    baseline_window = max(3, int(round(fs * 3.0)))
    if baseline_window % 2 == 0:
        baseline_window += 1
    if len(detection_activity) > baseline_window + 2:
        slow = savgol_filter(detection_activity, baseline_window, 2, mode="interp")
        detrended = detection_activity - slow + np.nanmedian(slow)
    else:
        detrended = detection_activity.copy()

    (
        frequency,
        power,
        dominant,
        autocorrelation_frequency,
        autocorrelation_peak,
    ) = estimate_pass_frequency(detrended, fs)
    if reference is None:
        used_frequency, frequency_source = choose_observed_frequency(
            dominant, autocorrelation_frequency, autocorrelation_peak
        )
    else:
        used_frequency = reference.used_pass_frequency_hz
        frequency_source = "locked to full-resolution reference"

    low, high = np.nanpercentile(detrended, [5, 99])
    dynamic_range = max(float(high - low), 1e-9)
    adaptive_prominence = max(
        0.08 * dynamic_range, 1.5 * float(robust_mad(detrended))
    )
    adaptive_height = float(np.nanpercentile(detrended, 35))
    if reference is not None and lock_thresholds:
        prominence = reference.peak_prominence_threshold
        height = reference.peak_height_threshold
    else:
        prominence = adaptive_prominence
        height = adaptive_height
    if reference is not None:
        minimum_distance = reference.minimum_peak_distance_samples
    else:
        minimum_distance = max(
            1,
            int(
                round(
                    spacing_fraction * fs / used_frequency
                )
            ),
        )
    peaks, _ = find_peaks(
        detrended,
        distance=minimum_distance,
        prominence=prominence,
        height=height,
    )

    # One refinement uses the detected median interval only when it agrees with
    # the independent PSD/autocorrelation estimate. This improves timing
    # without letting false subpeaks redefine the recurrence.
    if reference is None and len(peaks) >= 4:
        peak_intervals = np.diff(peaks).astype(float)
        median_interval = float(np.median(peak_intervals))
        regular_intervals = peak_intervals[
            (peak_intervals >= 0.50 * median_interval)
            & (peak_intervals <= 1.50 * median_interval)
        ]
        interval_frequency = float(fs / np.mean(regular_intervals))
        if (
            PASS_FREQUENCY_SEARCH_HZ[0]
            <= interval_frequency
            <= PASS_FREQUENCY_SEARCH_HZ[1]
            and 0.80 <= interval_frequency / used_frequency <= 1.25
        ):
            used_frequency = interval_frequency
            frequency_source += "; median-interval refinement"
            minimum_distance = max(
                1,
                int(
                    round(
                        spacing_fraction * fs / used_frequency
                    )
                ),
            )
            peaks, _ = find_peaks(
                detrended,
                distance=minimum_distance,
                prominence=prominence,
                height=height,
            )
    time_s = np.arange(len(activity), dtype=float) / fs
    return DetectionResult(
        activity=activity,
        detrended_activity=detrended,
        time_s=time_s,
        psd_frequency_hz=frequency,
        psd=power,
        dominant_frequency_hz=dominant,
        autocorrelation_frequency_hz=autocorrelation_frequency,
        autocorrelation_peak=autocorrelation_peak,
        used_pass_frequency_hz=float(used_frequency),
        used_frequency_source=frequency_source,
        peaks=peaks.astype(int),
        peak_prominence_threshold=float(prominence),
        peak_height_threshold=float(height),
        minimum_peak_distance_samples=int(minimum_distance),
    )


def extract_events(
    profiles: np.ndarray,
    peaks: np.ndarray,
    fs: float,
    pass_frequency_hz: float,
    half_period_fraction: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fraction = (
        EVENT_HALF_PERIOD_FRACTION
        if half_period_fraction is None
        else float(half_period_fraction)
    )
    if not 0.0 < fraction < 0.5:
        raise ValueError("half_period_fraction must be in (0, 0.5).")
    half = max(2, int(round(fraction * fs / pass_frequency_hz)))
    valid_peaks = peaks[(peaks >= half) & (peaks < len(profiles) - half)]
    relative_time = np.arange(-half, half + 1, dtype=float) / fs
    if not len(valid_peaks):
        return np.empty((0, len(relative_time), profiles.shape[1]), dtype=np.float32), valid_peaks, relative_time
    events = np.stack([profiles[peak - half : peak + half + 1] for peak in valid_peaks]).astype(np.float32)
    return events, valid_peaks, relative_time


def profile_width(x: np.ndarray, weight: np.ndarray) -> float:
    weight = np.asarray(weight, dtype=float)
    if len(weight) < 2 or not np.isfinite(weight).any() or np.nanmax(weight) <= 0:
        return math.nan
    peak = int(np.nanargmax(weight))
    threshold = 0.5 * weight[peak]
    left = peak
    while left > 0 and weight[left - 1] >= threshold:
        left -= 1
    right = peak
    while right < len(weight) - 1 and weight[right + 1] >= threshold:
        right += 1
    return float(x[right] - x[left])


def profile_asymmetry(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=float)
    if len(profile) < 5:
        return math.nan
    peak = int(np.nanargmax(np.abs(profile)))
    radius = min(peak, len(profile) - peak - 1)
    if radius < 2:
        return 1.0
    left = profile[peak - radius : peak]
    right = profile[peak + 1 : peak + radius + 1][::-1]
    denominator = np.linalg.norm(np.r_[left, right])
    return float(np.linalg.norm(left - right) / denominator) if denominator > 0 else 0.0


def event_profile(event: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    pre_count = max(1, event.shape[0] // 5)
    # Fragmented exports may contain gauges with no finite values in one
    # event window; those columns are removed by the common-support step.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = np.nanmedian(event[:pre_count], axis=0)
    residual = event - baseline
    activity = np.sqrt(np.nanmean(residual**2, axis=1))
    center = int(np.nanargmax(activity))
    return residual[center], activity, center


def event_feature_table(
    events: np.ndarray,
    peaks: np.ndarray,
    x: np.ndarray,
    fs: float,
    continuous_profile: bool = True,
    run_ids: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not len(events):
        return pd.DataFrame(), np.empty((0, len(x))), np.empty((0, events.shape[1] if events.ndim == 3 else 0))
    profiles, temporal = [], []
    rows: list[dict] = []
    for event_index, (event, peak) in enumerate(zip(events, peaks)):
        profile, activity, center = event_profile(event)
        profiles.append(profile)
        temporal.append(activity)
        absolute = np.abs(profile)
        peak_absolute_index = int(np.nanargmax(absolute))
        peak_negative_index = int(np.nanargmin(profile))
        peak_positive_index = int(np.nanargmax(profile))
        total = float(np.nansum(absolute))
        centroid = (
            float(np.nansum(x * absolute) / total)
            if continuous_profile and total > 0
            else math.nan
        )
        edge_count = max(1, int(round(0.15 * len(profile))))
        edge_energy = float(np.nansum(absolute[:edge_count]) + np.nansum(absolute[-edge_count:]))
        peak_activity = float(np.nanmax(activity))
        half_activity = 0.5 * peak_activity
        duration = float(np.sum(activity >= half_activity) / fs) if peak_activity > 0 else math.nan
        rows.append(
            {
                "pass_index": event_index,
                "peak_sample": int(peak),
                "peak_time_s": float(peak / fs),
                "local_event_peak_offset_s": float((center - event.shape[0] // 2) / fs),
                "peak_negative_strain_microstrain": float(profile[peak_negative_index]),
                "negative_peak_magnitude_microstrain": float(
                    max(0.0, -profile[peak_negative_index])
                ),
                "peak_negative_strain_location_m": float(x[peak_negative_index]),
                "peak_positive_strain_microstrain": float(profile[peak_positive_index]),
                "peak_positive_strain_location_m": float(x[peak_positive_index]),
                "peak_absolute_microstrain": float(absolute[peak_absolute_index]),
                "peak_absolute_location_m": float(x[peak_absolute_index]),
                "absolute_spatial_integral_microstrain_m": (
                    float(trapezoid(absolute, x)) if continuous_profile else math.nan
                ),
                "absolute_centroid_m": centroid,
                "spatial_fwhm_m": (
                    profile_width(x, absolute) if continuous_profile else math.nan
                ),
                "temporal_fwhm_s": duration,
                "edge_energy_ratio": (
                    edge_energy / total if continuous_profile and total > 0 else math.nan
                ),
                "profile_asymmetry": (
                    profile_asymmetry(profile) if continuous_profile else math.nan
                ),
                "event_activity_peak": peak_activity,
                "pre_event_baseline_rms": _pre_event_baseline_rms(event),
            }
        )
    profile_matrix = np.asarray(profiles)
    temporal_matrix = np.asarray(temporal)
    median_profile = np.nanmedian(profile_matrix, axis=0)
    median_centered = median_profile - np.nanmean(median_profile)
    median_norm = np.linalg.norm(median_centered)
    for row, profile in zip(rows, profile_matrix):
        centered = profile - np.nanmean(profile)
        norm = np.linalg.norm(centered) * median_norm
        diagnostic_correlation = float(np.dot(centered, median_centered) / norm) if norm > 0 else math.nan
        row["profile_correlation"] = diagnostic_correlation if continuous_profile else math.nan
        row["fragmented_gauge_profile_correlation"] = (
            diagnostic_correlation if not continuous_profile else math.nan
        )
        diagnostic_rmse = float(
            np.sqrt(np.mean((profile - median_profile) ** 2))
        )
        row["profile_rmse_to_median"] = (
            diagnostic_rmse if continuous_profile else math.nan
        )
        row["fragmented_gauge_profile_rmse_to_median_diagnostic"] = (
            diagnostic_rmse if not continuous_profile else math.nan
        )
    frame = pd.DataFrame(rows)
    if len(frame) > 1:
        frame["interpass_interval_s"] = frame["peak_time_s"].diff()
    else:
        frame["interpass_interval_s"] = np.nan
    return frame, profile_matrix, temporal_matrix


def _pre_event_baseline_rms(event: np.ndarray) -> float:
    """RMS of per-gage pre-event medians, safely skipping all-NaN gages."""
    pre_count = max(1, event.shape[0] // 5)
    baseline_window = event[:pre_count]
    has_finite_values = np.isfinite(baseline_window).any(axis=0)
    if not has_finite_values.any():
        return math.nan
    # np.nanmedian warns for columns that are entirely NaN. Restrict the
    # reduction to columns with at least one finite pre-event sample.
    baseline_medians = np.nanmedian(
        baseline_window[:, has_finite_values], axis=0
    )
    return float(np.sqrt(np.mean(baseline_medians**2)))


RESPONSE_ANOMALY_FEATURES = [
    "negative_peak_magnitude_microstrain",
    "peak_negative_strain_location_m",
    "absolute_spatial_integral_microstrain_m",
    "spatial_fwhm_m",
    "edge_energy_ratio",
    "profile_asymmetry",
    "profile_correlation",
    "profile_rmse_to_median",
]


def add_anomaly_scores(
    features: pd.DataFrame,
    profile_matrix: np.ndarray,
    continuous_profile: bool,
    z_threshold: float | None = None,
) -> pd.DataFrame:
    """Apply the prespecified robust scalar and profile-distance screens."""
    if features.empty:
        return features
    result = features.copy()
    threshold = ROBUST_Z_THRESHOLD if z_threshold is None else float(z_threshold)
    if threshold <= 0.0:
        raise ValueError("z_threshold must be positive.")
    available = [
        column
        for column in RESPONSE_ANOMALY_FEATURES
        if column in result and np.isfinite(result[column].to_numpy(float)).sum() >= 3
    ]
    if available:
        matrix = result[available].to_numpy(float)
        medians = np.nanmedian(matrix, axis=0)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        z = robust_z(matrix)
        scalar_score = np.nanmax(np.abs(z), axis=1)
    else:
        scalar_score = np.zeros(len(result), dtype=float)
    result["scalar_anomaly_score"] = scalar_score
    result["scalar_anomaly_flag"] = scalar_score > threshold

    # Baseline instability is an acquisition-quality indicator, not an
    # unusual structural response. Score it separately so it cannot create a
    # response-anomaly flag or change anomaly-exclusion sensitivity results.
    baseline_score = np.full(len(result), np.nan, dtype=float)
    baseline_flag = np.zeros(len(result), dtype=bool)
    if "pre_event_baseline_rms" in result:
        baseline_values = result["pre_event_baseline_rms"].to_numpy(float)
        if np.isfinite(baseline_values).sum() >= 3:
            baseline_score = robust_z(baseline_values[:, None]).ravel()
            baseline_flag = np.abs(baseline_score) > threshold
    result["baseline_quality_score"] = baseline_score
    result["baseline_quality_flag"] = baseline_flag

    profile_score = np.full(len(result), np.nan, dtype=float)
    profile_flag = np.zeros(len(result), dtype=bool)
    if continuous_profile and len(profile_matrix) == len(result) and len(profile_matrix):
        median_profile = np.nanmedian(profile_matrix, axis=0)
        denominator = float(np.linalg.norm(median_profile))
        if denominator > 1e-12:
            distances = np.linalg.norm(profile_matrix - median_profile, axis=1) / denominator
            distance_z = robust_z(distances[:, None]).ravel()
            profile_score = distance_z
            profile_flag = np.abs(distance_z) > threshold
            result["profile_distance_to_median"] = distances
        else:
            result["profile_distance_to_median"] = np.nan
    else:
        result["profile_distance_to_median"] = np.nan
    result["profile_anomaly_score"] = profile_score
    result["profile_anomaly_flag"] = profile_flag
    result["response_anomaly_flag"] = result["scalar_anomaly_flag"] | result["profile_anomaly_flag"]
    # Backward-compatible aliases used by existing plots and downstream files.
    result["robust_anomaly_score"] = result["scalar_anomaly_score"]
    result["robust_anomaly_flag"] = result["scalar_anomaly_flag"]
    result["consensus_anomaly_flag"] = result["response_anomaly_flag"]
    return result


def icc3_1(profile_matrix: np.ndarray) -> float:
    """Two-way mixed, consistency, single-measure ICC across pass profiles.

    ICC(3,1) requires a balanced gauge-by-pass matrix. Spatial gauges with a
    nonfinite event-profile value are removed as complete cases before the
    ANOVA components are calculated. Missing cells are not treated as zero.
    """
    data = np.asarray(profile_matrix, dtype=float).T  # spatial targets x repeated passes
    if data.ndim != 2:
        return math.nan
    data = data[np.isfinite(data).all(axis=1)]
    if min(data.shape) < 2:
        return math.nan
    n, k = data.shape
    grand = np.mean(data)
    target_means = np.mean(data, axis=1)
    rater_means = np.mean(data, axis=0)
    ss_targets = k * np.sum((target_means - grand) ** 2)
    ss_error = np.sum(
        (data - target_means[:, None] - rater_means[None, :] + grand) ** 2
    )
    ms_targets = ss_targets / (n - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_targets + (k - 1) * ms_error
    return float((ms_targets - ms_error) / denominator) if denominator != 0 else math.nan


def ordinary_cv_percent(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return math.nan
    mean = float(np.mean(finite))
    return float(100.0 * np.std(finite, ddof=1) / abs(mean)) if abs(mean) > 1e-12 else math.nan


def robust_cv_percent(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return math.nan
    median = float(np.median(finite))
    return float(100.0 * robust_mad(finite) / abs(median)) if abs(median) > 1e-12 else math.nan


def theil_sen_repeatability(values: np.ndarray, pass_index: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    pass_index = np.asarray(pass_index, dtype=float)
    finite = np.isfinite(values) & np.isfinite(pass_index)
    if finite.sum() < 3:
        return {
            "slope_per_pass": math.nan,
            "normalized_total_drift_percent": math.nan,
            "residual_scale": math.nan,
            "repeatability_coefficient": math.nan,
            "detrended_cv_percent": math.nan,
        }
    x = pass_index[finite]
    y = values[finite]
    slope, intercept, _, _ = stats.theilslopes(y, x)
    fitted = intercept + slope * x
    residual = y - fitted
    residual_scale = float(robust_mad(residual))
    median = float(np.median(y))
    total_drift = (
        float(100.0 * slope * (x.max() - x.min()) / median)
        if median > 1e-12
        else math.nan
    )
    return {
        "slope_per_pass": float(slope),
        "normalized_total_drift_percent": total_drift,
        "residual_scale": residual_scale,
        "repeatability_coefficient": float(1.96 * math.sqrt(2.0) * residual_scale),
        "detrended_cv_percent": (
            float(100.0 * residual_scale / abs(median)) if abs(median) > 1e-12 else math.nan
        ),
    }


def pairwise_profile_metrics(profile_matrix: np.ndarray) -> dict[str, float]:
    profiles = np.asarray(profile_matrix, dtype=float)
    if profiles.ndim != 2 or len(profiles) < 2:
        return {"median_pairwise_correlation": math.nan, "median_pairwise_nrmse": math.nan}
    median_profile = np.nanmedian(profiles, axis=0)
    scale = float(np.nanpercentile(median_profile, 95) - np.nanpercentile(median_profile, 5))
    correlations: list[float] = []
    nrmse: list[float] = []
    for first, second in combinations(range(len(profiles)), 2):
        a = profiles[first]
        b = profiles[second]
        valid = np.isfinite(a) & np.isfinite(b)
        if valid.sum() < 3:
            continue
        a_valid = a[valid]
        b_valid = b[valid]
        if np.std(a_valid) > 1e-12 and np.std(b_valid) > 1e-12:
            correlations.append(float(np.corrcoef(a_valid, b_valid)[0, 1]))
        if scale > 1e-12:
            nrmse.append(float(np.sqrt(np.mean((a_valid - b_valid) ** 2)) / scale))
    return {
        "median_pairwise_correlation": float(np.median(correlations)) if correlations else math.nan,
        "median_pairwise_nrmse": float(np.median(nrmse)) if nrmse else math.nan,
    }


def early_late_profile_metrics(profile_matrix: np.ndarray, x: np.ndarray) -> dict[str, float]:
    profiles = np.asarray(profile_matrix, dtype=float)
    if profiles.ndim != 2 or len(profiles) < 4:
        return {
            "early_late_profile_correlation": math.nan,
            "early_late_profile_nrmse": math.nan,
            "early_late_peak_shift_mm": math.nan,
            "early_late_peak_amplitude_difference_microstrain": math.nan,
        }
    split = len(profiles) // 2
    early = np.nanmedian(profiles[:split], axis=0)
    late = np.nanmedian(profiles[split:], axis=0)
    valid = np.isfinite(early) & np.isfinite(late)
    scale = float(np.nanpercentile(np.nanmedian(profiles, axis=0), 95) - np.nanpercentile(np.nanmedian(profiles, axis=0), 5))
    correlation = (
        float(np.corrcoef(early[valid], late[valid])[0, 1])
        if valid.sum() >= 3 and np.std(early[valid]) > 1e-12 and np.std(late[valid]) > 1e-12
        else math.nan
    )
    nrmse = (
        float(np.sqrt(np.mean((early[valid] - late[valid]) ** 2)) / scale)
        if valid.any() and scale > 1e-12
        else math.nan
    )
    early_peak = int(np.nanargmin(early))
    late_peak = int(np.nanargmin(late))
    return {
        "early_late_profile_correlation": correlation,
        "early_late_profile_nrmse": nrmse,
        "early_late_peak_shift_mm": float(1000.0 * (x[late_peak] - x[early_peak])),
        "early_late_peak_amplitude_difference_microstrain": float(late[late_peak] - early[early_peak]),
    }


def profile_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Pearson correlation between two profiles on their common finite gauges."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if valid.sum() < 3:
        return math.nan
    first_valid = first[valid]
    second_valid = second[valid]
    if np.std(first_valid) <= 1e-12 or np.std(second_valid) <= 1e-12:
        return math.nan
    return float(np.corrcoef(first_valid, second_valid)[0, 1])


def recurring_profile_analysis(
    configuration: str,
    features: pd.DataFrame,
    profile_matrix: np.ndarray,
    x: np.ndarray,
    continuous_profile: bool,
    minimum_profile_pairs: int | None = None,
    minimum_median_correlation: float | None = None,
    minimum_gain_over_lag_one: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Detect sequence-dependent recurring profile families without forcing them.

    The diagnostic first evaluates median profile correlation at event lags.
    Multi-family recurrence is retained only when pooled pairwise correlation is
    below the single-family threshold and a non-unit lag has both strong median
    correlation and a prespecified gain over lag one.  Family labels are
    descriptive sequence positions, not wheel or bogie identities.
    """
    min_pairs = RECURRENCE_MIN_PAIRS if minimum_profile_pairs is None else int(minimum_profile_pairs)
    min_correlation = (
        RECURRENCE_MIN_MEDIAN_CORRELATION
        if minimum_median_correlation is None
        else float(minimum_median_correlation)
    )
    min_gain = (
        RECURRENCE_MIN_GAIN_OVER_LAG_ONE
        if minimum_gain_over_lag_one is None
        else float(minimum_gain_over_lag_one)
    )
    annotated = features.copy()
    annotated["recurrence_family_id"] = pd.Series(
        pd.array([pd.NA] * len(annotated), dtype="Int64"), index=annotated.index
    )
    annotated["profile_correlation_with_family_median"] = np.nan
    profiles = np.asarray(profile_matrix, dtype=float)
    base_summary = {
        "configuration": configuration,
        "continuous_profile": bool(continuous_profile),
        "n_events": int(len(annotated)),
        "maximum_tested_lag_events": None,
        "pooled_median_pairwise_correlation": math.nan,
        "lag_one_median_correlation": math.nan,
        "detected_recurrence_lag_events": None,
        "detected_recurrence_median_correlation": math.nan,
        "gain_over_lag_one": math.nan,
        "n_profile_families": 0,
        "family_definition": "event index modulo the detected recurrence lag",
        "interpretation_warning": (
            "Sequence families are descriptive and are not wheel or bogie identities "
            "without an independently synchronized event-to-load record."
        ),
    }
    if not continuous_profile:
        return annotated, pd.DataFrame(), pd.DataFrame(), {
            **base_summary,
            "status": "not applicable to fragmented-gauge profiles",
            "reason": "A continuous common spatial profile is required.",
        }
    if profiles.ndim != 2 or len(profiles) != len(annotated) or len(profiles) < 4:
        return annotated, pd.DataFrame(), pd.DataFrame(), {
            **base_summary,
            "status": "insufficient complete continuous profiles",
            "reason": "At least four complete profiles are required.",
        }

    maximum_lag = min(RECURRENCE_MAX_LAG_EVENTS, len(profiles) - 1)
    lag_rows: list[dict] = []
    for lag in range(1, maximum_lag + 1):
        correlations = np.asarray(
            [
                profile_correlation(profiles[index], profiles[index + lag])
                for index in range(len(profiles) - lag)
            ],
            dtype=float,
        )
        finite = correlations[np.isfinite(correlations)]
        lag_rows.append(
            {
                "configuration": configuration,
                "lag_events": int(lag),
                "n_profile_pairs": int(len(finite)),
                "median_correlation": float(np.median(finite)) if len(finite) else math.nan,
                "q25_correlation": float(np.percentile(finite, 25)) if len(finite) else math.nan,
                "q75_correlation": float(np.percentile(finite, 75)) if len(finite) else math.nan,
                "minimum_correlation": float(np.min(finite)) if len(finite) else math.nan,
                "maximum_correlation": float(np.max(finite)) if len(finite) else math.nan,
            }
        )
    lag_frame = pd.DataFrame(lag_rows)
    pooled_correlation = pairwise_profile_metrics(profiles)[
        "median_pairwise_correlation"
    ]
    lag_one = float(lag_frame.loc[lag_frame["lag_events"] == 1, "median_correlation"].iloc[0])
    summary = {
        **base_summary,
        "maximum_tested_lag_events": int(maximum_lag),
        "pooled_median_pairwise_correlation": pooled_correlation,
        "lag_one_median_correlation": lag_one,
        "detection_thresholds": {
            "minimum_profile_pairs": min_pairs,
            "minimum_median_correlation": min_correlation,
            "minimum_gain_over_lag_one": min_gain,
            "single_family_correlation": RECURRENCE_SINGLE_FAMILY_CORRELATION,
            "fundamental_lag_tolerance": RECURRENCE_FUNDAMENTAL_TOLERANCE,
        },
    }

    detected_lag: int | None = None
    if np.isfinite(pooled_correlation) and pooled_correlation >= RECURRENCE_SINGLE_FAMILY_CORRELATION:
        status = "single stable profile family"
        family_count = 1
    else:
        candidates = lag_frame[
            (lag_frame["lag_events"] >= 2)
            & (lag_frame["n_profile_pairs"] >= min_pairs)
            & (lag_frame["median_correlation"] >= min_correlation)
            & (
                lag_frame["median_correlation"] - lag_one
                >= min_gain
            )
        ].copy()
        if candidates.empty:
            return annotated, lag_frame, pd.DataFrame(), {
                **summary,
                "status": "no strong recurring profile family detected",
                "reason": "No non-unit lag satisfied all prespecified recurrence criteria.",
            }
        best_correlation = float(candidates["median_correlation"].max())
        fundamental_candidates = candidates[
            candidates["median_correlation"]
            >= best_correlation - RECURRENCE_FUNDAMENTAL_TOLERANCE
        ]
        detected_lag = int(fundamental_candidates["lag_events"].min())
        status = "multi-family recurrence detected"
        family_count = detected_lag

    family_rows: list[dict] = []
    event_indices = np.arange(len(profiles), dtype=int)
    if detected_lag is None:
        family_assignments = np.zeros(len(profiles), dtype=int)
    else:
        family_assignments = event_indices % detected_lag
    annotated["recurrence_family_id"] = pd.array(family_assignments, dtype="Int64")

    for family_id in range(family_count):
        indices = event_indices[family_assignments == family_id]
        family_profiles = profiles[indices]
        family_median_profile = np.nanmedian(family_profiles, axis=0)
        family_correlations = np.asarray(
            [profile_correlation(profile, family_median_profile) for profile in family_profiles],
            dtype=float,
        )
        annotated.loc[
            annotated.index[indices], "profile_correlation_with_family_median"
        ] = family_correlations
        family_features = annotated.iloc[indices]
        family_pairwise = pairwise_profile_metrics(family_profiles)
        negative_peak = family_features[
            "negative_peak_magnitude_microstrain"
        ].to_numpy(float)
        spatial_integral = family_features[
            "absolute_spatial_integral_microstrain_m"
        ].to_numpy(float)
        family_rows.append(
            {
                "configuration": configuration,
                "recurrence_lag_events": int(detected_lag) if detected_lag else 1,
                "family_id": int(family_id),
                "event_indices": ";".join(str(int(index)) for index in indices),
                "n_events": int(len(indices)),
                "median_negative_peak_microstrain": float(np.nanmedian(negative_peak)),
                "negative_peak_cv_percent": ordinary_cv_percent(negative_peak),
                "negative_peak_robust_cv_percent": robust_cv_percent(negative_peak),
                "median_negative_peak_location_m": float(
                    np.nanmedian(family_features["peak_negative_strain_location_m"])
                ),
                "median_absolute_spatial_integral_microstrain_m": float(
                    np.nanmedian(spatial_integral)
                )
                if np.isfinite(spatial_integral).any()
                else math.nan,
                "spatial_integral_cv_percent": ordinary_cv_percent(spatial_integral),
                "median_absolute_centroid_m": float(
                    np.nanmedian(family_features["absolute_centroid_m"])
                ),
                "median_absolute_centroid_local_m": float(
                    np.nanmedian(family_features["absolute_centroid_m"]) - x[0]
                ),
                "median_spatial_fwhm_m": float(
                    np.nanmedian(family_features["spatial_fwhm_m"])
                ),
                "median_profile_correlation_with_family_median": float(
                    np.nanmedian(family_correlations)
                ),
                "median_within_family_pairwise_correlation": family_pairwise[
                    "median_pairwise_correlation"
                ],
                "within_family_icc3_1": icc3_1(family_profiles),
                "response_anomaly_count_from_pooled_screen": int(
                    family_features["response_anomaly_flag"].sum()
                ),
            }
        )
    family_frame = pd.DataFrame(family_rows)

    if detected_lag is not None:
        detected_row = lag_frame.loc[lag_frame["lag_events"] == detected_lag].iloc[0]
        detected_correlation = float(detected_row["median_correlation"])
        gain = float(detected_correlation - lag_one)
    else:
        detected_correlation = math.nan
        gain = math.nan
    summary.update(
        {
            "status": status,
            "detected_recurrence_lag_events": detected_lag,
            "detected_recurrence_median_correlation": detected_correlation,
            "gain_over_lag_one": gain,
            "n_profile_families": int(family_count),
            "families": family_frame.to_dict(orient="records"),
        }
    )
    return annotated, lag_frame, family_frame, summary


def repeatability_summary(
    configuration: str,
    features: pd.DataFrame,
    profile_matrix: np.ndarray,
    detection: DetectionResult,
    x: np.ndarray,
    continuous_profile: bool,
) -> dict:
    if features.empty:
        return {
            "configuration": configuration,
            "n_detected_peaks": int(len(detection.peaks)),
            "n_complete_events": 0,
            "status": "insufficient complete passes",
        }
    pass_index = features["pass_index"].to_numpy(float)
    negative_peak = features["negative_peak_magnitude_microstrain"].to_numpy(float)
    spatial_integral = features["absolute_spatial_integral_microstrain_m"].to_numpy(float)
    baseline_rms = features["pre_event_baseline_rms"].to_numpy(float)
    peak_location = features["peak_negative_strain_location_m"].to_numpy(float)

    peak_repeatability = theil_sen_repeatability(negative_peak, pass_index)
    integral_repeatability = theil_sen_repeatability(spatial_integral, pass_index)
    baseline_repeatability = theil_sen_repeatability(baseline_rms, pass_index)
    pairwise = pairwise_profile_metrics(profile_matrix) if continuous_profile else {
        "median_pairwise_correlation": math.nan,
        "median_pairwise_nrmse": math.nan,
    }
    early_late = early_late_profile_metrics(profile_matrix, x) if continuous_profile else {
        "early_late_profile_correlation": math.nan,
        "early_late_profile_nrmse": math.nan,
        "early_late_peak_shift_mm": math.nan,
        "early_late_peak_amplitude_difference_microstrain": math.nan,
    }
    intervals = features["interpass_interval_s"].dropna().to_numpy(float)
    interval_samples = np.diff(features["peak_sample"].to_numpy(float))
    return {
        "configuration": configuration,
        "description": CONFIGURATION_DESCRIPTIONS.get(configuration, ""),
        "n_detected_peaks": int(len(detection.peaks)),
        "n_complete_events": int(len(features)),
        "dominant_frequency_hz": detection.dominant_frequency_hz,
        "autocorrelation_frequency_hz": detection.autocorrelation_frequency_hz,
        "autocorrelation_peak": detection.autocorrelation_peak,
        "used_pass_frequency_hz": detection.used_pass_frequency_hz,
        "used_frequency_source": detection.used_frequency_source,
        "mean_interpass_interval_s": float(np.mean(intervals)) if len(intervals) else None,
        "median_interpass_interval_s": float(np.median(intervals)) if len(intervals) else None,
        "mean_interpass_interval_samples": float(np.mean(interval_samples))
        if len(interval_samples)
        else None,
        "median_interpass_interval_samples": float(np.median(interval_samples))
        if len(interval_samples)
        else None,
        "interpass_interval_cv_percent": ordinary_cv_percent(intervals),
        "observed_mean_interpass_frequency_hz": float(1.0 / np.mean(intervals))
        if len(intervals)
        else None,
        "observed_median_interpass_frequency_hz": float(1.0 / np.median(intervals))
        if len(intervals)
        else None,
        "negative_peak_mean_microstrain": float(np.nanmean(negative_peak)),
        "negative_peak_median_microstrain": float(np.nanmedian(negative_peak)),
        "negative_peak_std_microstrain": float(np.nanstd(negative_peak, ddof=1))
        if len(negative_peak) > 1
        else None,
        "negative_peak_cv_percent": ordinary_cv_percent(negative_peak),
        "negative_peak_robust_cv_percent": robust_cv_percent(negative_peak),
        "negative_peak_detrended_cv_percent": peak_repeatability["detrended_cv_percent"],
        "negative_peak_repeatability_coefficient_microstrain": peak_repeatability["repeatability_coefficient"],
        "median_negative_peak_location_m": float(np.nanmedian(peak_location)),
        "negative_peak_location_sd_mm": float(1000 * np.nanstd(peak_location, ddof=1))
        if len(peak_location) > 1
        else None,
        "spatial_integral_median_microstrain_m": float(np.nanmedian(spatial_integral)) if np.isfinite(spatial_integral).any() else math.nan,
        "spatial_integral_cv_percent": ordinary_cv_percent(spatial_integral),
        "spatial_integral_robust_cv_percent": robust_cv_percent(spatial_integral),
        "spatial_integral_repeatability_coefficient_microstrain_m": integral_repeatability["repeatability_coefficient"],
        "median_profile_correlation": float(np.nanmedian(features["profile_correlation"])) if continuous_profile else math.nan,
        "median_edge_energy_ratio": float(np.nanmedian(features["edge_energy_ratio"])) if continuous_profile else math.nan,
        "median_profile_asymmetry": float(np.nanmedian(features["profile_asymmetry"])) if continuous_profile else math.nan,
        "profile_icc3_1": icc3_1(profile_matrix) if continuous_profile else math.nan,
        "fragmented_gauge_icc3_1_diagnostic": icc3_1(profile_matrix) if not continuous_profile else math.nan,
        **pairwise,
        **early_late,
        "negative_peak_drift_microstrain_per_pass": peak_repeatability["slope_per_pass"],
        "negative_peak_total_drift_percent": peak_repeatability["normalized_total_drift_percent"],
        "spatial_integral_drift_microstrain_m_per_pass": integral_repeatability["slope_per_pass"],
        "spatial_integral_total_drift_percent": integral_repeatability["normalized_total_drift_percent"],
        "baseline_rms_drift_microstrain_per_pass": baseline_repeatability["slope_per_pass"],
        "baseline_rms_total_drift_percent": baseline_repeatability["normalized_total_drift_percent"],
        "scalar_anomaly_fraction": float(features["scalar_anomaly_flag"].mean()),
        "profile_anomaly_fraction": float(features["profile_anomaly_flag"].mean()),
        "response_anomaly_fraction": float(features["response_anomaly_flag"].mean()),
        "baseline_quality_flag_fraction": float(
            features["baseline_quality_flag"].mean()
        ),
        "status": "continuous-profile analysis" if continuous_profile else "fragmented-gauge diagnostic",
        "warning": (
            "Repeated passes from one test are correlated and are not independent specimens."
        ),
    }


def circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=math.ceil(n / block_length))
    blocks = [((start + np.arange(block_length)) % n) for start in starts]
    return np.concatenate(blocks)[:n]


def within_configuration_bootstrap(
    features: pd.DataFrame,
    profile_matrix: np.ndarray,
    continuous_profile: bool,
    iterations: int = BLOCK_BOOTSTRAP_ITERATIONS,
) -> tuple[pd.DataFrame, dict]:
    """Circular-block uncertainty for the observed repeated loading cycles."""
    n = len(features)
    if n < 4:
        return pd.DataFrame(), {"iterations": 0, "block_length_passes": None}
    block_length = max(2, int(math.ceil(n ** (1.0 / 3.0))))
    rng = np.random.default_rng(RANDOM_SEED)
    peak = features["negative_peak_magnitude_microstrain"].to_numpy(float)
    integral = features["absolute_spatial_integral_microstrain_m"].to_numpy(float)
    rows: list[dict] = []
    for iteration in range(iterations):
        indices = circular_block_indices(n, block_length, rng)
        sampled_peak = peak[indices]
        sampled_integral = integral[indices]
        sampled_profiles = profile_matrix[indices]
        peak_drift = theil_sen_repeatability(sampled_peak, np.arange(n, dtype=float))
        row = {
            "iteration": iteration,
            "negative_peak_median_microstrain": float(np.nanmedian(sampled_peak)),
            "negative_peak_cv_percent": ordinary_cv_percent(sampled_peak),
            "negative_peak_robust_cv_percent": robust_cv_percent(sampled_peak),
            "negative_peak_total_drift_percent": peak_drift["normalized_total_drift_percent"],
            "spatial_integral_median_microstrain_m": (
                float(np.nanmedian(sampled_integral))
                if np.isfinite(sampled_integral).any()
                else math.nan
            ),
            "profile_icc3_1": (
                icc3_1(sampled_profiles) if continuous_profile else math.nan
            ),
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = {
        "iterations": int(iterations),
        "block_length_passes": int(block_length),
        "interval_method": "2.5th and 97.5th percentiles of circular-block resamples",
    }
    for column in frame.columns:
        if column == "iteration":
            continue
        values = frame[column].to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values):
            summary[f"{column}_ci95_low"] = float(np.percentile(values, 2.5))
            summary[f"{column}_ci95_high"] = float(np.percentile(values, 97.5))
    return frame, summary


# =============================================================================
# Filter comparison and spatial degradation
# =============================================================================


def match_peaks(reference: np.ndarray, candidate: np.ndarray, tolerance_samples: int) -> tuple[int, float, float]:
    if not len(reference) or not len(candidate):
        return 0, 0.0, 0.0
    used: set[int] = set()
    matched = 0
    for ref in reference:
        distances = np.abs(candidate - ref)
        order = np.argsort(distances)
        for index in order:
            if int(index) not in used and distances[index] <= tolerance_samples:
                used.add(int(index))
                matched += 1
                break
    recall = matched / len(reference)
    precision = matched / len(candidate)
    return matched, recall, precision


def manual_detection_validation(
    configuration: str,
    detected_peaks: np.ndarray,
    fs: float,
    tolerance_samples: int,
) -> dict:
    """Compare detections with optional external manual annotations.

    An optional CSV named <configuration>.csv may contain a zero-based
    ``sample`` column or a ``time_s`` column. Annotation files are not bundled
    because they are data-derived and may require separate redistribution
    permission.
    """
    path = MANUAL_LABEL_DIR / f"{configuration}.csv"
    if not path.is_file():
        return {
            "status": "manual labels not provided",
            "expected_file": path.name,
            "required_column": "time_s or sample",
        }

    labels = pd.read_csv(path)
    if "sample" in labels:
        manual = pd.to_numeric(labels["sample"], errors="coerce")
        manual_peaks = np.rint(manual.dropna()).astype(int).to_numpy()
    elif "time_s" in labels:
        times = pd.to_numeric(labels["time_s"], errors="coerce").dropna().to_numpy(float)
        manual_peaks = np.rint(times * fs).astype(int)
    else:
        raise ValueError(f"{path.name} must contain a 'time_s' or 'sample' column")

    manual_peaks = np.unique(manual_peaks[manual_peaks >= 0])
    matched, recall, precision = match_peaks(manual_peaks, detected_peaks, tolerance_samples)
    f1 = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    timing_errors = []
    used: set[int] = set()
    for manual_peak in manual_peaks:
        distances = np.abs(detected_peaks - manual_peak)
        for candidate_index in np.argsort(distances):
            if int(candidate_index) not in used and distances[candidate_index] <= tolerance_samples:
                used.add(int(candidate_index))
                timing_errors.append(float((detected_peaks[candidate_index] - manual_peak) / fs))
                break
    return {
        "status": "computed",
        "manual_label_source": path.name,
        "n_manual": int(len(manual_peaks)),
        "n_detected": int(len(detected_peaks)),
        "true_positive": int(matched),
        "false_negative": int(len(manual_peaks) - matched),
        "false_positive": int(len(detected_peaks) - matched),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "median_absolute_timing_error_s": (
            float(np.median(np.abs(timing_errors))) if timing_errors else math.nan
        ),
        "match_tolerance_samples": int(tolerance_samples),
    }


def build_event_review_table(
    reference: DetectionResult,
    alternative_detections: dict[str, DetectionResult],
    complete_peaks: np.ndarray,
    fs: float,
) -> pd.DataFrame:
    complete_set = set(int(value) for value in complete_peaks)
    rows = []
    intervals = np.diff(reference.peaks)
    median_interval = float(np.median(intervals)) if len(intervals) else math.nan
    tolerance = max(
        1,
        int(round(PEAK_MATCH_TOLERANCE_PERIOD_FRACTION * fs / reference.used_pass_frequency_hz)),
    )
    for index, peak in enumerate(reference.peaks):
        matches = {"raw": True}
        for name, detection in alternative_detections.items():
            matches[name] = bool(
                len(detection.peaks) and np.min(np.abs(detection.peaks - peak)) <= tolerance
            )
        previous_interval = float(peak - reference.peaks[index - 1]) if index > 0 else math.nan
        gap_ratio = previous_interval / median_interval if index > 0 and median_interval > 0 else math.nan
        cadence_multiple = (
            max(1, int(round(gap_ratio))) if np.isfinite(gap_ratio) else None
        )
        cadence_residual = (
            abs(gap_ratio - cadence_multiple)
            if cadence_multiple is not None
            else math.nan
        )
        cadence_score = (
            max(0.0, 1.0 - cadence_residual / 0.30)
            if np.isfinite(cadence_residual)
            else 1.0
        )
        agreement_count = int(sum(matches.values()))
        agreement_score = agreement_count / max(1, len(matches))
        confidence_score = 0.75 * agreement_score + 0.25 * cadence_score
        if int(peak) not in complete_set:
            confidence_score *= 0.85
        confidence_class = (
            "high"
            if confidence_score >= 0.85
            else "moderate"
            if confidence_score >= 0.65
            else "low"
        )
        rows.append(
            {
                "candidate_index": index,
                "automatic_pass_id": index + 1,
                "automatic_pass_label": True,
                "sample": int(peak),
                "time_s": float(peak / fs),
                "complete_window": bool(int(peak) in complete_set),
                "matched_raw": True,
                "matched_gaussian": matches.get("gaussian", False),
                "matched_savgol": matches.get("savgol", False),
                "filter_agreement_count": agreement_count,
                "previous_interval_samples": previous_interval,
                "previous_gap_ratio_to_median": gap_ratio,
                "previous_cadence_multiple": cadence_multiple,
                "cadence_score": cadence_score,
                "automatic_confidence_score": confidence_score,
                "automatic_confidence_class": confidence_class,
                "confidence_warning": "Heuristic quality score; not a calibrated probability.",
                "manual_review_required": bool(
                    agreement_count < 2
                    or int(peak) not in complete_set
                    or (np.isfinite(cadence_residual) and cadence_residual > 0.30)
                    or (cadence_multiple is not None and cadence_multiple > 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def quick_feature_metrics(
    profiles: np.ndarray,
    x: np.ndarray,
    fs: float,
    activity_filter: str = "raw",
    continuous_profile: bool = True,
    run_ids: np.ndarray | None = None,
    reference_detection: DetectionResult | None = None,
    lock_detector: bool = False,
) -> dict:
    detection = detect_passes(
        profiles,
        fs,
        activity_filter=activity_filter,
        reference=reference_detection,
        lock_thresholds=lock_detector,
    )
    events, peaks, _ = extract_events(
        profiles, detection.peaks, fs, detection.used_pass_frequency_hz
    )
    features, profile_matrix, _ = event_feature_table(
        events,
        peaks,
        x,
        fs,
        continuous_profile=continuous_profile,
        run_ids=run_ids,
    )
    output = {
        "n_detected": int(len(detection.peaks)),
        "n_complete": int(len(features)),
        "dominant_frequency_hz": detection.dominant_frequency_hz,
        "autocorrelation_frequency_hz": detection.autocorrelation_frequency_hz,
        "used_pass_frequency_hz": detection.used_pass_frequency_hz,
        "used_frequency_source": detection.used_frequency_source,
        "median_negative_peak_magnitude_microstrain": float(
            np.nanmedian(features["negative_peak_magnitude_microstrain"])
        )
        if not features.empty
        else None,
        "negative_peak_cv_percent": float(
            100
            * np.nanstd(features["negative_peak_magnitude_microstrain"], ddof=1)
            / np.nanmean(features["negative_peak_magnitude_microstrain"])
        )
        if len(features) > 1
        and np.nanmean(features["negative_peak_magnitude_microstrain"]) > 0
        else None,
        "median_negative_peak_location_m": float(
            np.nanmedian(features["peak_negative_strain_location_m"])
        )
        if not features.empty
        else None,
        "median_spatial_integral_microstrain_m": float(
            np.nanmedian(features["absolute_spatial_integral_microstrain_m"])
        )
        if not features.empty
        and np.isfinite(
            features["absolute_spatial_integral_microstrain_m"].to_numpy(float)
        ).any()
        else None,
        "median_spatial_fwhm_m": float(np.nanmedian(features["spatial_fwhm_m"]))
        if not features.empty
        and np.isfinite(features["spatial_fwhm_m"].to_numpy(float)).any()
        else None,
        "median_profile_correlation": float(np.nanmedian(features["profile_correlation"]))
        if not features.empty
        and np.isfinite(features["profile_correlation"].to_numpy(float)).any()
        else None,
        "profile_icc3_1": icc3_1(profile_matrix) if continuous_profile else math.nan,
        "fragmented_gauge_icc3_1_diagnostic": (
            icc3_1(profile_matrix) if not continuous_profile else math.nan
        ),
    }
    if reference_detection is not None:
        tolerance = max(
            1,
            int(
                round(
                    PEAK_MATCH_TOLERANCE_PERIOD_FRACTION
                    * fs
                    / reference_detection.used_pass_frequency_hz
                )
            ),
        )
        matched, recall, precision = match_peaks(
            reference_detection.peaks, detection.peaks, tolerance
        )
        output.update(
            {
                "matched_peaks": matched,
                "detection_recall": recall,
                "detection_precision": precision,
                "peak_match_tolerance_samples": tolerance,
            }
        )
    return output


def filter_comparison(
    profiles: np.ndarray,
    x: np.ndarray,
    fs: float,
    continuous_profile: bool,
    run_ids: np.ndarray,
) -> pd.DataFrame:
    rows = []
    reference = detect_passes(profiles, fs, activity_filter="raw")
    for method in ("raw", "gaussian", "savgol"):
        rows.append(
            {
                "detection_filter": method,
                **quick_feature_metrics(
                    profiles,
                    x,
                    fs,
                    activity_filter=method,
                    continuous_profile=continuous_profile,
                    run_ids=run_ids,
                    reference_detection=reference,
                    lock_detector=False,
                ),
            }
        )
    return pd.DataFrame(rows)


def add_degradation_errors(metrics: dict, reference_metrics: dict) -> dict:
    result = dict(metrics)
    relative_pairs = [
        (
            "median_negative_peak_magnitude_microstrain",
            "negative_peak_error_percent",
        ),
        (
            "median_spatial_integral_microstrain_m",
            "spatial_integral_error_percent",
        ),
        ("median_spatial_fwhm_m", "spatial_fwhm_error_percent"),
    ]
    for source, target in relative_pairs:
        value = result.get(source)
        reference = reference_metrics.get(source)
        if value is None or reference is None or abs(reference) <= 1e-12:
            result[target] = None
        else:
            result[target] = float(100 * (value - reference) / reference)
    location = result.get("median_negative_peak_location_m")
    reference_location = reference_metrics.get("median_negative_peak_location_m")
    result["negative_peak_location_error_mm"] = (
        float(1000 * (location - reference_location))
        if location is not None and reference_location is not None
        else None
    )
    return result


def spatial_resolution_study(
    profiles: np.ndarray,
    x: np.ndarray,
    fs: float,
    reference_detection: DetectionResult,
) -> pd.DataFrame:
    rows = []
    base_pitch_mm = 1000 * float(np.nanmedian(np.diff(x)))
    reference_metrics = quick_feature_metrics(
        profiles,
        x,
        fs,
        reference_detection=reference_detection,
        lock_detector=True,
    )
    for factor in SPATIAL_PITCH_FACTORS:
        # Evaluate every possible grid phase; using only the first four offsets
        # can miss the worst placement at coarse pitch.
        for offset in range(factor):
            selected = np.arange(offset, len(x), factor)
            if len(selected) < 8:
                continue
            metrics = add_degradation_errors(
                quick_feature_metrics(
                    profiles[:, selected],
                    x[selected],
                    fs,
                    reference_detection=reference_detection,
                    lock_detector=True,
                ),
                reference_metrics,
            )
            rows.append(
                {
                    "degradation": "pitch",
                    "factor": factor,
                    "offset": offset,
                    "effective_pitch_mm": base_pitch_mm * factor,
                    "remaining_gages": int(len(selected)),
                    **metrics,
                }
            )

    rng = np.random.default_rng(RANDOM_SEED)
    x_span = x[-1] - x[0]
    for missing_mm in MISSING_SEGMENT_LENGTHS_MM:
        missing_m = missing_mm / 1000.0
        if missing_m >= 0.8 * x_span:
            continue
        for trial in range(MISSING_SEGMENT_TRIALS):
            start = rng.uniform(x[0], x[-1] - missing_m)
            keep = ~((x >= start) & (x <= start + missing_m))
            if keep.sum() < 8:
                continue
            metrics = add_degradation_errors(
                quick_feature_metrics(
                    profiles[:, keep],
                    x[keep],
                    fs,
                    reference_detection=reference_detection,
                    lock_detector=True,
                ),
                reference_metrics,
            )
            rows.append(
                {
                    "degradation": "missing_contiguous_segment",
                    "factor": None,
                    "offset": trial,
                    "effective_pitch_mm": base_pitch_mm,
                    "missing_length_mm": missing_mm,
                    "missing_start_m": start,
                    "remaining_gages": int(keep.sum()),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# Configuration comparison and figures
# =============================================================================


def block_bootstrap_median_difference(
    first: np.ndarray,
    second: np.ndarray,
    iterations: int = BLOCK_BOOTSTRAP_ITERATIONS,
) -> tuple[float, float, float, int | None]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    if min(len(first), len(second)) < 5:
        return math.nan, math.nan, math.nan, None
    rng = np.random.default_rng(RANDOM_SEED)

    def sample_blocks(values: np.ndarray) -> np.ndarray:
        block = max(2, int(math.ceil(len(values) ** (1.0 / 3.0))))
        return values[circular_block_indices(len(values), block, rng)]

    differences = np.empty(iterations)
    for index in range(iterations):
        differences[index] = np.median(sample_blocks(first)) - np.median(sample_blocks(second))
    estimate = float(np.median(first) - np.median(second))
    low, high = np.percentile(differences, [2.5, 97.5])
    block = max(
        max(2, int(math.ceil(len(first) ** (1.0 / 3.0)))),
        max(2, int(math.ceil(len(second) ** (1.0 / 3.0)))),
    )
    return estimate, float(low), float(high), block


def configuration_comparisons(features_by_config: dict[str, pd.DataFrame]) -> pd.DataFrame:
    measures = [
        "negative_peak_magnitude_microstrain",
        "profile_correlation",
        "edge_energy_ratio",
        "profile_asymmetry",
        "pre_event_baseline_rms",
    ]
    rows = []
    for first, second in combinations(features_by_config, 2):
        if features_by_config[first].empty or features_by_config[second].empty:
            continue
        for measure in measures:
            estimate, low, high, block = block_bootstrap_median_difference(
                features_by_config[first][measure].to_numpy(),
                features_by_config[second][measure].to_numpy(),
            )
            rows.append(
                {
                    "configuration_a": first,
                    "configuration_b": second,
                    "measure": measure,
                    "median_a_minus_b": estimate,
                    "block_bootstrap_ci95_low": low,
                    "block_bootstrap_ci95_high": high,
                    "maximum_block_size_passes": block,
                    "interpretation_warning": (
                        "Configuration is confounded with date/specimen/setup; this is descriptive, not causal."
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_roi_selection(
    item: ODiSIFile, roi_indices: np.ndarray, score: np.ndarray, output: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    axis.plot(item.distance_m, score, lw=1.0, color="#315a7d")
    axis.axvspan(
        item.distance_m[roi_indices[0]],
        item.distance_m[roi_indices[-1]],
        color="#d95f02",
        alpha=0.25,
        label="Selected interval",
    )
    axis.set(
        title=f"ROI selection: {item.configuration}",
        xlabel="ODiSI distance (m)",
        ylabel="Robust temporal range (microstrain)",
    )
    axis.legend()
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_configuration_overview(
    configuration: str,
    profiles: np.ndarray,
    x: np.ndarray,
    fs: float,
    detection: DetectionResult,
    events: np.ndarray,
    relative_time: np.ndarray,
    features: pd.DataFrame,
    profile_matrix: np.ndarray,
    output: Path,
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(14, 13), constrained_layout=True)
    limit = min(len(profiles), max(2, int(round(HEATMAP_SECONDS * fs))))
    image = axes[0, 0].imshow(
        profiles[:limit].T,
        aspect="auto",
        origin="lower",
        extent=[0, limit / fs, x[0], x[-1]],
        cmap="coolwarm",
        **percentile_norm(profiles[:limit]),
    )
    figure.colorbar(image, ax=axes[0, 0], label="Strain (microstrain)")
    axes[0, 0].set(title=f"Full-rate strain field (first {limit/fs:.1f} s)", xlabel="Time (s)", ylabel="Distance (m)")

    axes[0, 1].plot(detection.time_s, detection.detrended_activity, lw=0.8)
    axes[0, 1].scatter(
        detection.time_s[detection.peaks],
        detection.detrended_activity[detection.peaks],
        s=14,
        color="#d95f02",
        label=f"Detected passes: {len(detection.peaks)}",
    )
    axes[0, 1].set(title="Spatial activity and detected passes", xlabel="Time (s)", ylabel="RMS activity")
    axes[0, 1].legend()

    if len(detection.psd_frequency_hz):
        mask = detection.psd_frequency_hz <= max(
            8.0, 1.25 * detection.used_pass_frequency_hz
        )
        axes[1, 0].semilogy(detection.psd_frequency_hz[mask], detection.psd[mask])
        axes[1, 0].axvline(
            detection.used_pass_frequency_hz,
            ls="--",
            color="#D55E00",
            label=f"Observed {detection.used_pass_frequency_hz:.3f} Hz",
        )
        axes[1, 0].axvline(
            REPORT_NOMINAL_PASS_FREQUENCY_HZ,
            ls=":",
            color="0.35",
            label="Report reference: 2 Hz",
        )
        axes[1, 0].legend()
    axes[1, 0].set(title="Pass-frequency verification", xlabel="Frequency (Hz)", ylabel="PSD")

    if len(events):
        median_event = np.nanmedian(events, axis=0)
        image2 = axes[1, 1].imshow(
            median_event.T,
            aspect="auto",
            origin="lower",
            extent=[relative_time[0], relative_time[-1], x[0], x[-1]],
            cmap="coolwarm",
            **percentile_norm(median_event),
        )
        figure.colorbar(image2, ax=axes[1, 1], label="Strain (microstrain)")
    axes[1, 1].set(title="Median event-aligned strain field", xlabel="Time from pass center (s)", ylabel="Distance (m)")

    if len(profile_matrix):
        median = np.nanmedian(profile_matrix, axis=0)
        low, high = np.nanpercentile(profile_matrix, [2.5, 97.5], axis=0)
        for profile in profile_matrix[: min(10, len(profile_matrix))]:
            axes[2, 0].plot(x, profile, color="0.7", alpha=0.35, lw=0.7)
        axes[2, 0].fill_between(
            x,
            low,
            high,
            color="#56B4E9",
            alpha=0.25,
            label="Pointwise 95% pass envelope",
        )
        axes[2, 0].plot(x, median, color="#0072B2", lw=2, label="Median pass")
        axes[2, 0].legend()
    axes[2, 0].set(title="Event peak spatial profiles", xlabel="Distance (m)", ylabel="Baseline-corrected strain")

    if not features.empty:
        color = np.where(features["consensus_anomaly_flag"], "#D55E00", "#009E73")
        axes[2, 1].scatter(
            features["pass_index"],
            features["negative_peak_magnitude_microstrain"],
            c=color,
            s=20,
        )
    axes[2, 1].set(
        title="Negative-strain peak and response-anomaly flags",
        xlabel="Pass index",
        ylabel="Negative peak magnitude (microstrain)",
    )
    figure.suptitle(f"{configuration} (effective saved-data rate: {fs:g} Hz)", fontsize=15)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def percentile_norm(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"vmin": -1.0, "vmax": 1.0}
    low, high = np.percentile(finite, [1, 99])
    limit = max(abs(float(low)), abs(float(high)), 1e-9)
    return {"vmin": -limit, "vmax": limit}


def plot_within_run_stability(
    configuration: str, features: pd.DataFrame, output: Path
) -> None:
    if features.empty:
        return
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = features["pass_index"].to_numpy(float)
    panels = [
        (
            "negative_peak_magnitude_microstrain",
            "Negative peak magnitude",
            "Microstrain",
        ),
        (
            "absolute_spatial_integral_microstrain_m",
            "Absolute spatial integral",
            "Microstrain m",
        ),
        ("pre_event_baseline_rms", "Pre-event baseline RMS", "Microstrain"),
        ("absolute_centroid_m", "Absolute-response centroid", "Distance (m)"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), panels):
        y = features[column].to_numpy(float)
        axis.plot(x, y, "o-", ms=3, lw=0.8, color="#0072B2")
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() >= 3:
            fit = stats.linregress(x[finite], y[finite])
            axis.plot(
                x[finite],
                fit.intercept + fit.slope * x[finite],
                "--",
                color="#D55E00",
                lw=1.4,
                label=f"Slope={fit.slope:.3g}/pass",
            )
            axis.legend(fontsize=8)
        axis.set(title=title, xlabel="Pass index", ylabel=ylabel)
    figure.suptitle(f"Within-run stability: {configuration}")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_recurring_profile_analysis(
    configuration: str,
    x: np.ndarray,
    profile_matrix: np.ndarray,
    lag_frame: pd.DataFrame,
    recurrence_summary: dict,
    output: Path,
) -> None:
    """Visualize lag evidence and any detected sequence profile families."""
    if lag_frame.empty or np.asarray(profile_matrix).ndim != 2:
        return
    profiles = np.asarray(profile_matrix, dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    lag = lag_frame["lag_events"].to_numpy(int)
    median = lag_frame["median_correlation"].to_numpy(float)
    q25 = lag_frame["q25_correlation"].to_numpy(float)
    q75 = lag_frame["q75_correlation"].to_numpy(float)
    axes[0].plot(lag, median, "o-", color="#0072B2", lw=1.8, label="Median")
    axes[0].fill_between(lag, q25, q75, color="#56B4E9", alpha=0.25, label="IQR")
    detected_lag = recurrence_summary.get("detected_recurrence_lag_events")
    if detected_lag is not None:
        detected_value = float(
            lag_frame.loc[
                lag_frame["lag_events"] == int(detected_lag), "median_correlation"
            ].iloc[0]
        )
        axes[0].scatter(
            [detected_lag],
            [detected_value],
            s=80,
            marker="*",
            color="#D55E00",
            zorder=5,
            label=f"Detected recurrence: lag {detected_lag}",
        )
    axes[0].axhline(
        RECURRENCE_MIN_MEDIAN_CORRELATION,
        color="0.45",
        ls="--",
        lw=1.0,
        label="Recurrence threshold",
    )
    axes[0].set(
        title="Profile correlation by event lag",
        xlabel="Event lag",
        ylabel="Pearson correlation",
        xticks=lag,
        ylim=(-1.0, 1.05),
    )
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="best")

    local_x_mm = 1000.0 * (np.asarray(x, dtype=float) - float(x[0]))
    status = recurrence_summary.get("status", "")
    if status == "multi-family recurrence detected" and detected_lag is not None:
        colors = plt.get_cmap("tab10")
        event_indices = np.arange(len(profiles), dtype=int)
        for family_id in range(int(detected_lag)):
            indices = event_indices[event_indices % int(detected_lag) == family_id]
            family_profiles = profiles[indices]
            median_profile = np.nanmedian(family_profiles, axis=0)
            low = np.nanpercentile(family_profiles, 2.5, axis=0)
            high = np.nanpercentile(family_profiles, 97.5, axis=0)
            color = colors(family_id)
            axes[1].fill_between(local_x_mm, low, high, color=color, alpha=0.10)
            axes[1].plot(
                local_x_mm,
                median_profile,
                color=color,
                lw=1.8,
                label=f"Family {family_id + 1} (n={len(indices)})",
            )
        axes[1].set_title("Recurring profiles (descriptive sequence families)")
    else:
        median_profile = np.nanmedian(profiles, axis=0)
        low = np.nanpercentile(profiles, 2.5, axis=0)
        high = np.nanpercentile(profiles, 97.5, axis=0)
        axes[1].fill_between(
            local_x_mm, low, high, color="#56B4E9", alpha=0.25, label="95% envelope"
        )
        axes[1].plot(
            local_x_mm, median_profile, color="#0072B2", lw=2.0, label="Median profile"
        )
        axes[1].set_title("Single pooled profile family")
    axes[1].axhline(0.0, color="0.5", lw=0.8)
    axes[1].set(
        xlabel="Local sensing coordinate (mm)",
        ylabel="Baseline-corrected strain (microstrain)",
    )
    axes[1].grid(alpha=0.15)
    axes[1].legend(fontsize=8, loc="best")
    short_label = CONFIGURATION_SHORT_LABELS.get(configuration, configuration)
    figure.suptitle(f"Recurring-profile diagnostic: {short_label}")
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_configuration_comparison(features_by_config: dict[str, pd.DataFrame], output: Path) -> None:
    available = {key: value for key, value in features_by_config.items() if not value.empty}
    if not available:
        return
    measures = [
        (
            "negative_peak_magnitude_microstrain",
            "Negative peak magnitude (microstrain)",
        ),
        ("profile_correlation", "Correlation with median pass"),
        ("edge_energy_ratio", "Edge-energy ratio"),
        ("profile_asymmetry", "Profile asymmetry"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (measure, ylabel) in zip(axes.ravel(), measures):
        usable = [
            (key, frame[measure].dropna().to_numpy(float))
            for key, frame in available.items()
        ]
        usable = [(key, values) for key, values in usable if len(values)]
        if not usable:
            axis.text(
                0.5,
                0.5,
                "Not applicable to fragmented-gauge data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue
        data = [values for _, values in usable]
        labels = [CONFIGURATION_SHORT_LABELS.get(key, key) for key, _ in usable]
        # Matplotlib < 3.9 uses ``labels``; Matplotlib >= 3.9 renamed it to
        # ``tick_labels``. Keep the script compatible with both versions.
        try:
            axis.boxplot(data, tick_labels=labels, showfliers=True)
        except TypeError as error:
            if "tick_labels" not in str(error):
                raise
            axis.boxplot(data, labels=labels, showfliers=True)
        axis.set(title=measure.replace("_", " ").title(), ylabel=ylabel)
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Descriptive comparison of recorded MMLS configurations")
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_spatial_degradation(frame: pd.DataFrame, output: Path) -> None:
    if frame.empty:
        return
    frame = frame.copy()
    denominator = frame["detection_recall"] + frame["detection_precision"]
    frame["detection_f1"] = np.where(
        denominator > 0,
        2 * frame["detection_recall"] * frame["detection_precision"] / denominator,
        0.0,
    )
    frame["absolute_negative_peak_error_percent"] = np.abs(
        frame["negative_peak_error_percent"]
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

    def draw(
        axis,
        subset: pd.DataFrame,
        x_column: str,
        y_column: str,
        title: str,
        xlabel: str,
        ylabel: str,
        log_x: bool = False,
        bounded: bool = False,
    ) -> None:
        for configuration, group in subset.groupby("configuration"):
            xs, medians, lows, highs = [], [], [], []
            for x_value, values in group.groupby(x_column)[y_column]:
                values = values.dropna().to_numpy(float)
                if not len(values):
                    continue
                xs.append(float(x_value))
                medians.append(float(np.median(values)))
                lows.append(float(np.percentile(values, 2.5)))
                highs.append(float(np.percentile(values, 97.5)))
            if not xs:
                continue
            order = np.argsort(xs)
            xs = np.asarray(xs)[order]
            medians = np.asarray(medians)[order]
            lows = np.asarray(lows)[order]
            highs = np.asarray(highs)[order]
            label = configuration.split("_")[0]
            line = axis.plot(xs, medians, "o-", label=label)[0]
            axis.fill_between(xs, lows, highs, color=line.get_color(), alpha=0.15)
        if log_x:
            axis.set_xscale("log", base=2)
        if bounded:
            axis.set_ylim(0, 1.05)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        axis.legend()

    pitch = frame[frame["degradation"] == "pitch"]
    missing = frame[frame["degradation"] == "missing_contiguous_segment"]
    draw(
        axes[0, 0],
        pitch,
        "effective_pitch_mm",
        "detection_f1",
        "Pass detection under coarser pitch",
        "Effective pitch (mm)",
        "Detection F1",
        log_x=True,
        bounded=True,
    )
    draw(
        axes[0, 1],
        pitch,
        "effective_pitch_mm",
        "absolute_negative_peak_error_percent",
        "Peak-magnitude fidelity under coarser pitch",
        "Effective pitch (mm)",
        "Absolute median-peak error (%)",
        log_x=True,
    )
    draw(
        axes[1, 0],
        missing,
        "missing_length_mm",
        "detection_f1",
        "Pass detection with missing fiber",
        "Missing contiguous length (mm)",
        "Detection F1",
        bounded=True,
    )
    draw(
        axes[1, 1],
        missing,
        "missing_length_mm",
        "absolute_negative_peak_error_percent",
        "Peak-magnitude fidelity with missing fiber",
        "Missing contiguous length (mm)",
        "Absolute median-peak error (%)",
    )
    figure.suptitle("Spatial-degradation sensitivity by recorded configuration")
    figure.savefig(output, dpi=220)
    plt.close(figure)


# =============================================================================
# Command-line analysis workflow
# =============================================================================


def _feature_metric_summary(features: pd.DataFrame) -> dict:
    """Return stable, comparable medians for sensitivity tables."""
    mapping = {
        "negative_peak_magnitude_microstrain": "median_negative_peak_magnitude_microstrain",
        "peak_negative_strain_location_m": "median_negative_peak_location_m",
        "absolute_spatial_integral_microstrain_m": "median_spatial_integral_microstrain_m",
        "spatial_fwhm_m": "median_spatial_fwhm_m",
        "profile_correlation": "median_profile_correlation",
    }
    result = {"n_complete": int(len(features))}
    for source, target in mapping.items():
        if source not in features or features.empty:
            result[target] = None
            continue
        values = pd.to_numeric(features[source], errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        result[target] = float(np.median(values)) if len(values) else None
    if len(features) > 1 and "negative_peak_magnitude_microstrain" in features:
        values = pd.to_numeric(
            features["negative_peak_magnitude_microstrain"], errors="coerce"
        ).to_numpy(float)
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if len(values) else 0.0
        result["negative_peak_cv_percent"] = (
            float(100.0 * np.std(values, ddof=1) / mean)
            if len(values) > 1 and mean > 0.0
            else None
        )
    else:
        result["negative_peak_cv_percent"] = None
    return result


def _evaluate_profile_support(
    profiles: np.ndarray,
    x: np.ndarray,
    fs: float,
    *,
    gauge_completeness: float = GAUGE_MIN_VALID_FRACTION,
    activity_winsor_percentile: float = ACTIVITY_SPATIAL_WINSOR_PERCENTILE,
    peak_spacing_fraction: float = MIN_PEAK_DISTANCE_PERIOD_FRACTION,
    event_half_period_fraction: float = EVENT_HALF_PERIOD_FRACTION,
    anomaly_z_threshold: float = ROBUST_Z_THRESHOLD,
    reference_detection: DetectionResult | None = None,
    fixed_peaks: np.ndarray | None = None,
    nominal_pitch_m: float | None = None,
) -> tuple[dict, pd.DataFrame, DetectionResult | None]:
    """Recompute event features on one explicitly defined gauge support."""
    retained, keep, run_ids, audit = audit_gauge_validity(
        profiles, x, minimum_valid_fraction=gauge_completeness
    )
    retained_x = np.asarray(x, dtype=float)[keep]
    if fixed_peaks is not None:
        detection = reference_detection
        peaks = np.asarray(fixed_peaks, dtype=int)
    else:
        detection = detect_passes(
            retained,
            fs,
            activity_filter=PRIMARY_DETECTION_FILTER,
            minimum_peak_distance_fraction=peak_spacing_fraction,
            activity_spatial_winsor_percentile=activity_winsor_percentile,
        )
        peaks = detection.peaks
    frequency = (
        detection.used_pass_frequency_hz
        if detection is not None
        else (reference_detection.used_pass_frequency_hz if reference_detection else None)
    )
    if frequency is None or not np.isfinite(frequency) or frequency <= 0.0:
        return {
            "status": "not computed",
            "reason": "No valid pass frequency was available.",
            "n_detected": int(len(peaks)),
            "continuous_profile": bool(audit["continuous_full_interval"]),
        }, pd.DataFrame(), detection
    events, valid_peaks, _ = extract_events(
        retained,
        peaks,
        fs,
        frequency,
        half_period_fraction=event_half_period_fraction,
    )
    current_x = retained_x.copy()
    current_runs = run_ids.copy()
    continuous = bool(audit["continuous_full_interval"])
    features, profile_matrix, _ = event_feature_table(
        events,
        valid_peaks,
        current_x,
        fs,
        continuous_profile=continuous,
        run_ids=current_runs,
    )
    while len(profile_matrix):
        common = np.isfinite(profile_matrix).all(axis=0)
        if common.all():
            break
        if int(common.sum()) < 10:
            return {
                "status": "not computed",
                "reason": "Fewer than ten common finite gauges remain at event peaks.",
                "n_detected": int(len(peaks)),
                "n_complete": int(len(features)),
                "continuous_profile": continuous,
            }, pd.DataFrame(), detection
        events = events[:, :, common]
        current_x = current_x[common]
        current_runs = current_runs[common]
        # Keep the recording-level continuity classification from the
        # validity audit. Removing an occasional gauge that is incomplete at
        # an event peak must not reclassify an otherwise continuous recording
        # as fragmented; the retained physical coordinates are still used by
        # all spatial integrations and profile metrics.
        features, profile_matrix, _ = event_feature_table(
            events,
            valid_peaks,
            current_x,
            fs,
            continuous_profile=continuous,
            run_ids=current_runs,
        )
    if not features.empty:
        features = add_anomaly_scores(
            features, profile_matrix, continuous, z_threshold=anomaly_z_threshold
        )
    metrics = _feature_metric_summary(features)
    metrics.update(
        {
            "status": "computed" if len(features) else "no complete events",
            "n_detected": int(len(peaks)),
            "n_complete": int(len(features)),
            "n_response_anomalies": (
                int(features["response_anomaly_flag"].sum()) if len(features) else 0
            ),
            "n_recording_level_gages": int(keep.sum()),
            "continuous_profile": continuous,
            "profile_metric_status": (
                "continuous full-interval metrics"
                if continuous
                else "spatially fragmented; full-profile metrics suppressed"
            ),
        }
    )
    if detection is not None:
        metrics["used_pass_frequency_hz"] = float(detection.used_pass_frequency_hz)
    return metrics, features, detection


def _sensitivity_row(
    configuration: str,
    parameter: str,
    value,
    metrics: dict,
    reference_metrics: dict,
) -> dict:
    row = {
        "configuration": configuration,
        "parameter": parameter,
        "value": value,
        **metrics,
    }
    row.update(add_degradation_errors(metrics, reference_metrics))
    row["interpretation_warning"] = (
        "One-at-a-time computational sensitivity; no claim of invariance until the results are reviewed."
    )
    return row


def threshold_sensitivity_analysis(
    item: ODiSIFile,
    roi_profiles: np.ndarray,
    roi_x: np.ndarray,
    primary_detection: DetectionResult,
    primary_features: pd.DataFrame,
    primary_profile_matrix: np.ndarray,
    primary_x: np.ndarray,
    continuous_profile: bool,
) -> pd.DataFrame:
    """Run prespecified one-at-a-time alternatives around the primary settings."""
    nominal_pitch = float(np.nanmedian(np.diff(roi_x)))
    baseline, baseline_features, baseline_detection = _evaluate_profile_support(
        roi_profiles,
        roi_x,
        item.used_rate_hz,
        reference_detection=primary_detection,
        fixed_peaks=primary_detection.peaks,
        nominal_pitch_m=nominal_pitch,
    )
    rows = []
    rerun_groups = [
        ("gauge_completeness", SENSITIVITY_GAUGE_COMPLETENESS, "gauge_completeness"),
        (
            "activity_spatial_winsor_percentile",
            SENSITIVITY_ACTIVITY_WINSOR_PERCENTILES,
            "activity_winsor_percentile",
        ),
        ("minimum_peak_spacing_period_fraction", SENSITIVITY_PEAK_SPACING_FRACTIONS, "peak_spacing_fraction"),
        ("event_half_period_fraction", SENSITIVITY_EVENT_HALF_PERIOD_FRACTIONS, "event_half_period_fraction"),
    ]
    primary_values = {
        "gauge_completeness": GAUGE_MIN_VALID_FRACTION,
        "activity_spatial_winsor_percentile": ACTIVITY_SPATIAL_WINSOR_PERCENTILE,
        "minimum_peak_spacing_period_fraction": MIN_PEAK_DISTANCE_PERIOD_FRACTION,
        "event_half_period_fraction": EVENT_HALF_PERIOD_FRACTION,
    }
    for parameter, values, setting_name in rerun_groups:
        for value in values:
            if math.isclose(float(value), float(primary_values[parameter])):
                metrics = dict(baseline)
            else:
                kwargs = {
                    "gauge_completeness": GAUGE_MIN_VALID_FRACTION,
                    "activity_winsor_percentile": ACTIVITY_SPATIAL_WINSOR_PERCENTILE,
                    "peak_spacing_fraction": MIN_PEAK_DISTANCE_PERIOD_FRACTION,
                    "event_half_period_fraction": EVENT_HALF_PERIOD_FRACTION,
                }
                kwargs[setting_name] = float(value)
                if parameter == "event_half_period_fraction":
                    metrics, _, _ = _evaluate_profile_support(
                        roi_profiles,
                        roi_x,
                        item.used_rate_hz,
                        reference_detection=primary_detection,
                        fixed_peaks=primary_detection.peaks,
                        nominal_pitch_m=nominal_pitch,
                        **kwargs,
                    )
                else:
                    metrics, _, _ = _evaluate_profile_support(
                        roi_profiles,
                        roi_x,
                        item.used_rate_hz,
                        nominal_pitch_m=nominal_pitch,
                        **kwargs,
                    )
            rows.append(
                _sensitivity_row(item.configuration, parameter, value, metrics, baseline)
            )

    tolerance_rows = []
    for value in SENSITIVITY_MATCH_TOLERANCE_FRACTIONS:
        tolerance_samples = max(
            1,
            int(round(float(value) * item.used_rate_hz / primary_detection.used_pass_frequency_hz)),
        )
        validation = manual_detection_validation(
            item.configuration,
            primary_detection.peaks,
            item.used_rate_hz,
            tolerance_samples,
        )
        tolerance_metrics = {
            **baseline,
            "manual_label_status": validation.get("status"),
            "manual_precision": validation.get("precision"),
            "manual_recall": validation.get("recall"),
            "manual_f1": validation.get("f1"),
            "manual_timing_error_s": validation.get("median_absolute_timing_error_s"),
            "match_tolerance_samples": tolerance_samples,
        }
        tolerance_rows.append(
            _sensitivity_row(
                item.configuration,
                "peak_match_tolerance_period_fraction",
                value,
                tolerance_metrics,
                baseline,
            )
        )
    rows.extend(tolerance_rows)

    for value in SENSITIVITY_ANOMALY_Z_THRESHOLDS:
        scalar = (
            np.abs(baseline_features["scalar_anomaly_score"].to_numpy(float))
            if "scalar_anomaly_score" in baseline_features
            else np.zeros(len(baseline_features))
        )
        profile = (
            np.abs(baseline_features["profile_anomaly_score"].to_numpy(float))
            if "profile_anomaly_score" in baseline_features
            else np.zeros(len(baseline_features))
        )
        rescored_count = int(np.sum((scalar > float(value)) | (profile > float(value))))
        metrics = {
            **baseline,
            "n_response_anomalies": rescored_count,
        }
        rows.append(
            _sensitivity_row(
                item.configuration,
                "robust_anomaly_z_threshold",
                float(value),
                metrics,
                baseline,
            )
        )
    recurrence_sweeps = [
        ("recurrence_minimum_profile_pairs", SENSITIVITY_RECURRENCE_MIN_PAIRS, "minimum_profile_pairs"),
        ("recurrence_minimum_median_correlation", SENSITIVITY_RECURRENCE_MIN_CORRELATIONS, "minimum_median_correlation"),
        ("recurrence_minimum_gain_over_lag_one", SENSITIVITY_RECURRENCE_MIN_GAINS, "minimum_gain_over_lag_one"),
    ]
    for parameter, values, argument in recurrence_sweeps:
        for value in values:
            kwargs = {
                "minimum_profile_pairs": RECURRENCE_MIN_PAIRS,
                "minimum_median_correlation": RECURRENCE_MIN_MEDIAN_CORRELATION,
                "minimum_gain_over_lag_one": RECURRENCE_MIN_GAIN_OVER_LAG_ONE,
            }
            kwargs[argument] = value
            _, _, _, recurrence = recurring_profile_analysis(
                item.configuration,
                primary_features,
                primary_profile_matrix,
                primary_x,
                continuous_profile,
                **kwargs,
            )
            rows.append(
                {
                    "configuration": item.configuration,
                    "parameter": parameter,
                    "value": value,
                    "recurrence_status": recurrence.get("status"),
                    "recurrence_lag_events": recurrence.get("detected_recurrence_lag_events"),
                    "recurrence_family_count": recurrence.get("n_profile_families"),
                    "pooled_median_pairwise_correlation": recurrence.get("pooled_median_pairwise_correlation"),
                    "lag_one_median_correlation": recurrence.get("lag_one_median_correlation"),
                    "recurrence_gain_over_lag_one": recurrence.get("gain_over_lag_one"),
                    "minimum_profile_pairs": kwargs["minimum_profile_pairs"],
                    "minimum_median_correlation": kwargs["minimum_median_correlation"],
                    "minimum_gain_over_lag_one": kwargs["minimum_gain_over_lag_one"],
                    "interpretation_warning": "One-at-a-time recurrence-threshold sensitivity; family labels remain exploratory and do not identify wheels or loading mechanisms.",
                }
            )
    return pd.DataFrame(rows)


def _fixed_event_metrics_on_support(
    profiles: np.ndarray,
    x: np.ndarray,
    valid_fraction: np.ndarray,
    fs: float,
    peaks: np.ndarray,
    pass_frequency_hz: float,
    nominal_pitch_m: float,
    gauge_completeness: float = GAUGE_MIN_VALID_FRACTION,
) -> dict:
    """Compute features from event windows only for a fixed peak list."""
    keep = np.asarray(valid_fraction, dtype=float) >= gauge_completeness
    x_keep = np.asarray(x, dtype=float)[keep]
    if keep.sum() < 10:
        return {
            "status": "not computed",
            "reason": "Fewer than ten gauges pass the recording-level completeness rule.",
            "n_recording_level_gages": int(keep.sum()),
            "n_detected": int(len(peaks)),
        }
    half = max(2, int(round(EVENT_HALF_PERIOD_FRACTION * fs / pass_frequency_hz)))
    valid_peaks = np.asarray(peaks, dtype=int)
    valid_peaks = valid_peaks[
        (valid_peaks >= half) & (valid_peaks < len(profiles) - half)
    ]
    if not len(valid_peaks):
        return {
            "status": "no complete events",
            "n_detected": int(len(peaks)),
            "n_complete": 0,
            "continuous_profile": False,
        }
    events = np.stack(
        [profiles[peak - half : peak + half + 1, :][:, keep] for peak in valid_peaks]
    ).astype(np.float32)
    current_x = x_keep.copy()
    continuous = bool(
        len(current_x) > 1
        and np.all(np.diff(current_x) <= 1.5 * nominal_pitch_m)
    )
    features, profile_matrix, _ = event_feature_table(
        events,
        valid_peaks,
        current_x,
        fs,
        continuous_profile=continuous,
    )
    while len(profile_matrix):
        common = np.isfinite(profile_matrix).all(axis=0)
        if common.all():
            break
        if int(common.sum()) < 10:
            return {
                "status": "not computed",
                "reason": "Fewer than ten common finite gauges remain at event peaks.",
                "n_detected": int(len(peaks)),
                "n_complete": int(len(features)),
                "n_recording_level_gages": int(keep.sum()),
                "continuous_profile": False,
            }
        events = events[:, :, common]
        current_x = current_x[common]
        # As in the primary analysis, continuity is determined from the
        # recording-level gauge mask above, before event-complete-case
        # filtering. Keep that classification fixed while the common event
        # support is reduced.
        features, profile_matrix, _ = event_feature_table(
            events,
            valid_peaks,
            current_x,
            fs,
            continuous_profile=continuous,
        )
    if not features.empty:
        features = add_anomaly_scores(features, profile_matrix, continuous)
    metrics = _feature_metric_summary(features)
    metrics.update(
        {
            "status": "computed" if len(features) else "no complete events",
            "n_detected": int(len(peaks)),
            "n_complete": int(len(features)),
            "n_recording_level_gages": int(keep.sum()),
            "continuous_profile": continuous,
            "profile_metric_status": (
                "continuous full-interval metrics"
                if continuous
                else "spatially fragmented; full-profile metrics suppressed"
            ),
        }
    )
    return metrics


def roi_boundary_sensitivity_analysis(
    item: ODiSIFile,
    roi_indices: np.ndarray,
    roi_profiles: np.ndarray,
    primary_detection: DetectionResult,
) -> pd.DataFrame:
    """Shift each selected ROI endpoint independently while holding detected
    event times fixed, isolating how spatial support changes the reported
    profile metrics. The perturbation range is a stated scenario, not survey
    uncertainty; it must be revisited if bonded-fiber coordinates are recovered.
    """
    base_start = float(item.distance_m[roi_indices[0]])
    base_stop = float(item.distance_m[roi_indices[-1]])
    maximum_shift_m = max(abs(value) for value in ROI_BOUNDARY_OFFSETS_MM) / 1000.0
    extended_indices = np.flatnonzero(
        (item.distance_m >= base_start - maximum_shift_m)
        & (item.distance_m <= base_stop + maximum_shift_m)
    )
    base_profiles = np.asarray(item.data[:, extended_indices], dtype=np.float32)
    base_x = item.distance_m[extended_indices]
    base_valid_fraction = np.isfinite(base_profiles).mean(axis=0)
    nominal_pitch = float(np.nanmedian(np.diff(item.distance_m)))
    base_first = int(np.searchsorted(base_x, base_start, side="left"))
    base_after_last = int(np.searchsorted(base_x, base_stop, side="right"))
    baseline = _fixed_event_metrics_on_support(
        base_profiles[:, base_first:base_after_last],
        base_x[base_first:base_after_last],
        base_valid_fraction[base_first:base_after_last],
        item.used_rate_hz,
        primary_detection.peaks,
        primary_detection.used_pass_frequency_hz,
        nominal_pitch,
    )
    rows = []
    for start_shift in ROI_BOUNDARY_OFFSETS_MM:
        for stop_shift in ROI_BOUNDARY_OFFSETS_MM:
            start_m = base_start + start_shift / 1000.0
            stop_m = base_stop + stop_shift / 1000.0
            first = int(np.searchsorted(base_x, start_m, side="left"))
            after_last = int(np.searchsorted(base_x, stop_m, side="right"))
            if after_last - first < 10:
                rows.append(
                    {
                        "configuration": item.configuration,
                        "start_shift_mm": start_shift,
                        "stop_shift_mm": stop_shift,
                        "status": "not computed",
                        "reason": "Perturbed interval contains fewer than ten gauges.",
                    }
                )
                continue
            selected_x = base_x[first:after_last]
            metrics = _fixed_event_metrics_on_support(
                base_profiles[:, first:after_last],
                selected_x,
                base_valid_fraction[first:after_last],
                item.used_rate_hz,
                primary_detection.peaks,
                primary_detection.used_pass_frequency_hz,
                nominal_pitch,
            )
            row = _sensitivity_row(
                item.configuration,
                "roi_boundary_pair_mm",
                f"{start_shift:g},{stop_shift:g}",
                metrics,
                baseline,
            )
            row.update(
                {
                    "start_shift_mm": float(start_shift),
                    "stop_shift_mm": float(stop_shift),
                    "start_m": float(selected_x[0]),
                    "stop_m": float(selected_x[-1]),
                    "nominal_roi_start_m": base_start,
                    "nominal_roi_stop_m": base_stop,
                    "boundary_range_warning": (
                        "Exploratory +/-20 mm per endpoint; not a measured installation-coordinate uncertainty."
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def synthetic_ground_truth_benchmark() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exercise detection and profile features on deterministic known-truth data."""
    fs = SYNTHETIC_BENCHMARK_FS_HZ
    n_scans = int(round(SYNTHETIC_BENCHMARK_DURATION_S * fs))
    time_s = np.arange(n_scans, dtype=float) / fs
    x = np.linspace(0.0, 0.92, SYNTHETIC_BENCHMARK_GAUGES)
    rng = np.random.default_rng(SYNTHETIC_BENCHMARK_SEED)
    scenarios = {
        "clean_noise_and_drift": {"noise_sd": 3.0, "drift_scale": 6.0, "random_missing": 0.0, "gap": False},
        "random_missing_noise_and_drift": {"noise_sd": 6.0, "drift_scale": 18.0, "random_missing": 0.015, "gap": False},
        "contiguous_spatial_gap_noise_and_drift": {"noise_sd": 6.0, "drift_scale": 18.0, "random_missing": 0.005, "gap": True},
    }
    detail_rows = []
    frequency = 0.75
    period = 1.0 / frequency
    tolerance = max(1, int(round(PEAK_MATCH_TOLERANCE_PERIOD_FRACTION * fs / frequency)))
    for scenario, settings in scenarios.items():
        for replicate in range(SYNTHETIC_BENCHMARK_REPLICATES):
            centers = np.arange(2.0, SYNTHETIC_BENCHMARK_DURATION_S - 2.0, period)
            centers = centers + rng.uniform(-0.015, 0.015, size=len(centers))
            matrix = np.zeros((n_scans, len(x)), dtype=np.float32)
            x_norm = x / x[-1]
            drift = settings["drift_scale"] * (
                0.65 * np.sin(2.0 * np.pi * time_s / SYNTHETIC_BENCHMARK_DURATION_S)
                + 0.35 * time_s / SYNTHETIC_BENCHMARK_DURATION_S
            )
            matrix += (
                drift[:, None] * (0.5 + 0.5 * x_norm)[None, :]
            ).astype(np.float32)
            truth_profiles = []
            for event_index, center_time in enumerate(centers):
                amplitude = float(rng.uniform(90.0, 150.0))
                spatial_center = float(rng.uniform(0.30, 0.62))
                spatial_sigma = float(rng.uniform(0.045, 0.080))
                temporal_sigma = float(rng.uniform(0.055, 0.085))
                temporal = np.exp(
                    -0.5 * ((time_s - center_time) / temporal_sigma) ** 2
                )
                asymmetry = float(rng.uniform(-0.12, 0.12))
                scaled_x = (x - spatial_center) / spatial_sigma
                shape = np.exp(-0.5 * scaled_x**2) * (1.0 + asymmetry * np.tanh(scaled_x))
                profile = -amplitude * shape
                matrix += np.outer(temporal.astype(np.float32), profile.astype(np.float32))
                truth_profiles.append(profile)
            matrix += rng.normal(0.0, settings["noise_sd"], size=matrix.shape).astype(np.float32)
            if settings["random_missing"]:
                near_event = np.zeros(n_scans, dtype=bool)
                exclusion = max(2, int(round(0.25 * fs)))
                for center_sample in np.rint(centers * fs).astype(int):
                    near_event[
                        max(0, center_sample - exclusion) : min(
                            n_scans, center_sample + exclusion + 1
                        )
                    ] = True
                missing = rng.random(matrix.shape) < settings["random_missing"]
                missing[near_event, :] = False
                matrix[missing] = np.nan
            if settings["gap"]:
                gap_mask = (x >= 0.39) & (x <= 0.45)
                matrix[:, gap_mask] = np.nan

            metrics, features, detection = _evaluate_profile_support(
                matrix,
                x,
                fs,
                gauge_completeness=0.80,
                activity_winsor_percentile=ACTIVITY_SPATIAL_WINSOR_PERCENTILE,
                peak_spacing_fraction=MIN_PEAK_DISTANCE_PERIOD_FRACTION,
                event_half_period_fraction=EVENT_HALF_PERIOD_FRACTION,
                nominal_pitch_m=float(np.nanmedian(np.diff(x))),
            )
            detected = detection.peaks if detection is not None else np.array([], dtype=int)
            truth_samples = np.rint(centers * fs).astype(int)
            matched_pairs = []
            used_candidates: set[int] = set()
            for truth_index, truth_sample in enumerate(truth_samples):
                candidates = np.argsort(np.abs(detected - truth_sample))
                for candidate_index in candidates:
                    candidate_sample = int(detected[candidate_index])
                    if candidate_index not in used_candidates and abs(candidate_sample - truth_sample) <= tolerance:
                        used_candidates.add(int(candidate_index))
                        matched_pairs.append((truth_index, candidate_sample))
                        break
            recall = len(matched_pairs) / len(truth_samples) if len(truth_samples) else math.nan
            precision = len(matched_pairs) / len(detected) if len(detected) else 0.0
            f1 = 2.0 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
            timing_errors = [abs((candidate - truth_samples[i]) / fs) for i, candidate in matched_pairs]
            feature_lookup = (
                features.set_index("peak_sample", drop=False)
                if not features.empty and "peak_sample" in features
                else pd.DataFrame()
            )
            magnitude_errors, location_errors, integral_errors, width_errors = [], [], [], []
            for truth_index, candidate_sample in matched_pairs:
                if feature_lookup.empty or candidate_sample not in feature_lookup.index:
                    continue
                candidate = feature_lookup.loc[candidate_sample]
                if isinstance(candidate, pd.DataFrame):
                    candidate = candidate.iloc[0]
                truth_profile = truth_profiles[truth_index]
                truth_peak = int(np.argmin(truth_profile))
                true_magnitude = float(-truth_profile[truth_peak])
                magnitude_errors.append(
                    100.0 * abs(float(candidate["negative_peak_magnitude_microstrain"]) - true_magnitude) / max(true_magnitude, 1e-12)
                )
                location_errors.append(
                    1000.0 * abs(float(candidate["peak_negative_strain_location_m"]) - float(x[truth_peak]))
                )
                true_integral = float(trapezoid(np.abs(truth_profile), x))
                candidate_integral = candidate.get("absolute_spatial_integral_microstrain_m", math.nan)
                if np.isfinite(candidate_integral) and true_integral > 1e-12:
                    integral_errors.append(100.0 * abs(float(candidate_integral) - true_integral) / true_integral)
                true_width = profile_width(x, np.abs(truth_profile))
                candidate_width = candidate.get("spatial_fwhm_m", math.nan)
                if np.isfinite(candidate_width) and np.isfinite(true_width) and true_width > 1e-12:
                    width_errors.append(100.0 * abs(float(candidate_width) - true_width) / true_width)
            detail_rows.append(
                {
                    "scenario": scenario,
                    "replicate": replicate + 1,
                    "n_truth_events": int(len(truth_samples)),
                    "n_detected_events": int(len(detected)),
                    "n_matched_events": int(len(matched_pairs)),
                    "event_recall": float(recall),
                    "event_precision": float(precision),
                    "event_f1": float(f1),
                    "false_alarms_per_minute": float(max(0, len(detected) - len(matched_pairs)) / (SYNTHETIC_BENCHMARK_DURATION_S / 60.0)),
                    "median_absolute_timing_error_s": float(np.median(timing_errors)) if timing_errors else None,
                    "median_peak_magnitude_error_percent": float(np.median(magnitude_errors)) if magnitude_errors else None,
                    "median_peak_location_error_mm": float(np.median(location_errors)) if location_errors else None,
                    "median_spatial_integral_error_percent": float(np.median(integral_errors)) if integral_errors else None,
                    "median_spatial_fwhm_error_percent": float(np.median(width_errors)) if width_errors else None,
                    "spatial_metrics_status": metrics.get("profile_metric_status"),
                    "noise_sd_microstrain": settings["noise_sd"],
                    "baseline_drift_scale_microstrain": settings["drift_scale"],
                    "random_missing_fraction": settings["random_missing"],
                    "contiguous_gap_present": bool(settings["gap"]),
                    "interpretation_warning": "Synthetic algorithm benchmark only; not validation of physical MMLS measurements.",
                }
            )
    detail = pd.DataFrame(detail_rows)
    numeric = [
        "event_recall",
        "event_precision",
        "event_f1",
        "false_alarms_per_minute",
        "median_absolute_timing_error_s",
        "median_peak_magnitude_error_percent",
        "median_peak_location_error_mm",
        "median_spatial_integral_error_percent",
        "median_spatial_fwhm_error_percent",
    ]
    summary = (
        detail.groupby("scenario", as_index=False)[numeric]
        .median(numeric_only=True)
        .rename(columns={column: f"median_{column}" for column in numeric})
    )
    for column in numeric:
        quantiles = (
            detail.groupby("scenario")[column]
            .quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: f"replicate_q025_{column}", 0.975: f"replicate_q975_{column}"})
        )
        summary = summary.merge(quantiles, on="scenario", how="left")
    summary["n_replicates"] = SYNTHETIC_BENCHMARK_REPLICATES
    summary["interpretation_warning"] = (
        "Synthetic algorithm benchmark only; replicate quantiles describe "
        "simulation variability and are not confidence intervals or validation "
        "of physical MMLS measurements."
    )
    return detail, summary


def analyze_configuration(item: ODiSIFile) -> dict:
    config_dir = OUTPUT_DIR / clean_name(item.configuration)
    config_dir.mkdir(parents=True, exist_ok=True)
    roi_indices, roi_info = choose_roi(item)
    roi_score = np.asarray(roi_info.pop("score"))
    plot_roi_selection(
        item,
        roi_indices,
        roi_score,
        config_dir / "roi_selection.png",
    )
    roi_profiles = np.asarray(item.data[:, roi_indices], dtype=np.float32)
    roi_x = item.distance_m[roi_indices]
    raw_profiles, valid_columns, run_ids, gauge_audit = audit_gauge_validity(
        roi_profiles, roi_x
    )
    x = roi_x[valid_columns]
    # Preserve the recording-level support used by primary detection. The
    # event-profile complete-case loop below narrows ``raw_profiles`` and
    # ``x`` for event-level spatial metrics; filter comparisons must use the
    # same original support as primary detection instead of that narrowed set.
    filter_comparison_profiles = raw_profiles.copy()
    filter_comparison_x = x.copy()
    filter_comparison_run_ids = run_ids.copy()
    continuous_profile = bool(gauge_audit["continuous_full_interval"])

    validity_frame = pd.DataFrame(
        {
            "distance_m": roi_x,
            "eligible_at_80_percent": valid_columns,
            "valid_fraction": np.isfinite(roi_profiles).mean(axis=0),
        }
    )

    # Detection filters act only on the one-dimensional activity signal. All
    # accepted event fields and reported strain features use the raw profiles.
    detection = detect_passes(
        raw_profiles,
        item.used_rate_hz,
        activity_filter=PRIMARY_DETECTION_FILTER,
    )
    alternative_detections = {
        method: detect_passes(raw_profiles, item.used_rate_hz, activity_filter=method)
        for method in ("gaussian", "savgol")
    }
    events, valid_peaks, relative_time = extract_events(
        raw_profiles,
        detection.peaks,
        item.used_rate_hz,
        detection.used_pass_frequency_hz,
    )
    # First extract event-peak profiles on the recording-level 80% mask. Then
    # create a second fixed mask containing only gauges that are finite in
    # every accepted event-peak profile. All event-level features are
    # recomputed on that common spatial support; no values are imputed. A
    # fragmented configuration remains fragmented after this masking and its
    # full-profile structural metrics remain unavailable.
    features, profile_matrix, temporal_matrix = event_feature_table(
        events,
        valid_peaks,
        x,
        item.used_rate_hz,
        continuous_profile=continuous_profile,
        run_ids=run_ids,
    )
    # Iterate to a fixed point because removing incomplete gauges can slightly
    # change the event-local activity maximum and hence the selected profile
    # time. The cumulative mask always refers to the original 80%-eligible
    # coordinate set, while ``local_complete`` refers to the current arrays.
    event_complete_mask = np.ones(len(x), dtype=bool)
    while len(profile_matrix):
        local_complete = np.isfinite(profile_matrix).all(axis=0)
        if local_complete.all():
            break
        retained_indices = np.flatnonzero(event_complete_mask)
        event_complete_mask[retained_indices[~local_complete]] = False
        if event_complete_mask.sum() < 10:
            raise ValueError(
                "Fewer than ten common finite gauges remain across accepted "
                "event-peak profiles; fixed-support event analysis is invalid."
            )
        raw_profiles = raw_profiles[:, local_complete]
        events = events[:, :, local_complete]
        x = x[local_complete]
        run_ids = run_ids[local_complete]
        features, profile_matrix, temporal_matrix = event_feature_table(
            events,
            valid_peaks,
            x,
            item.used_rate_hz,
            continuous_profile=continuous_profile,
            run_ids=run_ids,
        )
    if len(profile_matrix) and not np.isfinite(profile_matrix).all():
        raise RuntimeError(
            "Internal error: the event-profile complete-case mask did not converge."
        )

    event_complete_roi = np.zeros(len(roi_x), dtype=bool)
    event_complete_roi[valid_columns] = event_complete_mask
    validity_frame["event_peak_complete_case"] = event_complete_roi
    validity_frame["used_for_event_level_metrics"] = event_complete_roi
    validity_frame["used_for_primary_continuous_profile_metrics"] = (
        event_complete_roi if continuous_profile else False
    )
    validity_frame.to_csv(config_dir / "gauge_validity_mask.csv", index=False)
    gauge_audit.update(
        {
            "event_profile_complete_case_gages": int(event_complete_mask.sum()),
            "event_profile_incomplete_gages": int((~event_complete_mask).sum()),
            "event_level_metric_gages": int(event_complete_mask.sum()),
            "primary_continuous_profile_gages": (
                int(event_complete_mask.sum()) if continuous_profile else None
            ),
            "event_profile_missing_values_imputed": False,
        }
    )
    pd.DataFrame(gauge_audit["runs"]).to_csv(
        config_dir / "contiguous_spatial_runs.csv", index=False
    )
    features = add_anomaly_scores(features, profile_matrix, continuous_profile)
    features, recurrence_lags, recurrence_families, recurrence_summary = (
        recurring_profile_analysis(
            item.configuration,
            features,
            profile_matrix,
            x,
            continuous_profile,
        )
    )
    summary = repeatability_summary(
        item.configuration,
        features,
        profile_matrix,
        detection,
        x,
        continuous_profile,
    )
    summary.update(
        {
            "recording_level_eligible_gages": int(valid_columns.sum()),
            "event_profile_complete_case_gages": int(event_complete_mask.sum()),
            "event_profile_incomplete_gages": int((~event_complete_mask).sum()),
            "event_level_metric_gages": int(event_complete_mask.sum()),
            "gauge_validity_rule": (
                f"recording-level finite fraction >= {GAUGE_MIN_VALID_FRACTION:.2f}; "
                "common finite event-peak support for all event-level metrics"
            ),
            "missing_values_imputed": False,
            "recurrence_status": recurrence_summary.get("status"),
            "recurrence_pooled_median_pairwise_correlation": recurrence_summary.get(
                "pooled_median_pairwise_correlation"
            ),
            "recurrence_lag_one_median_correlation": recurrence_summary.get(
                "lag_one_median_correlation"
            ),
            "detected_recurrence_lag_events": recurrence_summary.get(
                "detected_recurrence_lag_events"
            ),
            "detected_recurrence_median_correlation": recurrence_summary.get(
                "detected_recurrence_median_correlation"
            ),
            "recurrence_gain_over_lag_one": recurrence_summary.get(
                "gain_over_lag_one"
            ),
            "n_profile_families": recurrence_summary.get("n_profile_families"),
        }
    )
    recurrence_lags.to_csv(config_dir / "recurring_profile_lags.csv", index=False)
    recurrence_families.to_csv(
        config_dir / "recurring_profile_families.csv", index=False
    )
    write_json(
        config_dir / "recurring_profile_analysis.json", recurrence_summary
    )

    bootstrap_frame, bootstrap_summary = within_configuration_bootstrap(
        features,
        profile_matrix,
        continuous_profile,
    )
    summary.update({f"bootstrap_{key}": value for key, value in bootstrap_summary.items()})
    bootstrap_frame.insert(0, "configuration", item.configuration) if not bootstrap_frame.empty else None
    bootstrap_frame.to_csv(config_dir / "bootstrap_uncertainty.csv", index=False)

    anomaly_sensitivity: dict = {
        "status": "not computed",
        "reason": "Fewer than the required quality-eligible non-anomalous events.",
    }
    if not features.empty:
        keep_events = ~features["response_anomaly_flag"].to_numpy(bool)
        if keep_events.sum() >= MIN_EVENTS_FOR_ANALYSIS:
            sensitivity_summary = repeatability_summary(
                item.configuration,
                features.loc[keep_events].reset_index(drop=True),
                profile_matrix[keep_events],
                detection,
                x,
                continuous_profile,
            )
            anomaly_sensitivity = {
                "status": "computed",
                "n_events_primary": int(len(features)),
                "n_events_after_response_anomaly_exclusion": int(keep_events.sum()),
                "primary_negative_peak_cv_percent": summary.get("negative_peak_cv_percent"),
                "sensitivity_negative_peak_cv_percent": sensitivity_summary.get(
                    "negative_peak_cv_percent"
                ),
                "primary_profile_icc3_1": summary.get("profile_icc3_1"),
                "sensitivity_profile_icc3_1": sensitivity_summary.get("profile_icc3_1"),
                "warning": "Response anomalies were retained in the primary analysis.",
            }
    write_json(config_dir / "anomaly_exclusion_sensitivity.json", anomaly_sensitivity)

    tolerance_samples = max(
        1,
        int(
            round(
                PEAK_MATCH_TOLERANCE_PERIOD_FRACTION
                * item.used_rate_hz
                / detection.used_pass_frequency_hz
            )
        ),
    )
    manual_validation = manual_detection_validation(
        item.configuration,
        detection.peaks,
        item.used_rate_hz,
        tolerance_samples,
    )
    write_json(config_dir / "manual_detection_validation.json", manual_validation)
    event_review = build_event_review_table(
        detection,
        alternative_detections,
        valid_peaks,
        item.used_rate_hz,
    )
    source_timestamps = Path(item.timestamps_path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()
    if len(source_timestamps) == item.n_scans and not event_review.empty:
        event_review["source_timestamp"] = [
            source_timestamps[int(sample)] for sample in event_review["sample"]
        ]
    else:
        event_review["source_timestamp"] = ""
    event_review.insert(0, "configuration", item.configuration)
    event_review.to_csv(config_dir / "event_review.csv", index=False)
    event_review.to_csv(
        config_dir / "full_data_automatic_pass_labels.csv", index=False
    )

    features.insert(0, "configuration", item.configuration)
    features.to_csv(config_dir / "pass_features.csv", index=False)
    pd.DataFrame(
        {
            "sample": np.arange(item.n_scans),
            "time_s": detection.time_s,
            "activity": detection.activity,
            "detrended_activity": detection.detrended_activity,
            "detected_pass": np.isin(np.arange(item.n_scans), detection.peaks),
        }
    ).to_csv(config_dir / "activity_and_passes.csv", index=False)
    if len(detection.psd_frequency_hz):
        pd.DataFrame(
            {"frequency_hz": detection.psd_frequency_hz, "psd": detection.psd}
        ).to_csv(config_dir / "pass_frequency_psd.csv", index=False)
    filter_frame = filter_comparison(
        filter_comparison_profiles,
        filter_comparison_x,
        item.used_rate_hz,
        continuous_profile,
        filter_comparison_run_ids,
    )
    filter_frame.insert(0, "configuration", item.configuration)
    filter_frame.to_csv(config_dir / "filter_comparison.csv", index=False)

    if continuous_profile and len(detection.peaks) >= MIN_EVENTS_FOR_ANALYSIS:
        degradation = spatial_resolution_study(
            raw_profiles, x, item.used_rate_hz, detection
        )
    else:
        degradation = pd.DataFrame()
    degradation.insert(0, "configuration", item.configuration) if not degradation.empty else None
    degradation.to_csv(config_dir / "spatial_degradation.csv", index=False)

    threshold_sensitivity = threshold_sensitivity_analysis(
        item,
        roi_profiles,
        roi_x,
        detection,
        features,
        profile_matrix,
        x,
        continuous_profile,
    )
    threshold_sensitivity.to_csv(
        config_dir / "threshold_sensitivity.csv", index=False
    )
    roi_boundary_sensitivity = roi_boundary_sensitivity_analysis(
        item, roi_indices, roi_profiles, detection
    )
    roi_boundary_sensitivity.to_csv(
        config_dir / "roi_boundary_sensitivity.csv", index=False
    )
    write_json(
        config_dir / "sensitivity_analysis_scope.json",
        {
            "threshold_sensitivity": "One-at-a-time alternatives for gauge completeness, spatial activity winsorization, peak spacing, event half-window, manual-match tolerance, and robust anomaly threshold.",
            "roi_boundary_sensitivity": "Independent endpoint shifts with detected event times held fixed to isolate spatial support effects.",
            "roi_boundary_offsets_mm": list(ROI_BOUNDARY_OFFSETS_MM),
            "roi_boundary_warning": "The +/-20 mm endpoint range is exploratory and is not a surveyed installation-coordinate uncertainty.",
            "sensitivity_status": "computed; inspect each CSV for metric availability and actual changes before making invariance claims.",
        },
    )

    if len(events):
        np.savez_compressed(
            config_dir / "event_aligned_data.npz",
            events=events,
            relative_time_s=relative_time,
            distance_m=x,
            peak_samples=valid_peaks,
            profile_matrix=profile_matrix,
            temporal_activity=temporal_matrix,
            spatial_run_ids=run_ids,
            continuous_profile=np.asarray([continuous_profile]),
        )
    plot_configuration_overview(
        item.configuration,
        raw_profiles,
        x,
        item.used_rate_hz,
        detection,
        events,
        relative_time,
        features,
        profile_matrix,
        config_dir / "analysis_overview.png",
    )
    plot_within_run_stability(
        item.configuration,
        features,
        config_dir / "within_run_stability.png",
    )
    plot_recurring_profile_analysis(
        item.configuration,
        x,
        profile_matrix,
        recurrence_lags,
        recurrence_summary,
        config_dir / "recurring_profile_families.png",
    )

    audit = {
        "configuration": item.configuration,
        "description": CONFIGURATION_DESCRIPTIONS.get(item.configuration, ""),
        "interrogator_model": INSTRUMENT_MODEL_BY_CONFIGURATION.get(item.configuration),
        "interrogator_model_basis": INSTRUMENT_MODEL_BASIS_BY_CONFIGURATION.get(item.configuration),
        "source_file": Path(item.source_path).name,
        "source_size_bytes": item.source_size,
        "n_scans": item.n_scans,
        "n_gages_total": item.n_gages,
        "distance_total_m": [float(item.distance_m[0]), float(item.distance_m[-1])],
        "record_time_span_s": (item.n_scans - 1) / item.used_rate_hz,
        "units": item.units,
        "row_quantity": item.quantity,
        "configured_rate_override_hz": ACQUISITION_RATE_OVERRIDES_HZ.get(
            item.configuration
        ),
        "declared_rate_hz": item.declared_rate_hz,
        "timestamp_inferred_rate_hz": item.timestamp_rate_hz,
        "timestamp_interval_cv_percent": item.timestamp_interval_cv_percent,
        "used_rate_hz": item.used_rate_hz,
        "used_rate_source": item.used_rate_source,
        "roi": roi_info,
        "gauge_validity": gauge_audit,
        "n_gages_after_validity_filter": int(len(x)),
        "primary_detection_filter": PRIMARY_DETECTION_FILTER,
        "reported_profile_source": "raw locally baseline-corrected ODiSI values",
        "profile_metric_status": (
            "continuous full-interval metrics"
            if continuous_profile
            else "fragmented-gauge diagnostics; full-profile metrics suppressed"
        ),
        "recurring_profile_analysis": recurrence_summary,
        "manual_detection_validation": manual_validation,
        "detection": {
            "dominant_frequency_hz": detection.dominant_frequency_hz,
            "autocorrelation_frequency_hz": detection.autocorrelation_frequency_hz,
            "autocorrelation_peak": detection.autocorrelation_peak,
            "used_pass_frequency_hz": detection.used_pass_frequency_hz,
            "used_frequency_source": detection.used_frequency_source,
            "n_detected_peaks": int(len(detection.peaks)),
            "n_complete_events": int(len(valid_peaks)),
            "prominence_threshold": detection.peak_prominence_threshold,
            "height_threshold": detection.peak_height_threshold,
            "minimum_peak_distance_samples": detection.minimum_peak_distance_samples,
            "minimum_peak_distance_fraction_of_period": MIN_PEAK_DISTANCE_PERIOD_FRACTION,
        },
        "warnings": [],
    }
    if item.declared_rate_hz is None:
        audit["warnings"].append(
            "The raw header contains no acquisition rate; timestamp evidence was used when available, otherwise the documented 40 Hz saved-data fallback was used."
        )
    if item.n_scans < max(32, int(4 * item.used_rate_hz)):
        audit["warnings"].append(
            "This file contains too few scans for reliable frequency/pass analysis."
        )
    audit["warnings"].append(
        "Verify the selected ODiSI interval against the physical sensor map before interpreting pavement response."
    )
    audit["warnings"].append(
        "Negative and positive strain are reported neutrally; map them to tension/compression only after confirming sensor orientation and sign convention."
    )
    if not continuous_profile:
        audit["warnings"].append(
            "Disconnected valid gauges do not support full-interval integral, centroid, width, profile correlation, or primary ICC interpretation."
        )
    if manual_validation.get("status") != "computed":
        audit["warnings"].append(
            "No external manual annotations were supplied; automatic detections are not compared with independent manual labels."
        )
    write_json(config_dir / "data_audit.json", audit)
    write_json(config_dir / "repeatability_summary.json", summary)
    return {
        "audit": audit,
        "summary": summary,
        "features": features,
        "filter": filter_frame,
        "degradation": degradation,
        "threshold_sensitivity": threshold_sensitivity,
        "roi_boundary_sensitivity": roi_boundary_sensitivity,
        "bootstrap": bootstrap_frame,
        "recurrence_lags": recurrence_lags,
        "recurrence_families": recurrence_families,
        "recurrence_summary": recurrence_summary,
    }


def concat_result_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate outputs without pandas' all-NA dtype-inference warning."""
    usable = [frame for frame in frames if not frame.empty]
    if not usable:
        return pd.DataFrame()
    ordered_columns = list(dict.fromkeys(column for frame in usable for column in frame.columns))
    trimmed = [frame.dropna(axis=1, how="all") for frame in usable]
    return pd.concat(trimmed, ignore_index=True).reindex(columns=ordered_columns)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze native ODiSI exports without bundling measurement data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="directory containing the three expected ODiSI exports",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for derived tables, figures, audit files, and cache",
    )
    parser.add_argument(
        "--manual-labels-dir",
        type=Path,
        default=DEFAULT_MANUAL_LABEL_DIR,
        help="optional directory of authorized manual-annotation CSV files",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="rebuild only the parsed-data cache under the selected output directory",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global DATA_DIR, OUTPUT_DIR, CACHE_DIR, MANUAL_LABEL_DIR, INPUT_FILES
    args = parse_arguments(argv)
    DATA_DIR = args.data_dir.expanduser().resolve()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    CACHE_DIR = OUTPUT_DIR / "cache"
    MANUAL_LABEL_DIR = args.manual_labels_dir.expanduser().resolve()
    INPUT_FILES = {
        configuration: DATA_DIR / filename
        for configuration, filename in INPUT_FILENAMES.items()
    }
    if args.clear_cache and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    inputs = resolve_inputs()
    results: dict[str, dict] = {}
    for configuration, path in inputs.items():
        print(f"\n[{configuration}] Parsing/analyzing {path.name}")
        item = parse_and_cache_odisi(configuration, path)
        results[configuration] = analyze_configuration(item)
        print(
            f"  scans={item.n_scans}, gages={item.n_gages}, "
            f"rate={item.used_rate_hz:g} Hz ({item.used_rate_source}), "
            f"complete_events={results[configuration]['summary'].get('n_complete_events', 0)}"
        )

    synthetic_detail, synthetic_summary = synthetic_ground_truth_benchmark()
    synthetic_detail.to_csv(
        OUTPUT_DIR / "synthetic_benchmark_replicates.csv", index=False
    )
    synthetic_summary.to_csv(
        OUTPUT_DIR / "synthetic_benchmark_summary.csv", index=False
    )

    threshold_frames = [
        result["threshold_sensitivity"]
        for result in results.values()
        if not result["threshold_sensitivity"].empty
    ]
    all_threshold_sensitivity = (
        pd.concat(threshold_frames, ignore_index=True)
        if threshold_frames
        else pd.DataFrame()
    )
    all_threshold_sensitivity.to_csv(
        OUTPUT_DIR / "all_threshold_sensitivity.csv", index=False
    )
    roi_boundary_frames = [
        result["roi_boundary_sensitivity"]
        for result in results.values()
        if not result["roi_boundary_sensitivity"].empty
    ]
    all_roi_boundary_sensitivity = (
        pd.concat(roi_boundary_frames, ignore_index=True)
        if roi_boundary_frames
        else pd.DataFrame()
    )
    all_roi_boundary_sensitivity.to_csv(
        OUTPUT_DIR / "all_roi_boundary_sensitivity.csv", index=False
    )

    features_by_config = {key: result["features"] for key, result in results.items()}
    summaries = pd.DataFrame([result["summary"] for result in results.values()])
    summaries.to_csv(OUTPUT_DIR / "configuration_repeatability_summary.csv", index=False)
    comparisons = configuration_comparisons(features_by_config)
    comparisons.to_csv(OUTPUT_DIR / "configuration_block_bootstrap_comparisons.csv", index=False)
    all_features = concat_result_frames(list(features_by_config.values()))
    all_features.to_csv(OUTPUT_DIR / "all_pass_features.csv", index=False)
    all_filters = concat_result_frames([result["filter"] for result in results.values()])
    all_filters.to_csv(OUTPUT_DIR / "all_filter_comparisons.csv", index=False)
    bootstrap_frames = [
        result["bootstrap"] for result in results.values() if not result["bootstrap"].empty
    ]
    all_bootstrap = (
        pd.concat(bootstrap_frames, ignore_index=True)
        if bootstrap_frames
        else pd.DataFrame()
    )
    all_bootstrap.to_csv(OUTPUT_DIR / "all_bootstrap_uncertainty.csv", index=False)
    degradation_frames = [result["degradation"] for result in results.values() if not result["degradation"].empty]
    all_degradation = pd.concat(degradation_frames, ignore_index=True) if degradation_frames else pd.DataFrame()
    all_degradation.to_csv(OUTPUT_DIR / "all_spatial_degradation.csv", index=False)
    recurrence_lag_frames = [
        result["recurrence_lags"]
        for result in results.values()
        if not result["recurrence_lags"].empty
    ]
    all_recurrence_lags = (
        pd.concat(recurrence_lag_frames, ignore_index=True)
        if recurrence_lag_frames
        else pd.DataFrame()
    )
    all_recurrence_lags.to_csv(
        OUTPUT_DIR / "all_recurring_profile_lags.csv", index=False
    )
    recurrence_family_frames = [
        result["recurrence_families"]
        for result in results.values()
        if not result["recurrence_families"].empty
    ]
    all_recurrence_families = (
        pd.concat(recurrence_family_frames, ignore_index=True)
        if recurrence_family_frames
        else pd.DataFrame()
    )
    all_recurrence_families.to_csv(
        OUTPUT_DIR / "all_recurring_profile_families.csv", index=False
    )
    plot_configuration_comparison(features_by_config, OUTPUT_DIR / "configuration_comparison.png")
    plot_spatial_degradation(all_degradation, OUTPUT_DIR / "spatial_degradation_summary.png")
    write_json(
        OUTPUT_DIR / "master_audit.json",
        {
            "software": "mmls_ofdr_analysis.py",
            "nominal_test_conditions": {
                "wheel_load_lb": NOMINAL_WHEEL_LOAD_LB,
                "tire_pressure_psi": NOMINAL_TIRE_PRESSURE_PSI,
                "mmls_speed_m_s": NOMINAL_MMLS_SPEED_M_S,
                "saved_data_rate_fallback_hz": FALLBACK_ACQUISITION_RATE_HZ,
                "pass_frequency_hz_from_7200_per_hour_reference_only": REPORT_NOMINAL_PASS_FREQUENCY_HZ,
            },
            "configurations": {key: result["audit"] for key, result in results.items()},
            "synthetic_benchmark": {
                "replicates_per_scenario": SYNTHETIC_BENCHMARK_REPLICATES,
                "seed": SYNTHETIC_BENCHMARK_SEED,
                "summary_file": "synthetic_benchmark_summary.csv",
                "replicate_file": "synthetic_benchmark_replicates.csv",
                "interpretation": "Algorithm behavior under specified synthetic assumptions; not independent physical validation.",
            },
            "sensitivity_outputs": {
                "thresholds": "all_threshold_sensitivity.csv",
                "roi_boundaries": "all_roi_boundary_sensitivity.csv",
                "roi_boundary_offsets_mm": list(ROI_BOUNDARY_OFFSETS_MM),
            },
            "global_warnings": [
                "Configuration is confounded with acquisition date and potentially specimen/setup; comparisons are descriptive.",
                "Repeated wheel passes are temporally correlated and are not independent pavement specimens.",
                "Robust response-anomaly flags are screening indicators, not confirmed damage labels.",
                "Synthetic spatial degradation tests evaluate algorithm robustness, not experimentally observed fiber damage.",
                "Saved-data rate is resolved from file metadata and timestamps; a configured fallback is used only when neither is available.",
                "Full-profile metrics are suppressed for disconnected valid-gauge layouts and retained only as explicitly labeled diagnostics.",
                "Recurring profile families are post hoc sequence diagnostics and are not assigned to individual wheels or bogies without synchronized identifiers.",
                "Synthetic benchmark metrics must be checked against the final pipeline and are not physical MMLS validation.",
                "ROI-boundary sensitivity uses exploratory endpoint perturbations and fixed event detections; it does not establish bonded-fiber coordinate accuracy or boundary-dependent detection robustness.",

            ],
        },
    )
    print(f"\nCompleted. Results: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
