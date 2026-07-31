#!/usr/bin/env python
"""Physics-based synthetic vibration-signal generator.

Real CWRU/IMS datasets are large, license-gated downloads. To make the
entire TurboGuard pipeline runnable, testable, and demoable without them,
this script synthesizes small CWRU-like (fault classification, multiple
loads) and IMS-like (run-to-failure RUL) datasets that:

- use the *real* CWRU bearing geometry and shaft speeds (README section 7.3),
  so the resulting characteristic frequencies (BPFO/BPFI/BSF/FTF) match the
  literature exactly (e.g. BPFO = 107.36 Hz at 1797 rpm, drive-end bearing);
- model each bearing defect as a periodic train of resonance-ringing impacts
  at its characteristic frequency, amplitude-modulated as appropriate for the
  defect location (inner-race impacts are modulated by shaft rotation as the
  defect passes through the load zone; outer-race impacts are not);
- model IMS-style degradation as a growing-severity outer-race defect over a
  run-to-failure trajectory of snapshots, giving genuine (if small-scale)
  degradation curves for RUL estimation.

Downstream code (preprocessing, features, models) treats this data exactly
like real CWRU/IMS data — swapping in the real datasets later requires no
code changes, only pointing ``--input_dir`` at the real files.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.bearing_freqs import (
    CWRU_DRIVE_END_BEARING,
    IMS_BEARING,
    IMS_SHAFT_FREQ_HZ,
    BearingGeometry,
    bpfi,
    bpfo,
    bsf,
    ftf,
)
from src.utils.io import ensure_dir
from src.utils.logging_config import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)

# Real CWRU drive-end shaft speeds (rpm) for each motor load, from the
# CWRU Bearing Data Center documentation.
CWRU_LOAD_TO_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

FAULT_CLASSES = ["healthy", "inner_race", "outer_race", "ball", "compound"]

# Structural resonance excited by each impact (typical for small motor test rigs).
RESONANCE_FREQ_HZ = 3200.0
RESONANCE_DECAY_S = 0.0015


def _shaft_harmonics(t: np.ndarray, shaft_freq_hz: float, rng: np.random.Generator) -> np.ndarray:
    """Baseline imbalance/misalignment vibration present in every health state."""
    sig = np.zeros_like(t)
    for harmonic, amp in ((1, 0.06), (2, 0.03), (3, 0.015)):
        phase = rng.uniform(0, 2 * np.pi)
        sig += amp * np.sin(2 * np.pi * harmonic * shaft_freq_hz * t + phase)
    return sig


def _impulse_train(
    t: np.ndarray,
    sample_rate: float,
    defect_freq_hz: float,
    severity_amp: float,
    modulation_freq_hz: float | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """A train of resonance-ringing impacts at ``defect_freq_hz``, optionally
    amplitude-modulated at ``modulation_freq_hz`` (load-zone passage)."""
    duration_s = t[-1] + 1 / sample_rate
    dt_nominal = 1.0 / defect_freq_hz
    n_impulses = int(duration_s / dt_nominal) + 2
    jitter = rng.normal(0, 0.01 * dt_nominal, n_impulses)  # ~1% slip, as in real bearings
    impulse_times = np.cumsum(np.full(n_impulses, dt_nominal) + jitter)
    impulse_times = impulse_times[impulse_times < duration_s]

    sig = np.zeros_like(t)
    kernel_span_s = 6 * RESONANCE_DECAY_S
    kernel_span_n = int(kernel_span_s * sample_rate)

    for t0 in impulse_times:
        amp = severity_amp
        if modulation_freq_hz is not None:
            amp *= 1.0 + 0.7 * np.cos(2 * np.pi * modulation_freq_hz * t0)
            amp = max(amp, 0.0)
        start_idx = int(t0 * sample_rate)
        end_idx = min(start_idx + kernel_span_n, len(t))
        if start_idx >= end_idx:
            continue
        local_t = t[start_idx:end_idx] - t0
        kernel = amp * np.exp(-local_t / RESONANCE_DECAY_S) * np.sin(
            2 * np.pi * RESONANCE_FREQ_HZ * local_t
        )
        sig[start_idx:end_idx] += kernel
    return sig


def simulate_bearing_signal(
    *,
    duration_s: float,
    sample_rate: float,
    shaft_freq_hz: float,
    fault_type: str,
    severity: float,
    geom: BearingGeometry,
    rng: np.random.Generator,
    n_channels: int = 3,
) -> np.ndarray:
    """Simulate a triaxial vibration snapshot for one health state.

    ``severity`` in [0, 1]; 0 must correspond to the healthy baseline noise
    floor and increasing values raise impact amplitude (and hence crest
    factor / kurtosis / envelope-band energy — the standard early-fault
    indicators used throughout the feature-engineering module).
    """
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    base_noise_std = 0.05
    channels = np.stack(
        [rng.normal(0, base_noise_std, n_samples) for _ in range(n_channels)]
    )
    shaft_component = _shaft_harmonics(t, shaft_freq_hz, rng)
    channels += shaft_component[None, :]

    if fault_type == "healthy" or severity <= 0:
        return channels

    defect_specs: list[tuple[float, float | None, float]]
    if fault_type == "outer_race":
        defect_specs = [(bpfo(shaft_freq_hz, geom), None, 1.0)]
    elif fault_type == "inner_race":
        defect_specs = [(bpfi(shaft_freq_hz, geom), shaft_freq_hz, 1.0)]
    elif fault_type == "ball":
        defect_specs = [(bsf(shaft_freq_hz, geom), ftf(shaft_freq_hz, geom), 1.0)]
    elif fault_type == "compound":
        defect_specs = [
            (bpfo(shaft_freq_hz, geom), None, 0.6),
            (bpfi(shaft_freq_hz, geom), shaft_freq_hz, 0.6),
        ]
    else:
        raise ValueError(f"Unknown fault_type: {fault_type}")

    coupling = [1.0, 0.4, 0.25][:n_channels]
    for defect_freq, mod_freq, weight in defect_specs:
        base_amp = severity * weight * 1.2
        impulse = _impulse_train(t, sample_rate, defect_freq, base_amp, mod_freq, rng)
        for ch in range(n_channels):
            channels[ch] += coupling[ch] * impulse

    return channels


def gen_cwru_like(
    output_dir: Path,
    trials_per_condition: int,
    duration_s: float,
    sample_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out_dir = ensure_dir(output_dir / "cwru" / "synthetic")
    severity_by_class = {
        "healthy": 0.0,
        "inner_race": 1.0,
        "outer_race": 1.0,
        "ball": 1.0,
        "compound": 1.0,
    }
    rows = []
    for load, rpm in CWRU_LOAD_TO_RPM.items():
        shaft_freq_hz = rpm / 60.0
        for fault_type in FAULT_CLASSES:
            for trial in range(trials_per_condition):
                # Vary severity slightly per trial (like CWRU's multiple fault diameters).
                severity = severity_by_class[fault_type]
                if severity > 0:
                    severity *= float(rng.uniform(0.8, 1.3))
                signal = simulate_bearing_signal(
                    duration_s=duration_s,
                    sample_rate=sample_rate,
                    shaft_freq_hz=shaft_freq_hz,
                    fault_type=fault_type,
                    severity=severity,
                    geom=CWRU_DRIVE_END_BEARING,
                    rng=rng,
                )
                filename = f"{fault_type}_load{load}_trial{trial}.csv.gz"
                df = pd.DataFrame(signal.T, columns=["acc_x", "acc_y", "acc_z"])
                df.to_csv(out_dir / filename, index=False, compression="gzip")
                rows.append(
                    {
                        "filename": filename,
                        "health_state": fault_type,
                        "load_hp": load,
                        "rpm": rpm,
                        "shaft_freq_hz": shaft_freq_hz,
                        "severity": severity,
                        "sample_rate_hz": sample_rate,
                        "duration_s": duration_s,
                    }
                )
    labels = pd.DataFrame(rows)
    labels.to_csv(out_dir / "labels.csv", index=False)
    logger.info("Wrote %d synthetic CWRU-like files to %s", len(labels), out_dir)
    return labels


def gen_ims_like(
    output_dir: Path,
    n_bearings: int,
    n_snapshots: int,
    snapshot_duration_s: float,
    sample_rate: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    out_dir = ensure_dir(output_dir / "ims" / "synthetic")
    rows = []
    geom = IMS_BEARING

    for bearing_id in range(1, n_bearings + 1):
        bearing_dir = ensure_dir(out_dir / f"bearing{bearing_id}")
        # Exponential degradation health indicator, normalized to reach 1.0 at failure.
        growth_rate = float(rng.uniform(4.0, 6.0))
        cycles = np.arange(n_snapshots)
        raw_hi = np.exp(growth_rate * cycles / n_snapshots) - 1.0
        hi = raw_hi / raw_hi[-1]  # in [0, 1], severity proxy for the outer-race defect

        for snapshot_idx, severity in zip(cycles, hi):
            signal = simulate_bearing_signal(
                duration_s=snapshot_duration_s,
                sample_rate=sample_rate,
                shaft_freq_hz=IMS_SHAFT_FREQ_HZ,
                fault_type="outer_race" if severity > 0.02 else "healthy",
                severity=float(severity),
                geom=geom,
                rng=rng,
            )
            filename = f"snapshot_{snapshot_idx:05d}.csv.gz"
            df = pd.DataFrame(signal.T, columns=["acc_x", "acc_y", "acc_z"])
            df.to_csv(bearing_dir / filename, index=False, compression="gzip")
            rows.append(
                {
                    "bearing_id": bearing_id,
                    "filename": f"bearing{bearing_id}/{filename}",
                    "snapshot_index": int(snapshot_idx),
                    "rul_cycles": int(n_snapshots - 1 - snapshot_idx),
                    "health_indicator": float(severity),
                    "sample_rate_hz": sample_rate,
                    "duration_s": snapshot_duration_s,
                    "dominant_fault": "outer_race",
                }
            )
    labels = pd.DataFrame(rows)
    labels.to_csv(out_dir / "labels.csv", index=False)
    logger.info("Wrote %d synthetic IMS-like snapshots to %s", len(labels), out_dir)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cwru_trials_per_condition", type=int, default=1)
    parser.add_argument("--cwru_duration_s", type=float, default=1.0)
    parser.add_argument("--cwru_sample_rate", type=float, default=12000.0)
    parser.add_argument("--ims_n_bearings", type=int, default=2)
    parser.add_argument("--ims_n_snapshots", type=int, default=20)
    parser.add_argument(
        "--ims_snapshot_duration_s",
        type=float,
        default=1.0,
        help="Kept equal to the model's window_seconds (1.0) so CWRU and IMS "
        "windows share one shape and can feed the same multi-task encoder.",
    )
    parser.add_argument("--ims_sample_rate", type=float, default=12000.0)
    args = parser.parse_args()

    set_seed(args.seed)
    gen_cwru_like(
        args.output_dir,
        args.cwru_trials_per_condition,
        args.cwru_duration_s,
        args.cwru_sample_rate,
        args.seed,
    )
    gen_ims_like(
        args.output_dir,
        args.ims_n_bearings,
        args.ims_n_snapshots,
        args.ims_snapshot_duration_s,
        args.ims_sample_rate,
        args.seed,
    )


if __name__ == "__main__":
    main()
