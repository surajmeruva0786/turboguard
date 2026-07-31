"""Bearing characteristic fault frequencies (BPFO, BPFI, BSF, FTF).

These frequencies are the physical fingerprint of localized bearing defects:
every time a rolling element passes over a defect it produces a mechanical
impact, and those impacts repeat at a frequency fixed by the bearing's
geometry and the shaft speed. See README section 7.3 for the formulas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Geometric parameters of a rolling-element bearing.

    Attributes
    ----------
    n_elements: number of rolling elements (balls/rollers).
    element_diameter: rolling-element diameter, ``d``.
    pitch_diameter: pitch diameter, ``D`` (center-to-center of opposing elements).
    contact_angle_deg: contact angle, ``phi``, in degrees (0 for pure radial load).
    """

    n_elements: int
    element_diameter: float
    pitch_diameter: float
    contact_angle_deg: float = 0.0

    @property
    def contact_angle_rad(self) -> float:
        return math.radians(self.contact_angle_deg)

    @property
    def diameter_ratio(self) -> float:
        """``(d / D) * cos(phi)`` — the recurring geometric term in all four formulas."""
        return (self.element_diameter / self.pitch_diameter) * math.cos(self.contact_angle_rad)


# Published geometries for the two SKF deep-groove ball bearings used in the
# CWRU test rig (drive-end and fan-end), in inches. Source: Smith & Randall
# (2015), "Rolling element bearing diagnostics using the CWRU data".
CWRU_DRIVE_END_BEARING = BearingGeometry(n_elements=9, element_diameter=0.3126, pitch_diameter=1.537)
CWRU_FAN_END_BEARING = BearingGeometry(n_elements=9, element_diameter=0.2656, pitch_diameter=1.122)

# Rexnord ZA-2115 double-row bearing used in the NASA IMS test rig, in inches.
# Source: Qiu et al. (2006).
IMS_BEARING = BearingGeometry(n_elements=16, element_diameter=0.331, pitch_diameter=2.815)


def bpfo(shaft_freq_hz: float, geom: BearingGeometry) -> float:
    """Ball Pass Frequency – Outer race."""
    return (geom.n_elements / 2) * shaft_freq_hz * (1 - geom.diameter_ratio)


def bpfi(shaft_freq_hz: float, geom: BearingGeometry) -> float:
    """Ball Pass Frequency – Inner race."""
    return (geom.n_elements / 2) * shaft_freq_hz * (1 + geom.diameter_ratio)


def bsf(shaft_freq_hz: float, geom: BearingGeometry) -> float:
    """Ball Spin Frequency."""
    ratio = geom.diameter_ratio
    return (geom.pitch_diameter / (2 * geom.element_diameter)) * shaft_freq_hz * (1 - ratio**2)


def ftf(shaft_freq_hz: float, geom: BearingGeometry) -> float:
    """Fundamental Train Frequency (cage frequency)."""
    return 0.5 * shaft_freq_hz * (1 - geom.diameter_ratio)


def all_characteristic_frequencies(shaft_freq_hz: float, geom: BearingGeometry) -> dict[str, float]:
    """Return all four characteristic frequencies for a given shaft speed."""
    return {
        "BPFO": bpfo(shaft_freq_hz, geom),
        "BPFI": bpfi(shaft_freq_hz, geom),
        "BSF": bsf(shaft_freq_hz, geom),
        "FTF": ftf(shaft_freq_hz, geom),
    }


def closest_characteristic_frequency(
    peak_freq_hz: float, shaft_freq_hz: float, geom: BearingGeometry, max_harmonic: int = 3
) -> tuple[str, float, float]:
    """Find which characteristic frequency (or harmonic) best matches an observed peak.

    Returns ``(label, matched_frequency_hz, relative_error)`` for the closest
    match among BPFO/BPFI/BSF/FTF and their harmonics up to ``max_harmonic``.
    Used to give operators a physical justification for an alert (README 13.3).
    """
    freqs = all_characteristic_frequencies(shaft_freq_hz, geom)
    best_label, best_freq, best_err = "", 0.0, math.inf
    for label, base_freq in freqs.items():
        for h in range(1, max_harmonic + 1):
            candidate = base_freq * h
            if candidate <= 0:
                continue
            err = abs(peak_freq_hz - candidate) / candidate
            if err < best_err:
                best_label = label if h == 1 else f"{label} x{h}"
                best_freq = candidate
                best_err = err
    return best_label, best_freq, best_err
