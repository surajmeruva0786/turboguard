"""Frequency-band physical-justification annotator (README 13.3).

Every alert is annotated with the dominant envelope-spectrum peak and the
closest bearing characteristic frequency, giving the operator a
single-line physical justification for the prediction, e.g.:

    "Predicted: Outer-race fault. Confidence 0.94. Dominant envelope peak
     at 107.4 Hz matches BPFO (107.36 Hz) within 0.04%."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.bearing_freqs import BearingGeometry, closest_characteristic_frequency
from src.features.envelope import dominant_envelope_peak

_FAULT_LABELS = {
    "healthy": "Healthy",
    "inner_race": "Inner-race fault",
    "outer_race": "Outer-race fault",
    "ball": "Ball fault",
    "compound": "Compound fault",
}


@dataclass
class FrequencyAnnotation:
    predicted_class: str
    confidence: float
    dominant_peak_hz: float
    matched_label: str
    matched_freq_hz: float
    relative_error: float

    @property
    def message(self) -> str:
        label = _FAULT_LABELS.get(self.predicted_class, self.predicted_class)
        return (
            f"Predicted: {label}. Confidence {self.confidence:.2f}. "
            f"Dominant envelope peak at {self.dominant_peak_hz:.1f} Hz matches "
            f"{self.matched_label} ({self.matched_freq_hz:.2f} Hz) within {self.relative_error * 100:.2f}%."
        )


def annotate_alert(
    x: np.ndarray,
    sample_rate: float,
    shaft_freq_hz: float,
    geom: BearingGeometry,
    predicted_class: str,
    confidence: float,
    band: tuple[float, float] = (2000.0, 5900.0),
) -> FrequencyAnnotation:
    """Compute the dominant envelope peak for a single-axis window ``x`` and
    match it to the closest bearing characteristic frequency."""
    peak_freq, _peak_amp = dominant_envelope_peak(x, sample_rate, band)
    label, matched_freq, rel_err = closest_characteristic_frequency(peak_freq, shaft_freq_hz, geom)
    return FrequencyAnnotation(
        predicted_class=predicted_class,
        confidence=confidence,
        dominant_peak_hz=peak_freq,
        matched_label=label,
        matched_freq_hz=matched_freq,
        relative_error=rel_err,
    )
