import math

from src.features.bearing_freqs import (
    CWRU_DRIVE_END_BEARING,
    BearingGeometry,
    all_characteristic_frequencies,
    bpfi,
    bpfo,
    bsf,
    closest_characteristic_frequency,
    ftf,
)


def test_frequencies_positive_and_ordered():
    shaft_freq = 29.95  # ~1797 rpm, a CWRU drive-end operating point
    freqs = all_characteristic_frequencies(shaft_freq, CWRU_DRIVE_END_BEARING)
    assert all(v > 0 for v in freqs.values())
    # BPFI > BPFO for a fixed bearing under radial load (more balls loaded on inner race path)
    assert freqs["BPFI"] > freqs["BPFO"]


def test_bpfo_bpfi_symmetry_around_shaft_harmonic():
    # BPFO + BPFI should equal n * shaft_freq exactly, by construction of the formulas.
    shaft_freq = 30.0
    geom = BearingGeometry(n_elements=9, element_diameter=0.3126, pitch_diameter=1.537)
    total = bpfo(shaft_freq, geom) + bpfi(shaft_freq, geom)
    assert math.isclose(total, geom.n_elements * shaft_freq, rel_tol=1e-9)


def test_ftf_bounded_by_half_shaft_speed():
    shaft_freq = 30.0
    ftf_val = ftf(shaft_freq, CWRU_DRIVE_END_BEARING)
    assert 0 < ftf_val < shaft_freq / 2


def test_bsf_matches_known_value_zero_contact_angle():
    geom = BearingGeometry(n_elements=9, element_diameter=0.3126, pitch_diameter=1.537)
    shaft_freq = 30.0
    val = bsf(shaft_freq, geom)
    ratio = geom.element_diameter / geom.pitch_diameter
    expected = (geom.pitch_diameter / (2 * geom.element_diameter)) * shaft_freq * (1 - ratio**2)
    assert math.isclose(val, expected, rel_tol=1e-9)


def test_closest_characteristic_frequency_exact_match():
    shaft_freq = 29.95
    geom = CWRU_DRIVE_END_BEARING
    true_bpfo = bpfo(shaft_freq, geom)
    label, matched_freq, err = closest_characteristic_frequency(true_bpfo, shaft_freq, geom)
    assert label == "BPFO"
    assert err < 1e-6
    assert math.isclose(matched_freq, true_bpfo)


def test_closest_characteristic_frequency_harmonic():
    shaft_freq = 29.95
    geom = CWRU_DRIVE_END_BEARING
    true_bpfi_2nd = bpfi(shaft_freq, geom) * 2
    label, _, err = closest_characteristic_frequency(true_bpfi_2nd, shaft_freq, geom, max_harmonic=3)
    assert label == "BPFI x2"
    assert err < 1e-6
