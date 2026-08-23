#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FES2004 Ocean Tide Evaluation — Jolotundo Astronomical Observatory
==================================================================
Computes tidal harmonic constituents (amplitude and phase) at the
Jolotundo Observatory (7°37'05"S, 112°37'01"E) using the FES2004
global ocean tide model.

Input:
    fes2004_100_100_ell.txt  — FES2004 spherical harmonic coefficients

Output:
    Per-constituent amplitude (cm) and phase lag (degrees, UTC)
    Simulated hourly time series with synthetic observation gaps

Reference:
    Lyard, F., Lefevre, F., Letellier, T., Francis, O. (2006)
    "Modelling the global ocean tides: modern insights from FES2004"
    Ocean Dynamics, 56, 394–415

Author:  Jolotundo Research Consortium
Date:    2026-05-11
"""

import numpy as np
from scipy.special import lpmv
from math import lgamma, exp, pi, sin, cos, radians, atan2, sqrt, degrees


# =============================================================================
# FES2004 Spherical Harmonic Model Loader
# =============================================================================
class FES2004:
    """
    Loader and evaluator for FES2004 ocean tide spherical harmonics.

    Parses the standard IERS FES2004 coefficient file and provides
    pointwise evaluation of tidal height for each constituent.
    """

    def __init__(self, filepath):
        self.data = {}
        self.components = {}
        self._parse(filepath)

    def _parse(self, filepath):
        """Parse the FES2004 coefficient file."""
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip the three-line header
        for line in lines[3:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                continue

            try:
                doodson   = float(parts[0])
                name      = parts[1]
                l         = int(parts[2])
                m         = int(parts[3])
                c_plus    = float(parts[8])
                eps_plus  = float(parts[9])
                c_minus   = float(parts[10])
                eps_minus = float(parts[11])
            except (ValueError, IndexError):
                continue

            key = (doodson, name)
            if key not in self.data:
                self.data[key] = []
                self.components[name] = key
            self.data[key].append({
                'l': l, 'm': m,
                'c+': c_plus, 'eps+': eps_plus,
                'c-': c_minus, 'eps-': eps_minus,
            })

    def _plm_4pi(self, l, m, x):
        """4π-normalized associated Legendre function of degree l, order m."""
        m_abs = abs(m)
        delta = 1 if m_abs == 0 else 2
        log_norm = 0.5 * (np.log(delta) + np.log(2 * l + 1) +
                          lgamma(l - m_abs + 1) - lgamma(l + m_abs + 1))
        norm = exp(log_norm)
        p = lpmv(m_abs, l, x)
        if not np.isfinite(p):
            return 0.0
        return norm * p

    def evaluate_component(self, lat, lon, comp_key):
        """
        Evaluate a single tidal constituent at geographic coordinates.

        Parameters
        ----------
        lat, lon : float
            Geodetic latitude and longitude (degrees).
        comp_key : str or tuple
            Constituent name (e.g. 'M2') or (doodson, name) tuple.

        Returns
        -------
        amplitude_cm : float
            Tidal amplitude in centimetres.
        phase_deg : float
            Phase lag in degrees (0–360).
        """
        if isinstance(comp_key, str):
            if comp_key not in self.components:
                raise ValueError(f"Constituent '{comp_key}' not found.")
            comp_key = self.components[comp_key]

        entries = self.data.get(comp_key, [])
        phi = radians(lat)
        lam = radians(lon)
        sin_phi = sin(phi)

        sum_cos = 0.0
        sum_sin = 0.0

        for e in entries:
            l, m = e['l'], e['m']
            c_plus, eps_plus = e['c+'], radians(e['eps+'])
            c_minus, eps_minus = e['c-'], radians(e['eps-'])

            plm = self._plm_4pi(l, m, sin_phi)

            arg_plus = m * lam - eps_plus
            sum_cos += c_plus * cos(arg_plus) * plm
            sum_sin += c_plus * sin(arg_plus) * plm

            if c_minus != 0.0:
                arg_minus = m * lam - eps_minus
                sum_cos += c_minus * cos(arg_minus) * plm
                sum_sin += c_minus * sin(arg_minus) * plm

        amp = sqrt(sum_cos**2 + sum_sin**2)
        phase = degrees(atan2(sum_sin, sum_cos))
        if phase < 0:
            phase += 360.0
        return amp, phase

    def evaluate_all(self, lat, lon):
        """Evaluate all available constituents at a location."""
        return {name: self.evaluate_component(lat, lon, name)
                for name in self.components}

    def get_components(self):
        """Return list of available constituent names."""
        return list(self.components.keys())


# =============================================================================
# Tidal Harmonic Constants (Cartwright-Tayler convention)
# =============================================================================
TIDAL_CONSTITUENTS = {
    'Q1': {'frequency_cph': 0.649585,  'v0u_rad': radians(280.0)},
    'O1': {'frequency_cph': 0.675977,  'v0u_rad': radians(270.0)},
    'K1': {'frequency_cph': 0.729211,  'v0u_rad': radians(170.0)},
    'M2': {'frequency_cph': 1.405189,  'v0u_rad': radians(200.0)},
    'S2': {'frequency_cph': 1.454441,  'v0u_rad': radians(250.0)},
    'N2': {'frequency_cph': 1.378797,  'v0u_rad': radians(190.0)},
    'K2': {'frequency_cph': 1.458423,  'v0u_rad': radians(260.0)},
    'P1': {'frequency_cph': 0.725226,  'v0u_rad': radians(165.0)},
}


# =============================================================================
# Time Series Prediction
# =============================================================================
def predict_tide_series(model, lat, lon, hours, constituents=None):
    """
    Predict tidal height time series at a location.

    Parameters
    ----------
    model : FES2004
        Loaded FES2004 model instance.
    lat, lon : float
        Geodetic coordinates.
    hours : ndarray
        Time in hours since t0.
    constituents : dict, optional
        Tidal constituents with 'frequency_cph' and 'v0u_rad'.

    Returns
    -------
    tide : ndarray
        Predicted tidal height in cm.
    """
    if constituents is None:
        constituents = TIDAL_CONSTITUENTS

    total = np.zeros(len(hours), dtype=np.float64)
    for name, params in constituents.items():
        try:
            amp, phase_deg = model.evaluate_component(lat, lon, name)
        except ValueError:
            continue
        phase_rad = radians(phase_deg)
        freq = float(params['frequency_cph'])
        v0u = float(params['v0u_rad'])
        total += amp * np.cos(freq * hours + v0u - phase_rad)
    return total


# =============================================================================
# Synthetic Observation Generator
# =============================================================================
def generate_observation(times, true_series, noise_std=5.0, gap_prob=0.05):
    """
    Generate synthetic observations with Gaussian noise and random gaps.

    Parameters
    ----------
    times : ndarray
        Time vector (unused except for shape).
    true_series : ndarray
        True tidal signal.
    noise_std : float
        Standard deviation of Gaussian noise (cm).
    gap_prob : float
        Probability of a data gap (0 to 1).

    Returns
    -------
    obs : ndarray
        Synthetic observations with NaN values at gaps.
    """
    obs = true_series + np.random.normal(0, noise_std, len(times))
    mask = np.random.random(len(times)) < gap_prob
    obs[mask] = np.nan
    return obs


# =============================================================================
# Main: Jolotundo Observatory Evaluation
# =============================================================================
if __name__ == "__main__":
    # -----------------------------------------------------------------
    # 1. Load FES2004 model
    # -----------------------------------------------------------------
    MODEL_FILE = "fes2004_100_100_ell.txt"
    print("Loading FES2004 ocean tide model...")
    model = FES2004(MODEL_FILE)
    components = model.get_components()
    print(f"  Loaded {len(components)} tidal constituents: {components}\n")

    # -----------------------------------------------------------------
    # 2. Jolotundo Observatory coordinates (WGS84)
    # -----------------------------------------------------------------
    JOLO_LAT = -7.618055555555556   # 7°37'05.0" S
    JOLO_LON = 112.61694444444444   # 112°37'01.0" E

    print("=" * 65)
    print("  JOLOTUNDO ASTRONOMICAL OBSERVATORY")
    print(f"  Latitude  : {JOLO_LAT:+.6f}°")
    print(f"  Longitude : {JOLO_LON:+.6f}°")
    print("=" * 65)
    print()

    # -----------------------------------------------------------------
    # 3. Evaluate all available harmonic constituents
    # -----------------------------------------------------------------
    print("Tidal Harmonic Constituents (FES2004)")
    print("-" * 50)
    print(f"  {'Constituent':<12s} {'Amplitude (cm)':>14s}  {'Phase (°)':>10s}")
    print(f"  {'-'*12} {'-'*14}  {'-'*10}")

    constituent_data = {}
    for name in ['M2', 'S2', 'N2', 'K2', '2N2', 'K1', 'O1', 'P1', 'Q1',
                 'Mf', 'Mm', 'Ssa', 'Sa', 'Mtm', 'Msq', 'M4', 'Om1', 'Om2']:
        try:
            amp, phase = model.evaluate_component(JOLO_LAT, JOLO_LON, name)
            constituent_data[name] = {'amplitude_cm': amp, 'phase_deg': phase}
            print(f"  {name:<12s} {amp:14.2f}  {phase:10.1f}")
        except ValueError:
            pass
    print()

    # -----------------------------------------------------------------
    # 4. Time series prediction (30 days, 1-hour resolution)
    # -----------------------------------------------------------------
    HOURS = np.arange(0, 720, 1, dtype=np.float64)

    # Use constituents available in both model and our catalog
    available = {k: v for k, v in TIDAL_CONSTITUENTS.items()
                 if k in model.components}
    print(f"Constituents used for time series: {list(available.keys())}")

    predicted = predict_tide_series(model, JOLO_LAT, JOLO_LON, HOURS, available)

    # Synthetic observations from the same model
    observed = generate_observation(HOURS, predicted, noise_std=8.0, gap_prob=0.1)

    # -----------------------------------------------------------------
    # 5. Sample output
    # -----------------------------------------------------------------
    print(f"\n{'Time (h)':>8s}  {'Predicted (cm)':>14s}  {'Observed (cm)':>14s}")
    print(f"{'-'*8}  {'-'*14}  {'-'*14}")
    for t in range(0, 24, 3):
        obs_str = f"{observed[t]:14.2f}" if not np.isnan(observed[t]) else "           NaN"
        print(f"{t:8d}  {predicted[t]:14.2f}  {obs_str}")

    # -----------------------------------------------------------------
    # 6. Error statistics (on non-NaN values)
    # -----------------------------------------------------------------
    valid = ~np.isnan(observed)
    residuals = predicted[valid] - observed[valid]
    rmse = np.sqrt(np.mean(residuals**2))
    print(f"\n  RMSE (predicted vs observed): {rmse:.2f} cm")
    print(f"  Valid samples              : {np.sum(valid)} / {len(observed)}")
    print("\nDone.")
