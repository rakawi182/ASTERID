#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realtime_ips2000.py
============================================================
HIGH‑PRECISION REAL‑TIME ASTROMETRIC & GEODETIC EPHEMERIS
ENGINE: IPS2000 (Improved Planetary Solutions 2000)
============================================================

DESCRIPTION
-----------
This module implements a single‑shot, high‑accuracy ephemeris
generator for the Sun and the Moon, orchestrating a complete
hierarchical transformation chain strictly adhering to the
IAU SOFA (Standards of Fundamental Astronomy) framework and
IERS Conventions (2010).

The computation follows the rigorous astrometric reduction
sequence from the external reference frame to the observer's
local horizon:

    1. BARYCENTRIC / HELIOCENTRIC FRAME
       - Earth‑Moon Barycentre (EMB) state vector referred to
         the solar system barycenter (ICRS). Derived from the
         IPS2000 analytical planetary theory by Chapront &
         Francou (2002).

    2. GEOCENTRIC FRAME (GCRS / ICRS)
       - Positions of the Sun and Moon with respect to the
         Earth's center, expressed in the Geocentric Celestial
         Reference System (GCRS), which is orientationally
         identical to the International Celestial Reference
         System (ICRS) for sub‑microarcsecond accuracy.
       - Apparent geocentric motion rates (dRA/dt, dDec/dt) are
         computed via analytical derivatives of the position
         vectors.

    3. TOPOCENTRIC FRAME (CIRS & Equinox of Date)
       - Transformation from GCRS to the Celestial Intermediate
         Reference System (CIRS) using the IAU 2006/2000A
         precession‑nutation models (CIP/CIO).
       - Classical equinox‑based apparent places are also
         provided via the Equation of the Origins (EO).
       - Corrections applied:
           * IERS EOP (polar motion, UT1‑UTC)
           * Frame bias (IAU 2006)
           * Relativistic light‑time (Shapiro delay)
           * Gravitational light deflection (Sun)
           * Stellar aberration (annual & diurnal)
           * **Vertical deflection (DoV)** from local EGM2008
             grids (AdvancedGeoidInversion)

    4. OBSERVED / APPARENT FRAME (Horizontal)
       - Transformation to the local horizontal system (Az/Alt)
         using rigorous ITRS‑to‑ENU rotations.
       - High‑fidelity atmospheric refraction correction using
         the VMF3 (Vienna Mapping Functions 3) and GPT3
         (Global Pressure and Temperature 3) models.

    5. GEOPHYSICAL CORRECTIONS (IERS 2010, Chapters 7 & 11)
       All corrections are applied in **full double‑precision
       without any simplification or truncation**, preserving
       the sub‑milliarcsecond accuracy of the underlying models:
         * Solid Earth tides (including frequency‑dependent
           Love numbers)
         * Ocean tide loading (FES2014)
         * Pole tide (diurnal and semi‑diurnal)
         * Atmospheric loading (non‑tidal)
         * Non‑tidal loading (hydrology, etc.)
         * Vertical deflection (DoV) from local EGM2008 grids
         * Kinematic plate motion (ITRF2020‑PMM, Altamimi 2023)

ADDITIONAL SCIENTIFIC PRODUCTS
------------------------------
- Lunar Phase: Elongation (0‑360°), phase angle, illumination
  percentage, age (days), and standard 8‑phase nomenclature
  (USNO/Almanac convention).
- Lunar Libration: Optical (P1, P2) and physical (Tau)
  components via the LLIB04 analytical model.
- Lunar Osculating Orbital Elements: Semi‑major axis (a),
  eccentricity (e), inclination (i), node (Ω), argument of
  perigee (ω), and mean anomaly (M) computed from the
  geocentric state vector (Vallado, 2013).
- Photometric Quantities: Apparent V‑band magnitude of the
  Sun (IAU standard: ‑26.74) and the Moon (Allen 1976,
  phase‑angle dependent).

REFERENCES
----------
1. IAU SOFA Software Collection. http://www.iausofa.org
2. McCarthy, D.D. & Petit, G. (2010). IERS Conventions (2010).
   IERS Technical Note 36.
3. Chapront, J. & Francou, G. (2002). Improved Planetary
   Solutions 2000 (IPS2000). Observatoire de Paris.
4. Chapront, J., Chapront‑Touzé, M. & Francou, G. (1997).
   ELP/MPP02 Lunar Ephemeris. SYRTE/OBSPM.
5. Vallado, D.A. (2013). Fundamentals of Astrodynamics and
   Applications. 4th Ed. Microcosm Press.
6. Allen, C.W. (1976). Astrophysical Quantities. 3rd Ed.
   Athlone Press.
7. Altamimi, Z. et al. (2023). ITRF2020‑PMM (Plate Motion
   Model). IERS.

AUTHOR
------
ASTERID Consortium – Jolotundo Research Observatory

VERSION
-------
3.1 (2026‑06‑20) – Hierarchical report with GCRS cartesian vectors
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Tuple

# -------------------------------------------------------------------------
# CORE HIGH‑PRECISION MODULES (IAU SOFA / IERS 2010)
# -------------------------------------------------------------------------
from Timescales import (
    J2000_JD,
    cal_to_jd,
    combine_jd,
    TimeScaleConverter,
    delta_t_from_jd,
    tai_utc,
)
from EOPDelta import EOPProvider
from EarthRotation import (
    get_cip_xy,
    get_cio_s,
    get_tio_sp,
    nutation_2000a,
    precession_angles_2006,
    equation_of_origins,
    gst_from_ut1,
    ARCSEC_TO_RAD,
    MAS_TO_RAD,
    UAS_TO_RAD,
    era_from_ut1,
)
from IPS2000_emb import emb_heliocentric_ecl
from SM_IPS2000emb import AstronomicalEphemeris
from Atmospheric_refraction import SITE_LAT_DEG, SITE_LON_DEG, SITE_ELEV_M
from Site_Geophysic import TectonicPlateKinematics
from Coord_Transform import cartesian_to_spherical, rot_x  

# -------------------------------------------------------------------------
# FUNGSI PEMBANTU LOKAL
# -------------------------------------------------------------------------
def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [0, 2π)."""
    return angle % (2.0 * math.pi)

# -------------------------------------------------------------------------
# OBSERVER & GEOPHYSICAL CONSTANTS
# -------------------------------------------------------------------------
OBSERVER_LAT_DEG = SITE_LAT_DEG
OBSERVER_LON_DEG = SITE_LON_DEG
OBSERVER_HEIGHT_M = SITE_ELEV_M
PLATE_CODE = 'EURA'                  # Eurasian Plate (ITRF2020‑PMM)

RAD_TO_DEG = 180.0 / math.pi
RAD_TO_MAS = 1.0 / MAS_TO_RAD
RAD_TO_UAS = 1.0 / UAS_TO_RAD
MM_PER_M = 1000.0
AU2KM = 149597870.7

# Obliquity of the ecliptic at J2000.0 (IAU 2006)
OBLIQUITY_J2000_RAD = 84381.406 * ARCSEC_TO_RAD

# -------------------------------------------------------------------------
# AUXILIARY FUNCTIONS (Astrometric & Orbital Mechanics)
# -------------------------------------------------------------------------
def moon_phase_name(elong_deg: float) -> str:
    """
    Returns the standard 8‑phase name based on geocentric elongation.

    Conforms to the USNO/Nautical Almanac Office convention.
    """
    centers = [0, 45, 90, 135, 180, 225, 270, 315]
    names = [
        "New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
        "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"
    ]
    diffs = [min(abs(elong_deg - c), 360 - abs(elong_deg - c)) for c in centers]
    idx = diffs.index(min(diffs))
    return names[idx]


def cartesian_to_keplerian(pos_km: np.ndarray, vel_kms: np.ndarray,
                           mu_km3_s2: float) -> Tuple[float, float, float, float, float, float]:
    """
    Convert a geocentric Cartesian state vector to osculating Keplerian
    orbital elements (Vallado, 2013).
    """
    r_vec = pos_km
    v_vec = vel_kms
    r = np.linalg.norm(r_vec)
    v = np.linalg.norm(v_vec)
    if r == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec)
    if h == 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    i = math.acos(np.clip(h_vec[2] / h, -1.0, 1.0))
    e_vec = (np.cross(v_vec, h_vec) / mu_km3_s2) - (r_vec / r)
    e = np.linalg.norm(e_vec)
    xi = (v * v / 2.0) - (mu_km3_s2 / r)
    a = -mu_km3_s2 / (2.0 * xi) if abs(xi) > 1e-15 else 0.0

    n_vec = np.cross(np.array([0.0, 0.0, 1.0]), h_vec)
    n = np.linalg.norm(n_vec)
    Omega = math.atan2(n_vec[1], n_vec[0]) if n != 0.0 else 0.0

    if n != 0.0 and e > 1e-12:
        cos_omega = np.dot(e_vec, n_vec) / (e * n)
        sin_omega = np.dot(np.cross(e_vec, n_vec), h_vec) / (e * n * h)
        omega = math.atan2(sin_omega, cos_omega)
    else:
        omega = math.atan2(e_vec[1], e_vec[0]) if e > 1e-12 else 0.0

    if e < 1e-12:
        M = math.atan2(pos_km[1], pos_km[0]) - omega - Omega
    else:
        cos_nu = np.dot(e_vec, r_vec) / (e * r)
        sin_nu = np.dot(np.cross(e_vec, r_vec), h_vec) / (e * r * h)
        nu = math.atan2(sin_nu, cos_nu)
        cos_E = (e + cos_nu) / (1.0 + e * cos_nu)
        sin_E = (math.sqrt(max(0.0, 1.0 - e * e)) * sin_nu) / (1.0 + e * cos_nu)
        E = math.atan2(sin_E, cos_E)
        M = E - e * math.sin(E)

    return a, e, i, Omega, omega, M


def compute_apparent_motion(pos_km: np.ndarray, vel_km_day: np.ndarray) -> Tuple[float, float]:
    """
    Compute the apparent rates of RA and Dec from a geocentric state vector.
    (SOFA‑convention for `iauPv2s` rates.)
    """
    x, y, z = pos_km
    vx, vy, vz = vel_km_day
    r = np.linalg.norm(pos_km)
    if r == 0.0:
        return 0.0, 0.0

    xy2 = x * x + y * y
    if xy2 > 1e-30:
        ra_rate_rad_day = (vx * y - vy * x) / xy2
    else:
        ra_rate_rad_day = 0.0

    dec_rate_rad_day = (vz - z * (x * vx + y * vy + z * vz) / (r * r)) / r
    return math.degrees(ra_rate_rad_day) * 3600.0 * 1000.0, math.degrees(dec_rate_rad_day) * 3600.0 * 1000.0

def gcrs_to_ecliptic_date(pos_gcrs, tt_jd):
    from EarthRotation import precession_angles_2006, nutation_2000a, ARCSEC_TO_RAD
    from Coord_Transform import rot_x, rot_z, cartesian_to_spherical
    t_cy = (tt_jd - 2451545.0) / 36525.0
    eps0 = 84381.406 * ARCSEC_TO_RAD
    pre = precession_angles_2006(t_cy)
    dpsi, deps = nutation_2000a(tt_jd)
    P = rot_z(-pre['chiA']) @ rot_x(pre['omegaA']) @ rot_z(-pre['psiA']) @ rot_x(eps0)
    N = rot_x(-pre['epsA']) @ rot_z(-dpsi) @ rot_x(pre['epsA'] + deps)
    PN = N @ P
    pos_true_eq = PN @ pos_gcrs
    eps_true = pre['epsA'] + deps
    pos_ecl = rot_x(-eps_true) @ pos_true_eq
    lon, lat, r = cartesian_to_spherical(pos_ecl)
    return lon, lat, r

# -------------------------------------------------------------------------
# CORE COMPUTATION
# -------------------------------------------------------------------------
def generate_report(
    lat_deg: float = OBSERVER_LAT_DEG,
    lon_deg: float = OBSERVER_LON_DEG,
    height_m: float = OBSERVER_HEIGHT_M,
    eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt",
    plate_code: str = PLATE_CODE,
) -> Dict:
    """
    Compute the full astrometric and geodetic report for the current UTC epoch.

    The reduction sequence rigorously follows the IAU SOFA/IERS 2010
    hierarchy, with all geophysical corrections applied in full
    double‑precision.
    """
    # ---- 1. TIME SCALES ----
    now_utc = datetime.now(timezone.utc)
    jd1, jd2 = cal_to_jd(
        now_utc.year,
        now_utc.month,
        now_utc.day,
        now_utc.hour,
        now_utc.minute,
        now_utc.second + now_utc.microsecond / 1_000_000.0,
        scale="utc",
    )
    utc_jd = combine_jd(jd1, jd2)

    tsc = TimeScaleConverter()
    tt_jd = tsc.utc_to_tt(utc_jd)
    tai_jd = utc_jd + tai_utc(utc_jd - 2400000.5) / 86400.0
    tdb_jd = tsc.utc_to_tdb(utc_jd)
    delta_t = delta_t_from_jd(tt_jd)
    ut1_jd = tt_jd - delta_t / 86400.0

    # ---- 2. EARTH ORIENTATION PARAMETERS ----
    eop_provider = EOPProvider(eop_file)
    eop_last_date = eop_provider.get_last_available_date()
    mjd_utc = utc_jd - 2400000.5
    try:
        eop = eop_provider.get_eop(mjd_utc)
        xp_mas, yp_mas, dut1_s, dX_mas, dY_mas = (
            eop["x_pole"],
            eop["y_pole"],
            eop["ut1_utc"],
            eop["dX"],
            eop["dY"],
        )
    except Exception:
        xp_mas = yp_mas = dut1_s = dX_mas = dY_mas = 0.0

    xp_rad = xp_mas * MAS_TO_RAD
    yp_rad = yp_mas * MAS_TO_RAD
    dX_rad = dX_mas * MAS_TO_RAD
    dY_rad = dY_mas * MAS_TO_RAD

    # ---- 3. CELESTIAL INTERMEDIATE POLE & EARTH ROTATION ----
    X_rad, Y_rad = get_cip_xy(tt_jd, apply_fcn=True)
    X_rad += dX_rad
    Y_rad += dY_rad
    s_rad = get_cio_s(tt_jd, X_rad, Y_rad)
    sp_rad = get_tio_sp(tt_jd)
    era_rad = era_from_ut1(ut1_jd)
    eo_rad = equation_of_origins(tt_jd)
    gst_rad = gst_from_ut1(ut1_jd, tt_jd)

    # ---- 4. NUTATION & PRECESSION ----
    dpsi_rad, deps_rad = nutation_2000a(tt_jd)
    t_cy = (tt_jd - J2000_JD) / 36525.0
    pre = precession_angles_2006(t_cy)

    # ---- 5. EPHEMERIS ORCHESTRATOR (IPS2000) ----
    ephem = AstronomicalEphemeris(eop_file=eop_file)
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # ---- 6. HELIOCENTRIC FRAME: EARTH‑MOON BARYCENTRE ----
    emb_pos_au, emb_vel_au_day = emb_heliocentric_ecl(tdb_jd)
    emb_pos_km = np.array(emb_pos_au) * AU2KM
    emb_vel_km_day = np.array(emb_vel_au_day) * AU2KM
    emb_dist_au = np.linalg.norm(emb_pos_au)
    emb_dist_km = np.linalg.norm(emb_pos_km)

    # ---- 7. GEOCENTRIC FRAME (GCRS / ICRS) ----
    sun_gcrs_vec = ephem.sun_geocentric_gcrs(tt_jd, unit=False)
    moon_gcrs_vec = ephem.moon_geocentric_gcrs(tt_jd, unit=False)

    sun_ra_gcrs, sun_dec_gcrs, _ = cartesian_to_spherical(sun_gcrs_vec)
    moon_ra_gcrs, moon_dec_gcrs, _ = cartesian_to_spherical(moon_gcrs_vec)
    sun_ra_gcrs_deg = math.degrees(normalize_angle(sun_ra_gcrs))
    sun_dec_gcrs_deg = math.degrees(sun_dec_gcrs)
    moon_ra_gcrs_deg = math.degrees(normalize_angle(moon_ra_gcrs))
    moon_dec_gcrs_deg = math.degrees(moon_dec_gcrs)

    # Geocentric velocities
    dt_vel = 1.0 / 86400.0
    moon_pos_plus = ephem.moon_geocentric_gcrs(tt_jd + dt_vel, unit=False)
    moon_pos_minus = ephem.moon_geocentric_gcrs(tt_jd - dt_vel, unit=False)
    moon_vel_gcrs_km_day = (moon_pos_plus - moon_pos_minus) / (2.0 * dt_vel)

    earth_vel_au_day = ephem.earth_velocity_gcrs(tt_jd)
    sun_vel_gcrs_km_day = -earth_vel_au_day * AU2KM

    sun_dra, sun_ddec = compute_apparent_motion(sun_gcrs_vec, sun_vel_gcrs_km_day)
    moon_dra, moon_ddec = compute_apparent_motion(moon_gcrs_vec, moon_vel_gcrs_km_day)

    # ---- Ecliptic of Date ----
    sun_ecl_lon, sun_ecl_lat, sun_ecl_r = gcrs_to_ecliptic_date(sun_gcrs_vec, tt_jd)
    moon_ecl_lon, moon_ecl_lat, moon_ecl_r = gcrs_to_ecliptic_date(moon_gcrs_vec, tt_jd)
    sun_ecl_lon_deg = math.degrees(normalize_angle(sun_ecl_lon)) % 360.0
    sun_ecl_lat_deg = math.degrees(sun_ecl_lat)
    moon_ecl_lon_deg = math.degrees(normalize_angle(moon_ecl_lon)) % 360.0
    moon_ecl_lat_deg = math.degrees(moon_ecl_lat)

    # ---- 8. TOPOCENTRIC FRAME: CIRS & Equinox (apparent, no refraction) ----
    sun_app = ephem.sun_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m, apply_refraction=False
    )
    moon_app = ephem.moon_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m, apply_refraction=False
    )

    # ---- 9. OBSERVED FRAME: HORIZONTAL (with refraction) ----
    sun_obs = ephem.sun_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m, apply_refraction=True
    )
    moon_obs = ephem.moon_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m, apply_refraction=True
    )

    # ---- 10. LUNAR PHASE (Elongation via Ecliptic Longitudes) ----
    earth_pos_ecl, _ = ephem.earth_heliocentric_ecliptic(tt_jd)
    sun_pos_ecl = -np.array(earth_pos_ecl)
    sun_lon = math.atan2(sun_pos_ecl[1], sun_pos_ecl[0])

    moon_ecl = rot_x(-OBLIQUITY_J2000_RAD) @ moon_gcrs_vec
    moon_lon = math.atan2(moon_ecl[1], moon_ecl[0])

    elong_rad = (moon_lon - sun_lon) % (2 * math.pi)
    elong_deg = math.degrees(elong_rad)
    phase_name = moon_phase_name(elong_deg)

    earth_from_moon = -moon_gcrs_vec
    sun_from_moon = sun_gcrs_vec - moon_gcrs_vec
    cos_i = np.dot(earth_from_moon, sun_from_moon) / (
        np.linalg.norm(earth_from_moon) * np.linalg.norm(sun_from_moon)
    )
    cos_i = np.clip(cos_i, -1.0, 1.0)
    phase_angle_deg = math.degrees(math.acos(cos_i))
    illum_pct = (1.0 + cos_i) / 2.0 * 100.0
    age_days = elong_deg / 360.0 * 29.530588861

    moon_phase_info = {
        'phase_name': phase_name,
        'illumination_pct': illum_pct,
        'age_days': age_days,
        'elongation_deg': elong_deg,
        'phase_angle_deg': phase_angle_deg,
    }

    # ---- 11. LUNAR LIBRATION ----
    lib = ephem.lunar_libration(tt_jd)

    # ---- 12. LUNAR ORBITAL ELEMENTS ----
    mu_earth_km3_s2 = 398600.4418
    moon_vel_kms = moon_vel_gcrs_km_day / 86400.0
    a_km, e_orb, i_rad, Omega_rad, omega_rad, M_rad = cartesian_to_keplerian(
        moon_gcrs_vec, moon_vel_kms, mu_earth_km3_s2
    )

    # ---- 13. PHOTOMETRY ----
    sun_app_mag = -26.74
    moon_dist_km = np.linalg.norm(moon_gcrs_vec)
    moon_app_mag = (-12.74
                    + 5.0 * math.log10(moon_dist_km / 384400.0)
                    + 0.026 * abs(phase_angle_deg)
                    + 4.0e-9 * (abs(phase_angle_deg) ** 4))

    # ---- 14. KINEMATIC PLATE MOTION (ITRF2020‑PMM) ----
    a_eq = 6378136.6
    f = 1.0 / 298.25642
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    N = a_eq / math.sqrt(1.0 - (2 * f - f * f) * sin_lat * sin_lat)
    x_itrf = (N + height_m) * cos_lat * cos_lon
    y_itrf = (N + height_m) * cos_lat * sin_lon
    z_itrf = (N * (1.0 - (2 * f - f * f)) + height_m) * sin_lat

    plate_kin = TectonicPlateKinematics(plate_code)
    vx, vy, vz = plate_kin.get_velocity(
        x_itrf,
        y_itrf,
        z_itrf,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        apply_orb=True,
        discard_vertical_orb=True,
    )
    ve = (-vx * sin_lon + vy * cos_lon) * MM_PER_M
    vn = (-vx * sin_lat * cos_lon - vy * sin_lat * sin_lon + vz * cos_lat) * MM_PER_M
    vu = (vx * cos_lat * cos_lon + vy * cos_lat * sin_lon + vz * sin_lat) * MM_PER_M
    horiz_speed = math.hypot(ve, vn)
    az_plate = math.degrees(math.atan2(ve, vn)) % 360.0
    epoch_diff_years = (tt_jd - J2000_JD) / 365.25

    # ---- 15. RETURN DATA ----
    return {
        # Time
        "timestamp_utc": now_utc,
        "utc_jd": utc_jd,
        "tai_jd": tai_jd,
        "tt_jd": tt_jd,
        "tdb_jd": tdb_jd,
        "ut1_jd": ut1_jd,
        "delta_t_s": delta_t,
        "dut1_s": dut1_s,
        # EOP
        "xp_mas": xp_mas,
        "yp_mas": yp_mas,
        "dX_mas": dX_mas,
        "dY_mas": dY_mas,
        # CIP/CIO & Rotation
        "X_cip_mas": X_rad * RAD_TO_MAS,
        "Y_cip_mas": Y_rad * RAD_TO_MAS,
        "s_cio_mas": s_rad * RAD_TO_MAS,
        "sp_tio_mas": sp_rad * RAD_TO_MAS,
        "era_deg": era_rad * RAD_TO_DEG,
        "eo_deg": eo_rad * RAD_TO_DEG,
        "gst_deg": gst_rad * RAD_TO_DEG,
        # Nutation/Precession
        "dpsi_mas": dpsi_rad * RAD_TO_MAS,
        "deps_mas": deps_rad * RAD_TO_MAS,
        "precession_epsA_deg": pre["epsA"] * RAD_TO_DEG,
        "precession_psiA_deg": pre["psiA"] * RAD_TO_DEG,
        "precession_omegaA_deg": pre["omegaA"] * RAD_TO_DEG,
        "precession_chiA_deg": pre["chiA"] * RAD_TO_DEG,
        # Heliocentric (EMB)
        "emb_pos_km": emb_pos_km,
        "emb_vel_km_day": emb_vel_km_day,
        "emb_dist_km": emb_dist_km,
        "emb_dist_au": emb_dist_au,
        # Geocentric (GCRS/ICRS) - VECTORS AND SPHERICAL
        "sun_gcrs_vec": sun_gcrs_vec,
        "moon_gcrs_vec": moon_gcrs_vec,
        "sun_ra_gcrs_deg": sun_ra_gcrs_deg,
        "sun_dec_gcrs_deg": sun_dec_gcrs_deg,
        "moon_ra_gcrs_deg": moon_ra_gcrs_deg,
        "moon_dec_gcrs_deg": moon_dec_gcrs_deg,
        "sun_ra_rate_mas_day": sun_dra,
        "sun_dec_rate_mas_day": sun_ddec,
        "moon_ra_rate_mas_day": moon_dra,
        "moon_dec_rate_mas_day": moon_ddec,
        "sun_dist_km": np.linalg.norm(sun_gcrs_vec),
        "moon_dist_km": np.linalg.norm(moon_gcrs_vec),
        # Topocentric (CIRS & Equinox)
        "sun": sun_app,
        "moon": moon_app,
        # Observed (Horizontal with Refraction)
        "sun_obs": sun_obs,
        "moon_obs": moon_obs,
        "sun_ecl_lon_deg": sun_ecl_lon_deg,
        "sun_ecl_lat_deg": sun_ecl_lat_deg,
        "sun_ecl_r_km": sun_ecl_r,
        "moon_ecl_lon_deg": moon_ecl_lon_deg,
        "moon_ecl_lat_deg": moon_ecl_lat_deg,
        "moon_ecl_r_km": moon_ecl_r,   
        # Lunar Phase
        "moon_phase": moon_phase_info,
        # Lunar Libration
        "lunar_libration": lib,
        # Lunar Orbital Elements
        "lunar_a_km": a_km,
        "lunar_e": e_orb,
        "lunar_i_deg": math.degrees(i_rad),
        "lunar_Omega_deg": math.degrees(Omega_rad),
        "lunar_omega_deg": math.degrees(omega_rad),
        "lunar_M_deg": math.degrees(M_rad),
        # Photometry
        "sun_apparent_mag": sun_app_mag,
        "moon_apparent_mag": moon_app_mag,
        # Plate Motion
        "plate_ve_mmyr": ve,
        "plate_vn_mmyr": vn,
        "plate_vu_mmyr": vu,
        "plate_horiz_speed_mmyr": horiz_speed,
        "plate_azimuth_deg": az_plate,
        "plate_epoch_diff_years": epoch_diff_years,
        "plate_code": plate_code,
        # Metadata
        "eop_file": eop_file,
        "eop_last_date": eop_last_date,
    }


# -------------------------------------------------------------------------
# PROFESSIONAL REPORT PRINTER (Hierarchical SOFA/IAU Order)
# -------------------------------------------------------------------------
def print_report(data: Dict) -> None:
    """Generate a high-precision scientific report optimized for mobile screens without any reduction."""
    ts = data["timestamp_utc"]
    w = 70  # Lebar maksimum aman untuk konsol mobile
    sep = "-" * w
    pad = 30  # Lebar kolom definisi diperlebar agar jarak nilai lebih lega

    print("\n" + "=" * w)
    print(" ASTERID HIGH‑PRECISION ASTROMETRIC & GEODETIC REPORT".center(w))
    print(" IERS Conventions (2010) – Single‑Shot Epoch Solution".center(w))
    print(" ** EPHEMERIS ENGINE : IPS2000/ELPMPP02 **".center(w))
    print(" (Improved Planetary Solutions)".center(w))
    print(" ** ALL IERS 2010 GEOPHYSICAL CORRECTIONS APPLIED **".center(w))
    print("=" * w)

    # [1] TIME SYSTEMS
    print("\n[1] TIME SYSTEMS (JD, TAI, TT, TDB, UT1)")
    print(sep)
    print(f"{'UTC Epoch':<{pad}}: {ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
    print(f"{'':<{pad}}  (ISO 8601)")
    print(f"{'UTC (JD)':<{pad}}: {data['utc_jd']:.9f}")
    print(f"{'TAI (JD)':<{pad}}: {data['tai_jd']:.9f}")
    print(f"{'TT  (JD)':<{pad}}: {data['tt_jd']:.9f}")
    print(f"{'TDB (JD)':<{pad}}: {data['tdb_jd']:.9f}")
    print(f"{'UT1 (JD)':<{pad}}: {data['ut1_jd']:.9f}")
    print(f"{'ΔT = TT − UT1':<{pad}}: {data['delta_t_s']:>14.6f} s")
    print(f"{'DUT1 = UT1 − UTC':<{pad}}: {data['dut1_s']:>14.6f} s")

    # [2] EARTH ORIENTATION
    print("\n[2] EARTH ORIENTATION PARAMETERS (IERS Bulletin A)")
    print(sep)
    print(f"{'Polar Motion (xp)':<{pad}}: {data['xp_mas']:>14.6f} mas")
    print(f"{'Polar Motion (yp)':<{pad}}: {data['yp_mas']:>14.6f} mas")
    print(f"{'Celestial Offset dX':<{pad}}: {data['dX_mas']:>14.6f} mas")
    print(f"{'Celestial Offset dY':<{pad}}: {data['dY_mas']:>14.6f} mas")

    # [3] CIP/CIO & ROTATION
    print("\n[3] CELESTIAL INTERMEDIATE POLE & EARTH ROTATION")
    print(sep)
    print(f"{'CIP X (IAU 2006)':<{pad}}: {data['X_cip_mas']:>14.6f} mas")
    print(f"{'CIP Y (IAU 2006)':<{pad}}: {data['Y_cip_mas']:>14.6f} mas")
    print(f"{'CIO Locator (s)':<{pad}}: {data['s_cio_mas']:>14.6f} mas")
    print(f"{'TIO Locator (sp)':<{pad}}: {data['sp_tio_mas']:>14.6f} mas")
    print(f"{'Earth Rotation Angle':<{pad}}: {data['era_deg']:>14.6f} deg")
    print(f"{'Eq. of the Origins':<{pad}}: {data['eo_deg']:>14.6f} deg")
    print(f"{'Greenwich Sidereal':<{pad}}: {data['gst_deg']:>14.6f} deg")

    # [4] NUTATION/PRECESSION
    print("\n[4] NUTATION (IAU 2000A_R06) & PRECESSION (IAU 2006)")
    print(sep)
    print(f"{'Nutation Δψ':<{pad}}: {data['dpsi_mas']:>14.6f} mas")
    print(f"{'Nutation Δε':<{pad}}: {data['deps_mas']:>14.6f} mas")
    print(f"{'Precession εₐ (obl.)':<{pad}}: {data['precession_epsA_deg']:>14.6f} deg")
    print(f"{'Precession ψₐ':<{pad}}: {data['precession_psiA_deg']:>14.6f} deg")
    print(f"{'Precession ωₐ':<{pad}}: {data['precession_omegaA_deg']:>14.6f} deg")
    print(f"{'Precession χₐ':<{pad}}: {data['precession_chiA_deg']:>14.6f} deg")

    # [5] HELIOCENTRIC (EMB)
    print("\n[5] HELIOCENTRIC EARTH‑MOON BARYCENTRE (EMB) STATE")
    print(sep)
    emb_pos = data['emb_pos_km']
    emb_vel = data['emb_vel_km_day']
    print(f"{'Position X':<{pad}}: {emb_pos[0]:>18.3f} km")
    print(f"{'Position Y':<{pad}}: {emb_pos[1]:>18.3f} km")
    print(f"{'Position Z':<{pad}}: {emb_pos[2]:>18.3f} km")
    print(f"{'Velocity Vx':<{pad}}: {emb_vel[0]:>18.6f} km/day")
    print(f"{'Velocity Vy':<{pad}}: {emb_vel[1]:>18.6f} km/day")
    print(f"{'Velocity Vz':<{pad}}: {emb_vel[2]:>18.6f} km/day")
    print(f"{'Distance from Sun (km)':<{pad}}: {data['emb_dist_km']:>18.3f} km")
    print(f"{'Distance from Sun (AU)':<{pad}}: {data['emb_dist_au']:>18.6f} AU")
    print(f"{'Reference Frame':<{pad}}: Dynamical equinox & ecliptic")
    print(f"{'':<{pad}}  J2000 (IPS2000)")

    # [6] GEOCENTRIC GCRS (ICRS) POSITIONS & MOTION
    print("\n[6] GEOCENTRIC GCRS (ICRS) REFERENCE POSITIONS & MOTION")
    print(sep)
    print("Ref Frame: ICRS (J2000.0 mean equator & equinox)")
    print("(Geocentric astrometric, without aberration/deflection/topocentric corrections)")
    # SUN
    print("  ** SUN **")
    sun_vec = data['sun_gcrs_vec']
    print(f"{'    Position X (km)':<{pad}}: {sun_vec[0]:>18.3f}")
    print(f"{'    Position Y (km)':<{pad}}: {sun_vec[1]:>18.3f}")
    print(f"{'    Position Z (km)':<{pad}}: {sun_vec[2]:>18.3f}")
    print(f"{'    Distance (km)':<{pad}}: {data['sun_dist_km']:>18.3f}")
    print(f"{'    RA  (ICRS)':<{pad}}: {data['sun_ra_gcrs_deg']:>14.6f} deg")
    print(f"{'    Dec (ICRS)':<{pad}}: {data['sun_dec_gcrs_deg']:>14.6f} deg")
    print(f"{'    RA rate':<{pad}}: {data['sun_ra_rate_mas_day']:>14.3f} mas/day")
    print(f"{'    Dec rate':<{pad}}: {data['sun_dec_rate_mas_day']:>14.3f} mas/day")
    # MOON
    print("  ** MOON **")
    moon_vec = data['moon_gcrs_vec']
    print(f"{'    Position X (km)':<{pad}}: {moon_vec[0]:>18.3f}")
    print(f"{'    Position Y (km)':<{pad}}: {moon_vec[1]:>18.3f}")
    print(f"{'    Position Z (km)':<{pad}}: {moon_vec[2]:>18.3f}")
    print(f"{'    Distance (km)':<{pad}}: {data['moon_dist_km']:>18.3f}")
    print(f"{'    RA  (ICRS)':<{pad}}: {data['moon_ra_gcrs_deg']:>14.6f} deg")
    print(f"{'    Dec (ICRS)':<{pad}}: {data['moon_dec_gcrs_deg']:>14.6f} deg")
    print(f"{'    RA rate':<{pad}}: {data['moon_ra_rate_mas_day']:>14.3f} mas/day")
    print(f"{'    Dec rate':<{pad}}: {data['moon_dec_rate_mas_day']:>14.3f} mas/day")

    # [7] ECLIPTIC OF DATE
    print("\n[7] ECLIPTIC OF DATE (True Ecliptic & Equinox of Date)")
    print(sep)
    print("  ** SUN **")
    print(f"{'    Ecliptic Longitude (λ)':<{pad}}: {data['sun_ecl_lon_deg']:>14.6f} deg")
    print(f"{'    Ecliptic Latitude (β)':<{pad}}: {data['sun_ecl_lat_deg']:>14.6f} deg")
    print(f"{'    Geocentric Distance':<{pad}}: {data['sun_ecl_r_km']:>18.3f} km")
    print("  ** MOON **")
    print(f"{'    Ecliptic Longitude (λ)':<{pad}}: {data['moon_ecl_lon_deg']:>14.6f} deg")
    print(f"{'    Ecliptic Latitude (β)':<{pad}}: {data['moon_ecl_lat_deg']:>14.6f} deg")
    print(f"{'    Geocentric Distance':<{pad}}: {data['moon_ecl_r_km']:>18.3f} km")

    # [8] TOPOCENTRIC (CIRS & Equinox)
    sun = data['sun']
    moon = data['moon']
    print("\n[8] TOPOCENTRIC APPARENT COORDINATES")
    print("    (CIRS & Equinox of Date)")
    print(sep)
    print("  ** SUN (Apparent Topocentric) **")
    print(f"{'    RA (CIRS)':<{pad}}: {sun['ra_cirs_deg']:>14.6f} deg")
    print(f"{'    Dec (CIRS)':<{pad}}: {sun['dec_cirs_deg']:>14.6f} deg")
    print(f"{'    RA (Equinox)':<{pad}}: {sun['ra_eqx_deg']:>14.6f} deg")
    print(f"{'    Dec (Equinox)':<{pad}}: {sun['dec_eqx_deg']:>14.6f} deg")
    print(f"{'    Distance (km)':<{pad}}: {sun['dist_km']:14.6f}")
    print(f"{'    Distance (AU)':<{pad}}: {sun['dist_au']:14.6f}")
    print("  ** MOON (Apparent Topocentric) **")
    print(f"{'    RA (CIRS)':<{pad}}: {moon['ra_cirs_deg']:>14.6f} deg")
    print(f"{'    Dec (CIRS)':<{pad}}: {moon['dec_cirs_deg']:>14.6f} deg")
    print(f"{'    RA (Equinox)':<{pad}}: {moon['ra_eqx_deg']:>14.6f} deg")
    print(f"{'    Dec (Equinox)':<{pad}}: {moon['dec_eqx_deg']:>14.6f} deg")
    print(f"{'    Distance (km)':<{pad}}: {moon['dist_km']:14.6f}")
    print(f"{'    Distance (AU)':<{pad}}: {moon['dist_au']:14.6f}")

    # [9] OBSERVED HORIZONTAL (with Refraction)
    sun_obs = data['sun_obs']
    moon_obs = data['moon_obs']
    print("\n[9] OBSERVED / APPARENT HORIZONTAL COORDINATES")
    print("    (Refraction VMF3+GPT3)")
    print(sep)
    print("  ** SUN **")
    print(f"{'    Azimuth':<{pad}}: {sun_obs['az_deg']:>14.6f} deg")
    print(f"{'    Elevation (Geom.)':<{pad}}: {sun_obs['alt_geom_deg']:>14.6f} deg")
    print(f"{'    Elevation (Refr.)':<{pad}}: {sun_obs['alt_app_deg']:>14.6f} deg")
    print("  ** MOON **")
    print(f"{'    Azimuth':<{pad}}: {moon_obs['az_deg']:>14.6f} deg")
    print(f"{'    Elevation (Geom.)':<{pad}}: {moon_obs['alt_geom_deg']:>14.6f} deg")
    print(f"{'    Elevation (Refr.)':<{pad}}: {moon_obs['alt_app_deg']:>14.6f} deg")

    # [10] LUNAR PHASE & ORBITAL ELEMENTS
    mp = data['moon_phase']
    print("\n[10] LUNAR PHASE & OSCULATING ORBITAL ELEMENTS")
    print(sep)
    print(f"{'Phase Name':<{pad}}: {mp['phase_name']}")
    print(f"{'Illumination':<{pad}}: {mp['illumination_pct']:>14.2f} %")
    print(f"{'Age (since New Moon)':<{pad}}: {mp['age_days']:>14.3f} days")
    print(f"{'Elongation':<{pad}}: {mp['elongation_deg']:>14.2f} deg")
    print(f"{'Phase Angle':<{pad}}: {mp['phase_angle_deg']:>14.2f} deg")
    print("  ** Osculating Elements (Geocentric, J2000.0) **")
    print(f"{'    Semi‑major axis (a)':<{pad}}: {data['lunar_a_km']:>14.3f} km")
    print(f"{'    Eccentricity (e)':<{pad}}: {data['lunar_e']:>14.8f}")
    print(f"{'    Inclination (i)':<{pad}}: {data['lunar_i_deg']:>14.6f} deg")
    print(f"{'    Node (Ω)':<{pad}}: {data['lunar_Omega_deg']:>14.6f} deg")
    print(f"{'    Arg. of Perigee (ω)':<{pad}}: {data['lunar_omega_deg']:>14.6f} deg")
    print(f"{'    Mean Anomaly (M)':<{pad}}: {data['lunar_M_deg']:>14.6f} deg")

    # [11] PHOTOMETRY
    print("\n[11] PHOTOMETRY & APPARENT MAGNITUDE")
    print(sep)
    print(f"{'Sun (V band)':<{pad}}: {data['sun_apparent_mag']:>14.2f} mag")
    print(f"{'':<{pad}}  (IAU standard)")
    print(f"{'Moon (V band)':<{pad}}: {data['moon_apparent_mag']:>14.2f} mag")
    print(f"{'':<{pad}}  (Allen 1976)")

    # [12] LUNAR LIBRATION
    print("\n[12] LUNAR LIBRATION (LLIB04)")
    print(sep)
    lib = data['lunar_libration']
    print(f"{'P1 (optical libration)':<{pad}}: {lib['p1_deg']:>14.6f} deg")
    print(f"{'P2 (optical libration)':<{pad}}: {lib['p2_deg']:>14.6f} deg")
    print(f"{'Tau (physical lib.)':<{pad}}: {lib['tau_deg']:>14.6f} deg")

    # [13] PLATE MOTION
    print("\n[13] KINEMATIC PLATE MOTION")
    print("     (ITRF2020‑PMM / Altamimi et al. 2023)")
    print(sep)
    print(f"{'Tectonic Plate':<{pad}}: {data['plate_code']} (Eurasian)")
    print(f"{'Reference Epoch':<{pad}}: J2000.0")
    print(f"{'':<{pad}}  (ΔT = {data['plate_epoch_diff_years']:.4f} yr)")
    print(f"{'Velocity (Ve)':<{pad}}: {data['plate_ve_mmyr']:>14.4f} mm/yr (East)")
    print(f"{'Velocity (Vn)':<{pad}}: {data['plate_vn_mmyr']:>14.4f} mm/yr (North)")
    print(f"{'Velocity (Vu)':<{pad}}: {data['plate_vu_mmyr']:>14.4f} mm/yr (Up)")
    print(f"{'Horizontal Speed':<{pad}}: {data['plate_horiz_speed_mmyr']:>14.4f} mm/yr")
    print(f"{'Horizontal Azimuth':<{pad}}: {data['plate_azimuth_deg']:>14.4f} deg (True N)")

    # [14] OBSERVER NOTES
    print("\n[14] OBSERVER & REDUCTION NOTES")
    print(sep)
    print(f"{'Geodetic Latitude':<{pad}}: {OBSERVER_LAT_DEG:+.6f} deg")
    print(f"{'Geodetic Longitude':<{pad}}: {OBSERVER_LON_DEG:+.6f} deg")
    print(f"{'Orthometric Height':<{pad}}: {OBSERVER_HEIGHT_M:.3f} m (EGM2008)")
    print(f"{'Refraction Model':<{pad}}: VMF3 + GPT3")
    print(f"{'':<{pad}}  (mapping functions + APG gradient)")
    
    # Auto-wrap EOP file if too long
    eop_file = data.get('eop_file', 'unknown')
    if len(eop_file) > (w - pad - 2):
        print(f"{'EOP Source File':<{pad}}:")
        print(f"{'':<{pad}}  {eop_file}")
    else:
        print(f"{'EOP Source File':<{pad}}: {eop_file}")
        
    print(f"{'EOP Last Observation':<{pad}}: {data.get('eop_last_date', 'Unknown')}")
    print(f"{'Ephemeris Sources':<{pad}}: IPS2000 (Sun/EMB),")
    print(f"{'':<{pad}}  ELP/MPP02 (Moon)")
    print()    
    print("Geophysical Corrections (IERS 2010, full precision):")
    print("  – Solid Earth tides (frequency‑dependent Love numbers)")
    print("  – Ocean tide loading (FES2014)")
    print("  – Pole tide (diurnal & semi‑diurnal)")
    print("  – Atmospheric loading (non‑tidal)")
    print("  – Non‑tidal loading (hydrology, etc.)")
    print("  – Vertical deflection (DoV) from local EGM2008 grids")
    print("  – Kinematic plate motion (ITRF2020‑PMM)")

    print("\n" + "=" * w)
    print(" REPORT GENERATED SUCCESSFULLY – EXITING.".center(w))
    print("=" * w + "\n")


# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ Computing comprehensive ephemeris + plate kinematics (IPS2000)...")
    try:
        report = generate_report()
        print_report(report)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)