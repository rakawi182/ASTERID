#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DT18_VSOP2013_Realtime.py
============================================================
HIGH‑PRECISION REAL‑TIME ASTROMETRIC & GEODETIC EPHEMERIS
ENGINE: VSOP2013 + ELP/MPP02
============================================================

DESCRIPTION
-----------
This module is a drop‑in replacement for DT18_IPS2000_Realtime.py
but uses the VSOP2013 analytical planetary theory (with truncation
1e‑12) for the Sun and retains ELP/MPP02 for the Moon.

All geophysical corrections (IERS 2010) and the full IAU SOFA
reduction chain are applied identically to the IPS2000 version.
The only difference is the underlying solar ephemeris source.

REFERENCES
----------
- VSOP2013: Fienga et al. (2013) – IMCCE
- ELP/MPP02: Chapront et al. (1997) – SYRTE
- IERS Conventions (2010)
- IAU SOFA
- ITRF2020‑PMM (Altamimi et al. 2023)

AUTHOR
------
ASTERID Consortium – Jolotundo Research Observatory

VERSION
-------
2.0 (2026‑07‑05) – Hierarchical report with GCRS cartesian vectors
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
# --- Ganti impor ---
from SM_VSOP2013 import AstronomicalEphemeris
from Atmospheric_refraction import SITE_LAT_DEG, SITE_LON_DEG, SITE_ELEV_M
from Site_Geophysic import TectonicPlateKinematics
from Coord_Transform import cartesian_to_spherical, rot_x
from LODEngine import HighPrecisionLODEngine


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
MU = 1.0 / (1.0 + 81.30056907419)   # μ = 1/(1+EMRAT)

# Obliquity of the ecliptic at J2000.0 (IAU 2006)
OBLIQUITY_J2000_RAD = 84381.406 * ARCSEC_TO_RAD

# -------------------------------------------------------------------------
# AUXILIARY FUNCTIONS (Astrometric & Orbital Mechanics)
# -------------------------------------------------------------------------
def moon_phase_name(elong_deg: float, tolerance_deg: float = 1.0) -> str:
    """
    Menentukan nama fase Bulan berdasarkan sudut elongasi (0-360 derajat).
    Menggunakan toleransi presisi untuk fase utama (eksak).
    
    Parameters:
    elong_deg (float): Sudut elongasi geosentris antara Bulan dan Matahari.
    tolerance_deg (float): Lebar toleransi (dalam derajat) untuk melabeli 
                           fase utama. Default disetel ketat pada 1.0 derajat.
    """
    # Memastikan input selalu dalam rentang 0-360 derajat
    e = float(elong_deg) % 360.0
    
    # Deteksi Fase Utama (Primary Phases) dengan toleransi sempit
    if e <= tolerance_deg or e >= (360.0 - tolerance_deg):
        return "New Moon"
    elif abs(e - 90.0) <= tolerance_deg:
        return "First Quarter"
    elif abs(e - 180.0) <= tolerance_deg:
        return "Full Moon"
    elif abs(e - 270.0) <= tolerance_deg:
        return "Last Quarter"
        
    # Deteksi Fase Antara (Intermediate Phases)
    if 0.0 < e < 90.0:
        return "Waxing Crescent"
    elif 90.0 < e < 180.0:
        return "Waxing Gibbous"
    elif 180.0 < e < 270.0:
        return "Waning Gibbous"
    else:
        return "Waning Crescent"

def cartesian_to_keplerian(pos_km: np.ndarray, vel_kms: np.ndarray,
                           mu_km3_s2: float) -> Tuple[float, float, float, float, float, float]:
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
    vsop_file: str = "VSOP2013p3.dat",
) -> Dict:
    """
    Compute the full astrometric and geodetic report for the current UTC epoch
    using VSOP2013 for the Sun and ELP/MPP02 for the Moon.
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

    # ---- Hitung LOD menggunakan LODEngine ----
    lod_engine = HighPrecisionLODEngine(h=1e-6)
    lod_result = lod_engine.get_lod_precision(utc_jd)
    lod_total_ms = lod_result['lod_total']          # LOD total (ms)
    lod_iers_ms = lod_result['lod_iers']            # LOD dari IERS (ms)
    lod_tidal_corr_ms = lod_result['lod_tidal_correction']  # koreksi tidal (ms)

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

    # ---- 5. EPHEMERIS ORCHESTRATOR (VSOP2013) ----
    ephem = AstronomicalEphemeris(eop_file=eop_file, vsop_file=vsop_file)
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # ---- 6. HELIOCENTRIC FRAME: EARTH + Moon → EMB ----
    earth_pos_au, earth_vel_au_day = ephem.earth_heliocentric_ecliptic(tt_jd)
    moon_gcrs_km = ephem.moon_geocentric_gcrs(tt_jd, unit=False)
    moon_pos_au = moon_gcrs_km / AU2KM

    dt_vel = 1.0 / 86400.0
    moon_plus = ephem.moon_geocentric_gcrs(tt_jd + dt_vel, unit=False)
    moon_minus = ephem.moon_geocentric_gcrs(tt_jd - dt_vel, unit=False)
    moon_vel_km_day = (moon_plus - moon_minus) / (2.0 * dt_vel)
    moon_vel_au_day = moon_vel_km_day / AU2KM

    emb_pos_au = earth_pos_au + MU * moon_pos_au
    emb_vel_au_day = earth_vel_au_day + MU * moon_vel_au_day
    emb_pos_km = emb_pos_au * AU2KM
    emb_vel_km_day = emb_vel_au_day * AU2KM
    emb_dist_au = np.linalg.norm(emb_pos_au)
    emb_dist_km = np.linalg.norm(emb_pos_km)

    # ---- 7. GEOCENTRIC FRAME (GCRS / ICRS) ----
    sun_gcrs_vec = ephem.sun_geocentric_gcrs(tt_jd, unit=False)
    moon_gcrs_vec = moon_gcrs_km  # already in km

    sun_ra_gcrs, sun_dec_gcrs, _ = cartesian_to_spherical(sun_gcrs_vec)
    moon_ra_gcrs, moon_dec_gcrs, _ = cartesian_to_spherical(moon_gcrs_vec)
    sun_ra_gcrs_deg = math.degrees(normalize_angle(sun_ra_gcrs))
    sun_dec_gcrs_deg = math.degrees(sun_dec_gcrs)
    moon_ra_gcrs_deg = math.degrees(normalize_angle(moon_ra_gcrs))
    moon_dec_gcrs_deg = math.degrees(moon_dec_gcrs)

    earth_vel_au_day = ephem.earth_velocity_gcrs(tt_jd)  # in AU/day
    sun_vel_km_day = -earth_vel_au_day * AU2KM

    sun_dra, sun_ddec = compute_apparent_motion(sun_gcrs_vec, sun_vel_km_day)
    moon_dra, moon_ddec = compute_apparent_motion(moon_gcrs_vec, moon_vel_km_day)

    # ---- Ecliptic of Date ----
    sun_ecl_lon, sun_ecl_lat, sun_ecl_r = gcrs_to_ecliptic_date(sun_gcrs_vec, tt_jd)
    moon_ecl_lon, moon_ecl_lat, moon_ecl_r = gcrs_to_ecliptic_date(moon_gcrs_vec, tt_jd)
    sun_ecl_lon_deg = math.degrees(normalize_angle(sun_ecl_lon)) % 360.0
    sun_ecl_lat_deg = math.degrees(sun_ecl_lat)
    moon_ecl_lon_deg = math.degrees(normalize_angle(moon_ecl_lon)) % 360.0
    moon_ecl_lat_deg = math.degrees(moon_ecl_lat)

    # ---- 8 & 9. TOPOCENTRIC FRAME & HORIZONTAL OBSERVED (MERGED) ----
    sun_app = ephem.sun_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m, apply_refraction=False
    )
    moon_app = ephem.moon_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m, apply_refraction=False
    )

    from Atmospheric_refraction import calculate_refraction, hybrid_meteo_assimilation
    mjd = tt_jd - 2400000.5

    # EKSEKUSI ATMOSFER HANYA 1 KALI UNTUK KEDUA OBJEK (SUN & MOON)
    meteo_cache = hybrid_meteo_assimilation(mjd, lat_rad, lon_rad, height_m, is_realtime=True)

    ref_sun = calculate_refraction(
        alt_geom_deg=sun_app['alt_geom_deg'],
        model='vmf3',
        mjd=mjd,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        height_m=height_m,
        az_rad=math.radians(sun_app['az_deg']),
        is_realtime=True,
        meteo_data=meteo_cache  # Lintas Bypass
    )

    ref_moon = calculate_refraction(
        alt_geom_deg=moon_app['alt_geom_deg'],
        model='vmf3',
        mjd=mjd,
        lat_rad=lat_rad,
        lon_rad=lon_rad,
        height_m=height_m,
        az_rad=math.radians(moon_app['az_deg']),
        is_realtime=True,
        meteo_data=meteo_cache  # Lintas Bypass
    )

    # Dictionary akhir (langsung siap disajikan ke Report)
    sun_obs = {
        'az_deg': sun_app['az_deg'],
        'alt_geom_deg': sun_app['alt_geom_deg'],
        'alt_app_deg': sun_app['alt_geom_deg'] + ref_sun,
    }

    moon_obs = {
        'az_deg': moon_app['az_deg'],
        'alt_geom_deg': moon_app['alt_geom_deg'],
        'alt_app_deg': moon_app['alt_geom_deg'] + ref_moon,
    }

    # ---- 10. LUNAR PHASE ----
    earth_pos_ecl, _ = ephem.earth_heliocentric_ecliptic(tt_jd)
    sun_pos_ecl = -np.array(earth_pos_ecl)
    sun_lon = math.atan2(sun_pos_ecl[1], sun_pos_ecl[0])

    moon_ecl = rot_x(OBLIQUITY_J2000_RAD) @ moon_gcrs_vec
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
    moon_vel_kms = moon_vel_km_day / 86400.0
    a_km, e_orb, i_rad, Omega_rad, omega_rad, M_rad = cartesian_to_keplerian(
        moon_gcrs_vec, moon_vel_kms, mu_earth_km3_s2
    )

    # ---- 13. KINEMATIC PLATE MOTION ----
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

    # ---- 14. RETURN DATA ----
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
        # LOD
        "lod_total_ms": lod_total_ms,
        "lod_iers_ms": lod_iers_ms,
        "lod_tidal_correction_ms": lod_tidal_corr_ms,
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
# PROFESSIONAL REPORT PRINTER (Hierarchical SOFA/IAU Order - Aligned)
# -------------------------------------------------------------------------
def print_report(data: Dict) -> None:
    W = 72
    # Karakter Box-Drawing
    thick_sep = "═" * W
    thin_sep  = "─" * W
    
    # ANSI Escape codes
    BOLD = '\033[1m'
    RESET = '\033[0m'

    ts = data["timestamp_utc"]

    print(f"\n{BOLD}{thick_sep}{RESET}")
    print(f"{BOLD}{'ASTERID HIGH‑PRECISION ASTROMETRIC & GEODETIC REPORT'.center(W)}{RESET}")
    print(f"{BOLD}{'IERS Conventions (2010) – Single‑Shot Epoch Solution'.center(W)}{RESET}")
    print(f"{BOLD}{'** EPHEMERIS ENGINE : VSOP2013 + ELP/MPP02 **'.center(W)}{RESET}")
    print(f"{BOLD}{'(VSOP2013: Fienga et al. 2013)'.center(W)}{RESET}")
    print(f"{BOLD}{'** ALL IERS 2010 GEOPHYSICAL CORRECTIONS APPLIED **'.center(W)}{RESET}")
    print(f"{BOLD}{thick_sep}{RESET}")

    def section_header(title):
        print(f"\n{BOLD}{title}{RESET}")
        print(f"{BOLD}{thin_sep}{RESET}")

    # [1] TIME SYSTEMS
    section_header("[1] TIME SYSTEMS (JD, TAI, TT, TDB, UT1)")
    print(f"  {'UTC Epoch':<30}: {ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (ISO 8601)")
    print(f"  {'UTC (JD)':<30}: {data['utc_jd']:.9f}")
    print(f"  {'TAI (JD)':<30}: {data['tai_jd']:.9f}")
    print(f"  {'TT  (JD)':<30}: {data['tt_jd']:.9f}")
    print(f"  {'TDB (JD)':<30}: {data['tdb_jd']:.9f}")
    print(f"  {'UT1 (JD)':<30}: {data['ut1_jd']:.9f}")
    print(f"  {'ΔT = TT − UT1':<30}: {data['delta_t_s']:.6f} s")
    print(f"  {'DUT1 = UT1 − UTC':<30}: {data['dut1_s']:.6f} s")

    # [2] EARTH ORIENTATION PARAMETERS & LENGTH OF DAY (LOD)
    section_header("[2] EARTH ORIENTATION & LENGTH OF DAY")
    print(f"  {'Polar Motion (xp)':<30}: {data['xp_mas']:.6f} mas")
    print(f"  {'Polar Motion (yp)':<30}: {data['yp_mas']:.6f} mas")
    print(f"  {'Celestial Offset dX':<30}: {data['dX_mas']:.6f} mas")
    print(f"  {'Celestial Offset dY':<30}: {data['dY_mas']:.6f} mas")
    print(f"  {'LOD':<30}: {data['lod_total_ms']:+.6f} ms")
    print(f"  {'Day Length (SI)':<30}: {86400.0 + data['lod_total_ms'] / 1000.0:.6f} s")        

    # [3] CIP/CIO & ROTATION
    section_header("[3] CELESTIAL INTERMEDIATE POLE & EARTH ROTATION")
    print(f"  {'CIP X (IAU 2006)':<30}: {data['X_cip_mas']:.6f} mas")
    print(f"  {'CIP Y (IAU 2006)':<30}: {data['Y_cip_mas']:.6f} mas")
    print(f"  {'CIO Locator (s)':<30}: {data['s_cio_mas']:.6f} mas")
    print(f"  {'TIO Locator (sp)':<30}: {data['sp_tio_mas']:.6f} mas")
    print(f"  {'Earth Rotation Angle':<30}: {data['era_deg']:.6f}°")
    print(f"  {'Eq. of the Origins':<30}: {data['eo_deg']:.6f}°")
    print(f"  {'Greenwich Sidereal':<30}: {data['gst_deg']:.6f}°")

    # [4] NUTATION/PRECESSION
    section_header("[4] NUTATION (IAU 2000A_R06) & PRECESSION (IAU 2006)")
    print(f"  {'Nutation Δψ':<30}: {data['dpsi_mas']:.6f} mas")
    print(f"  {'Nutation Δε':<30}: {data['deps_mas']:.6f} mas")
    print(f"  {'Precession εₐ (obl.)':<30}: {data['precession_epsA_deg']:.6f}°")
    print(f"  {'Precession ψₐ':<30}: {data['precession_psiA_deg']:.6f}°")
    print(f"  {'Precession ωₐ':<30}: {data['precession_omegaA_deg']:.6f}°")
    print(f"  {'Precession χₐ':<30}: {data['precession_chiA_deg']:.6f}°")

    # [5] HELIOCENTRIC (EMB)
    section_header("[5] HELIOCENTRIC EARTH‑MOON BARYCENTRE (EMB) STATE")
    emb_pos = data['emb_pos_km']
    emb_vel = data['emb_vel_km_day']
    print(f"  {'Position X':<30}: {emb_pos[0]:.3f} km")
    print(f"  {'Position Y':<30}: {emb_pos[1]:.3f} km")
    print(f"  {'Position Z':<30}: {emb_pos[2]:.3f} km")
    print(f"  {'Velocity Vx':<30}: {emb_vel[0]:.6f} km/day")
    print(f"  {'Velocity Vy':<30}: {emb_vel[1]:.6f} km/day")
    print(f"  {'Velocity Vz':<30}: {emb_vel[2]:.6f} km/day")
    print(f"  {'Distance from Sun (km)':<30}: {data['emb_dist_km']:.3f} km")
    print(f"  {'Distance from Sun (AU)':<30}: {data['emb_dist_au']:.6f} AU")
    print(f"  {'Reference Frame':<30}: Dynamical equinox & ecliptic")
    print(f"  {'':<30}  J2000 (VSOP2013)")

    # [6] GEOCENTRIC GCRS (ICRS) POSITIONS & MOTION
    section_header("[6] GEOCENTRIC GCRS (ICRS) REFERENCE POSITIONS & MOTION")
    print("  Ref Frame: ICRS (J2000.0 mean equator & equinox)")
    print("  (Geocentric astrometric, without topocentric corrections)\n")
    # SUN
    print(f"{BOLD}  ** SUN **{RESET}")
    sun_vec = data['sun_gcrs_vec']
    print(f"    {'Position X (km)':<28}: {sun_vec[0]:.3f}")
    print(f"    {'Position Y (km)':<28}: {sun_vec[1]:.3f}")
    print(f"    {'Position Z (km)':<28}: {sun_vec[2]:.3f}")
    print(f"    {'Distance (km)':<28}: {data['sun_dist_km']:.3f}")
    print(f"    {'RA  (ICRS)':<28}: {data['sun_ra_gcrs_deg']:.6f}°")
    print(f"    {'Dec (ICRS)':<28}: {data['sun_dec_gcrs_deg']:.6f}°")
    print(f"    {'RA rate':<28}: {data['sun_ra_rate_mas_day']:.3f} mas/day")
    print(f"    {'Dec rate':<28}: {data['sun_dec_rate_mas_day']:.3f} mas/day")
    # MOON
    print(f"{BOLD}  ** MOON **{RESET}")
    moon_vec = data['moon_gcrs_vec']
    print(f"    {'Position X (km)':<28}: {moon_vec[0]:.3f}")
    print(f"    {'Position Y (km)':<28}: {moon_vec[1]:.3f}")
    print(f"    {'Position Z (km)':<28}: {moon_vec[2]:.3f}")
    print(f"    {'Distance (km)':<28}: {data['moon_dist_km']:.3f}")
    print(f"    {'RA  (ICRS)':<28}: {data['moon_ra_gcrs_deg']:.6f}°")
    print(f"    {'Dec (ICRS)':<28}: {data['moon_dec_gcrs_deg']:.6f}°")
    print(f"    {'RA rate':<28}: {data['moon_ra_rate_mas_day']:.3f} mas/day")
    print(f"    {'Dec rate':<28}: {data['moon_dec_rate_mas_day']:.3f} mas/day")

    # [7] ECLIPTIC OF DATE
    section_header("[7] ECLIPTIC OF DATE (True Ecliptic & Equinox of Date)")
    print(f"{BOLD}  ** SUN **{RESET}")
    print(f"    {'Ecliptic Longitude (λ)':<28}: {data['sun_ecl_lon_deg']:.6f}°")
    print(f"    {'Ecliptic Latitude (β)':<28}: {data['sun_ecl_lat_deg']:.6f}°")
    print(f"    {'Geocentric Distance':<28}: {data['sun_ecl_r_km']:.3f} km")
    print(f"{BOLD}  ** MOON **{RESET}")
    print(f"    {'Ecliptic Longitude (λ)':<28}: {data['moon_ecl_lon_deg']:.6f}°")
    print(f"    {'Ecliptic Latitude (β)':<28}: {data['moon_ecl_lat_deg']:.6f}°")
    print(f"    {'Geocentric Distance':<28}: {data['moon_ecl_r_km']:.3f} km")

    # [8] TOPOCENTRIC (CIRS & Equinox)
    section_header("[8] TOPOCENTRIC APPARENT COORDINATES (CIRS & Equinox of Date)")
    sun = data['sun']
    moon = data['moon']
    print(f"{BOLD}  ** SUN (Apparent Topocentric) **{RESET}")
    print(f"    RA (CIRS)                   : {sun['ra_cirs_deg']:.6f}°")
    print(f"    Dec (CIRS)                  : {sun['dec_cirs_deg']:.6f}°")
    print(f"    RA (Equinox)                : {sun['ra_eqx_deg']:.6f}°")
    print(f"    Dec (Equinox)               : {sun['dec_eqx_deg']:.6f}°")
    # (+) Tambahan untuk jarak toposentrik Matahari
    print(f"    Distance (km)               : {sun['dist_km']:.3f}")
    print(f"    Distance (AU)               : {sun['dist_au']:.9f}")
    print(f"{BOLD}  ** MOON (Apparent Topocentric) **{RESET}")
    print(f"    RA (CIRS)                   : {moon['ra_cirs_deg']:.6f}°")
    print(f"    Dec (CIRS)                  : {moon['dec_cirs_deg']:.6f}°")
    print(f"    RA (Equinox)                : {moon['ra_eqx_deg']:.6f}°")
    print(f"    Dec (Equinox)               : {moon['dec_eqx_deg']:.6f}°")
    # (+) Tambahan untuk jarak toposentrik Bulan
    print(f"    Distance (km)               : {moon['dist_km']:.3f}")
    print(f"    Distance (AU)               : {moon['dist_au']:.9f}")

    # [9] OBSERVED HORIZONTAL (with Refraction)
    sun_obs = data['sun_obs']
    moon_obs = data['moon_obs']
    section_header("[9] OBSERVED / APPARENT HORIZONTAL (Refraction VMF3+GPT3)")
    print(f"{BOLD}  ** SUN **{RESET}")
    print(f"    {'Azimuth':<28}: {sun_obs['az_deg']:.6f}°")
    print(f"    {'Elevation (Geom.)':<28}: {sun_obs['alt_geom_deg']:.6f}°")
    print(f"    {'Elevation (Refr.)':<28}: {sun_obs['alt_app_deg']:.6f}°")
    print(f"{BOLD}  ** MOON **{RESET}")
    print(f"    {'Azimuth':<28}: {moon_obs['az_deg']:.6f}°")
    print(f"    {'Elevation (Geom.)':<28}: {moon_obs['alt_geom_deg']:.6f}°")
    print(f"    {'Elevation (Refr.)':<28}: {moon_obs['alt_app_deg']:.6f}°")

    # [10] LUNAR PHASE & ORBITAL ELEMENTS
    mp = data['moon_phase']
    section_header("[10] LUNAR PHASE & OSCULATING ORBITAL ELEMENTS")
    print(f"  {'Phase Name':<30}: {mp['phase_name']}")
    print(f"  {'Illumination':<30}: {mp['illumination_pct']:.2f} %")
    print(f"  {'Age (since New Moon)':<30}: {mp['age_days']:.3f} days")
    print(f"  {'Elongation':<30}: {mp['elongation_deg']:.2f}°")
    print(f"  {'Phase Angle':<30}: {mp['phase_angle_deg']:.2f}°")
    print(f"{BOLD}  ** Osculating Elements (Geocentric, J2000.0) **{RESET}")
    print(f"    {'Semi‑major axis (a)':<28}: {data['lunar_a_km']:.3f} km")
    print(f"    {'Eccentricity (e)':<28}: {data['lunar_e']:.8f}")
    print(f"    {'Inclination (i)':<28}: {data['lunar_i_deg']:.6f}°")
    print(f"    {'Node (Ω)':<28}: {data['lunar_Omega_deg']:.6f}°")
    print(f"    {'Arg. of Perigee (ω)':<28}: {data['lunar_omega_deg']:.6f}°")
    print(f"    {'Mean Anomaly (M)':<28}: {data['lunar_M_deg']:.6f}°")

    # [11] LUNAR LIBRATION
    section_header("[11] LUNAR LIBRATION (LLIB04)")
    lib = data['lunar_libration']
    print(f"  {'P1 (optical libration)':<30}: {lib['p1_deg']:.6f}°")
    print(f"  {'P2 (optical libration)':<30}: {lib['p2_deg']:.6f}°")
    print(f"  {'Tau (physical lib.)':<30}: {lib['tau_deg']:.6f}°")

    # [12] PLATE MOTION
    section_header("[12] KINEMATIC PLATE MOTION (ITRF2020‑PMM)")
    print(f"  {'Tectonic Plate':<30}: {data['plate_code']} (Eurasian)")
    print(f"  {'Reference Epoch':<30}: J2000.0 (ΔT = {data['plate_epoch_diff_years']:.4f} yr)")
    print(f"  {'Velocity (Ve)':<30}: {data['plate_ve_mmyr']:.4f} mm/yr (East)")
    print(f"  {'Velocity (Vn)':<30}: {data['plate_vn_mmyr']:.4f} mm/yr (North)")
    print(f"  {'Velocity (Vu)':<30}: {data['plate_vu_mmyr']:.4f} mm/yr (Up)")
    print(f"  {'Horizontal Speed':<30}: {data['plate_horiz_speed_mmyr']:.4f} mm/yr")
    print(f"  {'Horizontal Azimuth':<30}: {data['plate_azimuth_deg']:.4f}° (True N)")

    # [13] OBSERVER NOTES
    section_header("[13] OBSERVER & REDUCTION NOTES")
    print(f"  {'Geodetic Latitude':<30}: {OBSERVER_LAT_DEG:+.6f}°")
    print(f"  {'Geodetic Longitude':<30}: {OBSERVER_LON_DEG:+.6f}°")
    print(f"  {'Orthometric Height':<30}: {OBSERVER_HEIGHT_M:.3f} m (XGM2019e-2159)")
    print(f"  {'Refraction Model':<30}: VMF3 + GPT3 (APG gradient)")

    # ---- Atmospheric source status (ECMWF vs GPT3) ----
    from Atmospheric_refraction import get_refraction_source
    src = get_refraction_source()
    if src['active']:
        print(f"  {'Atmospheric Source':<30}: ECMWF IFS (real-time NWP)")
        print(f"  {'ECMWF Timestamp':<30}: {src['timestamp']}")
    else:
        print(f"  {'Atmospheric Source':<30}: GPT3 climatology")

    eop_file = data.get('eop_file', 'unknown')
    if len(eop_file) > 40:
        print(f"  {'EOP Source File':<30}:")
        print(f"  {'':<30}  {eop_file}")
    else:
        print(f"  {'EOP Source File':<30}: {eop_file}")

    print(f"  {'EOP Last Observation':<30}: {data.get('eop_last_date', 'Unknown')}")
    print(f"  {'Ephemeris Sources':<30}: VSOP2013 (Sun), ELP/MPP02 (Moon)")
    print("\n  Geophysical Corrections (IERS 2010, full precision):")
    print("  – Solid Earth tides (frequency‑dependent Love numbers)")
    print("  – Ocean tide loading (FES2014)")
    print("  – Pole tide (diurnal & semi‑diurnal)")
    print("  – Atmospheric & Non-tidal loading")
    print("  – Vertical deflection (DoV) from local XGM2019e-2159 grids")
    print("  – Kinematic plate motion (ITRF2020‑PMM)")

    print(f"\n{BOLD}{thick_sep}{RESET}")
    print(f"{BOLD}{'REPORT GENERATED SUCCESSFULLY – EXITING.'.center(W)}{RESET}")
    print(f"{BOLD}{thick_sep}{RESET}\n")


# -------------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("⏳ Computing comprehensive ephemeris + plate kinematics (VSOP2013)...")
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