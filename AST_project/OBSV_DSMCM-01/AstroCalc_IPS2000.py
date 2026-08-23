#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroCalc_IPS2000.py – High-Precision Ephemeris (IPS2000 Engine)
================================================================
Fitur:
    - Input ISO 8601 dengan tahun negatif (misal -3120-06-07T04:57:18)
    - Opsi kalender: gregorian, julian, auto (year < 1582 → julian)
    - Semua koreksi IERS 2010 (EOP, CIP/CIO, nutasi, aberasi, defleksi)
    - Fase Bulan presisi tinggi (elongasi, phase angle, illuminasi)
    - RA/Dec Bulan dalam CIRS (CIO-based) dan apparent topocentric
    - Laporan lengkap dengan librasi, plate motion, dll.
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
import argparse
import re
from typing import Dict, Tuple

# -------------------------------------------------------------------------
# Impor modul eksternal (tidak diubah)
# -------------------------------------------------------------------------
from DateTime import DateTime, now_utc
from Timescales import (
    J2000_JD, delta_t_from_jd, tai_utc, tt_to_tdb, split_jd, combine_jd,
)
from EOPDelta import EOPProvider
from EarthRotation import (
    get_cip_xy, get_cio_s, get_tio_sp,
    nutation_2000a, precession_angles_2006,
    equation_of_origins, gst_from_ut1,
    ARCSEC_TO_RAD, MAS_TO_RAD, UAS_TO_RAD,
    era_from_ut1, Q_inverse, light_deflection_sun, bias_precession_nutation_matrix
)
from SM_IPS2000emb import AstronomicalEphemeris
from Atmospheric_refraction import SITE_LAT_DEG, SITE_LON_DEG, SITE_ELEV_M
from Site_Geophysic import TectonicPlateKinematics
from Coord_Transform import rot_x, unit_vector, cartesian_to_spherical

# -------------------------------------------------------------------------
# Ambil ELP_MPP02_full yang sudah diimpor oleh SM_IPS2000emb (hindari circular import)
# -------------------------------------------------------------------------
import sys
_elp_mod = sys.modules.get('ELP_MPP02_full')
if _elp_mod is None:
    import ELP_MPP02_full as _elp_mod
ELP_MPP02_full = _elp_mod

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
OBSERVER_LAT_DEG = SITE_LAT_DEG
OBSERVER_LON_DEG = SITE_LON_DEG
OBSERVER_HEIGHT_M = SITE_ELEV_M
PLATE_CODE = 'EURA'

RAD_TO_DEG = 180.0 / math.pi
RAD_TO_MAS = 1.0 / MAS_TO_RAD
RAD_TO_UAS = 1.0 / UAS_TO_RAD
MM_PER_M = 1000.0
C_KM_S = 299792.458
AU2KM = 149597870.7
KM2AU = 1.0 / AU2KM
C_AUDAY = 173.1446326846693
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)

# ============================================================================
# FUNGSI PEMBANTU LOKAL
# ============================================================================
def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [0, 2π)."""
    return angle % (2.0 * math.pi)

# ============================================================================
# 1. STELLAR ABERASI (copy dari SM_IPS2000emb agar tidak import error)
# ============================================================================
def stellar_aberration_full(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    beta = v / C_AUDAY
    beta2 = np.dot(beta, beta)
    if beta2 == 0.0:
        return p
    if beta2 >= 1.0:
        beta2 = 0.999999999999
    gamma_inv = math.sqrt(1.0 - beta2)
    p_dot_beta = np.dot(p, beta)
    factor = 1.0 / (1.0 + p_dot_beta)
    p_corr = (gamma_inv * p + (1.0 + p_dot_beta / (1.0 + gamma_inv)) * beta) * factor
    return p_corr / np.linalg.norm(p_corr)

# ============================================================================
# 2. FUNGSI FASE BULAN (presisi tinggi)
# ============================================================================
def moon_phase_name(elong_deg: float) -> str:
    names = [
        "New Moon", "Waxing Crescent", "First Quarter", "Waxing Gibbous",
        "Full Moon", "Waning Gibbous", "Last Quarter", "Waning Crescent"
    ]
    idx = int(round(elong_deg / 45.0)) % 8
    return names[idx]

# ============================================================================
# 3. KONVERSI KALENDER MANDIRI (jdcal – valid untuk semua tahun)
# ============================================================================
def _julian_to_jd_noon(year: int, month: int, day: int) -> int:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - 32083

def _gregorian_to_jd_noon(year: int, month: int, day: int) -> int:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

def julian_to_jd(year: int, month: int, day: int,
                 hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    jd_noon = _julian_to_jd_noon(year, month, day)
    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd_noon + day_frac - 0.5

def gregorian_to_jd(year: int, month: int, day: int,
                    hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    jd_noon = _gregorian_to_jd_noon(year, month, day)
    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd_noon + day_frac - 0.5

def parse_iso_and_calc_jd(iso_str: str, calendar: str) -> Tuple[float, str]:
    iso_str = iso_str.strip()
    # Pola utama: T sebagai pemisah, detik opsional
    pattern = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})(?:T(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d+(?:\.\d+)?))?)?$'
    m = re.match(pattern, iso_str)
    if not m:
        # Pola alternatif: spasi sebagai pemisah, detik opsional
        pattern2 = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})\s+(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d+(?:\.\d+)?))?$'
        m = re.match(pattern2, iso_str)
    if not m:
        # Pola tanpa detik sama sekali (hanya jam:menit)
        pattern3 = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})(?:T|\s+)(?P<hour>\d{2}):(?P<minute>\d{2})$'
        m = re.match(pattern3, iso_str)
    if not m:
        raise ValueError(f"Invalid ISO 8601: {repr(iso_str)}")
    
    sign = m.group('sign')
    year = int(sign + m.group('year'))
    month = int(m.group('month'))
    day = int(m.group('day'))
    hour = int(m.group('hour') or 0)
    minute = int(m.group('minute') or 0)
    second_str = m.group('second')
    second = float(second_str) if second_str is not None else 0.0

    if calendar == 'julian':
        jd = julian_to_jd(year, month, day, hour, minute, second)
        cal_used = 'julian'
    elif calendar == 'gregorian':
        jd = gregorian_to_jd(year, month, day, hour, minute, second)
        cal_used = 'gregorian'
    else:  # auto
        if year < 1582:
            jd = julian_to_jd(year, month, day, hour, minute, second)
            cal_used = 'julian'
        else:
            jd = gregorian_to_jd(year, month, day, hour, minute, second)
            cal_used = 'gregorian'
    return jd, cal_used

# ============================================================================
# 4. INTI PERHITUNGAN
# ============================================================================
def compute_for_date(
    iso_date: str,
    calendar: str = 'gregorian',
    eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt",
    lat_deg: float = OBSERVER_LAT_DEG,
    lon_deg: float = OBSERVER_LON_DEG,
    height_m: float = OBSERVER_HEIGHT_M,
) -> Dict:

    # --- 4a. Parse input ---
    jd_utc, cal_used = parse_iso_and_calc_jd(iso_date, calendar)
    dt = DateTime.from_jd(jd_utc)

    # --- 4b. Time scales ---
    delta_t = delta_t_from_jd(jd_utc)
    tt_jd = jd_utc + delta_t / 86400.0
    tai_jd = tt_jd - 32.184 / 86400.0
    tdb1, tdb2 = tt_to_tdb(*split_jd(tt_jd))
    tdb_jd = combine_jd(tdb1, tdb2)
    ut1_jd = jd_utc  # UT1 ≈ UTC for historical epochs

    # --- 4c. EOP ---
    eop_provider = EOPProvider(eop_file)
    eop_last_date = eop_provider.get_last_available_date()
    mjd_utc = jd_utc - 2400000.5
    try:
        eop = eop_provider.get_eop(mjd_utc)
        xp_mas, yp_mas, dut1_s, dX_mas, dY_mas = (
            eop["x_pole"], eop["y_pole"], eop["ut1_utc"],
            eop["dX"], eop["dY"]
        )
    except Exception:
        xp_mas = yp_mas = dut1_s = dX_mas = dY_mas = 0.0

    xp_rad = xp_mas * MAS_TO_RAD
    yp_rad = yp_mas * MAS_TO_RAD
    dX_rad = dX_mas * MAS_TO_RAD
    dY_rad = dY_mas * MAS_TO_RAD

    # --- 4d. CIP, CIO, ERA ---
    X_rad, Y_rad = get_cip_xy(tt_jd, apply_fcn=True)
    X_rad += dX_rad
    Y_rad += dY_rad
    s_rad = get_cio_s(tt_jd, X_rad, Y_rad)
    sp_rad = get_tio_sp(tt_jd)
    era_rad = era_from_ut1(ut1_jd)
    eo_rad = equation_of_origins(tt_jd)
    gst_rad = gst_from_ut1(ut1_jd, tt_jd)

    # --- 4e. Nutation & Precession ---
    dpsi_rad, deps_rad = nutation_2000a(tt_jd)
    t_cy = (tt_jd - J2000_JD) / 36525.0
    pre = precession_angles_2006(t_cy)

    # --- 4f. Ephemeris engine ---
    ephem = AstronomicalEphemeris(eop_file=eop_file)
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # --- 4g. Geocentric GCRS positions (km) ---
    sun_pos_gcrs = ephem.sun_geocentric_gcrs(tt_jd, unit=False)
    moon_pos_gcrs = ephem.moon_geocentric_gcrs(tt_jd, unit=False)
    earth_vel_gcrs = ephem.earth_velocity_gcrs(tt_jd)  # AU/day

    # --- Hitung RA/Dec GCRS (ICRS) ---
    sun_ra_gcrs, sun_dec_gcrs, _ = cartesian_to_spherical(sun_pos_gcrs)
    moon_ra_gcrs, moon_dec_gcrs, _ = cartesian_to_spherical(moon_pos_gcrs)

    obs_pos_gcrs = np.zeros(3)
    obs_vel_gcrs = earth_vel_gcrs * AU2KM   # km/day

    # --- Geocentric distances ---
    sun_dist_km = np.linalg.norm(sun_pos_gcrs)
    moon_dist_km = np.linalg.norm(moon_pos_gcrs)

    # --- True obliquity of date ---
    epsA_deg = pre["epsA"] * RAD_TO_DEG          # dari precession_angles_2006
    deps_deg = deps_rad * RAD_TO_DEG            # dari nutation_2000a
    eps_true_deg = epsA_deg + deps_deg
    eps_true_rad = math.radians(eps_true_deg)

    # --- BPN (GCRS -> true equator & equinox of date) ---
    BPN = bias_precession_nutation_matrix(tt_jd)

    # --- Rotasi ke ecliptic of date ---
    R_ecl = rot_x(-eps_true_rad)

    # Sun
    sun_true_eq = BPN @ sun_pos_gcrs
    sun_ecl_vec = R_ecl @ sun_true_eq
    sun_ecl_r = np.linalg.norm(sun_ecl_vec)
    sun_ecl_lon = math.atan2(sun_ecl_vec[1], sun_ecl_vec[0])
    sun_ecl_lat = math.asin(np.clip(sun_ecl_vec[2] / sun_ecl_r, -1.0, 1.0))
    sun_ecl_lon_deg = math.degrees(sun_ecl_lon) % 360.0
    sun_ecl_lat_deg = math.degrees(sun_ecl_lat)

    # Moon
    moon_true_eq = BPN @ moon_pos_gcrs
    moon_ecl_vec = R_ecl @ moon_true_eq
    moon_ecl_r = np.linalg.norm(moon_ecl_vec)
    moon_ecl_lon = math.atan2(moon_ecl_vec[1], moon_ecl_vec[0])
    moon_ecl_lat = math.asin(np.clip(moon_ecl_vec[2] / moon_ecl_r, -1.0, 1.0))
    moon_ecl_lon_deg = math.degrees(moon_ecl_lon) % 360.0
    moon_ecl_lat_deg = math.degrees(moon_ecl_lat)

    # =================================================================
    #  MOON : Light-time, deflection, aberration, CIRS and Equinox
    # =================================================================
    tau = 0.0
    for _ in range(5):
        t_emit = tt_jd - tau / 86400.0
        moon_pos_emit = ephem.moon_geocentric_gcrs(t_emit, unit=False)
        obs_pos_emit = obs_pos_gcrs + obs_vel_gcrs * (tau / 86400.0)
        topo = moon_pos_emit - obs_pos_emit
        dist_km = np.linalg.norm(topo)
        tau_new = dist_km / C_KM_S
        if abs(tau_new - tau) < 1e-11:
            tau = tau_new
            break
        tau = tau_new

    uv_topo = topo / dist_km

    sun_pos_emit = ephem.sun_geocentric_gcrs(t_emit, unit=False)
    p_sun = unit_vector(sun_pos_emit)
    uv_deflect = light_deflection_sun(uv_topo, p_sun)

    v_obs_au_day = obs_vel_gcrs / AU2KM
    uv_ab = stellar_aberration_full(uv_deflect, v_obs_au_day)

    # CIRS (CIO-based)
    Q_inv = Q_inverse(X_rad, Y_rad, s_rad)
    uv_cirs = Q_inv @ uv_ab
    ra_cirs = math.atan2(uv_cirs[1], uv_cirs[0])
    dec_cirs = math.asin(np.clip(uv_cirs[2], -1.0, 1.0))
    ra_cirs_deg = math.degrees(ra_cirs) % 360.0
    dec_cirs_deg = math.degrees(dec_cirs)

    # Equinox-based (using Equation of Origins)
    eo_deg = math.degrees(eo_rad) % 360.0
    ra_eqx_deg = (ra_cirs_deg - eo_deg) % 360.0
    dec_eqx_deg = dec_cirs_deg   # deklinasi tetap sama

    # =================================================================
    #  SUN  : Light-time, deflection, aberration, CIRS and Equinox
    # =================================================================
    tau_sun = 0.0
    for _ in range(5):
        t_emit_sun = tt_jd - tau_sun / 86400.0
        sun_pos_emit2 = ephem.sun_geocentric_gcrs(t_emit_sun, unit=False)
        obs_pos_emit_sun = obs_pos_gcrs + obs_vel_gcrs * (tau_sun / 86400.0)
        topo_sun = sun_pos_emit2 - obs_pos_emit_sun
        dist_sun_km = np.linalg.norm(topo_sun)
        tau_new_sun = dist_sun_km / C_KM_S
        if abs(tau_new_sun - tau_sun) < 1e-11:
            tau_sun = tau_new_sun
            break
        tau_sun = tau_new_sun

    uv_topo_sun = topo_sun / dist_sun_km
    p_sun_dir = unit_vector(sun_pos_emit2)
    uv_deflect_sun = light_deflection_sun(uv_topo_sun, p_sun_dir)
    uv_ab_sun = stellar_aberration_full(uv_deflect_sun, v_obs_au_day)

    uv_cirs_sun = Q_inv @ uv_ab_sun
    ra_cirs_sun = math.atan2(uv_cirs_sun[1], uv_cirs_sun[0])
    dec_cirs_sun = math.asin(np.clip(uv_cirs_sun[2], -1.0, 1.0))
    ra_cirs_sun_deg = math.degrees(ra_cirs_sun) % 360.0
    dec_cirs_sun_deg = math.degrees(dec_cirs_sun)

    ra_eqx_sun_deg = (ra_cirs_sun_deg - eo_deg) % 360.0
    dec_eqx_sun_deg = dec_cirs_sun_deg

    # --- 4h. Apparent topocentric (az/alt, refraction) from SM_IPS2000emb ---
    sun = ephem.sun_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction=True)
    moon = ephem.moon_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction=True)

    # --- 4i. Moon phase (high precision) ---
    earth_pos_ecl, _ = ephem.earth_heliocentric_ecliptic(tt_jd)
    sun_pos_ecl = -np.array(earth_pos_ecl)
    tj = tdb_jd - J2000_JD
    xyz_ecl_km = ELP_MPP02_full.elpmpp02(tj, icor=1)
    moon_pos_ecl = np.array(xyz_ecl_km[:3]) / AU2KM

    sun_lon = math.atan2(sun_pos_ecl[1], sun_pos_ecl[0])
    moon_lon = math.atan2(moon_pos_ecl[1], moon_pos_ecl[0])
    elong_rad = (moon_lon - sun_lon) % (2 * math.pi)
    elong_deg = math.degrees(elong_rad)
    phase_name = moon_phase_name(elong_deg)

    r_moon = moon_pos_gcrs
    r_sun  = sun_pos_gcrs
    v_ms = r_sun - r_moon
    v_me = -r_moon
    cos_i = np.clip(np.dot(v_ms, v_me) / (np.linalg.norm(v_ms) * np.linalg.norm(v_me)), -1.0, 1.0)
    phase_angle_deg = math.degrees(np.arccos(cos_i))
    illum_pct = (1.0 + cos_i) / 2.0 * 100.0
    age_days = elong_deg / 360.0 * 29.530588861

    moon_phase_info = {
        'phase_name': phase_name,
        'illumination_pct': illum_pct,
        'age_days': age_days,
        'elongation_deg': elong_deg,
        'phase_angle_deg': phase_angle_deg,
    }

    # --- 4j. Lunar libration ---
    lib = ephem.lunar_libration(tt_jd)

    # --- 4k. Plate motion ---
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

    plate_kin = TectonicPlateKinematics(PLATE_CODE)
    vx, vy, vz = plate_kin.get_velocity(
        x_itrf, y_itrf, z_itrf,
        lat_rad=lat_rad, lon_rad=lon_rad,
        apply_orb=True, discard_vertical_orb=True,
    )
    ve = (-vx * sin_lon + vy * cos_lon) * MM_PER_M
    vn = (-vx * sin_lat * cos_lon - vy * sin_lat * sin_lon + vz * cos_lat) * MM_PER_M
    vu = (vx * cos_lat * cos_lon + vy * cos_lat * sin_lon + vz * sin_lat) * MM_PER_M
    horiz_speed = math.hypot(ve, vn)
    az_plate = math.degrees(math.atan2(ve, vn)) % 360.0
    epoch_diff_years = (tt_jd - J2000_JD) / 365.25
    disp_e_mm = ve * epoch_diff_years
    disp_n_mm = vn * epoch_diff_years
    disp_u_mm = vu * epoch_diff_years

    # --- Return ---
    return {
        "timestamp_utc": dt,
        "calendar_input": cal_used,
        "utc_jd": jd_utc,
        "tai_jd": tai_jd,
        "tt_jd": tt_jd,
        "tdb_jd": tdb_jd,
        "ut1_jd": ut1_jd,
        "delta_t_s": delta_t,
        "dut1_s": dut1_s,
        "xp_mas": xp_mas,
        "yp_mas": yp_mas,
        "dX_mas": dX_mas,
        "dY_mas": dY_mas,
        "X_cip_mas": X_rad * RAD_TO_MAS,
        "Y_cip_mas": Y_rad * RAD_TO_MAS,
        "s_cio_mas": s_rad * RAD_TO_MAS,
        "sp_tio_mas": sp_rad * RAD_TO_MAS,
        "era_deg": era_rad * RAD_TO_DEG,
        "eo_deg": eo_rad * RAD_TO_DEG,
        "gst_deg": gst_rad * RAD_TO_DEG,
        "dpsi_mas": dpsi_rad * RAD_TO_MAS,
        "deps_mas": deps_rad * RAD_TO_MAS,
        "precession_epsA_deg": pre["epsA"] * RAD_TO_DEG,
        "precession_psiA_deg": pre["psiA"] * RAD_TO_DEG,
        "precession_omegaA_deg": pre["omegaA"] * RAD_TO_DEG,
        "precession_chiA_deg": pre["chiA"] * RAD_TO_DEG,
        "sun": sun,
        "moon": moon,
        "moon_phase": moon_phase_info,
        "lunar_libration": lib,
        # ===== GCRS (ICRS) astrometric positions =====
        "sun_ra_gcrs_deg": math.degrees(normalize_angle(sun_ra_gcrs)),
        "sun_dec_gcrs_deg": math.degrees(sun_dec_gcrs),
        "moon_ra_gcrs_deg": math.degrees(normalize_angle(moon_ra_gcrs)),
        "moon_dec_gcrs_deg": math.degrees(moon_dec_gcrs),
        "moon_ra_cirs_deg": moon['ra_cirs_deg'],
        "moon_dec_cirs_deg": moon['dec_cirs_deg'],
        "moon_ra_eqx_deg": moon['ra_eqx_deg'],
        "moon_dec_eqx_deg": moon['dec_eqx_deg'],
        "sun_ra_cirs_deg": sun['ra_cirs_deg'],
        "sun_dec_cirs_deg": sun['dec_cirs_deg'],
        "sun_ra_eqx_deg": sun['ra_eqx_deg'],
        "sun_dec_eqx_deg": sun['dec_eqx_deg'],
        "sun_dist_km": sun_dist_km,
        "moon_dist_km": moon_dist_km,
        "sun_ecl_lon_deg": sun_ecl_lon_deg,
        "sun_ecl_lat_deg": sun_ecl_lat_deg,
        "moon_ecl_lon_deg": moon_ecl_lon_deg,
        "moon_ecl_lat_deg": moon_ecl_lat_deg,
        "eps_true_deg": eps_true_deg,        
        "plate_ve_mmyr": ve,
        "plate_vn_mmyr": vn,
        "plate_vu_mmyr": vu,
        "plate_horiz_speed_mmyr": horiz_speed,
        "plate_azimuth_deg": az_plate,
        "plate_disp_e_mm": disp_e_mm,
        "plate_disp_n_mm": disp_n_mm,
        "plate_disp_u_mm": disp_u_mm,
        "plate_epoch_diff_years": epoch_diff_years,
        "plate_code": PLATE_CODE,
        "eop_file": eop_file,
        "eop_last_date": eop_last_date,
    }

# ============================================================================
# 5. CETAK LAPORAN
# ============================================================================
def print_report(data: Dict) -> None:
    sun, moon = data["sun"], data["moon"]
    ts = data["timestamp_utc"]
    cal = data["calendar_input"]

    print("\n" + "=" * 80)
    print(" ASTERID HIGH-PRECISION ASTROMETRIC & GEODETIC REPORT")
    print(" IERS Conventions (2010) – Single-Shot Epoch Solution")
    print(" ** EPHEMERIS ENGINE : IPS2000 (Improved Planetary Solutions) **")
    print("=" * 80)

    # =====================================================================
    # 1. TIME SYSTEMS
    # =====================================================================
    print("\n[1] TIME SYSTEMS (JD, TAI, TT, TDB, UT1)")
    print("-" * 80)
    print(f"  UTC Epoch            : {ts.to_iso(calendar=cal)} (ISO 8601, {cal} calendar)")
    print(f"  UTC (JD)             : {data['utc_jd']:.9f}")
    print(f"  TAI (JD)             : {data['tai_jd']:.9f}")
    print(f"  TT  (JD)             : {data['tt_jd']:.9f}")
    print(f"  TDB (JD)             : {data['tdb_jd']:.9f}")
    print(f"  UT1 (JD)             : {data['ut1_jd']:.9f}")
    print(f"  ΔT = TT − UT1        : {data['delta_t_s']:.6f} s")
    print(f"  DUT1 = UT1 − UTC     : {data['dut1_s']:.6f} s")

    # =====================================================================
    # 2. EARTH ORIENTATION PARAMETERS
    # =====================================================================
    print("\n[2] EARTH ORIENTATION PARAMETERS (IERS Bulletin A)")
    print("-" * 80)
    print(f"  Polar Motion (xp)    : {data['xp_mas']:12.6f} mas")
    print(f"  Polar Motion (yp)    : {data['yp_mas']:12.6f} mas")
    print(f"  Celestial Offset dX  : {data['dX_mas']:12.6f} mas")
    print(f"  Celestial Offset dY  : {data['dY_mas']:12.6f} mas")

    # =====================================================================
    # 3. CELESTIAL INTERMEDIATE POLE & EARTH ROTATION
    # =====================================================================
    print("\n[3] CELESTIAL INTERMEDIATE POLE & EARTH ROTATION")
    print("-" * 80)
    print(f"  CIP X (IAU 2006)     : {data['X_cip_mas']:15.6f} mas")
    print(f"  CIP Y (IAU 2006)     : {data['Y_cip_mas']:15.6f} mas")
    print(f"  CIO Locator (s)      : {data['s_cio_mas']:15.6f} mas")
    print(f"  TIO Locator (s')     : {data['sp_tio_mas']:15.6f} mas")
    print(f"  Earth Rotation Angle : {data['era_deg']:15.6f} deg")
    print(f"  Eq. of the Origins   : {data['eo_deg']:15.6f} deg")
    print(f"  Greenwich Sidereal   : {data['gst_deg']:15.6f} deg")

    # =====================================================================
    # 4. NUTATION & PRECESSION
    # =====================================================================
    print("\n[4] NUTATION (IAU 2000A_R06) & PRECESSION (IAU 2006)")
    print("-" * 80)
    print(f"  Nutation Δψ          : {data['dpsi_mas']:15.6f} mas")
    print(f"  Nutation Δε          : {data['deps_mas']:15.6f} mas")
    print(f"  Precession εₐ (obl.) : {data['precession_epsA_deg']:15.6f} deg")
    print(f"  Precession ψₐ        : {data['precession_psiA_deg']:15.6f} deg")
    print(f"  Precession ωₐ        : {data['precession_omegaA_deg']:15.6f} deg")
    print(f"  Precession χₐ        : {data['precession_chiA_deg']:15.6f} deg")

    # =====================================================================
    # 5. GEOCENTRIC GCRS (ICRS) ASTROMETRIC POSITIONS
    #    (Fundamental: no aberration, deflection, or topocentric corrections)
    # =====================================================================
    print("\n[5] GEOCENTRIC GCRS (ICRS) ASTROMETRIC POSITIONS")
    print("-" * 80)
    print("  (Without aberration, deflection, or topocentric corrections)")
    print("  ** SUN **")
    print(f"    RA  (GCRS) : {data['sun_ra_gcrs_deg']:15.6f} deg")
    print(f"    Dec (GCRS) : {data['sun_dec_gcrs_deg']:15.6f} deg")
    print("  ** MOON **")
    print(f"    RA  (GCRS) : {data['moon_ra_gcrs_deg']:15.6f} deg")
    print(f"    Dec (GCRS) : {data['moon_dec_gcrs_deg']:15.6f} deg")

    # =====================================================================
    # 6. GEOCENTRIC ECLIPTIC OF DATE
    #    (True ecliptic & equinox of date, geocentric)
    # =====================================================================
    print("\n[6] GEOCENTRIC ECLIPTIC COORDINATES OF DATE")
    print("-" * 80)
    print("  (True ecliptic & equinox of date, geocentric)")
    print("  ** SUN **")
    print(f"    Ecliptic Longitude (λ) : {data['sun_ecl_lon_deg']:15.6f} deg")
    print(f"    Ecliptic Latitude (β)  : {data['sun_ecl_lat_deg']:15.6f} deg")
    print(f"    Geocentric Distance    : {data['sun_dist_km']:15.3f} km  ({data['sun_dist_km'] / AU2KM:12.9f} AU)")
    print("  ** MOON **")
    print(f"    Ecliptic Longitude (λ) : {data['moon_ecl_lon_deg']:15.6f} deg")
    print(f"    Ecliptic Latitude (β)  : {data['moon_ecl_lat_deg']:15.6f} deg")
    print(f"    Geocentric Distance    : {data['moon_dist_km']:15.3f} km  ({data['moon_dist_km'] / AU2KM:12.9f} AU)")
    print(f"  True Obliquity (ε)       : {data['eps_true_deg']:15.6f} deg")

    # =====================================================================
    # 7. TOPOCENTRIC APPARENT (CIRS & Equinox)
    #    (With light-time, deflection, aberration, but NO refraction)
    # =====================================================================
    print("\n[7] TOPOCENTRIC APPARENT COORDINATES (CIRS & Equinox)")
    print("-" * 80)
    print("  (With light-time, gravitational deflection, stellar aberration, but NO refraction)")
    print("  ** SUN **")
    print(f"    RA  (CIRS)    : {data['sun_ra_cirs_deg']:15.6f} deg")
    print(f"    Dec (CIRS)    : {data['sun_dec_cirs_deg']:15.6f} deg")
    print(f"    RA  (Equinox) : {data['sun_ra_eqx_deg']:15.6f} deg")
    print(f"    Dec (Equinox) : {data['sun_dec_eqx_deg']:15.6f} deg")
    print("  ** MOON **")
    ra_cirs = data.get('moon_ra_cirs_deg', moon['ra_deg'])
    dec_cirs = data.get('moon_dec_cirs_deg', moon['dec_deg'])
    ra_eqx = data.get('moon_ra_eqx_deg', moon['ra_deg'])
    dec_eqx = data.get('moon_dec_eqx_deg', moon['dec_deg'])
    print(f"    RA  (CIRS)    : {ra_cirs:15.6f} deg")
    print(f"    Dec (CIRS)    : {dec_cirs:15.6f} deg")
    print(f"    RA  (Equinox) : {ra_eqx:15.6f} deg")
    print(f"    Dec (Equinox) : {dec_eqx:15.6f} deg")

    # =====================================================================
    # 8. OBSERVED HORIZONTAL (Az/El with Refraction)
    # =====================================================================
    print("\n[8] OBSERVED HORIZONTAL COORDINATES (With Refraction VMF3+GPT3)")
    print("-" * 80)
    print("  ** SUN **")
    print(f"    Azimuth (True N) : {sun['az_deg']:15.6f} deg")
    print(f"    Elevation (Geom.) : {sun['alt_geom_deg']:15.6f} deg")
    print(f"    Elevation (Refr.) : {sun['alt_app_deg']:15.6f} deg")
    print("  ** MOON **")
    print(f"    Azimuth (True N) : {moon['az_deg']:15.6f} deg")
    print(f"    Elevation (Geom.) : {moon['alt_geom_deg']:15.6f} deg")
    print(f"    Elevation (Refr.) : {moon['alt_app_deg']:15.6f} deg")

    # =====================================================================
    # 9. LUNAR PHASE & LIBRATION
    # =====================================================================
    mp = data['moon_phase']
    print("\n[9] LUNAR PHASE & LIBRATION")
    print("-" * 80)
    print(f"  Phase Name           : {mp['phase_name']}")
    print(f"  Illumination         : {mp['illumination_pct']:10.2f} %")
    print(f"  Age (since New Moon) : {mp['age_days']:10.3f} days")
    print(f"  Elongation           : {mp['elongation_deg']:10.2f} deg")
    print(f"  Phase Angle          : {mp['phase_angle_deg']:10.2f} deg")
    lib = data['lunar_libration']
    print(f"  P1 (optical lib.)    : {lib['p1_deg']:12.6f} deg")
    print(f"  P2 (optical lib.)    : {lib['p2_deg']:12.6f} deg")
    print(f"  Tau (physical lib.)  : {lib['tau_deg']:12.6f} deg")

    # =====================================================================
    # 10. KINEMATIC PLATE MOTION
    # =====================================================================
    print("\n[10] KINEMATIC PLATE MOTION (ITRF2020-PMM / Altamimi et al. 2023)")
    print("-" * 80)
    print(f"  Tectonic Plate       : {data['plate_code']} (Eurasian)")
    print(f"  Reference Epoch      : J2000.0 (ΔT = {data['plate_epoch_diff_years']:.4f} yr)")
    print(f"  Velocity (Ve)        : {data['plate_ve_mmyr']:12.4f} mm/yr  (East)")
    print(f"  Velocity (Vn)        : {data['plate_vn_mmyr']:12.4f} mm/yr  (North)")
    print(f"  Velocity (Vu)        : {data['plate_vu_mmyr']:12.4f} mm/yr  (Up, ORB=0)")
    print(f"  Horizontal Speed     : {data['plate_horiz_speed_mmyr']:12.4f} mm/yr")
    print(f"  Horizontal Azimuth   : {data['plate_azimuth_deg']:12.4f} deg  (True N)")
    print(f"  Accumulated Disp. (E): {data['plate_disp_e_mm']:12.4f} mm")
    print(f"  Accumulated Disp. (N): {data['plate_disp_n_mm']:12.4f} mm")
    print(f"  Accumulated Disp. (U): {data['plate_disp_u_mm']:12.4f} mm")
    print(f"  Note                 : Vertical ORB discarded (Altamimi 2023)")

    # =====================================================================
    # 11. OBSERVER & REDUCTION NOTES
    # =====================================================================
    print("\n[11] OBSERVER & REDUCTION NOTES")
    print("-" * 80)
    print(f"  Geodetic Latitude    : {OBSERVER_LAT_DEG:+.6f} deg")
    print(f"  Geodetic Longitude   : {OBSERVER_LON_DEG:+.6f} deg")
    print(f"  Orthometric Height   : {OBSERVER_HEIGHT_M:.3f} m (EGM2008)")
    print(f"  Refraction Model     : VMF3 + GPT3 (mapping functions + APG gradient)")
    print(f"  EOP Source File      : {data.get('eop_file', 'unknown')}")
    print(f"  EOP Last Observation : {data.get('eop_last_date', 'Unknown')}")
    print(f"  Ephemeris Sources    : IPS2000 (Sun/EMB), ELP/MPP02 (Moon)")
    print(f"  Calendar Input       : {data.get('calendar_input', 'gregorian')}")

    print("\n" + "=" * 80)
    print(" REPORT GENERATED SUCCESSFULLY – EXITING.")
    print("=" * 80 + "\n")

# ============================================================================
# 6. MAIN
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute high-precision ephemeris for a given date/time (IPS2000 engine).",
        epilog="Examples:\n"
               "  python AstroCalc_IPS2000.py                        # interactive\n"
               "  python AstroCalc_IPS2000.py 2026-06-17T15:30:00    # Gregorian (default)\n"
               "  python AstroCalc_IPS2000.py 1041-11-06T04:00:00 --calendar julian\n"
               "  python AstroCalc_IPS2000.py -4713-01-01T12:00:00   # negative year"
    )
    parser.add_argument(
        "date", nargs='?', default=None,
        help="ISO 8601 date/time. If omitted, interactive prompt."
    )
    parser.add_argument(
        "--calendar", choices=['gregorian', 'julian', 'auto'], default='auto',
        help="Input calendar: 'gregorian', 'julian', or 'auto' (Julian if year < 1582)."
    )
    parser.add_argument(
        "--eop", default="EOP_20u24_C04_one_file_1962-now.txt",
        help="EOP file path."
    )
    parser.add_argument(
        "--lat", type=float, default=OBSERVER_LAT_DEG,
        help=f"Observer latitude (deg, default: {OBSERVER_LAT_DEG})"
    )
    parser.add_argument(
        "--lon", type=float, default=OBSERVER_LON_DEG,
        help=f"Observer longitude (deg, default: {OBSERVER_LON_DEG})"
    )
    parser.add_argument(
        "--height", type=float, default=OBSERVER_HEIGHT_M,
        help=f"Observer height (m, default: {OBSERVER_HEIGHT_M})"
    )
    args = parser.parse_args()

    if args.date is None:
        print("Enter date/time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)")
        print("  Examples: 2026-06-17T15:30:00  or  -4713-01-01T12:00:00")
        print("  Press Enter to use current UTC time.")
        user_input = input("> ").strip()
        if user_input == "":
            dt = now_utc()
            iso_str = dt.to_iso()
            print(f"Using current UTC time: {iso_str}")
        else:
            iso_str = user_input
    else:
        iso_str = args.date

    try:
        data = compute_for_date(iso_str, args.calendar, args.eop,
                                args.lat, args.lon, args.height)
        print_report(data)
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()