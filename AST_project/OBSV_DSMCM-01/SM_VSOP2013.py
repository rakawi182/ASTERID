#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SM_VSOP2013.py – High‑Precision Astronomical Ephemeris Orchestrator (VSOP2013)
================================================================================
Menggunakan VSOP2013 (truncation 1e‑12) untuk posisi Matahari dan ELP/MPP02
untuk Bulan. Semua koreksi IERS 2010 (CIP‑CIO, nutasi, presesi, EOP, FCN),
light‑time, aberasi, defleksi gravitasi, refraksi (GPT3+VMF3), deformasi
kerak (solid tide, ocean tide, pole tide, loading), Defleksi Vertikal (DoV)
dinamis dari EGM2008 lokal, dan koreksi ketinggian elipsoidal untuk paralaks.

Perbedaan utama dengan SM_IPS2000emb.py:
    - Sumber ephemeris Matahari : VSOP2013 (bukan IPS2000)
    - Kecepatan Bumi dihitung dengan diferensiasi numerik (beda hingga)
    - Tetap menggunakan ELP/MPP02 untuk Bulan
    - Semua fungsionalitas lain identik (DoV, deformasi, refraksi, dll.)

Dependencies:
    - VSOP2013.py (dengan file VSOP2013p3_10e12.dat)
    - ELP_MPP02_full.py
    - llib04.py
    - Timescales.py, EarthRotation.py, Coord_Transform.py
    - Atmospheric_refraction.py, StationDispl.py, Site_Geophysic.py
    - IPS2000_emb (tidak digunakan lagi)
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ---------------------------------------------------------------------------
# IMPOR MODUL GEODINAMIKA & KOREKSI LOKAL JOLOTUNDO
# ---------------------------------------------------------------------------
from Timescales import (
    J2000_JD, split_jd, combine_jd,
    tt_to_tdb, delta_t_from_jd, tai_utc
)

# --- GANTI: impor VSOP2013, bukan IPS2000 ---
from VSOP2013 import VSOP2013
# --- tetap pakai ELP untuk Bulan ---
import ELP_MPP02_full
from EarthRotation import (
    get_cip_xy, get_cio_s, Q_matrix, Q_inverse, R_matrix, W_matrix,
    era_from_ut1, EarthOrientation, ARCSEC_TO_RAD, equation_of_origins,
    light_deflection_sun, diurnal_aberration,
    bias_precession_nutation_matrix, precession_angles_2006, nutation_2000a
)
from Coord_Transform import (
    CoordinateTransformer, unit_vector, cartesian_to_spherical,
    rot_x, rot_y, rot_z
)
from Atmospheric_refraction import (
    SITE_LAT_DEG, SITE_LON_DEG, SITE_ELEV_M,
    SITE_ELEV_ELLIPSOIDAL_M,
    JOLOTUNDO_GEOPHYSICS
)

# Injeksi Modul Presisi Tinggi IERS 2010 Bab 7 & 11
from StationDispl import StationDisplacement, JOLOTUNDO_FES2014_BLQ
from Site_Geophysic import LoadingResolver, LocalEGM2008Grids, AdvancedGeoidInversion

# Modul Lunar Libration
from llib04 import LLIB04

# ============================================================================
# KONSTANTA FISIKA (IERS 2010)
# ============================================================================
C_AUDAY = 173.1446326846693
C_KM_S = 299792.458
AU2KM = 149597870.7
KM2AU = 1.0 / AU2KM
MU = 1.0 / (1.0 + 81.30056907419)
TWO_PI = 2.0 * math.pi

GM_SUN_KM3_S2 = 1.32712442099e11
GM_EARTH_KM3_S2 = 398600.4418

# ---------------------------------------------------------------------------
# MATRIKS FRAME BIAS
# ---------------------------------------------------------------------------
def frame_bias_matrix() -> np.ndarray:
    dpsibi = -0.041775 * ARCSEC_TO_RAD
    depsbi = -0.0068192 * ARCSEC_TO_RAD
    dra    = -0.0146   * ARCSEC_TO_RAD
    return rot_x(-depsbi) @ rot_y(dpsibi) @ rot_z(-dra)

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

def normalize_angle(angle: float) -> float:
    return angle % TWO_PI

# ============================================================================
# ORCHESTRATOR CLASS (VSOP2013 untuk Matahari)
# ============================================================================
class AstronomicalEphemeris:
    def __init__(self, eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt",
                 load_elp_series: bool = True,
                 geoid_grid_dir: str = ".",
                 vsop_file: str = "VSOP2013p3_10e12.dat"):
        """
        Parameters
        ----------
        vsop_file : str
            Path ke file data VSOP2013 (truncation 1e‑12).
        """
        self.transformer = CoordinateTransformer(eop_file)
        self.eo_obj = self.transformer.eo

        # --- Inisialisasi VSOP2013 (Matahari) ---
        self.vsop = VSOP2013(vsop_file)

        # Inisialisasi Non-Tidal Loading Resolver
        self.loading_resolver = LoadingResolver(data_dir=".")

        if load_elp_series:
            if not ELP_MPP02_full.SERIES['long'].main:
                ELP_MPP02_full.load_all_series()

        self.bias_mat = frame_bias_matrix()
        self.mu = MU

        # ===================================================================
        # INISIALISASI DoV ENGINE (dinamis dari grid EGM2008 lokal)
        # ===================================================================
        self.dov_engine = None
        try:
            grids = LocalEGM2008Grids(data_dir=geoid_grid_dir)
            self.dov_engine = AdvancedGeoidInversion(grids)
            print("✅ DoV Engine initialized from local EGM2008 grids.")
        except Exception as e:
            print(f"⚠️ DoV Engine fallback to static constants: {e}")
            self._static_xi = JOLOTUNDO_GEOPHYSICS['vertical_deflections_arcsec']['xi_meridional']
            self._static_eta = JOLOTUNDO_GEOPHYSICS['vertical_deflections_arcsec']['eta_prime_vertical']

        try:
            self.lib = LLIB04()
        except Exception as e:
            print(f"⚠️ Gagal memuat LLIB04: {e}")
            self.lib = None

    # -----------------------------------------------------------------------
    # Metode Pembantu Waktu
    # -----------------------------------------------------------------------
    def tt_to_tdb(self, tt_jd: float) -> float:
        jd1, jd2 = split_jd(tt_jd)
        tdb1, tdb2 = tt_to_tdb(jd1, jd2)
        return combine_jd(tdb1, tdb2)

    # -----------------------------------------------------------------------
    # Metode Ephemeris Dasar (VSOP2013 untuk Matahari)
    # -----------------------------------------------------------------------
    def earth_heliocentric_ecliptic(self, tt_jd: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute heliocentric position and velocity of the Earth (not EMB)
        in the J2000 ecliptic frame, using VSOP2013 for the Sun and ELP/MPP02
        for the Moon.

        Returns
        -------
        earth_pos : np.ndarray, shape (3,)
            Heliocentric position of Earth (AU)
        earth_vel : np.ndarray, shape (3,)
            Heliocentric velocity of Earth (AU/day)
        """
        tdb = self.tt_to_tdb(tt_jd)

        # 1. Get geocentric Sun position and velocity from VSOP2013
        sun_pos, sun_vel = self.vsop.get_sun_geocentric_ecliptic(tdb)

        # 2. Heliocentric Earth (EMB) = -Sun
        emb_pos = -sun_pos
        emb_vel = -sun_vel

        # 3. Get Moon geocentric position and velocity from ELP/MPP02
        tj = tdb - J2000_JD
        xyz = ELP_MPP02_full.elpmpp02(tj, icor=1)
        moon_pos = np.array([xyz[0], xyz[1], xyz[2]]) / AU2KM
        moon_vel = np.array([xyz[3], xyz[4], xyz[5]]) / AU2KM

        # 4. Correct from EMB to Earth: Earth = EMB - MU * Moon
        earth_pos = emb_pos - self.mu * moon_pos
        earth_vel = emb_vel - self.mu * moon_vel

        return earth_pos, earth_vel

    def sun_geocentric_gcrs(self, tt_jd: float, unit: bool = False) -> np.ndarray:
        earth_pos_ecl, _ = self.earth_heliocentric_ecliptic(tt_jd)
        sun_pos_ecl = -earth_pos_ecl
        # Rotasi ekliptika J2000 → FK5 (mean equator J2000)
        R = rot_x(-84381.406 * ARCSEC_TO_RAD)
        sun_pos_fk5 = R @ sun_pos_ecl
        # Frame bias → GCRS
        sun_pos_gcrs = self.bias_mat @ sun_pos_fk5
        if unit:
            return unit_vector(sun_pos_gcrs)
        return sun_pos_gcrs * AU2KM

    def moon_geocentric_gcrs(self, tt_jd: float, unit: bool = False) -> np.ndarray:
        tdb = self.tt_to_tdb(tt_jd)
        tj = tdb - J2000_JD
        xyz_ecl = ELP_MPP02_full.elpmpp02(tj, icor=1)
        xyz_eq = ELP_MPP02_full.ecliptic_to_equator(xyz_ecl)
        pos_fk5 = np.array([xyz_eq[0], xyz_eq[1], xyz_eq[2]])
        pos_gcrs = self.bias_mat @ pos_fk5
        if unit:
            return unit_vector(pos_gcrs)
        return pos_gcrs

    def lunar_libration(self, tt_jd: float) -> Dict[str, float]:
        """
        Menghitung librasi Bulan (P1, P2, Tau) dalam radian dan derajat.
        Menggunakan model LLIB04 (analytical part) dengan TDB.
        """
        if self.lib is None:
            return {'p1_rad': 0.0, 'p2_rad': 0.0, 'tau_rad': 0.0,
                    'p1_deg': 0.0, 'p2_deg': 0.0, 'tau_deg': 0.0}
        tdb = self.tt_to_tdb(tt_jd)
        p1_rad, p2_rad, tau_rad = self.lib.compute(tdb)
        return {
            'p1_rad': p1_rad,
            'p2_rad': p2_rad,
            'tau_rad': tau_rad,
            'p1_deg': math.degrees(p1_rad),
            'p2_deg': math.degrees(p2_rad),
            'tau_deg': math.degrees(tau_rad),
        }

    def earth_velocity_gcrs(self, tt_jd: float) -> np.ndarray:
        # Ambil kecepatan Bumi dari earth_heliocentric_ecliptic (sudah dalam AU/hari)
        _, v_earth_ecl = self.earth_heliocentric_ecliptic(tt_jd)
        # Rotasi ekliptika J2000 → FK5
        R = rot_x(-84381.406 * ARCSEC_TO_RAD)
        v_earth_fk5 = R @ v_earth_ecl
        # Frame bias → GCRS
        return self.bias_mat @ v_earth_fk5

    # -----------------------------------------------------------------------
    # Helper ITRF Sun/Moon untuk deformasi pasang surut
    # -----------------------------------------------------------------------
    def _get_itrf_sun_moon_approx(self, tt_jd: float, ut1_jd: float,
                                  xp: float, yp: float) -> Tuple[np.ndarray, np.ndarray]:
        sun_gcrs_m = self.sun_geocentric_gcrs(tt_jd, unit=False) * 1000.0
        moon_gcrs_m = self.moon_geocentric_gcrs(tt_jd, unit=False) * 1000.0

        X, Y = get_cip_xy(tt_jd, apply_fcn=True)
        s = get_cio_s(tt_jd, X, Y)
        era = era_from_ut1(ut1_jd)

        Q_inv = Q_inverse(X, Y, s)
        W_inv = rot_y(-xp) @ rot_x(-yp)
        R_inv = rot_z(era)

        itrs_mat = W_inv @ R_inv @ Q_inv
        return itrs_mat @ sun_gcrs_m, itrs_mat @ moon_gcrs_m

    # -----------------------------------------------------------------------
    # OBSERVER GCRS (dengan koreksi ketinggian elipsoidal)
    # -----------------------------------------------------------------------
    def observer_gcrs(self, tt_jd: float, lat_rad: float, lon_rad: float,
                      height_m: float, h_ellip: Optional[float] = None,
                      xp_rad: float = 0.0, yp_rad: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Menghitung posisi dan kecepatan observer dalam GCRS.

        Parameters
        ----------
        height_m : float
            Ketinggian ortometrik (untuk refraksi dan deformasi).
        h_ellip : float, optional
            Ketinggian elipsoidal (untuk geometri XYZ). Jika None, gunakan height_m.
        """
        eop = self.eo_obj.get_eop_corrections(tt_jd)
        ut1_jd = self.eo_obj.ut1_jd_from_tt(tt_jd, eop['dut1'])
        xp = xp_rad if xp_rad != 0.0 else eop['xp']
        yp = yp_rad if yp_rad != 0.0 else eop['yp']

        # --- 1. Geodetik → ITRS Statis ---
        a_eq = 6378136.6
        f = 1.0 / 298.25642
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)

        # Gunakan h_ellip untuk geometri kartesian (paralaks)
        if h_ellip is None:
            h_ellip = height_m  # fallback

        u = math.sqrt(a_eq**2 * cos_lat**2 + (a_eq**2 * (1-f)**2) * sin_lat**2)

        x_itrs = (u + h_ellip) * cos_lat * cos_lon
        y_itrs = (u + h_ellip) * cos_lat * sin_lon
        z_itrs = (u * (1-f)**2 + h_ellip) * sin_lat
        pos_itrs_m = np.array([x_itrs, y_itrs, z_itrs], dtype=np.float64)

        # --- Injeksi Deformasi (Tidal, Loading, dll.) ---
        sun_itrf, moon_itrf = self._get_itrf_sun_moon_approx(tt_jd, ut1_jd, xp, yp)
        station_engine = StationDisplacement(pos_itrs_m, JOLOTUNDO_FES2014_BLQ)
        disp_tidal_m = station_engine.total_displacement(
            tt_jd, ut1_jd, xp, yp, sun_itrf, moon_itrf, include_atm=True
        )

        mjd_utc = (tt_jd - 2400000.5) - (tai_utc(tt_jd - 2400000.5) + 32.184) / 86400.0
        de_res, dn_res, du_res = self.loading_resolver.resolve_loading_at_mjd(mjd_utc)

        disp_nt_m = np.zeros(3, dtype=np.float64)
        disp_nt_m[0] = -sin_lon * de_res - sin_lat * cos_lon * dn_res + cos_lat * cos_lon * du_res
        disp_nt_m[1] =  cos_lon * de_res - sin_lat * sin_lon * dn_res + cos_lat * sin_lon * du_res
        disp_nt_m[2] =  cos_lat * dn_res + sin_lat * du_res

        pos_itrs_true_km = (pos_itrs_m + disp_tidal_m + disp_nt_m) / 1000.0

        # --- 2. ITRS → GCRS (CIP-CIO) ---
        X, Y = get_cip_xy(tt_jd, apply_fcn=True)
        s = get_cio_s(tt_jd, X, Y)
        era = era_from_ut1(ut1_jd)
        sp = 0.0
        Q = Q_matrix(X, Y, s)
        R = R_matrix(era)
        W = W_matrix(xp, yp, sp)

        pos_tirs = W @ pos_itrs_true_km
        pos_cirs = R @ pos_tirs
        pos_gcrs = Q @ pos_cirs

        # --- 3. Observer Velocity ---
        omega = 1.00273781191135448 * TWO_PI / 86400.0
        v_cirs_km_s = np.array([-omega * pos_cirs[1], omega * pos_cirs[0], 0.0])
        v_gcrs_km_s = Q @ v_cirs_km_s
        v_gcrs_km_day = v_gcrs_km_s * 86400.0

        v_earth_au = self.earth_velocity_gcrs(tt_jd)
        v_earth_km_day = v_earth_au * AU2KM
        v_gcrs_km_day += v_earth_km_day

        return pos_gcrs, v_gcrs_km_day, ut1_jd

    # -----------------------------------------------------------------------
    # KOREKSI VERTICAL DEFLECTION (DoV) – Dinamis + Fallback Statis
    # -----------------------------------------------------------------------
    def _apply_vertical_deflection(self, e: float, n: float, u: float,
                                   lat_deg: float, lon_deg: float) -> Tuple[float, float, float]:
        """
        Mengoreksi vektor ENU (Ellipsoidal) menjadi ENU Astronomis (Plumb Line)
        menggunakan Defleksi Vertikal (DoV) dari inversi gravimetri lokal.
        """
        if self.dov_engine is not None:
            try:
                dov = self.dov_engine.get_astro_geodetic_deflection(lat_deg, lon_deg)
                xi_arcsec = dov['xi_arcsec']
                eta_arcsec = dov['eta_arcsec']
            except Exception:
                xi_arcsec = self._static_xi
                eta_arcsec = self._static_eta
        else:
            xi_arcsec = self._static_xi
            eta_arcsec = self._static_eta

        xi_rad = math.radians(xi_arcsec / 3600.0)
        eta_rad = math.radians(eta_arcsec / 3600.0)

        e_ast = e - eta_rad * u
        n_ast = n - xi_rad * u
        u_ast = eta_rad * e + xi_rad * n + u

        return e_ast, n_ast, u_ast

    # -----------------------------------------------------------------------
    # SHAPIRO DELAY
    # -----------------------------------------------------------------------
    def _apply_shapiro_delay(self, r_obs: float, r_body: float, dist_km: float, is_sun: bool) -> float:
        r_j2 = r_body if not is_sun else max(r_body, 696000.0)
        shapiro_earth = (2 * GM_EARTH_KM3_S2 / C_KM_S**3) * math.log(
            (r_obs + r_j2 + dist_km) / (max(1e-6, r_obs + r_j2 - dist_km))
        )
        if is_sun:
            return shapiro_earth
        else:
            shapiro_sun = (2 * GM_SUN_KM3_S2 / C_KM_S**3) * math.log(
                (r_obs + r_j2 + dist_km) / (max(1e-6, r_obs + r_j2 - dist_km))
            )
            return shapiro_earth + shapiro_sun

    # -----------------------------------------------------------------------
    # SUN APPARENT TOPOCENTRIC (dengan DoV & ketinggian elipsoidal)
    # -----------------------------------------------------------------------
    def sun_apparent_topocentric(self, tt_jd: float, lat_rad: float, lon_rad: float,
                                 height_m: float, apply_refraction: bool = False,
                                 refraction_model: str = 'vmf3') -> Dict[str, float]:
        # ---- Observer dengan ketinggian elipsoidal ----
        obs_pos_gcrs, obs_vel_gcrs, ut1_jd = self.observer_gcrs(
            tt_jd, lat_rad, lon_rad, height_m,
            h_ellip=SITE_ELEV_ELLIPSOIDAL_M
        )

        tdb = self.tt_to_tdb(tt_jd)
        tau = 0.0

        # ---- Light-Time Relativistik ----
        for _ in range(5):
            t_emit = tdb - tau / 86400.0
            # Gunakan VSOP2013 untuk posisi Matahari pada t_emit
            earth_pos_ecl, _ = self.earth_heliocentric_ecliptic(t_emit)
            sun_pos_ecl = -earth_pos_ecl
            R = rot_x(-84381.406 * ARCSEC_TO_RAD)
            sun_pos_fk5 = R @ sun_pos_ecl
            sun_pos_gcrs_emit = self.bias_mat @ sun_pos_fk5 * AU2KM
            obs_pos_emit = obs_pos_gcrs + obs_vel_gcrs * (tau / 86400.0)
            topo = sun_pos_gcrs_emit - obs_pos_emit
            dist_km = np.linalg.norm(topo)

            shapiro_dt = self._apply_shapiro_delay(
                np.linalg.norm(obs_pos_emit),
                np.linalg.norm(sun_pos_gcrs_emit),
                dist_km, is_sun=True
            )
            tau_new = (dist_km / C_KM_S) + shapiro_dt
            if abs(tau_new - tau) < 1e-11:
                tau = tau_new
                break
            tau = tau_new

        uv_topo = topo / dist_km

        # ---- Defleksi Gravitasi & Aberasi ----
        p_sun_gcrs = unit_vector(sun_pos_gcrs_emit)
        uv_deflect = light_deflection_sun(uv_topo, p_sun_gcrs)
        v_obs_au_day = obs_vel_gcrs / AU2KM
        uv_ab = stellar_aberration_full(uv_deflect, v_obs_au_day)

        # ---- Transformasi ke ITRS ----
        X, Y = get_cip_xy(tt_jd, apply_fcn=True)
        s = get_cio_s(tt_jd, X, Y)
        era = era_from_ut1(ut1_jd)
        eop = self.eo_obj.get_eop_corrections(tt_jd)

        Q_inv = Q_inverse(X, Y, s)
        uv_cirs = Q_inv @ uv_ab
        uv_tirs = rot_z(era) @ uv_cirs
        W = rot_z(0.0) @ rot_y(eop['xp']) @ rot_x(eop['yp'])
        uv_itrs = W.T @ uv_tirs

        # ---- Transformasi Horizontal (dengan DoV) ----
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
        x, y, z = uv_itrs

        e_ell = -sin_lon * x + cos_lon * y
        n_ell = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
        u_ell = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z

        # Koreksi DoV
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)
        e_ast, n_ast, u_ast = self._apply_vertical_deflection(e_ell, n_ell, u_ell, lat_deg, lon_deg)

        az = math.atan2(e_ast, n_ast)
        if az < 0:
            az += TWO_PI
        alt_geom = math.asin(max(-1.0, min(1.0, u_ast)))
        alt_app_deg = math.degrees(alt_geom)

        # ---- Refraksi (dengan ketinggian ortometrik) ----
        if apply_refraction:
            from Atmospheric_refraction import calculate_refraction
            mjd = tt_jd - 2400000.5
            alt_app_deg += calculate_refraction(
                alt_app_deg, model=refraction_model,
                mjd=mjd, lat_rad=lat_rad, lon_rad=lon_rad,
                height_m=height_m, az_rad=az
            )

        # ---- RA/Dec CIRS dan Equinox ----
        eo_rad = equation_of_origins(tt_jd)
        ra_cirs = math.atan2(uv_cirs[1], uv_cirs[0])
        dec_cirs = math.asin(max(-1.0, min(1.0, uv_cirs[2])))
        ra_app = normalize_angle(ra_cirs - eo_rad)   # Equinox-based RA
        dec_app = dec_cirs

        ra_cirs_deg = math.degrees(ra_cirs) % 360.0
        dec_cirs_deg = math.degrees(dec_cirs)
        ra_eqx_deg = math.degrees(ra_app) % 360.0
        dec_eqx_deg = math.degrees(dec_app)

        return {
            # Kompatibilitas lama (Equinox-based)
            'ra_deg': ra_eqx_deg,
            'dec_deg': dec_eqx_deg,
            # Koordinat CIRS murni
            'ra_cirs_deg': ra_cirs_deg,
            'dec_cirs_deg': dec_cirs_deg,
            # Koordinat Equinox-based (eksplisit)
            'ra_eqx_deg': ra_eqx_deg,
            'dec_eqx_deg': dec_eqx_deg,
            # Horizontal
            'az_deg': math.degrees(az),
            'alt_geom_deg': math.degrees(alt_geom),
            'alt_app_deg': alt_app_deg,
            'dist_km': dist_km,
            'dist_au': dist_km / AU2KM
        }

    # -----------------------------------------------------------------------
    # MOON APPARENT TOPOCENTRIC (dengan DoV & ketinggian elipsoidal)
    # -----------------------------------------------------------------------
    def moon_apparent_topocentric(self, tt_jd: float, lat_rad: float, lon_rad: float,
                                  height_m: float, apply_refraction: bool = False,
                                  refraction_model: str = 'vmf3') -> Dict[str, float]:
        # ---- Observer dengan ketinggian elipsoidal ----
        obs_pos_gcrs, obs_vel_gcrs, ut1_jd = self.observer_gcrs(
            tt_jd, lat_rad, lon_rad, height_m,
            h_ellip=SITE_ELEV_ELLIPSOIDAL_M
        )

        tdb = self.tt_to_tdb(tt_jd)
        tau = 0.0

        # ---- Light-Time Relativistik ----
        for _ in range(5):
            t_emit = tdb - tau / 86400.0
            tj = t_emit - J2000_JD
            xyz_ecl = ELP_MPP02_full.elpmpp02(tj, icor=1)
            xyz_eq = ELP_MPP02_full.ecliptic_to_equator(xyz_ecl)
            pos_fk5 = np.array([xyz_eq[0], xyz_eq[1], xyz_eq[2]])
            pos_gcrs_emit = self.bias_mat @ pos_fk5
            obs_pos_emit = obs_pos_gcrs + obs_vel_gcrs * (tau / 86400.0)
            topo = pos_gcrs_emit - obs_pos_emit
            dist_km = np.linalg.norm(topo)

            shapiro_dt = self._apply_shapiro_delay(
                np.linalg.norm(obs_pos_emit),
                np.linalg.norm(pos_gcrs_emit),
                dist_km, is_sun=False
            )
            tau_new = (dist_km / C_KM_S) + shapiro_dt
            if abs(tau_new - tau) < 1e-11:
                tau = tau_new
                break
            tau = tau_new

        uv_topo = topo / dist_km

        # ---- Defleksi Gravitasi & Aberasi ----
        sun_pos_emit = self.sun_geocentric_gcrs(t_emit, unit=False)
        p_sun_gcrs = unit_vector(sun_pos_emit)
        uv_deflect = light_deflection_sun(uv_topo, p_sun_gcrs)
        v_obs_au_day = obs_vel_gcrs / AU2KM
        uv_ab = stellar_aberration_full(uv_deflect, v_obs_au_day)

        # ---- Transformasi ke ITRS ----
        X, Y = get_cip_xy(tt_jd, apply_fcn=True)
        s = get_cio_s(tt_jd, X, Y)
        era = era_from_ut1(ut1_jd)
        eop = self.eo_obj.get_eop_corrections(tt_jd)

        Q_inv = Q_inverse(X, Y, s)
        uv_cirs = Q_inv @ uv_ab
        uv_tirs = rot_z(era) @ uv_cirs
        W = rot_z(0.0) @ rot_y(eop['xp']) @ rot_x(eop['yp'])
        uv_itrs = W.T @ uv_tirs

        # ---- Transformasi Horizontal (dengan DoV) ----
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
        x, y, z = uv_itrs

        e_ell = -sin_lon * x + cos_lon * y
        n_ell = -sin_lat * cos_lon * x - sin_lat * sin_lon * y + cos_lat * z
        u_ell = cos_lat * cos_lon * x + cos_lat * sin_lon * y + sin_lat * z

        # Koreksi DoV
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)
        e_ast, n_ast, u_ast = self._apply_vertical_deflection(e_ell, n_ell, u_ell, lat_deg, lon_deg)

        az = math.atan2(e_ast, n_ast)
        if az < 0:
            az += TWO_PI
        alt_geom = math.asin(max(-1.0, min(1.0, u_ast)))
        alt_app_deg = math.degrees(alt_geom)

        # ---- Refraksi (dengan ketinggian ortometrik) ----
        if apply_refraction:
            from Atmospheric_refraction import calculate_refraction
            mjd = tt_jd - 2400000.5
            alt_app_deg += calculate_refraction(
                alt_app_deg, model=refraction_model,
                mjd=mjd, lat_rad=lat_rad, lon_rad=lon_rad,
                height_m=height_m, az_rad=az
            )

        # ---- RA/Dec CIRS dan Equinox ----
        eo_rad = equation_of_origins(tt_jd)
        ra_cirs = math.atan2(uv_cirs[1], uv_cirs[0])
        dec_cirs = math.asin(max(-1.0, min(1.0, uv_cirs[2])))
        ra_app = normalize_angle(ra_cirs - eo_rad)   # Equinox-based RA
        dec_app = dec_cirs

        ra_cirs_deg = math.degrees(ra_cirs) % 360.0
        dec_cirs_deg = math.degrees(dec_cirs)
        ra_eqx_deg = math.degrees(ra_app) % 360.0
        dec_eqx_deg = math.degrees(dec_app)

        return {
            # Kompatibilitas lama (Equinox-based)
            'ra_deg': ra_eqx_deg,
            'dec_deg': dec_eqx_deg,
            # Koordinat CIRS murni
            'ra_cirs_deg': ra_cirs_deg,
            'dec_cirs_deg': dec_cirs_deg,
            # Koordinat Equinox-based (eksplisit)
            'ra_eqx_deg': ra_eqx_deg,
            'dec_eqx_deg': dec_eqx_deg,
            # Horizontal
            'az_deg': math.degrees(az),
            'alt_geom_deg': math.degrees(alt_geom),
            'alt_app_deg': alt_app_deg,
            'dist_km': dist_km,
            'dist_au': dist_km / AU2KM
        }

    # -----------------------------------------------------------------------
    # Ecliptic of Date (true ecliptic & equinox of date)
    # -----------------------------------------------------------------------
    def sun_ecliptic_of_date(self, tt_jd: float, unit: bool = False,
                             degrees: bool = False):
        """
        Hitung bujur, lintang, dan jarak Matahari dalam sistem ekliptika tanggal.
        """
        pos_gcrs = self.sun_geocentric_gcrs(tt_jd, unit=unit)
        return self._ecliptic_of_date_from_gcrs(pos_gcrs, tt_jd, degrees)

    def moon_ecliptic_of_date(self, tt_jd: float, unit: bool = False,
                              degrees: bool = False):
        pos_gcrs = self.moon_geocentric_gcrs(tt_jd, unit=unit)
        return self._ecliptic_of_date_from_gcrs(pos_gcrs, tt_jd, degrees)

    def _ecliptic_of_date_from_gcrs(self, pos_gcrs, tt_jd, degrees=False):
        t_cy = (tt_jd - J2000_JD) / 36525.0
        pre = precession_angles_2006(t_cy)
        dpsi, deps = nutation_2000a(tt_jd)
        eps0 = 84381.406 * ARCSEC_TO_RAD

        P = rot_z(-pre['chiA']) @ rot_x(pre['omegaA']) @ rot_z(-pre['psiA']) @ rot_x(eps0)
        N = rot_x(-pre['epsA']) @ rot_z(-dpsi) @ rot_x(pre['epsA'] + deps)
        PN = N @ P
        pos_true_eq = PN @ pos_gcrs
        eps_true = pre['epsA'] + deps
        pos_ecl = rot_x(-eps_true) @ pos_true_eq
        lon, lat, r = cartesian_to_spherical(pos_ecl)
        if degrees:
            lon = math.degrees(lon) % 360.0
            lat = math.degrees(lat)
        return lon, lat, r

    # -----------------------------------------------------------------------
    # Metode tambahan untuk kompatibilitas dengan modul lain
    # -----------------------------------------------------------------------
    def sun_radec(self, tt_jd: float, apparent: bool = False,
                  lat_rad: float = 0.0, lon_rad: float = 0.0,
                  height_m: float = 0.0) -> Tuple[float, float]:
        if apparent:
            res = self.sun_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction=False)
            return math.radians(res['ra_deg']), math.radians(res['dec_deg'])
        pos = self.sun_geocentric_gcrs(tt_jd, unit=True)
        ra, dec, _ = cartesian_to_spherical(pos)
        return normalize_angle(ra), dec

    def moon_radec(self, tt_jd: float, apparent: bool = False,
                   lat_rad: float = 0.0, lon_rad: float = 0.0,
                   height_m: float = 0.0) -> Tuple[float, float]:
        if apparent:
            res = self.moon_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction=False)
            return math.radians(res['ra_deg']), math.radians(res['dec_deg'])
        pos = self.moon_geocentric_gcrs(tt_jd, unit=True)
        ra, dec, _ = cartesian_to_spherical(pos)
        return normalize_angle(ra), dec

    def sun_horizontal(self, tt_jd: float, lat_rad: float, lon_rad: float,
                       height_m: float, apply_refraction: bool = False,
                       refraction_model: str = 'vmf3') -> Tuple[float, float]:
        res = self.sun_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m,
                                            apply_refraction, refraction_model)
        return math.radians(res['az_deg']), math.radians(res['alt_app_deg'])

    def moon_horizontal(self, tt_jd: float, lat_rad: float, lon_rad: float,
                        height_m: float, apply_refraction: bool = False,
                        refraction_model: str = 'vmf3') -> Tuple[float, float]:
        res = self.moon_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m,
                                             apply_refraction, refraction_model)
        return math.radians(res['az_deg']), math.radians(res['alt_app_deg'])


# ============================================================================
# SELF-TEST
# ============================================================================
if __name__ == "__main__":
    from Timescales import delta_t_from_jd
    from Atmospheric_refraction import SITE_LAT_DEG, SITE_LON_DEG, SITE_ELEV_M

    lat_rad = math.radians(SITE_LAT_DEG)
    lon_rad = math.radians(SITE_LON_DEG)
    height_m = SITE_ELEV_M
    tt_jd = J2000_JD - 0.25

    print("Initializing High-Precision Orchestrator with VSOP2013, DoV & Ellipsoidal Height...")
    ephem = AstronomicalEphemeris(eop_file="EOP_20u24_C04_one_file_1962-now.txt", geoid_grid_dir=".")

    delta_t = delta_t_from_jd(tt_jd)
    ut1_jd = tt_jd - (delta_t / 86400.0)

    sun_gcrs_pos = ephem.sun_geocentric_gcrs(tt_jd, unit=True)
    sun_ra_gcrs, sun_dec_gcrs, _ = cartesian_to_spherical(sun_gcrs_pos)
    sun_app = ephem.sun_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction=True)

    moon_gcrs_pos = ephem.moon_geocentric_gcrs(tt_jd, unit=True)
    moon_ra_gcrs, moon_dec_gcrs, _ = cartesian_to_spherical(moon_gcrs_pos)
    moon_app = ephem.moon_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction=True)

    print("\n" + "=" * 80)
    print(" ASTROMETRIC REDUCTION REPORT – HIGH PRECISION IERS 2010")
    print(" WITH VSOP2013 (SUN), ELP/MPP02 (MOON), DOPPLER (DoV) & ELLIPSOIDAL HEIGHT")
    print("=" * 80)

    print("\n [ EPOCH & STATION PARAMETERS ]")
    print("-" * 80)
    print(f"  Target Julian Date (TT) : {tt_jd:.9f}")
    print(f"  Target Julian Date (UT1): {ut1_jd:.9f} (ΔT = {delta_t:.3f} s)")
    print(f"  Geodetic Coordinates    : Lat {SITE_LAT_DEG:+.6f}°, Lon {SITE_LON_DEG:+.6f}°")
    print(f"  Orthometric Elevation   : {SITE_ELEV_M:.3f} m")
    print(f"  Ellipsoidal Elevation   : {SITE_ELEV_ELLIPSOIDAL_M:.3f} m")

    print("\n [ SUN (SOLAR) KINEMATICS ]")
    print("-" * 80)
    print("  GCRS (Geocentric Astrometric - J2000.0 Mean Equator)")
    print(f"    Right Ascension (α)   : {math.degrees(normalize_angle(sun_ra_gcrs)):12.6f}°")
    print(f"    Declination (δ)       : {math.degrees(sun_dec_gcrs):12.6f}°")
    print("\n  CIRS (Celestial Intermediate - Apparent Topocentric)")
    print(f"    Right Ascension (α)   : {sun_app['ra_cirs_deg']:12.6f}°")
    print(f"    Declination (δ)       : {sun_app['dec_cirs_deg']:12.6f}°")
    print("\n  Equinox (True Equator & Equinox of Date)")
    print(f"    Right Ascension (α)   : {sun_app['ra_eqx_deg']:12.6f}°")
    print(f"    Declination (δ)       : {sun_app['dec_eqx_deg']:12.6f}°")
    print(f"    Vector Distance (Δ)   : {sun_app['dist_au']:12.9f} AU  |  {sun_app['dist_km']:15.3f} km")
    print("\n  Topocentric Horizontal Frame (DoV-corrected)")
    print(f"    Azimuth (True North)  : {sun_app['az_deg']:12.6f}°")
    print(f"    Altitude (Geometric)  : {sun_app['alt_geom_deg']:12.6f}°")
    print(f"    Altitude (Refracted)  : {sun_app['alt_app_deg']:12.6f}°")

    print("\n [ MOON (LUNAR) KINEMATICS ]")
    print("-" * 80)
    print("  GCRS (Geocentric Astrometric - J2000.0 Mean Equator)")
    print(f"    Right Ascension (α)   : {math.degrees(normalize_angle(moon_ra_gcrs)):12.6f}°")
    print(f"    Declination (δ)       : {math.degrees(moon_dec_gcrs):12.6f}°")
    print("\n  CIRS (Celestial Intermediate - Apparent Topocentric)")
    print(f"    Right Ascension (α)   : {moon_app['ra_cirs_deg']:12.6f}°")
    print(f"    Declination (δ)       : {moon_app['dec_cirs_deg']:12.6f}°")
    print("\n  Equinox (True Equator & Equinox of Date)")
    print(f"    Right Ascension (α)   : {moon_app['ra_eqx_deg']:12.6f}°")
    print(f"    Declination (δ)       : {moon_app['dec_eqx_deg']:12.6f}°")
    print(f"    Vector Distance (Δ)   : {moon_app['dist_au']:12.9f} AU  |  {moon_app['dist_km']:15.3f} km")
    print("\n  Topocentric Horizontal Frame (DoV-corrected)")
    print(f"    Azimuth (True North)  : {moon_app['az_deg']:12.6f}°")
    print(f"    Altitude (Geometric)  : {moon_app['alt_geom_deg']:12.6f}°")
    print(f"    Altitude (Refracted)  : {moon_app['alt_app_deg']:12.6f}°")

    # Koordinat ekliptika tanggal (true ecliptic of date)
    sun_ecl = ephem.sun_ecliptic_of_date(tt_jd, unit=False, degrees=True)
    moon_ecl = ephem.moon_ecliptic_of_date(tt_jd, unit=False, degrees=True)

    print("\n [ SUN (ECLIPTIC OF DATE) ]")
    print("-" * 80)
    print(f"  Ecliptic Longitude (λ) : {sun_ecl[0]:12.6f}°")
    print(f"  Ecliptic Latitude (β)  : {sun_ecl[1]:12.6f}°")
    print(f"  Geocentric Distance    : {sun_ecl[2]:15.3f} km")

    print("\n [ MOON (ECLIPTIC OF DATE) ]")
    print("-" * 80)
    print(f"  Ecliptic Longitude (λ) : {moon_ecl[0]:12.6f}°")
    print(f"  Ecliptic Latitude (β)  : {moon_ecl[1]:12.6f}°")
    print(f"  Geocentric Distance    : {moon_ecl[2]:15.3f} km")

    print("\n" + "=" * 80)
    print(" END OF EPHEMERIS REPORT")
    print("=" * 80 + "\n")
