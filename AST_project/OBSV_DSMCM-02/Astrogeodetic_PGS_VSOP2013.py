#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTERID_PGS_VSOP2013.py — Integrated Geodetic-Gravimetric Inversion Engine (VSOP2013 Edition)
===============================================================================================

Target          : Kompleks Candi Pawitra (Gunung Penanggungan), Jawa Timur
Architecture    : Standalone, Double-Precision (64-bit), Vectorized NumPy
Ephemeris Core  : VSOP2013 (Fienga et al., 2013) for the Sun + ELP/MPP02 for the Moon.

Scientific Rationale:
    This module replaces the previous IPS2000-based orchestrator with the VSOP2013
    semi-analytical planetary theory. VSOP2013 is fitted to the INPOP10a numerical
    integration over [1890, 2000], providing sub‑milliarcsecond accuracy for the
    Earth‑Moon barycenter. This update improves long‑term stability and reduces
    systematic uncertainties in archaeo‑astronomical positioning.

Functional Scope (IERS Conventions 2010 & IAU 2006/2000A):
    1. Absolute Solar & Lunar ephemerides (GCRS, CIRS, ITRS, Topocentric).
    2. Exact geoidal undulation inversion (Local XGM2019e-2159 grids).
    3. Dynamic vertical deflection (DoV) from local gravimetric inversion.
    4. High‑resolution terrain corrections (Nagy prism + line‑mass).
    5. 4‑D tropospheric ray‑tracing (GPT3 + VMF3) with ECMWF real-time data.
    6. Non‑tidal crustal loading (GFZ‑Potsdam harmonic resolver).
    7. Solid Earth tide, Ocean tide loading (FES2014b), Pole tide, and Atmospheric loading.
    8. OTL gravity & tilt corrections (FES2014b, specific to Jolotundo).
    9. ITRF2020 crustal kinematics (Altamimi et al., 2023).
    10. DORIS/IDS precise geodetic tie (ITRF2020 SINEX).
    11. Real-time atmospheric profile display from ECMWF IFS HRES.

Critical Unit Fix (vs. initial prototype):
    The VSOP2013 orchestrator (SM_VSOP2013.py) returns heliocentric/geocentric states
    in KILOMETERS. However, the CIP‑CIO transformation routines (gcrs_to_itrs_cip)
    and the Solid Earth Tide algorithms strictly require inputs in METERS.
    This module explicitly performs the km → m scaling before any astronomical
    or geodetic transformation, preventing catastrophic scaling errors.

References:
    [1] Fienga, A., et al. (2013). VSOP2013: A new planetary theory. IMCCE Tech. Note.
    [2] IERS Conventions (2010). IERS Technical Note 36.
    [3] Altamimi, Z., et al. (2023). ITRF2020 plate motion model. GRL, 50, e2023GL106373.
    [4] Paripurno, E. T., et al. (2018). Geological map of Mount Penanggungan.
        IOP Conf. Ser. Earth Environ. Sci. 212:012045.
"""

import sys
sys.dont_write_bytecode = True

import math
import textwrap
import numpy as np
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional

# ==============================================================================
# 1. DEPENDENCIES (IERS 2010, VSOP2013 Orchestrator, Station Displacement)
# ==============================================================================
from Timescales import (
    J2000_JD, MJD_ZERO, split_jd, combine_jd, cal_to_jd, jd_to_cal,
    tai_utc, delta_t_from_jd, tt_to_tdb, utc_to_ut1, ut1_to_tt,
)

from EarthRotation import (
    get_cip_xy, get_cio_s, get_tio_sp,
    Q_matrix, Q_inverse, R_matrix, R_inverse, W_matrix, W_inverse,
    gcrs_to_itrs_cip, itrs_to_gcrs_cip,
    EarthOrientation,
    ARCSEC_TO_RAD, MAS_TO_RAD, UAS_TO_RAD,
    light_deflection_sun, diurnal_aberration,
    bias_precession_nutation_matrix, precession_angles_2006, nutation_2000a,
    equation_of_origins, era_from_ut1,
    quaternion_to_matrix, quaternion_earth_rotation_exact,
    quaternion_earth_rotation_approx,
    gcrs_to_itrs_quaternion, itrs_to_gcrs_quaternion,
)

from StationDispl import (
    StationDisplacement,
    JOLOTUNDO_FES2014_BLQ,
    JOLOTUNDO_FES2014_GRAV_BLQ,
    Asterid342Engine_FES2014,
    solid_earth_tide,
    pole_tide,
    atm_loading_displacement,
    ocean_pole_tide_loading,
    geodetic_to_itrf,
    itrf_to_geodetic,
    mean_pole_iers2010,
    compute_fundamental_arguments,
)

# VSOP2013-based orchestrator (replaces SM_IPS2000emb)
from SM_VSOP2013 import AstronomicalEphemeris

from Site_Geophysic import (
    LocalEGM2008Grids,
    PawitraStratigraphy,
    LocalDEMEngine,
    AdvancedGeoidInversion,
    PawitraGeophysics,
    DynamicRayTracing,
    DORIS_ITRF_Engine,
    TectonicPlateKinematics,
    EmbeddedGeodeticNetwork,
    LoadingResolver,
)

# ECMWF real‑time module
import ecmwf_realtime as ecmwf

# ==============================================================================
# DISPLAY STYLE (aligned with DT18_VSOP2013_Realtime)
# ==============================================================================
W = 72
THICK_SEP = "═" * W
THIN_SEP  = "─" * W
BOLD = '\033[1m'
RESET = '\033[0m'

# ==============================================================================
# 2. GEODETIC & GEOPHYSICAL CONSTANTS (WGS84/GRS80/ITRF)
# ==============================================================================
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2
WGS84_OMEGA = 7.292115e-5

GAMMA_E = 9.7803267715
R_EARTH_MEAN = 6371000.0
GRAVITATIONAL_CONSTANT = 6.67430e-11

ITRF_A = 6378136.6
ITRF_F = 1.0 / 298.25642
ITRF_E2 = 2.0 * ITRF_F - ITRF_F**2

# ==============================================================================
# 3. CRITICAL WRAPPER: SUN & MOON ITRF POSITIONS (VSOP2013 with Unit Fix)
# ==============================================================================
def get_sun_moon_itrf_vsop2013(
    tt_jd: float,
    ut1_jd: float,
    xp_rad: float,
    yp_rad: float,
    dX_rad: float = 0.0,
    dY_rad: float = 0.0,
    vsop2013_file: str = "VSOP2013p3.dat"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute geocentric Sun and Moon positions in the ITRF (meters) using the
    VSOP2013 analytical ephemeris and the CIP-CIO transformation (IERS 2010).

    CRITICAL UNIT FIX:
        The underlying VSOP2013 orchestrator returns positions in KILOMETERS.
        Since the CIP-CIO transform (gcrs_to_itrs_cip) and the Solid Earth Tide
        routines (StationDisplacement) require METERS, we perform an explicit
        scaling by 1000.0 prior to transformation.

    Parameters:
        tt_jd (float): Terrestrial Time as Julian Date.
        ut1_jd (float): Universal Time UT1 as Julian Date.
        xp_rad, yp_rad (float): Polar motion coordinates (radians).
        dX_rad, dY_rad (float): Celestial pole offsets (radians).
        vsop2013_file (str): Path to the VSOP2013 full series file.

    Returns:
        sun_itrf, moon_itrf (np.ndarray): Position vectors in the ITRF (meters).
    """
    ephem = AstronomicalEphemeris(
        eop_file="EOP_20u24_C04_one_file_1962-now.txt",
        geoid_grid_dir=".",
        vsop_file=vsop2013_file
    )

    sun_gcrs_km = ephem.sun_geocentric_gcrs(tt_jd, unit=False)
    moon_gcrs_km = ephem.moon_geocentric_gcrs(tt_jd, unit=False)

    sun_gcrs_m = sun_gcrs_km * 1000.0
    moon_gcrs_m = moon_gcrs_km * 1000.0

    sun_itrf = gcrs_to_itrs_cip(
        sun_gcrs_m, tt_jd, ut1_jd, xp_rad, yp_rad, dX_rad, dY_rad
    )
    moon_itrf = gcrs_to_itrs_cip(
        moon_gcrs_m, tt_jd, ut1_jd, xp_rad, yp_rad, dX_rad, dY_rad
    )

    return sun_itrf, moon_itrf


# ==============================================================================
# 4. DORIS DEFORMATION ANALYSER (Secular Coordinate Extraction)
# ==============================================================================
class DORIS_Deformation_Analyzer:
    """
    Remove non-secular displacements (tides, loading, pole tide) from observed
    ITRF coordinates to recover the secular (tide-free) position.

    The correction follows the IERS Conventions 2010:
        pos_secular = pos_obs - (solid_tide + ocean_tide + pole_tide + atm_loading)

    This implementation utilizes VSOP2013 for the Sun's position, ensuring
    consistency with the updated ephemeris core.
    """

    def __init__(self, eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt",
                 vsop2013_file: str = "VSOP2013p3.dat"):
        self.eo = EarthOrientation(eop_file)
        self.loading_resolver = LoadingResolver(data_dir=".")
        self.vsop_file = vsop2013_file

    def extract_secular_coordinates(
        self,
        x_obs: float, y_obs: float, z_obs: float,
        lat_deg: float, lon_deg: float, h_ell_m: float,
        year: int, month: int, day: int, fhr: float,
        jd_tt: float, mjd_utc: float,
        xp_rad: float, yp_rad: float,
        dut1: float = 0.0,
        dX_rad: float = 0.0, dY_rad: float = 0.0
    ) -> Tuple[float, float, float]:
        """
        Extract secular (tide-free) coordinates from observed ITRF coordinates
        by removing solid Earth tides, ocean tide loading, pole tide, and
        non‑tidal atmospheric loading (using ESMGFZ only, not Ray & Ponte).
        """
        # 1. Compute UT1 from TT
        ut1_jd = self.eo.ut1_jd_from_tt(jd_tt, dut1)

        # 2. Get Sun and Moon positions in ITRF (VSOP2013)
        sun_itrf, moon_itrf = get_sun_moon_itrf_vsop2013(
            jd_tt, ut1_jd, xp_rad, yp_rad, dX_rad, dY_rad,
            vsop2013_file=self.vsop_file
        )

        # 3. Observed position vector
        pos_itrf = np.array([x_obs, y_obs, z_obs], dtype=np.float64)

        # 4. Station displacement (solid tide, ocean tide, atmospheric tide, pole tide)
        station = StationDisplacement(pos_itrf, blq_data=JOLOTUNDO_FES2014_BLQ)
        disp_tidal = station.total_displacement(
            jd_tt, ut1_jd, xp_rad, yp_rad, sun_itrf, moon_itrf,
            include_atm=True          # <--- KEMBALIKAN KE TRUE: Ray & Ponte (Tidal) dibutuhkan
        )

        # 5. Non‑tidal loading from ESMGFZ (atmosfer, ocean, hidrologi, sea level)
        de_res, dn_res, du_res = self.loading_resolver.resolve_loading_at_mjd(mjd_utc)

        # 6. Convert ENU loading to Cartesian
        lat_r = math.radians(lat_deg)
        lon_r = math.radians(lon_deg)
        sin_lat, cos_lat = math.sin(lat_r), math.cos(lat_r)
        sin_lon, cos_lon = math.sin(lon_r), math.cos(lon_r)

        disp_nt = np.array([
            -sin_lon * de_res - sin_lat * cos_lon * dn_res + cos_lat * cos_lon * du_res,
             cos_lon * de_res - sin_lat * sin_lon * dn_res + cos_lat * sin_lon * du_res,
             cos_lat * dn_res + sin_lat * du_res
        ], dtype=np.float64)

        # 7. Total correction = tidal + non‑tidal
        disp_correction = disp_tidal + disp_nt

        # 8. Secular position = observed – correction
        pos_sec = pos_itrf - disp_correction

        return pos_sec[0], pos_sec[1], pos_sec[2]


# ==============================================================================
# 5. MAIN REPORTING FUNCTION (FULL VSOP2013 PIPELINE with ECMWF Assimilation)
# ==============================================================================
def generate_advanced_geodetic_report_vsop2013(
    lat: float,
    lon: float,
    h_ellipsoid_tide_free: float,
    density: float,
    jd_tt: float,
    jd_ut1: float,
    mjd_utc: float,
    dut1: float,
    local_grids: LocalEGM2008Grids,
    dem_engine: Optional[LocalDEMEngine] = None,
    ids_integrator: Optional[EmbeddedGeodeticNetwork] = None,
    loading_disp: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    x_itrf2020_m: Optional[float] = None,
    y_itrf2020_m: Optional[float] = None,
    z_itrf2020_m: Optional[float] = None,
    loading_resolver: Optional[LoadingResolver] = None,
    xp_rad: float = 0.0,
    yp_rad: float = 0.0,
    dX_rad: float = 0.0,
    dY_rad: float = 0.0,
    use_station_displacement: bool = True,
    sun_itrf: Optional[np.ndarray] = None,
    moon_itrf: Optional[np.ndarray] = None,
    vsop2013_file: str = "VSOP2013p3.dat"
) -> None:
    """
    Generate a comprehensive geodetic-gravimetric report for the Jolotundo
    site, fully compliant with the ASTERID pipeline but powered by the
    VSOP2013 ephemeris for the Sun. Real-time ECMWF data are assimilated
    into tropospheric delay computations and displayed in full detail.

    All computations strictly follow IERS Conventions 2010 and IAU 2006/2000A.
    """
    # ------------------------------------------------------------------
    # Initialization of core geophysical engines.
    # ------------------------------------------------------------------
    if ids_integrator is None:
        ids_integrator = EmbeddedGeodeticNetwork()
    if loading_resolver is None:
        loading_resolver = LoadingResolver(data_dir=".")

    geoid_inv = AdvancedGeoidInversion(local_grids)
    geophysics = PawitraGeophysics(
        local_grids,
        dem_engine=dem_engine,
        default_density_kg_m3=density
    )
    ray_tracer = DynamicRayTracing()

    # ==================================================================
    # OCEAN TIDE LOADING – GRAVITY & TILT (FES2014b)
    # ==================================================================
    otl_grav_engine = Asterid342Engine_FES2014(JOLOTUNDO_FES2014_GRAV_BLQ)
    delta_t_sec = delta_t_from_jd(jd_tt)

    dg_nm_s2, dtilt_ew_nrad, dtilt_ns_nrad = otl_grav_engine.compute_displacement(
        mjd_tt=jd_tt - 2400000.5,
        delta_t=delta_t_sec
    )

    otl_gravity_correction_mgal = dg_nm_s2 / 10000.0
    RAD_TO_ARCSEC = 206264.80624709636
    otl_tilt_ew_arcsec = dtilt_ew_nrad * 1e-9 * RAD_TO_ARCSEC
    otl_tilt_ns_arcsec = dtilt_ns_nrad * 1e-9 * RAD_TO_ARCSEC

    # ---- Sun/Moon ITRF positions ----
    if sun_itrf is None or moon_itrf is None:
        sun_itrf, moon_itrf = get_sun_moon_itrf_vsop2013(
            jd_tt, jd_ut1, xp_rad, yp_rad, dX_rad, dY_rad,
            vsop2013_file=vsop2013_file
        )

    # ---- 0. GFZ-POTSDAM CRUSTAL LOADING RESOLVER ----
    de_res, dn_res, du_res = loading_resolver.resolve_loading_at_mjd(mjd_utc)
    de_res_mm, dn_res_mm, du_res_mm = de_res * 1000.0, dn_res * 1000.0, du_res * 1000.0

    # ---- 1. EXACT HEIGHT INVERSION & ASTRO-GEODETIC DEFLECTION ----
    N_undulation = geoid_inv.get_undulation(lat, lon)
    h_ortho = h_ellipsoid_tide_free - N_undulation   # Ketinggian ortometrik (MSL)

    lat_rad = math.radians(lat)
    tide_free_correction = -0.198 * (0.5 * (3.0 * math.sin(lat_rad)**2 - 1.0))
    h_mean_tide = h_ellipsoid_tide_free + tide_free_correction

    deflection = geoid_inv.get_astro_geodetic_deflection(
        lat, lon,
        otl_tilt_xi_arcsec=otl_tilt_ns_arcsec,
        otl_tilt_eta_arcsec=otl_tilt_ew_arcsec
    )

    # ---- 2. LOCALISED GEOPHYSICS (GRAVITY FIELD) ----
    M, N = geophysics.local_curvature_radii(lat)
    anomalies = geophysics.dynamic_gravity_anomalies(
        lat, lon, h_ortho,
        otl_gravity_correction_mgal=otl_gravity_correction_mgal,
        use_grid_g_obs=True
    )

    g_mean_mgal = anomalies['g_obs_surface_mgal'] + (anomalies['fac_mgal'] / 2.0)
    geopotential_number = (h_ortho * g_mean_mgal) / 1e6  # kGal·m

    # ---- ECMWF real-time atmospheric data (full dataset) ----
    ecmwf_data = None
    ecmwf_status = {'source': 'GPT3', 'active': False, 'timestamp': None}
    try:
        from datetime import datetime, timezone
        unix_ts = (mjd_utc - 40587.0) * 86400.0
        utc_time = datetime.fromtimestamp(unix_ts, tz=timezone.utc)
        ecmwf_data = ecmwf.get_ecmwf_at_point(lat, lon, h_ortho, utc_time)

        # CRITICAL FIX: Only activate if data is valid (not None)
        if ecmwf_data is not None:
            ecmwf_status = {
                'source': 'ECMWF IFS HRES (9 km)',
                'active': True,
                'timestamp': utc_time.isoformat(),
            }
        else:
            ecmwf_status['source'] = 'GPT3 climatology'
            ecmwf_status['active'] = False
    except Exception:
        ecmwf_status['source'] = 'GPT3 climatology'
        ecmwf_status['active'] = False

    # ---- 3. 4-D TROPOSPHERIC RAY-TRACING (GPT3 + VMF3 with ECMWF) ----
    tropo = ray_tracer.compute_tropospheric_slant(
        mjd_utc, lat, lon, h_ellipsoid_tide_free,
        is_realtime=ecmwf_status['active']
    )
    
    # Ambil surface_meteo untuk digunakan di seluruh fungsi
    met = tropo['surface_meteo']

    # ---- Sinkronisasi status ECMWF dengan hasil aktual ----
    # Jika perhitungan troposfer menggunakan GPT3, maka kita anggap ECMWF tidak aktif,
    # sehingga data tambahan tidak ditampilkan.
    if tropo.get('source') == 'GPT3':
        ecmwf_status['active'] = False
        ecmwf_data = None

    # ---- 4. SOLID EARTH TIDE SYNCHRONISATION (for diagnostic display) ----
    if x_itrf2020_m is not None and sun_itrf is not None and moon_itrf is not None:
        xsta = np.array([x_itrf2020_m, y_itrf2020_m, z_itrf2020_m])
        cal = jd_to_cal(*split_jd(jd_ut1), scale='utc')
        fhr = cal['hour'] + cal['minute'] / 60.0 + cal['second'] / 3600.0

        dxtide = solid_earth_tide(
            xsta, cal['year'], cal['month'], cal['day'], fhr, sun_itrf, moon_itrf
        )

        lat_r, lon_r = math.radians(lat), math.radians(lon)
        sin_p, cos_p = math.sin(lat_r), math.cos(lat_r)
        sin_l, cos_l = math.sin(lon_r), math.cos(lon_r)

        de_tide = (-sin_l * dxtide[0] + cos_l * dxtide[1]) * 1000.0
        dn_tide = (-sin_p * cos_l * dxtide[0] - sin_p * sin_l * dxtide[1] + cos_p * dxtide[2]) * 1000.0
        du_tide = (cos_p * cos_l * dxtide[0] + cos_p * sin_l * dxtide[1] + sin_p * dxtide[2]) * 1000.0
        total_tide_mm = math.hypot(du_tide, math.hypot(de_tide, dn_tide))
    else:
        de_tide = dn_tide = du_tide = total_tide_mm = 0.0

    # ---- 5. STATION DISPLACEMENT (Total Vector for Coordinate Correction) ----
    if use_station_displacement and x_itrf2020_m is not None:
        station = StationDisplacement(
            np.array([x_itrf2020_m, y_itrf2020_m, z_itrf2020_m]),
            blq_data=JOLOTUNDO_FES2014_BLQ
        )
        disp_total_vec = station.total_displacement(
            jd_tt, jd_ut1, xp_rad, yp_rad, sun_itrf, moon_itrf, include_atm=True
        )

        lat_r, lon_r = math.radians(lat), math.radians(lon)
        sin_p, cos_p = math.sin(lat_r), math.cos(lat_r)
        sin_l, cos_l = math.sin(lon_r), math.cos(lon_r)

        de_sta = (-sin_l * disp_total_vec[0] + cos_l * disp_total_vec[1]) * 1000.0
        dn_sta = (-sin_p * cos_l * disp_total_vec[0] - sin_p * sin_l * disp_total_vec[1] + cos_p * disp_total_vec[2]) * 1000.0
        du_sta = (cos_p * cos_l * disp_total_vec[0] + cos_p * sin_l * disp_total_vec[1] + sin_p * disp_total_vec[2]) * 1000.0
    else:
        de_sta = dn_sta = du_sta = 0.0

    # ---- 6. DORIS/IDS PRECISE GEODETIC TIE (ITRF2020 SINEX) ----
    epoch_year = 2000.0 + (jd_tt - 2451545.0) / 365.25
    nearest = ids_integrator.get_nearest_station(lat, lon, epoch_year)

    if nearest is not None:
        station_code = nearest['code']
        station_data = ids_integrator.stations.get(station_code)
        if station_data is not None:
            v_obs_x = station_data.get('Vx', 0.0)
            v_obs_y = station_data.get('Vy', 0.0)
            v_obs_z = station_data.get('Vz', 0.0)
        else:
            v_obs_x = v_obs_y = v_obs_z = 0.0
    else:
        v_obs_x = v_obs_y = v_obs_z = 0.0

    # ---- 7. TECTONIC KINEMATICS & CRUSTAL DEFORMATION (ITRF2020-PMM) ----
    nearest_plate = 'EURA'
    plate_kinematics = TectonicPlateKinematics(nearest_plate)

    if x_itrf2020_m is not None:
        x_sta, y_sta, z_sta = x_itrf2020_m, y_itrf2020_m, z_itrf2020_m
        lat_rad_st, lon_rad_st = math.radians(lat), math.radians(lon)
        used_point = "Jolotundo (ITRF2020 tide-free, VSOP2013)"
    else:
        if nearest is not None:
            x_sta, y_sta, z_sta = nearest['X'], nearest['Y'], nearest['Z']
            lat_rad_st, lon_rad_st = math.radians(nearest['lat']), math.radians(nearest['lon'])
            used_point = f"{nearest['name']} ({nearest['code']})"
        else:
            x_sta, y_sta, z_sta = 0.0, 0.0, 0.0
            lat_rad_st, lon_rad_st = math.radians(lat), math.radians(lon)
            used_point = "Unknown (no reference station)"

    vx_orb, vy_orb, vz_orb = plate_kinematics.get_velocity(
        x_sta, y_sta, z_sta,
        lat_rad=lat_rad_st, lon_rad=lon_rad_st,
        apply_orb=True, discard_vertical_orb=True
    )
    vx_no_orb, vy_no_orb, vz_no_orb = plate_kinematics.get_velocity(
        x_sta, y_sta, z_sta,
        lat_rad=lat_rad_st, lon_rad=lon_rad_st,
        apply_orb=False, discard_vertical_orb=False
    )

    sin_lat, cos_lat = math.sin(lat_rad_st), math.cos(lat_rad_st)
    sin_lon, cos_lon = math.sin(lon_rad_st), math.cos(lon_rad_st)

    ve_orb = -vx_orb * sin_lon + vy_orb * cos_lon
    vn_orb = -vx_orb * sin_lat * cos_lon - vy_orb * sin_lat * sin_lon + vz_orb * cos_lat
    vu_orb =  vx_orb * cos_lat * cos_lon + vy_orb * cos_lat * sin_lon + vz_orb * sin_lat

    ve_no_orb = -vx_no_orb * sin_lon + vy_no_orb * cos_lon
    vn_no_orb = -vx_no_orb * sin_lat * cos_lon - vy_no_orb * sin_lat * sin_lon + vz_no_orb * cos_lat
    vu_no_orb =  vx_no_orb * cos_lat * cos_lon + vy_no_orb * cos_lat * sin_lon + vz_no_orb * sin_lat

    ve_obs = -v_obs_x * sin_lon + v_obs_y * cos_lon
    vn_obs = -v_obs_x * sin_lat * cos_lon - v_obs_y * sin_lat * sin_lon + v_obs_z * cos_lat
    vu_obs =  v_obs_x * cos_lat * cos_lon + v_obs_y * cos_lat * sin_lon + v_obs_z * sin_lat

    res_e = (ve_obs - ve_orb) * 1000.0
    res_n = (vn_obs - vn_orb) * 1000.0
    res_u = (vu_obs - vu_orb) * 1000.0

    v_horiz_orb = math.hypot(ve_orb, vn_orb) * 1000.0
    azimuth_orb = math.degrees(math.atan2(ve_orb, vn_orb)) % 360.0
    v_horiz_no_orb = math.hypot(ve_no_orb, vn_no_orb) * 1000.0
    azimuth_no_orb = math.degrees(math.atan2(ve_no_orb, vn_no_orb)) % 360.0

    compass_points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    compass_bearings = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                        180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    best_idx = min(range(len(compass_bearings)), key=lambda i: min(abs(azimuth_orb - compass_bearings[i]), 360 - abs(azimuth_orb - compass_bearings[i])))
    plate_direction = compass_points[best_idx]
    best_idx_no = min(range(len(compass_bearings)), key=lambda i: min(abs(azimuth_no_orb - compass_bearings[i]), 360 - abs(azimuth_no_orb - compass_bearings[i])))
    plate_direction_no = compass_points[best_idx_no]

    # ---- 8. GEODETIC DIAGNOSTIC & UNCERTAINTY (IAG/Standar Profesional) ----
    # ------------------------------------------------------------
    # 1. KOMPONEN A PRIORI (Statis, berdasarkan model global)
    # ------------------------------------------------------------
    sigma_geoid_apriori = 0.080   # 8 cm (standar IAG untuk XGM2019e-2159 di daerah tropis)
    sigma_tide_apriori   = 0.003   # 3 mm (akurasi FES2014 untuk vertikal)

    # ------------------------------------------------------------
    # 2. KOMPONEN A POSTERIORI (Adaptif, dari data aktual)
    # ------------------------------------------------------------
    # 2a. Sigma troposfer adaptif berdasarkan RH (dari ECMWF jika ada)
    if ecmwf_data is not None:
        rh = ecmwf_data.get('rh_2m', 60.0)
        if rh > 85:
            sigma_tropo_aposteriori = 0.035
        elif rh > 70:
            sigma_tropo_aposteriori = 0.025
        elif rh > 50:
            sigma_tropo_aposteriori = 0.018
        else:
            sigma_tropo_aposteriori = 0.012
    else:
        sigma_tropo_aposteriori = 0.020  # fallback

    # 2b. Sigma dari variabilitas defleksi vertikal
    otl_tilt_total = math.hypot(otl_tilt_ns_arcsec, otl_tilt_ew_arcsec)
    sigma_defleksi = 0.0005 * otl_tilt_total
    sigma_defleksi = min(0.010, sigma_defleksi)

    # 2c. Sigma dari perbedaan GPT3 vs ECMWF (jika keduanya tersedia)
    if ecmwf_data is not None and tropo.get('source') == 'ECMWF':
        p_diff = abs(met['p_hpa'] - ecmwf_data['p'])
        t_diff = abs(met['t_c'] - ecmwf_data['T'])
        sigma_model_diff = math.hypot(p_diff * 0.0023, t_diff * 0.0005)
        sigma_model_diff = min(0.015, sigma_model_diff)
    else:
        sigma_model_diff = 0.005

    # ------------------------------------------------------------
    # 3. KOMBINASI A PRIORI + A POSTERIORI (Root-Sum-Square)
    # ------------------------------------------------------------
    sigma_1sigma = math.sqrt(
        sigma_geoid_apriori**2 +
        sigma_tide_apriori**2 +
        sigma_tropo_aposteriori**2 +
        sigma_defleksi**2 +
        sigma_model_diff**2
    )
    sigma_95 = sigma_1sigma * 1.96
    bias_max = 0.020  # estimasi bias sistematis

    # ------------------------------------------------------------
    # 4. DIAGNOSTIK LAINNYA
    # ------------------------------------------------------------
    azimuth_dev = math.degrees(math.atan2(deflection['eta_arcsec'], deflection['xi_arcsec']))
    if azimuth_dev < 0:
        azimuth_dev += 360.0

    rho_local = anomalies.get('local_density_used', density)
    unit_code = anomalies.get('local_unit_code', 'Unknown')
    material_desc = str(anomalies.get('local_material_description', 'No description'))

    # ======================================================================
    # 9. CETAK LAPORAN DENGAN GAYA BOX-DRAWING
    # ======================================================================
    print(f"\n{BOLD}{THICK_SEP}{RESET}")
    print(f"{BOLD}{'ASTERID GEODETIC-GRAVIMETRIC DATUM JOLOTUNDO | MT.PAWITRA'.center(W)}{RESET}")
    print(f"{BOLD}{'** VSOP2013/ELPMPP02 EPHEMERIS + IERS CONVENTIONS (2010) **'.center(W)}{RESET}")
    print(f"{BOLD}{THICK_SEP}{RESET}")

    def section_header(title):
        print(f"\n{BOLD}{title}{RESET}")
        print(f"{BOLD}{THIN_SEP}{RESET}")

    # [0] EPOCH & DATUM
    section_header("[0] COMPUTATION EPOCH & ABSOLUTE DATUM")
    print(f"  {'Computation UTC':<30}: {datetime.now(timezone.utc).isoformat()}")
    print(f"  {'TT/TDB Epoch':<30}: {epoch_year:.6f} (ITRF2000 scale)")
    print(f"  {'Geodetic Latitude':<30}: {lat:.6f}°")
    print(f"  {'Geodetic Longitude':<30}: {lon:.6f}°")
    print(f"  {'Input Tide-Free H':<30}: {h_ellipsoid_tide_free:.3f} m")

    # [1] GFZ-POTSDAM CRUSTAL LOADING RESOLVER
    section_header("[1] GFZ-POTSDAM CRUSTAL LOADING RESOLVER (Zero Double-Counting)")
    print(f"  {'Env. non-tidal loading':<30}: maintained as transient vectors")
    print(f"  {'Target MJD':<30}: {mjd_utc:.5f}")
    print(f"  {'dE (East-West)':<30}: {de_res_mm:+.4f} mm")
    print(f"  {'dN (North-South)':<30}: {dn_res_mm:+.4f} mm")
    print(f"  {'dU (Vertical)':<30}: {du_res_mm:+.4f} mm")
    print(f"  {'Resolver Status':<30}: COMPLETED [O(1) Binary Lock]")

    # [2] EXACT HEIGHT INVERSION & ASTRO-GEODETIC DEFLECTION
    section_header("[2] HEIGHT INVERSION & ASTRO-GEODETIC DEFLECTION")
    print(f"  {'XGM2019e-2159 Undulation (N)':<30}: {N_undulation:.3f} m (Tide-Free)")
    print(f"  {'Derived Orthometric (H)':<30}: {h_ortho:.3f} m")
    print(f"  {'Mean-Tide Ellipsoid':<30}: {h_mean_tide:.3f} m (reference only)")
    print(f"  {'Vertical Deflection ξ (NS)':<30}: {deflection['xi_arcsec']:+.4f}″")
    print(f"  {'Vertical Deflection η (EW)':<30}: {deflection['eta_arcsec']:+.4f}″")
    print(f"  {'Total Deflection θ':<30}: {deflection['total_theta_arcsec']:.4f}″")
    print(f"  {'OTL Tilt (NS, EW)':<30}: {otl_tilt_ns_arcsec:+.4f}, {otl_tilt_ew_arcsec:+.4f} arcsec")

    # [3] LOCAL GEOPHYSICS
    section_header("[3] LOCAL GEOPHYSICS & GRAVITY FIELD")
    print(f"  {'Density':<30}: {rho_local:.1f} kg/m³  ({unit_code})")
    prefix_awal = f"  {'Material Description':<30}: "
    spasi_penahan = " " * len(prefix_awal)
    deskripsi_rapi = textwrap.fill(
        material_desc,
        width=72,
        initial_indent=prefix_awal,
        subsequent_indent=spasi_penahan
    )
    print(deskripsi_rapi)
    print(f"  {'Meridional Radius (M)':<30}: {M:.3f} m")
    print(f"  {'Prime Vertical Radius (N)':<30}: {N:.3f} m")
    print(f"  {'Normal Gravity γ₀':<30}: {anomalies['gamma_0_mgal']:.4f} mGal")
    print(f"  {'Free-Air Anomaly (XGM2019e)':<30}: {anomalies['delta_g_fa_egm_mgal']:+.4f} mGal")
    print(f"  {'2nd-Order Free-Air Corr.':<30}: {anomalies['fac_mgal']:+.4f} mGal")
    print(f"  {'Bouguer Slab Corr.':<30}: {anomalies['bc_slab_mgal']:+.4f} mGal (ρ={rho_local:.0f})")
    print(f"  {'DEM Terrain Correction':<30}: {anomalies['terrain_correction_mgal']:+.4f} mGal")
    print(f"  {'OTL Gravity Correction':<30}: {otl_gravity_correction_mgal:+.4f} mGal")
    print(f"  {'Observed Surface Gravity':<30}: {anomalies['g_obs_surface_mgal']:.4f} mGal")
    print(f"  {'Complete Bouguer Anomaly':<30}: {anomalies['complete_bouguer_anomaly_mgal']:+.4f} mGal")
    print(f"  {'Bouguer-Reduced Gravity':<30}: {anomalies['reduced_bouguer_gravity_mgal']:.4f} mGal")
    print(f"  {'Local VGG (T_zz) Anomaly':<30}: {anomalies['vgg_anomaly_eotvos']:+.2f} Eötvös")
    print(f"  {'Total VGG (W_zz)':<30}: {anomalies['total_vgg_eotvos']:.2f} Eötvös")
    print(f"  {'Local Free-Air Gradient':<30}: {anomalies['local_fag_mgal_m']:.5f} mGal/m")

    # [4] TROPOSPHERIC RAY-TRACING (with ECMWF data and full display)
    section_header("[4] 4-D TROPOSPHERIC RAY-TRACING (GPT3 + VMF3)")
    met = tropo['surface_meteo']
    print(f"  {'Surface Pressure (GPT3)':<30}: {met['p_hpa']:.1f} hPa")
    print(f"  {'Surface Temperature (GPT3)':<30}: {met['t_c']:.1f} °C")
    print(f"  {'Water Vapour Pressure (GPT3)':<30}: {met['e_hpa']:.2f} hPa")
    print(f"  {'Total Refractivity (N)':<30}: {tropo['refractivity']['N_total']:.2f}")
    print(f"  {'Zenith Total Delay (ZTD)':<30}: {tropo['zenith_delays']['ztd_m']:.4f} m")
    print(f"  {'Slant Delay @ 10° elev.':<30}: {tropo['slant_delay_10deg_m']:.4f} m")
    print(f"  {'Slant Delay @ 30° elev.':<30}: {tropo['slant_delay_30deg_m']:.4f} m")
    print(f"  {'Data Source (troposphere)':<30}: {tropo.get('source', 'GPT3')}")

    # ---- ECMWF real-time data (full display) ----
    if ecmwf_data is not None and ecmwf_status['active']:
        print(f"\n  ECMWF IFS HRES (9 km) Real-Time Atmospheric Profile:")
        print(f"    {'Model Orography':<30}: {ecmwf_data['h_surface']:.1f} m")
        print(f"    {'Surface Pressure':<30}: {ecmwf_data['p_surface']:.2f} hPa")
        print(f"    {'Surface Temperature':<30}: {ecmwf_data['T_surface']:.2f} °C")
        print(f"    {'Timestamp (UTC)':<30}: {ecmwf_status['timestamp']}")
        print(f"\n    Extrapolated to Site Elevation ({ecmwf_data['metadata']['target_height_m']:.1f} m):")
        print(f"      {'Pressure (P)':<28}: {ecmwf_data['p']:.2f} hPa")
        print(f"      {'Temperature (T)':<28}: {ecmwf_data['T']:.2f} °C")
        print(f"      {'Dew Point (Td)':<28}: {ecmwf_data['Td']:.2f} °C")
        print(f"      {'Vapour Pressure (e)':<28}: {ecmwf_data['e']:.2f} hPa")
        print(f"      {'Relative Humidity (RH)':<28}: {ecmwf_data.get('rh_2m', 0.0):.1f} %")
        print(f"      {'Cloud Cover':<28}: {ecmwf_data.get('cloud_cover', 0.0):.1f} %")
        print(f"      {'Wind Speed (10m)':<28}: {ecmwf_data.get('wind_speed_10m', 0.0):.1f} km/h")
        print(f"      {'Wind Direction':<28}: {ecmwf_data.get('wind_direction_10m', 0.0):.0f}°")
        print(f"      {'Precipitation':<28}: {ecmwf_data.get('precipitation', 0.0):.2f} mm/h")
        print(f"    {'Data Points':<30}: {ecmwf_data['metadata']['data_points']} hourly")

    # [5] SOLID EARTH TIDE
    section_header("[5] SOLID EARTH TIDE SYNCHRONISATION")
    print(f"  {'Radial (Up) Displacement':<30}: {du_tide:+.4f} mm")
    print(f"  {'East Displacement':<30}: {de_tide:+.4f} mm")
    print(f"  {'North Displacement':<30}: {dn_tide:+.4f} mm")
    print(f"  {'Total Vector Magnitude':<30}: {total_tide_mm:.4f} mm")

    # [6] DORIS/IDS PRECISE TIE
    section_header("[6] DORIS/IDS PRECISE GEODETIC TIE (ITRF2020 SINEX)")
    if nearest is not None:
        print(f"  {'Nearest IDS Station':<30}: {nearest['name']} ({nearest['code']})")
        print(f"  {'Reference Epoch':<30}: {nearest['epoch']:.3f}")
        print(f"  {'X (ITRF2020)':<30}: {nearest['X']:.4f} m")
        print(f"  {'Y (ITRF2020)':<30}: {nearest['Y']:.4f} m")
        print(f"  {'Z (ITRF2020)':<30}: {nearest['Z']:.4f} m")
        print(f"  {'Sigmas (σX, σY, σZ)':<30}: {nearest['sigX']*1000:.2f}, {nearest['sigY']*1000:.2f}, {nearest['sigZ']*1000:.2f} mm")
        print(f"  {'Approx Geodetic Coords':<30}: Lat {nearest['lat']:.6f}°, Lon {nearest['lon']:.6f}°")
        if x_itrf2020_m is not None:
            dx = x_itrf2020_m - nearest['X']
            dy = y_itrf2020_m - nearest['Y']
            dz = z_itrf2020_m - nearest['Z']
            baseline = math.hypot(dx, math.hypot(dy, dz))
            sigma_baseline = math.hypot(nearest['sigX'], math.hypot(nearest['sigY'], nearest['sigZ']))
            print(f"  {'Baseline (survey-DORIS)':<30}: {baseline/1000:.3f} km ± {sigma_baseline*1000:.2f} mm")
        else:
            if ids_integrator is not None:
                baseline = ids_integrator._haversine(lat, lon, nearest['lat'], nearest['lon'])
                print(f"  {'Baseline (approx)':<30}: {baseline/1000:.3f} km")
            else:
                print(f"  {'Baseline (approx)':<30}: N/A")
    else:
        print(f"  {'Nearest IDS Station':<30}: None found")

    # [7] TECTONIC KINEMATICS (ITRF2020-PMM)
    section_header("[7] TECTONIC KINEMATICS (ITRF2020-PMM)")
    orb_tx, orb_ty, orb_tz = TectonicPlateKinematics._ORB_MM_YR
    print(f"  {'Plate Assignment':<30}: {nearest_plate} Plate")
    print(f"  {'Origin Rate Bias (ORB)':<30}: Tx={orb_tx:+.2f}, Ty={orb_ty:+.2f}, Tz={orb_tz:+.2f} mm/yr")

    prefix_ref = f"  {'Reference Point':<30}: "
    wrapped_ref = textwrap.fill(
        used_point,
        width=72,
        initial_indent=prefix_ref,
        subsequent_indent=" " * len(prefix_ref)
    )
    print(wrapped_ref)

    prefix_orb = f"  {'NNR Euler w/ ORB':<30}: "
    print(f"{prefix_orb}Ve = {ve_orb*1000:+.2f} mm/yr")
    indent_orb = " " * len(prefix_orb)
    print(f"{indent_orb}Vn = {vn_orb*1000:+.2f} mm/yr")
    print(f"{indent_orb}Vu = {vu_orb*1000:+.2f} mm/yr")

    prefix_no_orb = f"  {'NNR Euler w/o ORB':<30}: "
    print(f"{prefix_no_orb}Ve = {ve_no_orb*1000:+.2f} mm/yr")
    indent_no_orb = " " * len(prefix_no_orb)
    print(f"{indent_no_orb}Vn = {vn_no_orb*1000:+.2f} mm/yr")
    print(f"{indent_no_orb}Vu = {vu_no_orb*1000:+.2f} mm/yr")

    print(f"  {'Horiz. Resultant (ORB)':<30}: {v_horiz_orb:.2f} mm/yr -> {plate_direction} (Az {azimuth_orb:.1f}°)")
    print(f"  {'Horiz. Resultant (no ORB)':<30}: {v_horiz_no_orb:.2f} mm/yr -> {plate_direction_no} (Az {azimuth_no_orb:.1f}°)")
    print(f"  {'Local Deformation (obs-model)':<30}: dVe={res_e:+.2f}, dVn={res_n:+.2f}, dVu={res_u:+.2f} mm/yr")
    print(f"  {'Note':<30}: Vertical ORB discarded (Altamimi 2023)")

    # [8] GEODETIC DIAGNOSTIC & UNCERTAINTY
    section_header("[8] GEODETIC DIAGNOSTIC & UNCERTAINTY")
    print(f"  {'Geopotential Number (C)':<30}: {geopotential_number:.4f} kGal·m")
    print(f"  {'H Uncertainty (1σ, 68%)':<30}: ±{sigma_1sigma:.4f} m")
    print(f"  {'H Uncertainty (2σ, 95%)':<30}: ±{sigma_95:.4f} m")
    print(f"  {'Estimated Systematic Bias':<30}: ±{bias_max:.3f} m (max)")
    print(f"  {'Plumb-line Azimuthal Bias':<30}: {azimuth_dev:.2f}° (Ref. North)")


# ==============================================================================
# 6. MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # ---- Site constants (Jolotundo observatory) ----
    LAT_JOL, LON_JOL = -7.609444, 112.595556
    H_ELLIPSOID_TIDE_FREE = 583.355          # Preliminary value (m)
    ROCK_DENSITY_PAWITRA = 2450.0          # Fallback density (kg/m³)

    # ---- Initialise local data ----
    print("⚙️ [DIAGNOSTIC] Initialising local grids and DEM...")
    local_grids = LocalEGM2008Grids(data_dir=".")
    dem_engine = LocalDEMEngine(dem_filepath="output_hh.asc")
    strat_engine = PawitraStratigraphy()
    dem_engine.stratigraphy_engine = strat_engine
    loading_resolver = LoadingResolver(data_dir=".")

    # ---- Current time (UTC) ----
    now = datetime.now(timezone.utc)
    jd_utc1, jd_utc2 = cal_to_jd(
        now.year, now.month, now.day,
        now.hour, now.minute,
        now.second + now.microsecond / 1e6,
        scale='utc'
    )
    jd_utc = jd_utc1 + jd_utc2
    mjd_utc = jd_utc - MJD_ZERO

    # ---- Earth Orientation Parameters ----
    print("⚙️ [DIAGNOSTIC] Retrieving EOP from IERS C04/Bulletin-A...")
    eo = EarthOrientation()
    eop = eo.get_eop_corrections(jd_utc)   # TT
    dut1 = eop['dut1']
    xp_rad = eop['xp']
    yp_rad = eop['yp']
    dX_rad = eop['dX']
    dY_rad = eop['dY']

    # ---- Time systems ----
    ut1_jd = eo.ut1_jd_from_tt(jd_utc, dut1)
    jd_tt = jd_utc

    cal_ut1 = jd_to_cal(*split_jd(ut1_jd), scale='utc')
    fhr = cal_ut1['hour'] + cal_ut1['minute'] / 60.0 + cal_ut1['second'] / 3600.0

    # ---- Step 1: Geodetic → ITRF (raw) ----
    print("⚙️ [DIAGNOSTIC] Converting geodetic coordinates to ITRF...")
    sin_lat = math.sin(math.radians(LAT_JOL))
    cos_lat = math.cos(math.radians(LAT_JOL))
    sin_lon = math.sin(math.radians(LON_JOL))
    cos_lon = math.cos(math.radians(LON_JOL))

    N_radius = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
    x_raw = (N_radius + H_ELLIPSOID_TIDE_FREE) * cos_lat * cos_lon
    y_raw = (N_radius + H_ELLIPSOID_TIDE_FREE) * cos_lat * sin_lon
    z_raw = (N_radius * (1.0 - WGS84_E2) + H_ELLIPSOID_TIDE_FREE) * sin_lat
    pos_itrf_raw = np.array([x_raw, y_raw, z_raw])

    # ---- Step 2: Secular coordinate extraction (DORIS deformation analyser) ----
    print("⚙️ [DIAGNOSTIC] Activating DORIS_Deformation_Analyzer (IERS 2010) using VSOP2013...")
    deformation_analyzer = DORIS_Deformation_Analyzer()
    x_sec, y_sec, z_sec = deformation_analyzer.extract_secular_coordinates(
        x_obs=x_raw,
        y_obs=y_raw,
        z_obs=z_raw,
        lat_deg=LAT_JOL,
        lon_deg=LON_JOL,
        h_ell_m=H_ELLIPSOID_TIDE_FREE,
        year=cal_ut1['year'],
        month=cal_ut1['month'],
        day=cal_ut1['day'],
        fhr=fhr,
        jd_tt=jd_tt,
        mjd_utc=mjd_utc,
        xp_rad=xp_rad,
        yp_rad=yp_rad,
        dut1=dut1,
        dX_rad=dX_rad,
        dY_rad=dY_rad
    )

    # ---- Step 3: ITRF transformation (for scale comparison) ----
    print("⚙️ [DIAGNOSTIC] Applying DORIS_ITRF_Engine (14-parameter Helmert)...")
    itrf_engine = DORIS_ITRF_Engine()
    epoch_year = 2000.0 + (jd_tt - 2451545.0) / 365.25
    x_trans, y_trans, z_trans = itrf_engine.transform_itrf2020_to_target(
        target_frame="ITRF2000",
        x_m=x_sec,
        y_m=y_sec,
        z_m=z_sec,
        obs_epoch_year=epoch_year
    )

    # ---- Step 4: Prepare final outputs ----
    x_jol_itrf2020 = x_sec
    y_jol_itrf2020 = y_sec
    z_jol_itrf2020 = z_sec

    disp_total = pos_itrf_raw - np.array([x_sec, y_sec, z_sec])
    
    # PERBAIKAN: Proyeksi XYZ (ITRF) ke komponen Up (Vertikal Lokal)
    lat_r = math.radians(LAT_JOL)
    lon_r = math.radians(LON_JOL)
    sin_p, cos_p = math.sin(lat_r), math.cos(lat_r)
    sin_l, cos_l = math.sin(lon_r), math.cos(lon_r)
    
    du_total = (cos_p * cos_l * disp_total[0] + 
                cos_p * sin_l * disp_total[1] + 
                sin_p * disp_total[2])
                
    h_final_tide_free = H_ELLIPSOID_TIDE_FREE - du_total

    # ---- Step 5: Obtain Sun & Moon ITRF positions using VSOP2013 ----
    sun_itrf, moon_itrf = get_sun_moon_itrf_vsop2013(
        jd_tt, ut1_jd, xp_rad, yp_rad, dX_rad, dY_rad
    )

    # ---- Step 6: Generate the full report ----
    print("[SUCCESS] Spatio-temporal reduction completed (VSOP2013 ephemeris).")
    print("[PROCESSING] Assembling geophysical inversion matrix...")

    generate_advanced_geodetic_report_vsop2013(
        lat=LAT_JOL,
        lon=LON_JOL,
        h_ellipsoid_tide_free=h_final_tide_free,
        density=ROCK_DENSITY_PAWITRA,
        jd_tt=jd_tt,
        jd_ut1=ut1_jd,
        mjd_utc=mjd_utc,
        dut1=dut1,
        local_grids=local_grids,
        dem_engine=dem_engine,
        ids_integrator=None,
        loading_disp=(0.0, 0.0, 0.0),
        x_itrf2020_m=x_jol_itrf2020,
        y_itrf2020_m=y_jol_itrf2020,
        z_itrf2020_m=z_jol_itrf2020,
        loading_resolver=loading_resolver,
        xp_rad=xp_rad,
        yp_rad=yp_rad,
        dX_rad=dX_rad,
        dY_rad=dY_rad,
        use_station_displacement=True,
        sun_itrf=sun_itrf,
        moon_itrf=moon_itrf,
        vsop2013_file="VSOP2013p3.dat"
    )