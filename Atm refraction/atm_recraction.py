# atm_refraction.py
# Corrected to import cal_to_jd and tai_utc from Timescales instead of EOPDelta

import sys
sys.dont_write_bytecode = True

import re
import math
import bisect
import numpy as np
import os
from typing import Tuple, Dict, Optional

# GPT3 & VMF3
from gpt3 import load_gpt3_grid, gpt3_interpolate
from vmf3 import vmf3_ht

# Replace EOPDelta with Timescales for cal_to_jd and tai_utc
from Timescales import cal_to_jd, tai_utc


# -----------------------------------------------------------------------------
# SITE CONSTANTS – JOLOTUNDO OBSERVATORY
# Updated from ASTERID Geodetic-Gravimetric Datum (2026-06-12)
# -----------------------------------------------------------------------------
SITE_LAT_DEG = -7.609444          # Geodetic latitude (WGS84)
SITE_LON_DEG = 112.595556         # Geodetic longitude (WGS84)
SITE_ELEV_M = 554.509                    # COP30 DEM (orthometric)
SITE_ELEV_ELLIPSOIDAL_M = 583.355 # Ellipsoidal height (WGS84, tide-free)

# Lokasi lengkap (untuk referensi dan dokumentasi)
JOLOTUNDO_LOCATION = {
    'name': 'Jolotundo Archaeological Observatory',
    'latitude_deg': SITE_LAT_DEG,
    'longitude_deg': SITE_LON_DEG,
    'elevation_ellipsoidal_m': 583.355,
    'elevation_orthometric_m': 554.509,
    'geoid_height_m': 28.8464,           # XGM2019e-2159
    'reference': 'WGS84/EGM2008 (ASTERID inversion)'
}

# Parameter atmosfer standar Jolotundo – hasil GPT3 + VMF3 pada epoch 2026.445
JOLOTUNDO_ATMOSPHERE = {
    'standard_pressure_hpa': 952.5,    # Tekanan permukaan (hPa)
    'standard_temperature_c': 22.9,    # Suhu udara (°C)
    'mean_humidity': 0.728,            # Kelembaban relatif (dari e=20.27 hPa, psat≈27.83)
    'water_vapor_pressure_hpa': 20.27, # Tekanan uap air (ditambahkan untuk referensi)
    'reference': 'GPT3 + VMF3 (epoch 2026.445)'
}


# -----------------------------------------------------------------------------
# GEOPHYSICAL & GEODYNAMIC RESULTS – ASTERID INVERSION (Epoch 2026.445)
# -----------------------------------------------------------------------------
JOLOTUNDO_GEOPHYSICS = {
    'epoch': '2026-06-12T15:32:28.677736+00:00',
    'epoch_mjd': 61203.64729,
    'density_kg_m3': 2250.0,
    'material_description': (
        "Masjedong pyroclastic flow: youngest unit, brown, poorly sorted, "
        "andesite fragments 2-15 cm, sand matrix, spreads westwards to the flank."
    ),
    'geodetic_radii': {
        'meridional_radius_m': 6336555.028,
        'prime_vertical_radius_m': 6378511.385
    },
    'gravity': {
        'normal_gravity_mgal': 978108.9982,
        'free_air_anomaly_mgal': 12.2666,
        'second_order_free_air_corr_mgal': 163.3846,
        'simple_bouguer_slab_corr_mgal': 50.1922,
        'terrain_correction_mgal': 2.2044,
        'inferred_surface_gravity_mgal': 977958.6976,
        'complete_bouguer_anomaly_mgal': -35.7212,
        'bouguer_reduced_gravity_mgal': 978073.2770
    },
    'vertical_gravity_gradient': {
        'local_vgg_anomaly_eotvos': -14.58,
        'total_vgg_eotvos': 3071.42,
        'local_free_air_gradient_mgal_per_m': 0.30714
    },
    'vertical_deflections_arcsec': {
        'xi_meridional': 1.9644,
        'eta_prime_vertical': -1.8685,
        'total_theta': 2.7111
    },
    'solid_earth_tide_mm': {
        'radial_up': 254.1324,
        'tangential_east': -27.6869,
        'tangential_north': -34.4517,
        'total_vector_magnitude': 257.9472
    },
    'crustal_loading_mm': {
        'dE_east': 0.0674,
        'dN_north': -1.6708,
        'dU_up': 1.7631
    },
    'geopotential': {
        'geopotential_number_kGal_m': 520.3380,
        'height_uncertainty_m': 0.0114,          # 95% confidence
        'plumb_line_azimuth_deg': 316.43
    }
}

# -----------------------------------------------------------------------------
# GEODYNAMIC PARAMETERS – ITRF2020 / NNR-MORVEL56 PLATE KINEMATICS
# -----------------------------------------------------------------------------
JOLOTUNDO_GEODYNAMICS = {
    # Reference frame and tectonic plate assignment
    'reference_frame': 'ITRF2020',
    'plate': 'EURA',                      # Eurasian Plate
    
    # Origin rate bias (mm/yr) – Altamimi et al. 2023
    'origin_rate_bias_mm_per_yr': {
        'Tx': 0.37,                       # X-component
        'Ty': 0.35,                       # Y-component
        'Tz': 0.74,                       # Z-component
    },
    
    # NNR Euler pole velocities (mm/yr)
    'nnr_euler_velocities_mm_per_yr': {
        'with_orb': {                     # Including ORB correction
            'Ve': 20.79,                  # East velocity
            'Vn': -7.84,                  # North velocity
            'Vu': 0.00,                   # Up velocity
        },
        'without_orb': {                  # Excluding ORB correction
            'Ve': 21.27,
            'Vn': -8.59,
            'Vu': 0.01,
        },
    },
    
    # Horizontal resultant velocities
    'horizontal_resultant': {
        'with_orb': {
            'speed_mm_per_yr': 22.22,
            'azimuth_deg': 110.7,         # ESE direction
        },
        'without_orb': {
            'speed_mm_per_yr': 22.94,
            'azimuth_deg': 112.0,
        },
    },
    
    # Local crustal deformation relative to NNR (mm/yr)
    'local_deformation_mm_per_yr': {
        'dVe_east': 7.29,                 # East residual
        'dVn_north': -0.41,               # North residual
        'dVu_up': -1.32,                  # Up residual
    },
    
    # DORIS/IDS precise geodetic tie (ITRF2020 SINEX)
    'doris_tie': {
        'nearest_station': 'CMJT (CORS Mojokerto BIG)',
        'reference_epoch': 2026.445,
        'itrf2020_coordinates_m': {
            'X': -2414318.8533,
            'Y': 5845521.5267,
            'Z': -823220.6979,
        },
        'coordinate_sigmas_mm': {
            'sigmaX': 0.30,
            'sigmaY': 0.60,
            'sigmaZ': 0.20,
        },
        'approximate_geodetic': {
            'lat_deg': -7.465579,
            'lon_deg': 112.441616,
        },
        'baseline_km': 23.321,
        'baseline_sigma_mm': 0.70,
    },
    
    # Additional notes
    'note': 'Vertical ORB discarded (Altamimi 2023)',
}


# -----------------------------------------------------------------------------
# JPL STANDARD REFRACTION CONSTANTS (NOAA GFS 0.25° Model)
# -----------------------------------------------------------------------------
JPL_ATMOSPHERE = {
    'standard_pressure_hpa': 1012.31,    # Tekanan standar JPL (sea-level)
    'standard_temperature_c': 27.68,      # Suhu standar JPL (sea-level)
    'reference': 'JPL Horizons default yellow-light refraction'
}

# =============================================================================
# ATMOSPHERIC SOURCE STATE TRACKER
# =============================================================================
# Global variable for tracking the origin of atmospheric parameters used
# in refraction calculations. Updated by hybrid_meteo_assimilation().
#
# Fields:
#   source    : str  ('GPT3' | 'ECMWF') — indicates which model provided P, T, e
#   active    : bool — True if ECMWF data was successfully assimilated
#   timestamp : str  — ISO-8601 timestamp of the ECMWF data (if active)
# =============================================================================
_REFRACTION_SOURCE = {
    'source': 'GPT3',
    'active': False,
    'timestamp': None,
}

def get_refraction_source():
    """
    Return a copy of the current atmospheric source state.

    Returns
    -------
    dict
        A dictionary containing keys: source, active, timestamp.
    """
    return _REFRACTION_SOURCE.copy()


# =============================================================================
# GPT3 GRID RESOLUSI 1° (gpt3_1.npz)
# Modul ini menggunakan grid GPT3 dengan resolusi 1°x1° yang disimpan dalam
# format .npz (hasil perbaikan dari gpt3.py). File ini menggantikan versi
# CSV 5° sebelumnya untuk akurasi spasial yang lebih tinggi.
# =============================================================================

_gpt3_grid_cache = None

def _get_gpt3_grid():
    """Muat grid GPT3 resolusi 1° dari file gpt3_1.npz (lazy loading)."""
    global _gpt3_grid_cache
    if _gpt3_grid_cache is None:
        base_dir = os.path.dirname(__file__)
        grid_path = os.path.join(base_dir, "gpt3_1.npz")
        if not os.path.exists(grid_path):
            grid_path = "gpt3_1.npz"   # fallback ke working directory
        _gpt3_grid_cache = load_gpt3_grid(grid_path)
    return _gpt3_grid_cache

def gpt3_full(mjd, lat_rad, lon_rad, height_m):
    """
    Wrapper for gpt3_interpolate, returns a tuple compatible with old gpt2_full:
    (pressure_hpa, temperature_c, dT_k_per_km, e_hpa, ah, aw, undu)
    plus extra dictionary with all GPT3 parameters.
    """
    grid = _get_gpt3_grid()
    res = gpt3_interpolate(mjd, lat_rad, lon_rad, height_m, grid, it=0)
    return (res['p'], res['T'], res['dT'], res['e'],
            res['ah'], res['aw'], res['undu']), res


def gpt3_vmf3(mjd, lat_rad, lon_rad, h_ell_m, zd_rad):
    """
    Kombinasi GPT3 + VMF3: mengembalikan parameter troposfer lengkap
    beserta hydrostatic & wet mapping factors.
    """
    (p, T, dT, e, ah, aw, undu), extra = gpt3_full(mjd, lat_rad, lon_rad, h_ell_m)
    mfh, mfw = vmf3_ht(mjd, lat_rad, lon_rad, h_ell_m, zd_rad, ah, aw)
    return {
        'p': p, 'T': T, 'dT': dT, 'e': e,
        'ah': ah, 'aw': aw, 'undu': undu,
        'mfh': mfh, 'mfw': mfw,
        **extra
    }

def hybrid_meteo_assimilation(mjd, lat_rad, lon_rad, height_m, is_realtime=False):
    """
    Retrieve atmospheric parameters with optional ECMWF assimilation.

    Parameters
    ----------
    mjd : float
        Modified Julian Date (UTC).
    lat_rad, lon_rad : float
        Geodetic coordinates in radians.
    height_m : float
        Ellipsoidal height in metres.
    is_realtime : bool, default False
        If True, attempt to override P, T, e with ECMWF data.

    Returns
    -------
    tuple
        (p_hpa, t_c, dT, e_hpa, ah, aw, undu, extra_dict)
        Where ah and aw are from GPT3, while p, T, e may be from ECMWF.
    """
    global _REFRACTION_SOURCE

    # 1. Always get GPT3 parameters (including ah, aw)
    (p_hpa, t_c, dT, e_hpa, ah, aw, undu), extra = gpt3_full(mjd, lat_rad, lon_rad, height_m)

    # 2. Default state: GPT3
    _REFRACTION_SOURCE['source'] = 'GPT3'
    _REFRACTION_SOURCE['active'] = False
    _REFRACTION_SOURCE['timestamp'] = None

    # 3. ECMWF assimilation if requested
    if is_realtime:
        try:
            import ecmwf_realtime as ecmwf
            from datetime import datetime, timezone
            import math

            unix_ts = (mjd - 40587.0) * 86400.0
            utc_time = datetime.fromtimestamp(unix_ts, tz=timezone.utc)

            lat_deg = math.degrees(lat_rad)
            lon_deg = math.degrees(lon_rad)

            # Convert to orthometric height using undulation from GPT3
            h_ortho = height_m - undu   # undu already obtained from gpt3_full above

            ecmwf_data = ecmwf.get_ecmwf_at_point(lat_deg, lon_deg, h_ortho, utc_time)

            # CRITICAL FIX: Only extract if data is valid (not None)
            if ecmwf_data is not None:
                p_hpa = ecmwf_data['p']
                t_c   = ecmwf_data['T']
                e_hpa = ecmwf_data['e']

                _REFRACTION_SOURCE['source'] = 'ECMWF'
                _REFRACTION_SOURCE['active'] = True
                _REFRACTION_SOURCE['timestamp'] = utc_time.isoformat()

        except Exception:
            # Any failure (import, network, parsing) leaves GPT3 values intact
            pass

    return p_hpa, t_c, dT, e_hpa, ah, aw, undu, extra

# -----------------------------------------------------------------------------
# Normalisasi sudut dan konstanta WGS84 (SOFA VM & IERS TN36)
# -----------------------------------------------------------------------------
def iauAnp(a: float) -> float:
    """Normalize angle into range 0 to 2π (IAU SOFA)."""
    w = math.fmod(a, 2.0 * math.pi)
    return w + 2.0 * math.pi if w < 0 else w

def iauAnpm(a: float) -> float:
    """Normalize angle into range -π to +π (IAU SOFA)."""
    w = math.fmod(a, 2.0 * math.pi)
    if abs(w) >= math.pi:
        w -= math.copysign(2.0 * math.pi, a)
    return w

# Konstanta WGS84 sesuai IERS TN36
WGS84_A = 6378137.0                # meter
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2

def geodetic_to_geocentric(lat_deg: float, lon_deg: float, elev_m: float,
                           xp: float = 0.0, yp: float = 0.0) -> np.ndarray:
    a = 6378.137  # km (WGS84)
    f = 1.0 / 298.257223563
    e2 = 2*f - f*f

    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    h = elev_m / 1000.0

    x = (N + h) * cos_lat * math.cos(lon_rad)
    y = (N + h) * cos_lat * math.sin(lon_rad)
    z = ((1 - e2) * N + h) * sin_lat

    if xp != 0.0 or yp != 0.0:
        x_obs = x
        y_obs = y
        x = x_obs + y_obs * xp
        y = y_obs - x_obs * xp
        z = z + (x_obs * yp - y_obs * xp)

    return np.array([x, y, z])

def sofa_refraction_constants(phpa: float, tc: float, rh: float, wl: float) -> tuple:
    """
    Hitung koefisien refraksi A dan B (dalam radian) sesuai model SOFA `iauRefco`.
    
    Parameters
    ----------
    phpa : float
        Tekanan atmosfer di lokasi pengamat (hPa = mbar).
    tc   : float
        Suhu udara (°C).
    rh   : float
        Kelembaban relatif (0.0 – 1.0).
    wl   : float
        Panjang gelombang efektif (mikrometer). wl > 100 → radio.
    
    Returns
    -------
    refa, refb : float
        Koefisien untuk model Δζ = A tan ζ + B tan³ζ (radian).
    """
    tk = tc + 273.15

    # Tekanan uap air saturasi (Gill 1982)
    if tc > 0.0:
        psat = 6.1121 * math.exp((18.678 - tc / 234.5) * tc / (257.14 + tc))
    else:
        psat = 6.1121 * math.exp((23.036 - tc / 333.7) * tc / (279.82 + tc))
    pwat = rh * psat

    # Indeks bias udara (SOFA `iauRefco`)
    if wl <= 100.0:                     # Optik / IR
        sigma = 1.0 / wl
        n1 = 287.6155 + 4.88660 * sigma**2 + 0.06800 * sigma**4
        n2 = 1.6289 + 0.01360 * sigma**2
    else:                               # Radio
        n1 = 77.6890
        n2 = 5.3180

    n = 1e-6 * (n1 * (phpa / tk) - n2 * (pwat / tk))

    beta = 4.4474e-6 * tk
    if wl > 100.0:
        beta *= (1.0 + 0.5054 * (pwat / tk))

    refa = n
    refb = n * (n / 2.0 - beta)
    return refa, refb

def calculate_refraction(alt_geom_deg: float,
                         pressure: float = JOLOTUNDO_ATMOSPHERE['standard_pressure_hpa'],
                         temp_c: float = JOLOTUNDO_ATMOSPHERE['standard_temperature_c'],
                         model: str = 'bennett',
                         mjd: Optional[float] = None,
                         lat_rad: Optional[float] = None,
                         lon_rad: Optional[float] = None,
                         height_m: Optional[float] = None,
                         az_rad: Optional[float] = None,
                         is_realtime: bool = False,
                         meteo_data: Optional[tuple] = None) -> float:
    """
    Koreksi refraksi atmosfer dengan physical clamping di bawah ufuk.
    Selaras dengan pembatasan horizon JPL Horizons.
    """
    # PHYSICAL CLAMP: 
    # Sinar cahaya dari objek langit tidak akan mencapai pengamat 
    # jika elevasi geometris terlalu jauh di bawah horizon.
    # JPL memotong refraksi di bawah elevasi tertentu untuk menghindari ledakan asimtotik.
    PHYSICAL_HORIZON_LIMIT = -1.0
    
    if alt_geom_deg < PHYSICAL_HORIZON_LIMIT:
        return 0.0

    if model == 'bennett':
        weather_factor = (pressure / 1010.0) * (283.0 / (273.0 + temp_c))
        # Sæmundsson modification of Bennett's formula
        term = alt_geom_deg + 10.3 / (alt_geom_deg + 5.11)
        tan_val = math.tan(math.radians(term))
        # Penanganan khusus agar pembagian tidak meledak di sekitar horizon limit
        if abs(tan_val) < 1e-6:
            tan_val = 1e-6 if tan_val >= 0 else -1e-6
        r_arcmin = 1.02 / tan_val
        
        ref_deg = (r_arcmin * weather_factor) / 60.0
        return max(ref_deg, 0.0)

    elif model == 'jpl':
        # Tekanan standar JPL (sea-level) dan Suhu 15C
        weather_factor = (1013.25 / 1010.0) * (283.0 / (273.0 + 15.0))
        term = alt_geom_deg + 10.3 / (alt_geom_deg + 5.11)
        tan_val = math.tan(math.radians(term))
        if abs(tan_val) < 1e-6:
            tan_val = 1e-6 if tan_val >= 0 else -1e-6
        r_arcmin = 1.02 / tan_val
        
        ref_deg = (r_arcmin * weather_factor) / 60.0
        return max(ref_deg, 0.0)

    elif model == 'simpson':
        return auer_standish_refraction(alt_geom_deg, pressure, temp_c, quadrature='simpson')

    elif model == 'gauss':
        return auer_standish_refraction(alt_geom_deg, pressure, temp_c, quadrature='gauss')

    elif model == 'gpt':
        if None in (mjd, lat_rad, lon_rad, height_m):
            raise ValueError("Parameter MJD, lat, lon, height diperlukan untuk model GPT3.")
        return calculate_refraction_gpt(alt_geom_deg, mjd, lat_rad, lon_rad, height_m,
                                        az_rad, is_realtime)

    elif model == 'vmf3':
        if None in (mjd, lat_rad, lon_rad, height_m):
            raise ValueError("Parameter MJD, lat, lon, height diperlukan untuk model VMF3.")
        # Teruskan argumen meteo_data ke fungsi VMF3
        return calculate_refraction_vmf3(alt_geom_deg, mjd, lat_rad, lon_rad, height_m,
                                         az_rad, is_realtime, meteo_data)               

    else:
        raise ValueError(f"Unknown refraction model: {model}")

# =============================================================================
# Fungsi GMF: Global Mapping Function (Boehm et al., 2006)
# =============================================================================
def gmf_mapping_function(mjd, lat_rad, lon_rad, height_m, zd_rad):
    """
    Menghitung hydrostatic dan wet mapping function menggunakan model GMF.

    Parameters
    ----------
    mjd : float
        Modified Julian Date.
    lat_rad : float
        Lintang (rad, utara+).
    lon_rad : float
        Bujur (rad, timur+).
    height_m : float
        Tinggi ellipsoidal dalam meter.
    zd_rad : float
        Jarak zenith dalam radian (π/2 - elevasi).

    Returns
    -------
    gmfh : float
        Hydrostatic mapping function.
    gmfw : float
        Wet mapping function.
    """
    TWOPI = 2.0 * math.pi
    doy = mjd - 44239.0 + 1 - 28  # seperti Niell (1996)

    nmax = 9

    # ---------- koefisien GMF ----------
    ah_mean = [
        1.2517e+02, 8.503e-01, 6.936e-02,-6.760e+00, 1.771e-01,
        1.130e-02, 5.963e-01, 1.808e-02, 2.801e-03,-1.414e-03,
       -1.212e+00, 9.300e-02, 3.683e-03, 1.095e-03, 4.671e-05,
        3.959e-01,-3.867e-02, 5.413e-03,-5.289e-04, 3.229e-04,
        2.067e-05, 3.000e-01, 2.031e-02, 5.900e-03, 4.573e-04,
       -7.619e-05, 2.327e-06, 3.845e-06, 1.182e-01, 1.158e-02,
        5.445e-03, 6.219e-05, 4.204e-06,-2.093e-06, 1.540e-07,
       -4.280e-08,-4.751e-01,-3.490e-02, 1.758e-03, 4.019e-04,
       -2.799e-06,-1.287e-06, 5.468e-07, 7.580e-08,-6.300e-09,
       -1.160e-01, 8.301e-03, 8.771e-04, 9.955e-05,-1.718e-06,
       -2.012e-06, 1.170e-08, 1.790e-08,-1.300e-09, 1.000e-10
    ]
    bh_mean = [
        0.000e+00, 0.000e+00, 3.249e-02, 0.000e+00, 3.324e-02,
        1.850e-02, 0.000e+00,-1.115e-01, 2.519e-02, 4.923e-03,
        0.000e+00, 2.737e-02, 1.595e-02,-7.332e-04, 1.933e-04,
        0.000e+00,-4.796e-02, 6.381e-03,-1.599e-04,-3.685e-04,
        1.815e-05, 0.000e+00, 7.033e-02, 2.426e-03,-1.111e-03,
       -1.357e-04,-7.828e-06, 2.547e-06, 0.000e+00, 5.779e-03,
        3.133e-03,-5.312e-04,-2.028e-05, 2.323e-07,-9.100e-08,
       -1.650e-08, 0.000e+00, 3.688e-02,-8.638e-04,-8.514e-05,
       -2.828e-05, 5.403e-07, 4.390e-07, 1.350e-08, 1.800e-09,
        0.000e+00,-2.736e-02,-2.977e-04, 8.113e-05, 2.329e-07,
        8.451e-07, 4.490e-08,-8.100e-09,-1.500e-09, 2.000e-10
    ]
    ah_amp = [
       -2.738e-01,-2.837e+00, 1.298e-02,-3.588e-01, 2.413e-02,
        3.427e-02,-7.624e-01, 7.272e-02, 2.160e-02,-3.385e-03,
        4.424e-01, 3.722e-02, 2.195e-02,-1.503e-03, 2.426e-04,
        3.013e-01, 5.762e-02, 1.019e-02,-4.476e-04, 6.790e-05,
        3.227e-05, 3.123e-01,-3.535e-02, 4.840e-03, 3.025e-06,
       -4.363e-05, 2.854e-07,-1.286e-06,-6.725e-01,-3.730e-02,
        8.964e-04, 1.399e-04,-3.990e-06, 7.431e-06,-2.796e-07,
       -1.601e-07, 4.068e-02,-1.352e-02, 7.282e-04, 9.594e-05,
        2.070e-06,-9.620e-08,-2.742e-07,-6.370e-08,-6.300e-09,
        8.625e-02,-5.971e-03, 4.705e-04, 2.335e-05, 4.226e-06,
        2.475e-07,-8.850e-08,-3.600e-08,-2.900e-09, 0.000e+00
    ]
    bh_amp = [
        0.000e+00, 0.000e+00,-1.136e-01, 0.000e+00,-1.868e-01,
       -1.399e-02, 0.000e+00,-1.043e-01, 1.175e-02,-2.240e-03,
        0.000e+00,-3.222e-02, 1.333e-02,-2.647e-03,-2.316e-05,
        0.000e+00, 5.339e-02, 1.107e-02,-3.116e-03,-1.079e-04,
       -1.299e-05, 0.000e+00, 4.861e-03, 8.891e-03,-6.448e-04,
       -1.279e-05, 6.358e-06,-1.417e-07, 0.000e+00, 3.041e-02,
        1.150e-03,-8.743e-04,-2.781e-05, 6.367e-07,-1.140e-08,
       -4.200e-08, 0.000e+00,-2.982e-02,-3.000e-03, 1.394e-05,
       -3.290e-05,-1.705e-07, 7.440e-08, 2.720e-08,-6.600e-09,
        0.000e+00, 1.236e-02,-9.981e-04,-3.792e-05,-1.355e-05,
        1.162e-06,-1.789e-07, 1.470e-08,-2.400e-09,-4.000e-10
    ]

    aw_mean = [
        5.640e+01, 1.555e+00,-1.011e+00,-3.975e+00, 3.171e-02,
        1.065e-01, 6.175e-01, 1.376e-01, 4.229e-02, 3.028e-03,
        1.688e+00,-1.692e-01, 5.478e-02, 2.473e-02, 6.059e-04,
        2.278e+00, 6.614e-03,-3.505e-04,-6.697e-03, 8.402e-04,
        7.033e-04,-3.236e+00, 2.184e-01,-4.611e-02,-1.613e-02,
       -1.604e-03, 5.420e-05, 7.922e-05,-2.711e-01,-4.406e-01,
       -3.376e-02,-2.801e-03,-4.090e-04,-2.056e-05, 6.894e-06,
        2.317e-06, 1.941e+00,-2.562e-01, 1.598e-02, 5.449e-03,
        3.544e-04, 1.148e-05, 7.503e-06,-5.667e-07,-3.660e-08,
        8.683e-01,-5.931e-02,-1.864e-03,-1.277e-04, 2.029e-04,
        1.269e-05, 1.629e-06, 9.660e-08,-1.015e-07,-5.000e-10
    ]
    bw_mean = [
        0.000e+00, 0.000e+00, 2.592e-01, 0.000e+00, 2.974e-02,
       -5.471e-01, 0.000e+00,-5.926e-01,-1.030e-01,-1.567e-02,
        0.000e+00, 1.710e-01, 9.025e-02, 2.689e-02, 2.243e-03,
        0.000e+00, 3.439e-01, 2.402e-02, 5.410e-03, 1.601e-03,
        9.669e-05, 0.000e+00, 9.502e-02,-3.063e-02,-1.055e-03,
       -1.067e-04,-1.130e-04, 2.124e-05, 0.000e+00,-3.129e-01,
        8.463e-03, 2.253e-04, 7.413e-05,-9.376e-05,-1.606e-06,
        2.060e-06, 0.000e+00, 2.739e-01, 1.167e-03,-2.246e-05,
       -1.287e-04,-2.438e-05,-7.561e-07, 1.158e-06, 4.950e-08,
        0.000e+00,-1.344e-01, 5.342e-03, 3.775e-04,-6.756e-05,
       -1.686e-06,-1.184e-06, 2.768e-07, 2.730e-08, 5.700e-09
    ]
    aw_amp = [
        1.023e-01,-2.695e+00, 3.417e-01,-1.405e-01, 3.175e-01,
        2.116e-01, 3.536e+00,-1.505e-01,-1.660e-02, 2.967e-02,
        3.819e-01,-1.695e-01,-7.444e-02, 7.409e-03,-6.262e-03,
       -1.836e+00,-1.759e-02,-6.256e-02,-2.371e-03, 7.947e-04,
        1.501e-04,-8.603e-01,-1.360e-01,-3.629e-02,-3.706e-03,
       -2.976e-04, 1.857e-05, 3.021e-05, 2.248e+00,-1.178e-01,
        1.255e-02, 1.134e-03,-2.161e-04,-5.817e-06, 8.836e-07,
       -1.769e-07, 7.313e-01,-1.188e-01, 1.145e-02, 1.011e-03,
        1.083e-04, 2.570e-06,-2.140e-06,-5.710e-08, 2.000e-08,
       -1.632e+00,-6.948e-03,-3.893e-03, 8.592e-04, 7.577e-05,
        4.539e-06,-3.852e-07,-2.213e-07,-1.370e-08, 5.800e-09
    ]
    bw_amp = [
        0.000e+00, 0.000e+00,-8.865e-02, 0.000e+00,-4.309e-01,
        6.340e-02, 0.000e+00, 1.162e-01, 6.176e-02,-4.234e-03,
        0.000e+00, 2.530e-01, 4.017e-02,-6.204e-03, 4.977e-03,
        0.000e+00,-1.737e-01,-5.638e-03, 1.488e-04, 4.857e-04,
       -1.809e-04, 0.000e+00,-1.514e-01,-1.685e-02, 5.333e-03,
       -7.611e-05, 2.394e-05, 8.195e-06, 0.000e+00, 9.326e-02,
       -1.275e-02,-3.071e-04, 5.374e-05,-3.391e-05,-7.436e-06,
        6.747e-07, 0.000e+00,-8.637e-02,-3.807e-03,-6.833e-04,
       -3.861e-05,-2.268e-05, 1.454e-06, 3.860e-07,-1.068e-07,
        0.000e+00,-2.658e-02,-1.947e-03, 7.131e-04,-3.506e-05,
        1.885e-07, 5.792e-07, 3.990e-08, 2.000e-08,-5.700e-09
    ]

    # ----- Polinomial Legendre (dengan faktorial seperti GMF.F) -----
    dfac = [1]
    for i in range(1, 2*nmax + 2):
        dfac.append(dfac[-1] * i)

    p = [[0.0]*(nmax+2) for _ in range(nmax+2)]
    t = math.sin(lat_rad)
    for i in range(nmax+1):
        for j in range(min(i, nmax)+1):
            ir = (i - j)//2
            sum1 = 0.0
            for k in range(ir+1):
                sum1 += ((-1)**k * dfac[2*i - 2*k + 1] / dfac[k+1] /
                         dfac[i - k + 1] / dfac[i - j - 2*k + 1] *
                         t**(i - j - 2*k))
            p[i+1][j+1] = (1.0 / 2**i) * math.sqrt((1 - t*t)**j) * sum1

    # Spherical Harmonics
    ap_list = []
    bp_list = []
    for n in range(nmax+1):
        for m in range(n+1):
            ap_list.append(p[n+1][m+1] * math.cos(m * lon_rad))
            bp_list.append(p[n+1][m+1] * math.sin(m * lon_rad))

    # ---------- Hydrostatic mapping function ----------
    bh = 0.0029
    c0h = 0.062
    if lat_rad < 0.0:
        phh = math.pi
        c11h = 0.007
        c10h = 0.002
    else:
        phh = 0.0
        c11h = 0.005
        c10h = 0.001
    ch = c0h + ((math.cos(doy/365.25*TWOPI + phh) + 1.0) * c11h/2.0 + c10h) * (1.0 - math.cos(lat_rad))

    ahm = 0.0
    aha = 0.0
    for i in range(55):
        ahm += (ah_mean[i]*ap_list[i] + bh_mean[i]*bp_list[i]) * 1e-5
        aha += (ah_amp[i]*ap_list[i]  + bh_amp[i]*bp_list[i])  * 1e-5
    ah = ahm + aha * math.cos(doy/365.25 * TWOPI)

    sine = math.sin(math.pi/2.0 - zd_rad)
    beta = bh / (sine + ch)
    gamma = ah / (sine + beta)
    topcon = (1.0 + ah/(1.0 + bh/(1.0 + ch)))
    gmfh = topcon / (sine + gamma)

    # Koreksi ketinggian untuk GMFH (Niell 1996)
    a_ht = 2.53e-5
    b_ht = 5.49e-3
    c_ht = 1.14e-3
    hs_km = height_m / 1000.0
    beta = b_ht / (sine + c_ht)
    gamma = a_ht / (sine + beta)
    topcon = (1.0 + a_ht/(1.0 + b_ht/(1.0 + c_ht)))
    ht_corr_coef = 1.0/sine - topcon/(sine + gamma)
    ht_corr = ht_corr_coef * hs_km
    gmfh += ht_corr

    # ---------- Wet mapping function ----------
    bw = 0.00146
    cw = 0.04391
    awm = 0.0
    awa = 0.0
    for i in range(55):
        awm += (aw_mean[i]*ap_list[i] + bw_mean[i]*bp_list[i]) * 1e-5
        awa += (aw_amp[i]*ap_list[i]  + bw_amp[i]*bp_list[i])  * 1e-5
    aw = awm + awa * math.cos(doy/365.25 * TWOPI)

    beta = bw / (sine + cw)
    gamma = aw / (sine + beta)
    topcon = (1.0 + aw/(1.0 + bw/(1.0 + cw)))
    gmfw = topcon / (sine + gamma)

    return gmfh, gmfw


# =============================================================================
# Model gradien atmosfer asimetris APG (Boehm et al., IERS Conventions 2010)
# Porting dari APG.F
# =============================================================================

# Koefisien spherical harmonics dari APG.F (55 elemen, degree/order maks 9)
_APG_AN = [
    2.8959e-02, -4.6440e-01, -8.6531e-03,  1.1836e-01, -2.4168e-02,
   -6.9072e-05,  2.6783e-01, -1.1697e-03, -2.3396e-03, -1.6206e-03,
   -7.4883e-02,  1.3583e-02,  1.7750e-03,  3.2496e-04,  8.8051e-05,
    9.6532e-02,  1.3192e-02,  5.5250e-04,  4.0507e-04, -5.4758e-06,
    9.4260e-06, -1.0872e-01,  5.7551e-03,  5.3986e-05, -2.3753e-04,
   -3.8241e-05,  1.7377e-06, -4.4135e-08,  2.1863e-01,  2.0228e-02,
   -2.0127e-04, -3.3669e-04,  8.7575e-06,  7.0461e-07, -4.0001e-08,
   -4.5911e-08, -3.1945e-03, -5.1369e-03,  3.0684e-04,  2.4459e-05,
    7.6575e-06, -5.5319e-07,  3.5133e-08,  1.1074e-08,  3.4623e-09,
   -1.5845e-01, -2.0376e-02, -4.0081e-04,  2.2062e-04, -7.9179e-06,
   -1.6441e-07, -5.0004e-08,  8.0689e-10, -2.3813e-10, -2.4483e-10
]

_APG_BN = [
    0.0000e+00,  0.0000e+00, -1.1930e-02,  0.0000e+00,  9.8349e-03,
   -1.6861e-03,  0.0000e+00,  4.3338e-03,  6.1707e-03,  7.4635e-04,
    0.0000e+00,  3.5124e-03,  2.1967e-03,  4.2029e-04,  2.4476e-06,
    0.0000e+00,  4.1373e-04, -2.3281e-03,  2.7382e-04, -8.5220e-05,
    1.4204e-05,  0.0000e+00, -8.0076e-03,  4.5587e-05, -5.8053e-05,
   -1.1021e-05,  7.2338e-07, -1.9827e-07,  0.0000e+00, -3.9229e-03,
   -4.0697e-04, -1.6992e-04,  5.4705e-06, -4.4594e-06,  2.0121e-07,
   -7.7840e-08,  0.0000e+00, -3.2916e-03, -1.2302e-03, -6.5735e-06,
   -3.1840e-06, -8.9836e-07,  1.1870e-07, -5.8781e-09, -2.9124e-09,
    0.0000e+00,  1.0759e-02, -6.6074e-05, -4.0635e-05,  8.7141e-06,
    6.4567e-07, -4.4684e-08, -5.0293e-11,  2.7723e-10,  1.6903e-10
]

_APG_AE = [
   -2.4104e-03,  1.1408e-04, -3.4621e-04,  1.6565e-03, -4.0620e-03,
   -6.8424e-03, -3.3718e-04,  7.3857e-03, -1.3324e-03, -1.5645e-03,
    4.6444e-03,  1.0296e-03,  3.6253e-03,  4.0329e-04,  3.1943e-04,
   -7.1992e-04,  4.8706e-03,  9.4300e-04,  2.0765e-04, -5.0987e-06,
   -7.1741e-06, -1.3131e-02,  2.9099e-04, -2.2509e-04,  2.6716e-04,
   -8.1815e-05,  8.4297e-06, -9.2378e-07, -5.8095e-04,  2.7501e-03,
    4.3659e-04, -8.2990e-06, -1.4808e-05,  2.2033e-06, -3.3215e-07,
    2.8858e-08,  9.9968e-03,  4.9291e-04,  3.3739e-05,  2.4696e-06,
   -8.1749e-06, -9.0052e-07,  2.0153e-07, -1.0271e-08,  1.8249e-09,
    3.0578e-03,  1.1229e-03, -1.9977e-04,  4.4581e-06, -7.6921e-06,
   -2.8308e-07,  1.0305e-07, -6.9026e-09,  1.5523e-10, -1.0395e-10
]

_APG_BE = [
    0.0000e+00,  0.0000e+00, -2.5396e-03,  0.0000e+00,  9.2146e-03,
   -7.5836e-03,  0.0000e+00,  1.2765e-02, -1.1436e-03,  1.7909e-04,
    0.0000e+00,  2.9318e-03, -6.8541e-04,  9.5775e-04,  2.4596e-05,
    0.0000e+00,  3.5662e-03, -1.3949e-03, -3.4597e-04, -5.8236e-05,
    5.6956e-06,  0.0000e+00, -5.0164e-04, -6.5585e-04,  1.1134e-05,
    2.3315e-05, -4.0521e-06, -4.1747e-07,  0.0000e+00,  5.1650e-04,
   -1.0483e-03,  5.8109e-06,  1.6406e-05, -1.6261e-06,  6.2992e-07,
    1.3134e-08,  0.0000e+00, -6.1449e-03, -3.2511e-04,  1.7646e-04,
    7.5326e-06, -1.1946e-06,  5.1217e-08,  2.4618e-08,  3.6290e-09,
    0.0000e+00,  3.6769e-03, -9.7683e-04, -3.2096e-07,  1.3860e-06,
   -6.2832e-09,  2.6918e-09,  2.5705e-09, -2.4401e-09, -3.7917e-11
]


def apg_gradient_delay(lat_rad, lon_rad, az_rad, el_rad):
    """
    Menghitung koreksi asimetris troposfer menggunakan model APG
    (IERS Conventions 2010, Bab 9.2, subroutine APG.F).

    Parameters
    ----------
    lat_rad, lon_rad : float
        Koordinat geodetik stasiun (radian).
    az_rad : float
        Azimuth pengamatan dari utara (radian).
    el_rad : float
        Elevasi geometris (radian).

    Returns
    -------
    D_m : float
        Delay asimetris dalam meter.
    GRN_mm : float
        Gradien utara dalam mm.
    GRE_mm : float
        Gradien timur dalam mm.
    """
    NMAX = 9
    MMAX = 9

    # Vektor satuan stasiun
    x = math.cos(lat_rad) * math.cos(lon_rad)
    y = math.cos(lat_rad) * math.sin(lon_rad)
    z = math.sin(lat_rad)

    # Inisialisasi Legendre polynomials
    V = [[0.0] * (MMAX + 2) for _ in range(NMAX + 2)]
    W = [[0.0] * (MMAX + 2) for _ in range(NMAX + 2)]
    V[1][1] = 1.0
    W[1][1] = 0.0
    V[2][1] = z * V[1][1]
    W[2][1] = 0.0

    for n in range(2, NMAX + 1):
        V[n + 1][1] = ((2 * n - 1) * z * V[n][1] - (n - 1) * V[n - 1][1]) / n
        W[n + 1][1] = 0.0

    for m in range(1, MMAX + 1):
        V[m + 1][m + 1] = (2 * m - 1) * (x * V[m][m] - y * W[m][m])
        W[m + 1][m + 1] = (2 * m - 1) * (x * W[m][m] + y * V[m][m])
        if m < MMAX:
            V[m + 2][m + 1] = (2 * m + 1) * z * V[m + 1][m + 1]
            W[m + 2][m + 1] = (2 * m + 1) * z * W[m + 1][m + 1]
        for n in range(m + 2, NMAX + 1):
            V[n + 1][m + 1] = ((2 * n - 1) * z * V[n][m + 1] - (n + m - 1) * V[n - 1][m + 1]) / (n - m)
            W[n + 1][m + 1] = ((2 * n - 1) * z * W[n][m + 1] - (n + m - 1) * W[n - 1][m + 1]) / (n - m)

    # Hitung gradien utara dan timur
    GRN = 0.0
    GRE = 0.0
    idx = 0
    for n in range(0, NMAX + 1):
        for m in range(0, n + 1):
            GRN += (_APG_AN[idx] * V[n + 1][m + 1] + _APG_BN[idx] * W[n + 1][m + 1])
            GRE += (_APG_AE[idx] * V[n + 1][m + 1] + _APG_BE[idx] * W[n + 1][m + 1])
            idx += 1

    # Delay asimetris (Chen & Herring, 1997)
    sin_el = math.sin(el_rad)
    tan_el = math.tan(el_rad)
    D_mm = (1.0 / (sin_el * tan_el + 0.0031)) * (GRN * math.cos(az_rad) + GRE * math.sin(az_rad))
    D_m = D_mm / 1000.0   # mm -> meter

    return D_m, GRN, GRE

def gpt3_optical_refraction(alt_geom_deg, mjd, lat_rad, lon_rad, height_m,
                            az_rad=None, is_realtime=False):
    """
    Refraksi optik presisi tinggi dengan asimilasi ECMWF-GPT3
    dan integrator Gauss‑Legendre 24 titik.
    """
    if alt_geom_deg < -2.0:
        return 0.0

    # === Asimilasi data meteo (P, T, e) dari GPT3 + override ECMWF jika diperlukan ===
    p_hpa, t_c, _, e_hpa, _, _, _, _ = hybrid_meteo_assimilation(
        mjd, lat_rad, lon_rad, height_m, is_realtime
    )

    R_earth = 6371.0e3
    H_hydro = 8400.0
    H_wet   = 2000.0
    n_dry = 1.000292
    n_wet_ref = 1.000256
    P0 = 1013.25

    n0_hydro = 1.0 + (n_dry - 1.0) * (p_hpa - e_hpa) / P0
    n0_wet   = 1.0 + (n_wet_ref - 1.0) * e_hpa / P0

    z_obs = math.radians(90.0 - alt_geom_deg)

    # === Integrand ===
    def integrand(y, n0, H_scale):
        y = np.asarray(y)
        n = 1.0 + (n0 - 1.0) * np.exp(-y)
        r = R_earth + y * H_scale
        sin_chi = (n0 * R_earth * np.sin(z_obs)) / (n * r)
        sin_chi = np.clip(sin_chi, -1.0, 1.0)
        tan_chi = sin_chi / np.sqrt(1.0 - sin_chi**2)
        return tan_chi * np.exp(-y) / n

    # === Integrator Gauss‑Legendre 24 titik (SciPy) ===
    from scipy.integrate import fixed_quad

    def integrate_component(n0, H_scale):
        s1 = fixed_quad(lambda y: integrand(y, n0, H_scale), 0.0, 1.0, n=24)[0]
        s2 = fixed_quad(lambda y: integrand(y, n0, H_scale), 1.0, 4.0, n=24)[0]
        s3 = fixed_quad(lambda y: integrand(y, n0, H_scale), 4.0, 10.0, n=24)[0]
        return s1 + s2 + s3

    integral_hydro = integrate_component(n0_hydro, H_hydro)
    integral_wet   = integrate_component(n0_wet,   H_wet)

    R_rad = (n0_hydro - 1.0) * integral_hydro + (n0_wet - 1.0) * integral_wet

    # === Koreksi APG (opsional) ===
    if az_rad is not None and alt_geom_deg < 15.0:
        try:
            D_apg_m, _, _ = apg_gradient_delay(lat_rad, lon_rad, az_rad, math.radians(alt_geom_deg))
            R_earth_km = 6371.0
            zd_rad = math.radians(90.0 - alt_geom_deg)
            if math.tan(zd_rad) > 1e-6:
                delta_apg_rad = (D_apg_m / 1000.0) / (R_earth_km * math.tan(zd_rad))
                R_rad += delta_apg_rad
        except Exception:
            pass

    return math.degrees(R_rad)

def calculate_refraction_gpt(alt_geom_deg, mjd, lat_rad, lon_rad, height_m,
                             az_rad=None, is_realtime=False):
    """
    Refraksi optik menggunakan GPT (tekanan & suhu) + integrator Auer-Standish (Gauss).
    """
    return gpt3_optical_refraction(alt_geom_deg, mjd, lat_rad, lon_rad, height_m,
                                   az_rad=az_rad, is_realtime=is_realtime)

def calculate_refraction_vmf3(alt_geom_deg, mjd, lat_rad, lon_rad, height_m,
                              az_rad=None, is_realtime=False, meteo_data=None):
    """
    Refraksi berbasis VMF3 + GPT3 untuk aplikasi geodesi, dengan asimilasi ECMWF.
    """
    if alt_geom_deg < -1.0:
        return 0.0
    import math
    zd_rad = math.radians(90.0 - alt_geom_deg)

    # === Bypass Asimilasi Jika Data Statis Disuplai ===
    if meteo_data is not None:
        p_hpa, t_c, dT, e_hpa, ah, aw, undu = meteo_data[:7]
    else:
        p_hpa, t_c, dT, e_hpa, ah, aw, undu, _ = hybrid_meteo_assimilation(
            mjd, lat_rad, lon_rad, height_m, is_realtime
        )
        
    mfh, mfw = vmf3_ht(mjd, lat_rad, lon_rad, height_m, zd_rad, ah, aw)

    zhd = 0.0022768 * p_hpa / (1 - 0.00266 * math.cos(2*lat_rad) - 0.28e-6 * height_m)
    zwd = 0.0022768 * (1255.0 / (t_c + 273.15) + 0.05) * e_hpa
    slant_delay_m = zhd * mfh + zwd * mfw

    if az_rad is not None and alt_geom_deg < 15.0:
        D_apg_m, _, _ = apg_gradient_delay(lat_rad, lon_rad, az_rad, math.radians(alt_geom_deg))
        slant_delay_m += D_apg_m

    R_earth_km = 6371.0
    ref_rad = (slant_delay_m / 1000.0) / (R_earth_km * math.tan(zd_rad))
    return math.degrees(ref_rad)


# -----------------------------------------------------------------------------
# Kuadratur Simpson 1/3 (via SciPy)
# -----------------------------------------------------------------------------
from scipy.integrate import simpson as _scipy_simpson

def simpson_n(f, a, b, n=200):
    if n % 2 != 0:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = np.vectorize(f)(x)
    return _scipy_simpson(y, x=x)


# -----------------------------------------------------------------------------
# Kuadratur Gauss‑Legendre orde 16 (via NumPy)
# -----------------------------------------------------------------------------
from numpy.polynomial.legendre import leggauss

def gauss_legendre_16(f, a, b):
    nodes, weights = leggauss(16)
    mid, half = 0.5 * (a + b), 0.5 * (b - a)
    total = 0.0
    for node, w in zip(nodes, weights):
        total += w * f(mid + half * node)
    return half * total


# -----------------------------------------------------------------------------
# Refraksi Auer‑Standish dengan atmosfer eksponensial seragam
# -----------------------------------------------------------------------------
def auer_standish_refraction(alt_deg, pressure_hpa=1013.25, temp_c=15.0,
                             quadrature='simpson'):
    """
    Refraksi Auer‑Standish menggunakan model atmosfer eksponensial seragam.
    
    Parameters
    ----------
    alt_deg : float
        Elevasi geometris (derajat).
    pressure_hpa : float
        Tekanan permukaan (hPa).
    temp_c : float
        Suhu permukaan (°C).
    quadrature : str
        'simpson' untuk Simpson 10‑titik, 'gauss' untuk Gauss‑Legendre 16‑titik.
    
    Returns
    -------
    ref_deg : float
        Koreksi refraksi dalam derajat.
    """
    if alt_deg < -1.0:
        return 0.0

    import math

    # Konstanta fisik
    R_earth = 6371.0e3          # meter
    H_scale = 8400.0            # meter (skala tinggi atmosfer)
    n0 = 1.000292               # indeks bias permukaan (λ = 550 nm)

    # Koreksi tekanan dan suhu
    n0 = 1.0 + (n0 - 1.0) * (pressure_hpa / 1013.25) * (273.15 / (temp_c + 273.15))

    z_obs = math.radians(90.0 - alt_deg)

    def integrand(y):
        # Indeks bias pada ketinggian y (dalam satuan scale height)
        n = 1.0 + (n0 - 1.0) * math.exp(-y)
        
        # Jarak radial dari pusat bumi
        r = R_earth + y * H_scale
        
        # Hukum Snellius untuk atmosfer sferis
        sin_chi = (n0 * R_earth * math.sin(z_obs)) / (n * r)
        
        # Menghindari error pembagian dengan nol di horizon
        if sin_chi >= 1.0:
            tan_chi = 1e6 
        else:
            tan_chi = sin_chi / math.sqrt(1.0 - sin_chi**2)
        
        # Integran: tan(chi) * (1/n) * e^(-y)
        return tan_chi * math.exp(-y) / n

    # Pilih metode kuadratur dengan penanganan khusus untuk presisi tinggi
    if quadrature == 'gauss':
        # Domain Splitting: 
        # Memusatkan 16 titik pertama di lapisan paling padat (0 - 2 scale heights / ~0-17 km)
        # dan 16 titik kedua di sisa atmosfer (2 - 10 scale heights / ~17-84 km)
        integral_bawah = gauss_legendre_16(integrand, 0.0, 2.0)
        integral_atas  = gauss_legendre_16(integrand, 2.0, 10.0)
        integral = integral_bawah + integral_atas
    else:  
        # Default simpson dengan interval jauh lebih rapat (200 interval)
        integral = simpson_n(integrand, 0.0, 10.0, n=200)

    # Kalikan faktor (n0 - 1) yang dikeluarkan dari turunan dn
    R_rad = (n0 - 1.0) * integral
    return math.degrees(R_rad)
