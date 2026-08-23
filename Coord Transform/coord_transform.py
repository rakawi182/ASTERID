#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
coord_transform – High-precision celestial coordinate transformations
====================================================================

Provides (all in radians, JD in TT unless stated):

1. Spherical <-> Cartesian
2. Rotation matrices (Rx, Ry, Rz)
3. Equatorial <-> Ecliptic (IAU 2006 obliquity)
4. GCRS <-> ITRS (CIP‑CIO or Equinox‑based) – using EarthRotation.py if available
5. Horizontal (az/alt) with atmospheric refraction (GPT3/VMF3 or Bennett)
6. Diurnal corrections (aberration, light deflection) – if EarthRotation available
7. Sun and Moon directions (using VSOP87 and ELP/MPP02 if available)

If optional modules are missing, the corresponding functions will
raise a clear error or use simplified fallbacks.

Author:   ASTERID Project
Version:  2.1 (Robust standalone)
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any

# ============================================================================
# 1. Constants
# ============================================================================
J2000_JD = 2451545.0
MJD_ZERO = 2400000.5
OBLIQUITY_J2000_RAD = 84381.406 * (math.pi / (180.0 * 3600.0))   # IAU 2006

# ============================================================================
# 2. Basic vector and rotation utilities
# ============================================================================
def spherical_to_cartesian(lon: float, lat: float, r: float = 1.0) -> np.ndarray:
    """Convert spherical (lon, lat, r) to Cartesian (x, y, z)."""
    cl = math.cos(lat)
    return np.array([r * cl * math.cos(lon),
                     r * cl * math.sin(lon),
                     r * math.sin(lat)])

def cartesian_to_spherical(vec: np.ndarray) -> Tuple[float, float, float]:
    """Convert Cartesian (x, y, z) to spherical (lon, lat, r)."""
    x, y, z = vec[0], vec[1], vec[2]
    r = math.hypot(x, math.hypot(y, z))
    if r == 0.0:
        return (0.0, 0.0, 0.0)
    lon = math.atan2(y, x)
    lat = math.asin(z / r)
    return lon, lat, r

def unit_vector(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0.0 else vec

def rot_x(angle: float) -> np.ndarray:
    c = math.cos(angle); s = math.sin(angle)
    return np.array([[1.0, 0.0, 0.0],
                     [0.0,   c,   s],
                     [0.0,  -s,   c]])

def rot_y(angle: float) -> np.ndarray:
    c = math.cos(angle); s = math.sin(angle)
    return np.array([[  c, 0.0,  -s],
                     [0.0, 1.0, 0.0],
                     [  s, 0.0,   c]])

def rot_z(angle: float) -> np.ndarray:
    c = math.cos(angle); s = math.sin(angle)
    return np.array([[  c,   s, 0.0],
                     [ -s,   c, 0.0],
                     [0.0, 0.0, 1.0]])

# ============================================================================
# 3. Equatorial <-> Ecliptic (IAU 2006)
# ============================================================================
def equatorial_to_ecliptic(ra: float, dec: float, obliquity: Optional[float] = None) -> Tuple[float, float]:
    if obliquity is None:
        obliquity = OBLIQUITY_J2000_RAD
    vec = spherical_to_cartesian(ra, dec)
    vec_ec = rot_x(-obliquity) @ vec
    lon, lat, _ = cartesian_to_spherical(vec_ec)
    return lon, lat

def ecliptic_to_equatorial(lon_ecl: float, lat_ecl: float, obliquity: Optional[float] = None) -> Tuple[float, float]:
    if obliquity is None:
        obliquity = OBLIQUITY_J2000_RAD
    vec = spherical_to_cartesian(lon_ecl, lat_ecl)
    vec_eq = rot_x(obliquity) @ vec
    ra, dec, _ = cartesian_to_spherical(vec_eq)
    return ra, dec

# ============================================================================
# 4. GCRS <-> ITRS (using EarthRotation if available)
# ============================================================================
try:
    from EarthRotation import (
        EarthOrientation,
        gcrs_to_itrs_cip, itrs_to_gcrs_cip,
        gcrs_to_itrs_cip_pv, itrs_to_gcrs_cip_pv,
        apply_diurnal_corrections,
    )
    _EARTHROT_AVAILABLE = True
except ImportError:
    _EARTHROT_AVAILABLE = False
    print("WARNING: EarthRotation.py not found. GCRS<->ITRS and diurnal corrections disabled.")

class CoordinateTransformer:
    """
    Unified interface for coordinate transformations.
    Uses EarthRotation if available, otherwise falls back to basic functions.
    """
    def __init__(self, eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt"):
        self.eop_file = eop_file
        if _EARTHROT_AVAILABLE:
            self.eo = EarthOrientation(eop_file)
        else:
            self.eo = None

    # ---- Equatorial <-> Ecliptic ----
    @staticmethod
    def equatorial_to_ecliptic(ra: float, dec: float) -> Tuple[float, float]:
        return equatorial_to_ecliptic(ra, dec)

    @staticmethod
    def ecliptic_to_equatorial(lon_ecl: float, lat_ecl: float) -> Tuple[float, float]:
        return ecliptic_to_equatorial(lon_ecl, lat_ecl)

    # ---- GCRS <-> ITRS ----
    def gcrs_to_itrs(self, gcrs_vec: np.ndarray, tt_jd: float,
                     paradigm: str = 'cip', use_eop: bool = True) -> np.ndarray:
        if not _EARTHROT_AVAILABLE:
            raise RuntimeError("EarthRotation module not available.")
        return self.eo.gcrs_to_itrs(gcrs_vec, tt_jd, paradigm, use_eop)

    def itrs_to_gcrs(self, itrs_vec: np.ndarray, tt_jd: float,
                     paradigm: str = 'cip', use_eop: bool = True) -> np.ndarray:
        if not _EARTHROT_AVAILABLE:
            raise RuntimeError("EarthRotation module not available.")
        return self.eo.itrs_to_gcrs(itrs_vec, tt_jd, paradigm, use_eop)

    def gcrs_to_itrs_pv(self, pv_gcrs: np.ndarray, tt_jd: float,
                        use_eop: bool = True) -> np.ndarray:
        if not _EARTHROT_AVAILABLE:
            raise RuntimeError("EarthRotation module not available.")
        return self.eo.gcrs_to_itrs_pv_analytic(pv_gcrs, tt_jd, use_eop)

    def itrs_to_gcrs_pv(self, pv_itrs: np.ndarray, tt_jd: float,
                        use_eop: bool = True) -> np.ndarray:
        if not _EARTHROT_AVAILABLE:
            raise RuntimeError("EarthRotation module not available.")
        return self.eo.itrs_to_gcrs_pv_analytic(pv_itrs, tt_jd, use_eop)

    # ---- Horizontal (az/alt) ----
    def gcrs_to_horizontal(self, gcrs_vec: np.ndarray, tt_jd: float,
                           lat_rad: float, lon_rad: float, height_m: float,
                           apply_refraction: bool = True,
                           refraction_model: str = 'bennett',
                           apply_diurnal: bool = True,
                           sun_gcrs: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """
        Convert GCRS unit vector to horizontal (azimuth, altitude).

        Parameters
        ----------
        gcrs_vec : array (3,)
            Unit vector in GCRS.
        tt_jd : float
            Julian date in TT.
        lat_rad, lon_rad : float
            Observer latitude, longitude (radians, north/east positive).
        height_m : float
            Height above ellipsoid (meters).
        apply_refraction : bool
            Apply atmospheric refraction correction.
        refraction_model : str
            'bennett', 'gpt', 'vmf3' (gpt/vmf3 require Atmospheric_refraction.py).
        apply_diurnal : bool
            Apply aberration and light deflection (requires EarthRotation).
        sun_gcrs : array (3,), optional
            Unit vector to Sun in GCRS (needed for light deflection).

        Returns
        -------
        az, alt : float
            Azimuth (radians from north, eastward) and altitude (radians).
        """
        # 1. Diurnal corrections if requested
        if apply_diurnal:
            if not _EARTHROT_AVAILABLE:
                raise RuntimeError("Diurnal corrections require EarthRotation.py.")
            eop = self.eo.get_eop_corrections(tt_jd)
            ut1_jd = self.eo.ut1_jd_from_tt(tt_jd, eop['dut1'])
            if sun_gcrs is None:
                sun_gcrs = self.sun_direction_gcrs(tt_jd)
            gcrs_vec = apply_diurnal_corrections(
                gcrs_vec, tt_jd, ut1_jd, eop['xp'], eop['yp'],
                lat_rad, lon_rad, height_m, sun_gcrs
            )

        # 2. GCRS -> ITRS (always CIP‑CIO)
        if _EARTHROT_AVAILABLE:
            eop = self.eo.get_eop_corrections(tt_jd)
            ut1_jd = self.eo.ut1_jd_from_tt(tt_jd, eop['dut1'])
            itrs_vec = gcrs_to_itrs_cip(gcrs_vec, tt_jd, ut1_jd,
                                        eop['xp'], eop['yp'], eop['dX'], eop['dY'])
        else:
            # Fallback: no rotation (assume GCRS == ITRS) – only for testing
            itrs_vec = gcrs_vec

        # 3. ITRS -> local horizontal (ENU)
        x, y, z = itrs_vec[0], itrs_vec[1], itrs_vec[2]
        sl = math.sin(lat_rad); cl = math.cos(lat_rad)
        so = math.sin(lon_rad); co = math.cos(lon_rad)

        east = -so * x + co * y
        north = -sl * co * x - sl * so * y + cl * z
        up = cl * co * x + cl * so * y + sl * z

        alt_geom = math.asin(max(-1.0, min(1.0, up)))
        az = math.atan2(east, north)
        if az < 0.0:
            az += 2.0 * math.pi

        # 4. Atmospheric refraction
        if apply_refraction:
            alt_deg = math.degrees(alt_geom)
            if refraction_model in ('gpt', 'vmf3'):
                try:
                    from Atmospheric_refraction import calculate_refraction
                    mjd = tt_jd - MJD_ZERO
                    ref_deg = calculate_refraction(alt_deg, model=refraction_model,
                                                   mjd=mjd, lat_rad=lat_rad,
                                                   lon_rad=lon_rad, height_m=height_m,
                                                   az_rad=az)
                except ImportError:
                    raise RuntimeError("Atmospheric_refraction.py required for GPT/VMF3 model.")
            else:
                # Bennett's formula (standard)
                weather = (1013.25 / 1010.0) * (283.0 / (273.0 + 15.0))
                term = alt_deg + 10.3 / (alt_deg + 5.11)
                tan_term = math.tan(math.radians(term))
                if abs(tan_term) < 1e-6:
                    tan_term = 1e-6 if tan_term >= 0 else -1e-6
                ref_arcmin = 1.02 / tan_term
                ref_deg = (ref_arcmin * weather) / 60.0
            alt = alt_geom + math.radians(max(ref_deg, 0.0))
        else:
            alt = alt_geom

        return az, alt

    def horizontal_to_gcrs(self, az: float, alt: float, tt_jd: float,
                           lat_rad: float, lon_rad: float, height_m: float,
                           remove_refraction: bool = True,
                           refraction_model: str = 'bennett') -> np.ndarray:
        """
        Convert horizontal (az, alt) to GCRS unit vector.
        """
        # 1. Remove refraction
        alt_geom = alt
        if remove_refraction:
            alt_deg = math.degrees(alt)
            if refraction_model in ('gpt', 'vmf3'):
                try:
                    from Atmospheric_refraction import calculate_refraction
                    mjd = tt_jd - MJD_ZERO
                    ref_deg = calculate_refraction(alt_deg, model=refraction_model,
                                                   mjd=mjd, lat_rad=lat_rad,
                                                   lon_rad=lon_rad, height_m=height_m,
                                                   az_rad=az)
                except ImportError:
                    raise RuntimeError("Atmospheric_refraction.py required for GPT/VMF3 model.")
            else:
                # Bennett's formula (standard)
                weather = (1013.25 / 1010.0) * (283.0 / (273.0 + 15.0))
                term = alt_deg + 10.3 / (alt_deg + 5.11)
                tan_term = math.tan(math.radians(term))
                if abs(tan_term) < 1e-6:
                    tan_term = 1e-6 if tan_term >= 0 else -1e-6
                ref_arcmin = 1.02 / tan_term
                ref_deg = (ref_arcmin * weather) / 60.0
            alt_geom = alt - math.radians(max(ref_deg, 0.0))

        # 2. ENU -> ITRS
        east = math.sin(az) * math.cos(alt_geom)
        north = math.cos(az) * math.cos(alt_geom)
        up = math.sin(alt_geom)

        sl = math.sin(lat_rad); cl = math.cos(lat_rad)
        so = math.sin(lon_rad); co = math.cos(lon_rad)
        x = -so * east - sl * co * north + cl * co * up
        y = co * east - sl * so * north + cl * so * up
        z = cl * north + sl * up
        itrs_vec = np.array([x, y, z])

        # 3. ITRS -> GCRS
        if _EARTHROT_AVAILABLE:
            eop = self.eo.get_eop_corrections(tt_jd)
            ut1_jd = self.eo.ut1_jd_from_tt(tt_jd, eop['dut1'])
            gcrs_vec = itrs_to_gcrs_cip(itrs_vec, tt_jd, ut1_jd,
                                        eop['xp'], eop['yp'], eop['dX'], eop['dY'])
        else:
            gcrs_vec = itrs_vec  # fallback

        return gcrs_vec

    # ---- Ephemeris (Sun, Moon) ----
    def sun_direction_gcrs(self, tt_jd: float) -> np.ndarray:
        """
        Unit vector from Earth to Sun in GCRS.
        Requires VSOP87A.py with file VSOP87A_ear.txt.
        """
        try:
            from Timescales import tt_to_tdb, split_jd, combine_jd
            from VSOP87A import VSOP87A
            tdb_jd = tt_to_tdb(*split_jd(tt_jd))
            tdb = combine_jd(*tdb_jd)
            earth = VSOP87A(VSOP87A.find_file("ear"))
            x, y, z, _, _, _ = earth.compute(tdb)
            sun_gcrs = -np.array([x, y, z])
            return unit_vector(sun_gcrs)
        except ImportError as e:
            raise RuntimeError(f"VSOP87A module not available: {e}")

    def moon_position_gcrs(self, tt_jd: float, icor: int = 1) -> np.ndarray:
        """
        Geocentric position of the Moon in GCRS (km).
        Requires ELP_MPP02_full.py.
        """
        try:
            from Timescales import tt_to_tdb, split_jd, combine_jd
            from ELP_MPP02_full import elpmpp02_icrs
            tdb_jd = tt_to_tdb(*split_jd(tt_jd))
            tdb = combine_jd(*tdb_jd)
            tj = tdb - J2000_JD
            xyz = elpmpp02_icrs(tj, icor=icor)
            return xyz[:3]
        except ImportError as e:
            raise RuntimeError(f"ELP_MPP02_full module not available: {e}")

    def moon_direction_gcrs(self, tt_jd: float) -> np.ndarray:
        pos = self.moon_position_gcrs(tt_jd)
        return unit_vector(pos)

# ============================================================================
# Convenience functions (standalone)
# ============================================================================
def equatorial_to_ecliptic_quick(ra: float, dec: float) -> Tuple[float, float]:
    return equatorial_to_ecliptic(ra, dec)

def ecliptic_to_equatorial_quick(lon_ecl: float, lat_ecl: float) -> Tuple[float, float]:
    return ecliptic_to_equatorial(lon_ecl, lat_ecl)

def gcrs_to_itrs_quick(gcrs_vec: np.ndarray, tt_jd: float,
                       paradigm: str = 'cip', use_eop: bool = True,
                       eop_file: str = "finals2000A.all.csv") -> np.ndarray:
    ct = CoordinateTransformer(eop_file)
    return ct.gcrs_to_itrs(gcrs_vec, tt_jd, paradigm, use_eop)

def itrs_to_gcrs_quick(itrs_vec: np.ndarray, tt_jd: float,
                       paradigm: str = 'cip', use_eop: bool = True,
                       eop_file: str = "finals2000A.all.csv") -> np.ndarray:
    ct = CoordinateTransformer(eop_file)
    return ct.itrs_to_gcrs(itrs_vec, tt_jd, paradigm, use_eop)

def gcrs_to_horizontal_quick(gcrs_vec: np.ndarray, tt_jd: float,
                             lat_rad: float, lon_rad: float, height_m: float,
                             **kwargs) -> Tuple[float, float]:
    ct = CoordinateTransformer(kwargs.pop('eop_file', "finals2000A.all.csv"))
    return ct.gcrs_to_horizontal(gcrs_vec, tt_jd, lat_rad, lon_rad, height_m, **kwargs)

# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("COORDINATE TRANSFORMER – Self Test")
    print("=" * 60)

    ct = CoordinateTransformer()

    # Test spherical <-> Cartesian
    lon, lat = 1.2, 0.5
    vec = spherical_to_cartesian(lon, lat)
    lon2, lat2, r2 = cartesian_to_spherical(vec)
    print(f"Spherical -> Cartesian -> Spherical: {lon:.8f}->{lon2:.8f}, {lat:.8f}->{lat2:.8f}")

    # Test equatorial <-> ecliptic
    ra, dec = 1.2, 0.5
    l_ecl, b_ecl = ct.equatorial_to_ecliptic(ra, dec)
    ra2, dec2 = ct.ecliptic_to_equatorial(l_ecl, b_ecl)
    print(f"Equatorial <-> Ecliptic: RA {ra:.6f}->{ra2:.6f}, Dec {dec:.6f}->{dec2:.6f}")

    # Test GCRS -> ITRS if EarthRotation available
    try:
        vec = np.array([1.0, 0.0, 0.0])
        tt = J2000_JD
        vec_itrs = ct.gcrs_to_itrs(vec, tt, use_eop=False)
        print(f"GCRS vector {vec} -> ITRS: {vec_itrs}")
    except Exception as e:
        print(f"GCRS->ITRS skipped: {e}")

    # Test horizontal with Bennett refraction
    try:
        lat_obs = math.radians(45.0)
        lon_obs = math.radians(10.0)
        h_obs = 100.0
        az, alt = ct.gcrs_to_horizontal(vec, tt, lat_obs, lon_obs, h_obs,
                                        apply_refraction=True,
                                        refraction_model='bennett',
                                        apply_diurnal=False)
        print(f"Horizontal: az={math.degrees(az):.3f}°, alt={math.degrees(alt):.3f}°")
    except Exception as e:
        print(f"Horizontal skipped: {e}")

    print("=" * 60)
    print("Test completed.")
