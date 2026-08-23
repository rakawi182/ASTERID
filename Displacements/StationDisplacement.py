#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StationDisplacement – IERS 2010 Conventional Displacement Models (High Precision)
================================================================================

Implementasi lengkap model perpindahan stasiun geodetik sesuai IERS Conventions 2010,
dengan tingkat presisi yang setara dengan modul IERS_2010.py.

Model yang disertakan:
    1. Solid Earth tides (Dehant, Mathews & Gipson) – lengkap dengan koreksi ST1, STEP2.
    2. Ocean tide loading (FES2014b) – melalui OTL_fes2014.
    3. Ocean pole tide loading (Desai 2002).
    4. Pole tide (rotational deformation) – IERS 2010 Sec 7.1.4 dengan Love numbers yang benar.
    5. Atmospheric pressure loading (Ray & Ponte 2003) – dengan fundamental arguments lengkap.
"""

import sys
sys.dont_write_bytecode = True

import os
import math
import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.special import lpmv, gammaln

# --------------------------------------------------------------------------
# High-precision time modules (required)
# --------------------------------------------------------------------------
try:
    from Timescales import (
        J2000_JD, cal_to_jd, combine_jd, jd_to_cal, 
        tai_utc, delta_t_from_jd, split_jd
    )
except ImportError:
    raise ImportError("Module 'Timescales' is required for time conversions.")

try:
    from EarthRotation import gst_from_ut1   # only needed for GMST if not recomputed
except ImportError:
    gst_from_ut1 = None


# =============================================================================
# INISIALISASI DATA OCEAN POLE TIDE LOADING (DESAI 2002)
# =============================================================================
_OPOLE_PATH = os.path.join(os.path.dirname(__file__), 'opoleloadcoefcmcor.npz')
try:
    _opole_archive = np.load(_OPOLE_PATH)
    # Asumsi: array disimpan dengan key 'load_coef'
    _OPOLE_GRID = _opole_archive['load_coef'] 
except Exception as e:
    _OPOLE_GRID = None
    print(f"Peringatan: Gagal memuat grid Ocean Pole Tide: {e}")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
MAS_TO_RAD = 1e-3 * ARCSEC_TO_RAD
UAS_TO_RAD = 1e-6 * ARCSEC_TO_RAD
RAD_TO_ARCSEC = 1.0 / ARCSEC_TO_RAD

A_E = 6378136.6          # Equatorial radius ITRS (m)
GM_EARTH = 3.986004418e14 
OMEGA_E = 7.292115e-5    
G_EQ = 9.7803278         
RHO_W = 1025.0           # Density of sea water (kg/m³) for ocean pole tide

# --------------------------------------------------------------------------
# Fundamental arguments for nutation and tides (IERS 2010, Eqs. 5.43, 5.44)
# --------------------------------------------------------------------------
FUNDAMENTAL_ARGS = {
    'l':   [134.96340251 * 3600, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
    'lp':  [357.52910918 * 3600, 129596581.0481, -0.5532, 0.000136, -0.00001149],
    'F':   [93.27209062 * 3600,  1739527262.8478, -12.7512, -0.001037, 0.00000417],
    'D':   [297.85019547 * 3600, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
    'Om':  [125.04455501 * 3600, -6962890.5431, 7.4722, 0.007702, -0.00005939],
}

PLANETARY_ARGS = {
    'Me': [4.402608842, 2608.7903141574],
    'Ve': [3.176146697, 1021.3285546211],
    'E':  [1.753470314, 628.3075849991],
    'Ma': [6.203480913, 334.0612426700],
    'J':  [0.599546497, 52.9690962641],
    'Sa': [0.874016757, 21.3299104960],
    'U':  [5.481293872, 7.4781598567],
    'Ne': [5.311886287, 3.8133035638],
}
PRECESSION_RATE = 0.02438175   # p_A coefficient (rad/Julian century)

def compute_fundamental_arguments(t_cy: float) -> Dict[str, float]:
    """Compute fundamental arguments (l, l', F, D, Ω, planets, p_A) in radians."""
    args = {}
    for name, coeffs in FUNDAMENTAL_ARGS.items():
        val_arcsec = sum(c * (t_cy ** i) for i, c in enumerate(coeffs))
        args[name] = math.radians(val_arcsec / 3600.0) % (2.0 * math.pi)
    for name, (const, rate) in PLANETARY_ARGS.items():
        args[name] = (const + rate * t_cy) % (2.0 * math.pi)
    args['pa'] = (PRECESSION_RATE * t_cy + 0.00000538691 * t_cy**2) % (2.0 * math.pi)
    return args

# --------------------------------------------------------------------------
# Geodetic transformations (GRS80)
# --------------------------------------------------------------------------
def geodetic_to_itrf(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    a = 6378137.0
    f = 1.0 / 298.257222101
    e2 = 2*f - f*f
    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    N = a / math.sqrt(1.0 - e2 * sin_lat**2)
    x = (N + h_m) * cos_lat * math.cos(lon)
    y = (N + h_m) * cos_lat * math.sin(lon)
    z = (N * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z])

def itrf_to_geodetic(xyz: np.ndarray) -> Tuple[float, float, float]:
    a = 6378137.0
    f = 1.0 / 298.257222101
    b = a * (1.0 - f)
    e2 = (a*a - b*b) / (a*a)
    x, y, z = xyz
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat0 = math.atan2(z, p * (1.0 - e2))
    for _ in range(5):
        sin_lat = math.sin(lat0)
        N = a / math.sqrt(1.0 - e2 * sin_lat**2)
        h = p / math.cos(lat0) - N
        lat = math.atan2(z, p * (1.0 - e2 * N / (N + h)))
        if abs(lat - lat0) < 1e-12:
            break
        lat0 = lat
    return math.degrees(lat), math.degrees(lon), h

# --------------------------------------------------------------------------
# Mean pole model (IERS 2010, Section 7.1.4)
# --------------------------------------------------------------------------
def mean_pole_iers2010(tt_jd: float) -> Tuple[float, float]:
    t_yr = (tt_jd - 2451545.0) / 365.25
    if t_yr <= 10.0:
        xp_mas = 55.974 + 1.8243*t_yr - 0.62872*t_yr**2 + 0.030*t_yr**3
        yp_mas = 346.346 + 1.7896*t_yr - 0.10729*t_yr**2 - 0.000908*t_yr**3
    else:
        xp_mas = 23.513 + 7.6141 * (t_yr - 10.0)
        yp_mas = 358.891 - 0.6287 * (t_yr - 10.0)
    return xp_mas * MAS_TO_RAD, yp_mas * MAS_TO_RAD

# --------------------------------------------------------------------------
# Solid Earth Tides (full implementation from IERS_2010.py)
# --------------------------------------------------------------------------
def solid_earth_tide(xsta, yr, month, day, fhr, xsun, xmon):
    """
    Menghitung perpindahan stasiun akibat pasang surut Bumi padat
    sesuai model IERS Conventions 2010 (Dehant-Mathews-Gipson).
    
    Ini adalah porting dari DEHANTTIDEINEL.F ke Python.
    
    Parameters
    ----------
    xsta : array(3)  [meter]  posisi stasiun dalam ITRF
    yr, month, day : int      tanggal UTC
    fhr : float              jam dalam hari UTC (jam + menit/60 + detik/3600)
    xsun, xmon : array(3) [meter]  posisi Matahari dan Bulan dalam ITRF
    
    Returns
    -------
    dxtide : array(3) [meter]  vektor perpindahan dalam ITRF
    """
    PI = math.pi
    D2PI = 2.0 * PI
    
    # =====================================================================
    # Konstanta dari IERS Conventions 2010
    # =====================================================================
    H20 = 0.6078          # Nominal Love number h2
    L20 = 0.0847          # Nominal Shida number l2
    H3  = 0.292           # Degree 3 Love number
    L3  = 0.015           # Degree 3 Shida number
    MASS_RATIO_SUN = 332946.0482
    MASS_RATIO_MOON = 0.0123000371
    RE = 6378136.6        # Equatorial radius [meter]
    
    # =====================================================================
    # Fungsi bantu: norm dan zero vector
    # =====================================================================
    def norm8(v):
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    
    def zero_vec8(v):
        v[0] = 0.0
        v[1] = 0.0
        v[2] = 0.0
        return v
    
    # =====================================================================
    # Step 1: Degree 2 dan Degree 3 utama
    # =====================================================================
    rsta = norm8(xsta)
    cosphi = math.sqrt(xsta[0]**2 + xsta[1]**2) / rsta
    
    h2 = H20 - 0.0006 * (1.0 - 1.5 * cosphi**2)
    l2 = L20 + 0.0002 * (1.0 - 1.5 * cosphi**2)
    
    # Scalar product stasiun dengan Matahari dan Bulan
    scs = xsta[0]*xsun[0] + xsta[1]*xsun[1] + xsta[2]*xsun[2]
    scm = xsta[0]*xmon[0] + xsta[1]*xmon[1] + xsta[2]*xmon[2]
    rsun = norm8(xsun)
    rmon = norm8(xmon)
    scsun = scs / (rsta * rsun)
    scmon = scm / (rsta * rmon)
    
    # Term P2 dan P3
    p2sun = 3.0 * (h2/2.0 - l2) * scsun**2 - h2/2.0
    p2mon = 3.0 * (h2/2.0 - l2) * scmon**2 - h2/2.0
    p3sun = 2.5 * (H3 - 3.0*L3) * scsun**3 + 1.5 * (L3 - H3) * scsun
    p3mon = 2.5 * (H3 - 3.0*L3) * scmon**3 + 1.5 * (L3 - H3) * scmon
    
    # Term dalam arah Matahari/Bulan
    x2sun = 3.0 * l2 * scsun
    x2mon = 3.0 * l2 * scmon
    x3sun = 1.5 * L3 * (5.0 * scsun**2 - 1.0)
    x3mon = 1.5 * L3 * (5.0 * scmon**2 - 1.0)
    
    # Faktor
    fac2sun = MASS_RATIO_SUN * RE * (RE / rsun)**3
    fac2mon = MASS_RATIO_MOON * RE * (RE / rmon)**3
    fac3sun = fac2sun * (RE / rsun)
    fac3mon = fac2mon * (RE / rmon)
    
    # Total displacement
    dxtide = [0.0, 0.0, 0.0]
    for i in range(3):
        dxtide[i] = (fac2sun * (x2sun * xsun[i]/rsun + p2sun * xsta[i]/rsta) +
                     fac2mon * (x2mon * xmon[i]/rmon + p2mon * xsta[i]/rsta) +
                     fac3sun * (x3sun * xsun[i]/rsun + p3sun * xsta[i]/rsta) +
                     fac3mon * (x3mon * xmon[i]/rmon + p3mon * xsta[i]/rsta))
    
    # =====================================================================
    # Koreksi ST1IDIU (out-of-phase diurnal)
    # =====================================================================
    def st1idiu(xsta, xsun, xmon, fac2sun, fac2mon):
        dhi = -0.0025
        dli = -0.0007
        
        rsta = norm8(xsta)
        sinphi = xsta[2] / rsta
        cosphi = math.sqrt(xsta[0]**2 + xsta[1]**2) / rsta
        cos2phi = cosphi**2 - sinphi**2
        sinla = xsta[1] / (cosphi * rsta)
        cosla = xsta[0] / (cosphi * rsta)
        rmon = norm8(xmon)
        rsun = norm8(xsun)
        
        drsun = (-3.0 * dhi * sinphi * cosphi * fac2sun *
                 xsun[2] * (xsun[0]*sinla - xsun[1]*cosla) / rsun**2)
        drmon = (-3.0 * dhi * sinphi * cosphi * fac2mon *
                 xmon[2] * (xmon[0]*sinla - xmon[1]*cosla) / rmon**2)
        dnsun = (-3.0 * dli * cos2phi * fac2sun *
                 xsun[2] * (xsun[0]*sinla - xsun[1]*cosla) / rsun**2)
        dnmon = (-3.0 * dli * cos2phi * fac2mon *
                 xmon[2] * (xmon[0]*sinla - xmon[1]*cosla) / rmon**2)
        desun = (-3.0 * dli * sinphi * fac2sun *
                 xsun[2] * (xsun[0]*cosla + xsun[1]*sinla) / rsun**2)
        demon = (-3.0 * dli * sinphi * fac2mon *
                 xmon[2] * (xmon[0]*cosla + xmon[1]*sinla) / rmon**2)
        
        dr = drsun + drmon
        dn = dnsun + dnmon
        de = desun + demon
        
        xcor = [0.0, 0.0, 0.0]
        xcor[0] = dr*cosla*cosphi - de*sinla - dn*sinphi*cosla
        xcor[1] = dr*sinla*cosphi + de*cosla - dn*sinphi*sinla
        xcor[2] = dr*sinphi + dn*cosphi
        return xcor
    
    xcorsta = st1idiu(xsta, xsun, xmon, fac2sun, fac2mon)
    for i in range(3):
        dxtide[i] += xcorsta[i]
    
    # =====================================================================
    # Koreksi ST1ISEM (out-of-phase semi-diurnal)
    # =====================================================================
    def st1isem(xsta, xsun, xmon, fac2sun, fac2mon):
        dhi = -0.0022
        dli = -0.0007
        
        rsta = norm8(xsta)
        sinphi = xsta[2] / rsta
        cosphi = math.sqrt(xsta[0]**2 + xsta[1]**2) / rsta
        sinla = xsta[1] / (cosphi * rsta)
        cosla = xsta[0] / (cosphi * rsta)
        costwola = cosla**2 - sinla**2
        sintwola = 2.0 * cosla * sinla
        rmon = norm8(xmon)
        rsun = norm8(xsun)
        
        term_sun = ((xsun[0]**2 - xsun[1]**2) * sintwola -
                    2.0 * xsun[0]*xsun[1] * costwola) / rsun**2
        term_mon = ((xmon[0]**2 - xmon[1]**2) * sintwola -
                    2.0 * xmon[0]*xmon[1] * costwola) / rmon**2
        
        drsun = -0.75 * dhi * cosphi**2 * fac2sun * term_sun
        drmon = -0.75 * dhi * cosphi**2 * fac2mon * term_mon
        dnsun = 1.5 * dli * sinphi * cosphi * fac2sun * term_sun
        dnmon = 1.5 * dli * sinphi * cosphi * fac2mon * term_mon
        
        term_sun2 = ((xsun[0]**2 - xsun[1]**2) * costwola +
                     2.0 * xsun[0]*xsun[1] * sintwola) / rsun**2
        term_mon2 = ((xmon[0]**2 - xmon[1]**2) * costwola +
                     2.0 * xmon[0]*xmon[1] * sintwola) / rmon**2
        
        desun = -1.5 * dli * cosphi * fac2sun * term_sun2
        demon = -1.5 * dli * cosphi * fac2mon * term_mon2
        
        dr = drsun + drmon
        dn = dnsun + dnmon
        de = desun + demon
        
        xcor = [0.0, 0.0, 0.0]
        xcor[0] = dr*cosla*cosphi - de*sinla - dn*sinphi*cosla
        xcor[1] = dr*sinla*cosphi + de*cosla - dn*sinphi*sinla
        xcor[2] = dr*sinphi + dn*cosphi
        return xcor
    
    xcorsta = st1isem(xsta, xsun, xmon, fac2sun, fac2mon)
    for i in range(3):
        dxtide[i] += xcorsta[i]
    
    # =====================================================================
    # Koreksi ST1L1 (latitude dependence L¹)
    # =====================================================================
    def st1l1(xsta, xsun, xmon, fac2sun, fac2mon):
        l1d = 0.0012
        l1sd = 0.0024
        
        rsta = norm8(xsta)
        sinphi = xsta[2] / rsta
        cosphi = math.sqrt(xsta[0]**2 + xsta[1]**2) / rsta
        sinla = xsta[1] / (cosphi * rsta)
        cosla = xsta[0] / (cosphi * rsta)
        rmon = norm8(xmon)
        rsun = norm8(xsun)
        
        xcor = [0.0, 0.0, 0.0]
        
        # Diurnal
        l1 = l1d
        dnsun = (-l1 * sinphi**2 * fac2sun * xsun[2] *
                 (xsun[0]*cosla + xsun[1]*sinla) / rsun**2)
        dnmon = (-l1 * sinphi**2 * fac2mon * xmon[2] *
                 (xmon[0]*cosla + xmon[1]*sinla) / rmon**2)
        desun = (l1 * sinphi * (cosphi**2 - sinphi**2) * fac2sun *
                 xsun[2] * (xsun[0]*sinla - xsun[1]*cosla) / rsun**2)
        demon = (l1 * sinphi * (cosphi**2 - sinphi**2) * fac2mon *
                 xmon[2] * (xmon[0]*sinla - xmon[1]*cosla) / rmon**2)
        
        de = 3.0 * (desun + demon)
        dn = 3.0 * (dnsun + dnmon)
        
        xcor[0] = -de*sinla - dn*sinphi*cosla
        xcor[1] = de*cosla - dn*sinphi*sinla
        xcor[2] = dn*cosphi
        
        # Semi-diurnal
        l1 = l1sd
        costwola = cosla**2 - sinla**2
        sintwola = 2.0 * cosla * sinla
        
        term_sun = ((xsun[0]**2 - xsun[1]**2)*costwola +
                    2.0*xsun[0]*xsun[1]*sintwola) / rsun**2
        term_mon = ((xmon[0]**2 - xmon[1]**2)*costwola +
                    2.0*xmon[0]*xmon[1]*sintwola) / rmon**2
        
        dnsun = -l1/2.0 * sinphi*cosphi * fac2sun * term_sun
        dnmon = -l1/2.0 * sinphi*cosphi * fac2mon * term_mon
        
        term_sun2 = ((xsun[0]**2 - xsun[1]**2)*sintwola -
                     2.0*xsun[0]*xsun[1]*costwola) / rsun**2
        term_mon2 = ((xmon[0]**2 - xmon[1]**2)*sintwola -
                     2.0*xmon[0]*xmon[1]*costwola) / rmon**2
        
        desun = -l1/2.0 * sinphi**2 * cosphi * fac2sun * term_sun2
        demon = -l1/2.0 * sinphi**2 * cosphi * fac2mon * term_mon2
        
        de = 3.0 * (desun + demon)
        dn = 3.0 * (dnsun + dnmon)
        
        xcor[0] += -de*sinla - dn*sinphi*cosla
        xcor[1] += de*cosla - dn*sinphi*sinla
        xcor[2] += dn*cosphi
        
        return xcor
    
    xcorsta = st1l1(xsta, xsun, xmon, fac2sun, fac2mon)
    for i in range(3):
        dxtide[i] += xcorsta[i]
    
    # =====================================================================
    # Step 2: Konversi waktu
    # =====================================================================
    fhrd = fhr / 24.0
    # Hitung Julian date menggunakan cal_to_jd yang sudah ada
    jd1, jd2 = cal_to_jd(yr, month, day, hour=0, minute=0, second=0.0, scale='utc')
    jd = jd1 + jd2
    t = ((jd - 2451545.0) + jd2 - jd1 + fhrd) / 36525.0
    
    # Hitung DTT = TAI-UTC + 32.184 (menggunakan tai_utc yang sudah ada)
    mjd = jd - 2400000.5 + fhrd
    tai_offset = tai_utc(mjd)
    dtt = tai_offset + 32.184
    t = t + dtt / (3600.0 * 24.0 * 36525.0)
    
    # =====================================================================
    # STEP2DIU (in-phase dan out-of-phase diurnal)
    # =====================================================================
    def step2diu(xsta, fhr, t):
        deg2rad = D2PI / 360.0
        
        # Tabel koefisien dari STEP2DIU.F
        datdi = [
            (-3,0,2,0,0,-0.01,0,0,0), (-3,2,0,0,0,-0.01,0,0,0),
            (-2,0,1,-1,0,-0.02,0,0,0), (-2,0,1,0,0,-0.08,0,-0.01,0.01),
            (-2,2,-1,0,0,-0.02,0,0,0), (-1,0,0,-1,0,-0.10,0,0,0),
            (-1,0,0,0,0,-0.51,0,-0.02,0.03), (-1,2,0,0,0,0.01,0,0,0),
            (0,-2,1,0,0,0.01,0,0,0), (0,0,-1,0,0,0.02,0,0,0),
            (0,0,1,0,0,0.06,0,0,0), (0,0,1,1,0,0.01,0,0,0),
            (0,2,-1,0,0,0.01,0,0,0), (1,-3,0,0,1,-0.06,0,0,0),
            (1,-2,0,-1,0,0.01,0,0,0), (1,-2,0,0,0,-1.23,-0.07,0.06,0.01),
            (1,-1,0,0,-1,0.02,0,0,0), (1,-1,0,0,1,0.04,0,0,0),
            (1,0,0,-1,0,-0.22,0.01,0.01,0), (1,0,0,0,0,12.00,-0.80,-0.67,-0.03),
            (1,0,0,1,0,1.73,-0.12,-0.10,0), (1,0,0,2,0,-0.04,0,0,0),
            (1,1,0,0,-1,-0.50,-0.01,0.03,0), (1,1,0,0,1,0.01,0,0,0),
            (0,1,0,1,-1,-0.01,0,0,0), (1,2,-2,0,0,-0.01,0,0,0),
            (1,2,0,0,0,-0.11,0.01,0.01,0), (2,-2,1,0,0,-0.01,0,0,0),
            (2,0,-1,0,0,-0.02,0,0,0), (3,0,0,0,0,0,0,0,0),
            (3,0,0,1,0,0,0,0,0)
        ]
        
        # Argumen fase
        s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139*t)*t)*t
        tau = fhr*15.0 + 280.4606184 + (36000.7700536 + (0.00038793 - 0.0000000258*t)*t)*t - s
        pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007*t)*t)*t)*t
        s = s + pr
        h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 - 0.00000000654*t)*t)*t)*t
        p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263*t)*t)*t)*t
        zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650*t)*t)*t)*t
        ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 - 0.00000000334*t)*t)*t)*t
        
        s = s % 360.0
        tau = tau % 360.0
        h = h % 360.0
        p = p % 360.0
        zns = zns % 360.0
        ps = ps % 360.0
        
        rsta = norm8(xsta)
        sinphi = xsta[2] / rsta
        cosphi = math.sqrt(xsta[0]**2 + xsta[1]**2) / rsta
        cosla = xsta[0] / (cosphi * rsta)
        sinla = xsta[1] / (cosphi * rsta)
        zla = math.atan2(xsta[1], xsta[0])
        
        xcor = [0.0, 0.0, 0.0]
        for j in range(31):
            (n1, n2, n3, n4, n5, a6, a7, a8, a9) = datdi[j]
            thetaf = (tau + n1*s + n2*h + n3*p + n4*zns + n5*ps) * deg2rad
            dr = (a6 * 2.0 * sinphi*cosphi * math.sin(thetaf+zla) +
                  a7 * 2.0 * sinphi*cosphi * math.cos(thetaf+zla))
            dn = (a8 * (cosphi**2 - sinphi**2) * math.sin(thetaf+zla) +
                  a9 * (cosphi**2 - sinphi**2) * math.cos(thetaf+zla))
            de = (a8 * sinphi * math.cos(thetaf+zla) -
                  a9 * sinphi * math.sin(thetaf+zla))
            
            xcor[0] += dr*cosla*cosphi - de*sinla - dn*sinphi*cosla
            xcor[1] += dr*sinla*cosphi + de*cosla - dn*sinphi*sinla
            xcor[2] += dr*sinphi + dn*cosphi
        
        for i in range(3):
            xcor[i] /= 1000.0
        return xcor
    
    xcorsta = step2diu(xsta, fhr, t)
    for i in range(3):
        dxtide[i] += xcorsta[i]
    
    # =====================================================================
    # STEP2LON (in-phase dan out-of-phase long-period)
    # =====================================================================
    def step2lon(xsta, t):
        deg2rad = D2PI / 360.0
        
        datdi = [
            (0,0,0,1,0, 0.47,0.23,0.16,0.07),
            (0,2,0,0,0, -0.20,-0.12,-0.11,-0.05),
            (1,0,-1,0,0, -0.11,-0.08,-0.09,-0.04),
            (2,0,0,0,0, -0.13,-0.11,-0.15,-0.07),
            (2,0,0,1,0, -0.05,-0.05,-0.06,-0.03)
        ]
        
        s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139*t)*t)*t
        pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007*t)*t)*t)*t
        s = s + pr
        h = 280.46645 + (36000.7697489 + (0.00030322222 + (0.000000020 - 0.00000000654*t)*t)*t)*t
        p = 83.35324312 + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263*t)*t)*t)*t
        zns = 234.95544499 + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650*t)*t)*t)*t
        ps = 282.93734098 + (1.71945766667 + (0.00045688889 + (-0.00000001778 - 0.00000000334*t)*t)*t)*t
        
        s = s % 360.0
        h = h % 360.0
        p = p % 360.0
        zns = zns % 360.0
        ps = ps % 360.0
        
        rsta = norm8(xsta)
        sinphi = xsta[2] / rsta
        cosphi = math.sqrt(xsta[0]**2 + xsta[1]**2) / rsta
        cosla = xsta[0] / (cosphi * rsta)
        sinla = xsta[1] / (cosphi * rsta)
        
        xcor = [0.0, 0.0, 0.0]
        for j in range(5):
            (n1, n2, n3, n4, n5, a6, a7, a8, a9) = datdi[j]
            thetaf = (n1*s + n2*h + n3*p + n4*zns + n5*ps) * deg2rad
            dr = (a6 * (3.0*sinphi**2 - 1.0)/2.0 * math.cos(thetaf) +
                  a8 * (3.0*sinphi**2 - 1.0)/2.0 * math.sin(thetaf))
            dn = (a7 * (cosphi*sinphi*2.0) * math.cos(thetaf) +
                  a9 * (cosphi*sinphi*2.0) * math.sin(thetaf))
            de = 0.0
            
            xcor[0] += dr*cosla*cosphi - de*sinla - dn*sinphi*cosla
            xcor[1] += dr*sinla*cosphi + de*cosla - dn*sinphi*sinla
            xcor[2] += dr*sinphi + dn*cosphi
        
        for i in range(3):
            xcor[i] /= 1000.0
        return xcor
    
    xcorsta = step2lon(xsta, t)
    for i in range(3):
        dxtide[i] += xcorsta[i]
    
    # Catatan: Step 3 untuk permanent tide TIDAK diterapkan,
    # konsisten dengan keputusan IERS untuk menghindari lompatan pada reference frame
    
    return dxtide

# --------------------------------------------------------------------------
# Pole Tide (IERS 2010, Section 7.1.4) – dengan Love numbers yang benar
# --------------------------------------------------------------------------
def pole_tide(xp_rad, yp_rad, lat_rad, lon_rad, elev_m):
    """
    Pole tide site displacement (IERS Conventions 2010, Sec. 7.1.4).

    Parameters
    ----------
    xp_rad, yp_rad : float
        Pole coordinates (radians).
    lat_rad, lon_rad : float
        Geodetic latitude and longitude of the station (radians).
    elev_m : float
        Ellipsoidal height (meters).  Not used in the model but included
        for uniform interface.

    Returns
    -------
    dX, dY, dZ : float
        Displacement vector in ITRS (meters).  The correction should be
        **subtracted** from the observed station position to obtain the
        tide‑free position.
    """
    # Love numbers (IERS 2010, Table 7.2)
    H2 = 0.6027
    L2 = 0.0836

    # Earth parameters
    OMEGA = 7.292115e-5          # mean angular velocity (rad/s)
    A_EQ  = 6378136.6            # equatorial radius (m)
    G_EQ  = 9.7803278            # equatorial gravity (m/s²)
    SCALE = (OMEGA**2 * A_EQ) / (2.0 * G_EQ)   # ~0.0053

    sin_phi = math.sin(lat_rad)
    cos_phi = math.cos(lat_rad)
    sin_lam = math.sin(lon_rad)
    cos_lam = math.cos(lon_rad)

    # Combination of pole coordinates (IERS 2010, Eq. 7.25)
    xp_cos = xp_rad * cos_lam - yp_rad * sin_lam
    yp_sin = xp_rad * sin_lam + yp_rad * cos_lam

    # Local displacement components
    dr = -SCALE * H2 * math.sin(2.0 * lat_rad) * xp_cos
    dn =  SCALE * L2 * math.cos(2.0 * lat_rad) * xp_cos * 2.0
    de = -SCALE * L2 * sin_phi * yp_sin * 2.0

    # Rotate to ITRS (local N‑E‑U → cartesian XYZ)
    dX = dr * cos_phi * cos_lam - dn * sin_phi * cos_lam - de * sin_lam
    dY = dr * cos_phi * sin_lam - dn * sin_phi * sin_lam + de * cos_lam
    dZ = dr * sin_phi + dn * cos_phi

    return dX, dY, dZ

# --------------------------------------------------------------------------
# Ocean Pole Tide Loading (Desai 2002 Equilibrium Model)
# --------------------------------------------------------------------------
def ocean_pole_tide_loading(lat_rad: float, lon_rad: float, m1: float, m2: float) -> np.ndarray:
    """
    Menghitung deformasi Ocean Pole Tide Loading (IERS TN36 Eq 7.29)
    dengan interpolasi bilinear dari grid Desai (2002).

    Input:
        lat_rad, lon_rad : Koordinat stasiun dalam radian
        m1, m2           : Komponen Polar Motion (xp - xp_mean, -(yp - yp_mean)) dalam RADIAN

    Output:
        np.array([de, dn, du]) : Komponen perpindahan dalam meter (East, North, Up)
    """
    if _OPOLE_GRID is None:
        return np.zeros(3)

    # Konversi koordinat stasiun
    lon_deg = math.degrees(lon_rad) % 360.0
    lat_deg = math.degrees(lat_rad)

    # Hitung indeks untuk interpolasi (Grid 0.5 derajat)
    f_lon = (lon_deg - 0.25) / 0.5
    f_lat = (lat_deg - (-89.75)) / 0.5

    lon0 = int(f_lon) % 720
    lon1 = (lon0 + 1) % 720
    lat0 = int(max(0, min(f_lat, 358)))
    lat1 = lat0 + 1

    tx = f_lon - int(f_lon)
    ty = f_lat - lat0

    def get_val(lo, la): return _OPOLE_GRID[la * 720 + lo]

    p00, p01 = get_val(lon0, lat0), get_val(lon1, lat0)
    p10, p11 = get_val(lon0, lat1), get_val(lon1, lat1)

    def interp(v00, v01, v10, v11):
        return (1-tx)*(1-ty)*v00 + tx*(1-ty)*v01 + (1-tx)*ty*v10 + tx*ty*v11

    urR = interp(p00[2], p01[2], p10[2], p11[2])
    urI = interp(p00[3], p01[3], p10[3], p11[3])
    unR = interp(p00[4], p01[4], p10[4], p11[4])
    unI = interp(p00[5], p01[5], p10[5], p11[5])
    ueR = interp(p00[6], p01[6], p10[6], p11[6])
    ueI = interp(p00[7], p01[7], p10[7], p11[7])

    # ----- PERBAIKAN: Terapkan K dan GAMMA sesuai IERS Eq 7.29 -----
    GAMMA_R = 0.6870
    GAMMA_I = 0.0036
    K = 5.3394043696e3  # meters/radian

    term1 = m1 * GAMMA_R + m2 * GAMMA_I
    term2 = m2 * GAMMA_R - m1 * GAMMA_I

    du = K * (term1 * urR + term2 * urI)
    dn = K * (term1 * unR + term2 * unI)
    de = K * (term1 * ueR + term2 * ueI)

    return np.array([de, dn, du])

# =============================================================================
# ATMOSPHERIC PRESSURE LOADING (Ray & Ponte 2003) - EMBEDDED
# =============================================================================
from scipy.special import lpmv, gammaln

_APL_COMPONENTS = None
_LOAD_H = [
    -0.309, -1.032, -1.340, -1.564, -1.739, -1.877, -1.986, -2.073,
    -2.141, -2.195, -2.238, -2.272, -2.299, -2.320, -2.338, -2.352,
    -2.364, -2.374, -2.382, -2.389
]

def _get_apl_components():
    global _APL_COMPONENTS
    # Jika sudah dimuat di memori (dalam satu sesi run), gunakan yang ada
    if _APL_COMPONENTS is not None:
        return _APL_COMPONENTS
        
    components = {}
    data_file = os.path.join(os.path.dirname(__file__), 'Ray_Ponte_2003_mbar.txt')
    if not os.path.exists(data_file):
        _APL_COMPONENTS = components
        return components
        
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Atmospheric') or line.startswith('normalized') or line.startswith('Doodson'):
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            try:
                doodson = float(parts[0])
                name    = parts[1]
                l       = int(parts[2])
                m       = int(parts[3])
                c_plus  = float(parts[8])
                eps_plus = float(parts[9])
                c_minus = float(parts[10])
                eps_minus = float(parts[11])
            except (ValueError, IndexError):
                continue
                
            if name not in components:
                components[name] = {'doodson': doodson, 'terms': []}
            components[name]['terms'].append({
                'l': l, 'm': m,
                'c+': c_plus, 'eps+': eps_plus,
                'c-': c_minus, 'eps-': eps_minus,
            })
    _APL_COMPONENTS = components
    return components

def _apl_norm_factor(l, m):
    """Normalisasi 4π untuk harmonik sferis (Geodesi)."""
    m_abs = abs(m)
    delta = 1.0 if m_abs == 0 else 2.0
    log_norm = 0.5 * (math.log(delta) + math.log(2*l + 1) +
                      gammaln(l - m_abs + 1) - gammaln(l + m_abs + 1))
    return math.exp(log_norm)

def _apl_plm_all(lat_rad, max_deg=20):
    sin_lat = math.sin(lat_rad)
    P = {}
    for l in range(0, max_deg+1):
        P[l] = {}
        for m in range(0, l+1):
            P[l][m] = _apl_norm_factor(l, m) * lpmv(m, l, sin_lat)
    return P

def _apl_exact_doodson_phase(doodson_val, args, gmst_rad):
    """Mengekstrak angka Doodson menjadi fase astronomis."""
    d_str = f"{doodson_val:07.3f}".replace('.', '')
    n1, n2, n3, n4, n5, n6 = (int(d_str[i]) for i in range(6))
    
    l, lp, F, D, Om = args['l'], args['lp'], args['F'], args['D'], args['Om']
    s_lunar = F + Om
    h_sun   = F + Om - D
    p_lunar = F + Om - l
    N_prime = -Om
    ps_sun  = F + Om - D - lp
    
    tau = gmst_rad + math.pi - s_lunar
    
    theta_f = (n1 * tau + (n2 - 5) * s_lunar + (n3 - 5) * h_sun + 
               (n4 - 5) * p_lunar + (n5 - 5) * N_prime + (n6 - 5) * ps_sun)
    return theta_f % (2.0 * math.pi)

def atm_loading_displacement(mjd_utc, lat_rad, lon_rad):
    """Fungsi utama APL - menghasilkan dX, dY, dZ (meter)."""
    components = _get_apl_components()
    if not components:
        return 0.0, 0.0, 0.0
        
    t_cy = (mjd_utc - 51544.5) / 36525.0
    args = compute_fundamental_arguments(t_cy)
    
    # GMST (Aoki 1982 formulation)
    gmst_sec = 67310.54841 + t_cy * ((8640184.812866 + 3155760000.0) +
               t_cy * (0.093104 + t_cy * (-0.0000062)))
    gmst_rad = (gmst_sec % 86400.0) * (2.0 * math.pi / 86400.0)
    
    P = _apl_plm_all(lat_rad, 20)
    RHO_E = 5514.0
    G_EQ = 9.80665
    
    dX = dY = dZ = 0.0

    for name, comp in components.items():
        theta_f = _apl_exact_doodson_phase(comp['doodson'], args, gmst_rad)
        
        for term in comp['terms']:
            l_deg = term['l']
            m_ord = term['m']
            if l_deg > 20: continue
                
            factor_l = (3.0 / ((2.0 * l_deg + 1.0) * RHO_E * G_EQ)) * 100.0
            arg_plus  = theta_f + m_ord * lon_rad - math.radians(term['eps+'])
            arg_minus = theta_f - m_ord * lon_rad - math.radians(term['eps-'])
            
            press_lm = term['c+'] * math.cos(arg_plus) + term['c-'] * math.cos(arg_minus)
            
            if abs(press_lm) < 1e-12: continue
                
            h_l = _LOAD_H[l_deg-1] if l_deg <= len(_LOAD_H) else -0.3
            plm_val = P[l_deg][abs(m_ord)]
            dr = h_l * factor_l * press_lm * plm_val
            
            dX += dr * math.cos(lat_rad) * math.cos(lon_rad)
            dY += dr * math.cos(lat_rad) * math.sin(lon_rad)
            dZ += dr * math.sin(lat_rad)

    return dX, dY, dZ

# ==============================================================================
# Ocean Tide Loading
# Diperbarui untuk model FES2014b - Stasiun Jolotundo
# ==============================================================================

import numpy as np
import math

class Asterid342Engine_FES2014:
    def __init__(self, blq_data):
        self.blq = blq_data
        
        # 11 Gelombang Utama Standar IERS
        self.main_waves = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm', 'Ssa']
        
        # Doodson 11 Gelombang Utama
        self.main_doodson = np.array([
            [2,  0,  0,  0,  0,  0], # M2
            [2,  2, -2,  0,  0,  0], # S2
            [2, -1,  0,  1,  0,  0], # N2
            [2,  2,  0,  0,  0,  0], # K2
            [1,  1,  0,  0,  0,  0], # K1
            [1, -1,  0,  0,  0,  0], # O1
            [1,  1, -2,  0,  0,  0], # P1
            [1, -2,  0,  1,  0,  0], # Q1
            [0,  2,  0,  0,  0,  0], # Mf
            [0,  1,  0, -1,  0,  0], # Mm
            [0,  0,  2,  0,  0,  0]  # Ssa
        ])

        # Ekstraksi absolut Cartwright-Edden Amplitude (TAMP) dari ADMINT.F IERS
        self.tamp_342 = np.array([
            .632208, .294107, .121046, .079915, .023818,-.023589, .022994, .019333,-.017871, .017192, 
            .016018, .004671,-.004662,-.004519, .004470, .004467, .002589,-.002455,-.002172, .001972, 
            .001947, .001914,-.001898, .001802, .001304, .001170, .001130, .001061,-.001022,-.001017, 
            .001014, .000901,-.000857, .000855, .000855, .000772, .000741, .000741,-.000721, .000698, 
            .000658, .000654,-.000653, .000633, .000626,-.000598, .000590, .000544, .000479,-.000464, 
            .000413,-.000390, .000373, .000366, .000366,-.000360,-.000355, .000354, .000329, .000328, 
            .000319, .000302, .000279,-.000274,-.000272, .000248,-.000225, .000224,-.000223,-.000216,
            .000211, .000209, .000194, .000185,-.000174,-.000171, .000159, .000131, .000127, .000120, 
            .000118, .000117, .000108, .000107, .000105,-.000102, .000102, .000099,-.000096, .000095,
            -.000089,-.000085,-.000084,-.000081,-.000077,-.000072,-.000067, .000066, .000064, .000063, 
            .000063, .000063, .000062, .000062,-.000060, .000056, .000053, .000051, .000050, .368645,
            -.262232,-.121995,-.050208, .050031,-.049470, .020620, .020613, .011279,-.009530,-.009469,
            -.008012, .007414,-.007300, .007227,-.007131,-.006644, .005249, .004137, .004087, .003944, 
            .003943, .003420, .003418, .002885, .002884, .002160,-.001936, .001934,-.001798, .001690,
            .001689, .001516, .001514,-.001511, .001383, .001372, .001371,-.001253,-.001075, .001020, 
            .000901, .000865,-.000794, .000788, .000782,-.000747,-.000745, .000670,-.000603,-.000597, 
            .000542, .000542,-.000541,-.000469,-.000440, .000438, .000422, .000410,-.000374,-.000365, 
            .000345, .000335,-.000321,-.000319, .000307, .000291, .000290,-.000289, .000286, .000275, 
            .000271, .000263,-.000245, .000225, .000225, .000221,-.000202,-.000200,-.000199, .000192, 
            .000183, .000183, .000183,-.000170, .000169, .000168, .000162, .000149,-.000147,-.000141, 
            .000138, .000136, .000136, .000127, .000127,-.000126,-.000121,-.000121, .000117,-.000116,
            -.000114,-.000114,-.000114, .000114, .000113, .000109, .000108, .000106,-.000106,-.000106, 
            .000105, .000104,-.000103,-.000100,-.000100,-.000100, .000099,-.000098, .000093, .000093, 
            .000090,-.000088, .000083,-.000083,-.000082,-.000081,-.000079,-.000077,-.000075,-.000075,
            -.000075, .000071, .000071,-.000071, .000068, .000068, .000065, .000065, .000064, .000064, 
            .000064,-.000064,-.000060, .000056, .000056, .000053, .000053, .000053,-.000053, .000053, 
            .000053, .000052, .000050,-.066607,-.035184,-.030988, .027929,-.027616,-.012753,-.006728,
            -.005837,-.005286,-.004921,-.002884,-.002583,-.002422, .002310, .002283,-.002037, .001883,
            -.001811,-.001687,-.001004,-.000925,-.000844, .000766, .000766,-.000700,-.000495,-.000492, 
            .000491, .000483, .000437,-.000416,-.000384, .000374,-.000312,-.000288,-.000273, .000259, 
            .000245,-.000232, .000229,-.000216, .000206,-.000204,-.000202, .000200, .000195,-.000190, 
            .000187, .000180,-.000179, .000170, .000153,-.000137,-.000119,-.000119,-.000112,-.000110,
            -.000110, .000107,-.000095,-.000095,-.000091,-.000090,-.000081,-.000079,-.000079, .000077,
            -.000073, .000069,-.000067,-.000066, .000065, .000064,-.000062, .000060, .000059,-.000056, 
            .000055,-.000051
        ])

        # Ekstraksi absolut Doodson Number 342 Konstituen (IDD) dari ADMINT.F IERS
        self.idd_342_flat = np.array([
            2,0,0,0,0,0, 2,2,-2,0,0,0, 2,-1,0,1,0,0, 2,2,0,0,0,0, 2,2,0,0,1,0, 2,0,0,0,-1,0, 2,-1,2,-1,0,0,
            2,-2,2,0,0,0, 2,1,0,-1,0,0, 2,2,-3,0,0,1, 2,-2,0,2,0,0, 2,-3,2,1,0,0, 2,1,-2,1,0,0, 2,-1,0,1,-1,0,
            2,3,0,-1,0,0, 2,1,0,1,0,0, 2,2,0,0,2,0, 2,2,-1,0,0,-1, 2,0,-1,0,0,1, 2,1,0,1,1,0, 2,3,0,-1,1,0,
            2,0,1,0,0,-1, 2,0,-2,2,0,0, 2,-3,0,3,0,0, 2,-2,3,0,0,-1, 2,4,0,0,0,0, 2,-1,1,1,0,-1, 2,-1,3,-1,0,-1,
            2,2,0,0,-1,0, 2,-1,-1,1,0,1, 2,4,0,0,1,0, 2,-3,4,-1,0,0, 2,-1,2,-1,-1,0, 2,3,-2,1,0,0, 2,1,2,-1,0,0,
            2,-4,2,2,0,0, 2,4,-2,0,0,0, 2,0,2,0,0,0, 2,-2,2,0,-1,0, 2,2,-4,0,0,2, 2,2,-2,0,-1,0, 2,1,0,-1,-1,0,
            2,-1,1,0,0,0, 2,2,-1,0,0,1, 2,2,1,0,0,-1, 2,-2,0,2,-1,0, 2,-2,4,-2,0,0, 2,2,2,0,0,0, 2,-4,4,0,0,0,
            2,-1,0,-1,-2,0, 2,1,2,-1,1,0, 2,-1,-2,3,0,0, 2,3,-2,1,1,0, 2,4,0,-2,0,0, 2,0,0,2,0,0, 2,0,2,-2,0,0,
            2,0,2,0,1,0, 2,-3,3,1,0,-1, 2,0,0,0,-2,0, 2,4,0,0,2,0, 2,4,-2,0,1,0, 2,0,0,0,0,2, 2,1,0,1,2,0,
            2,0,-2,0,-2,0, 2,-2,1,0,0,1, 2,-2,1,2,0,-1, 2,-1,1,-1,0,1, 2,5,0,-1,0,0, 2,1,-3,1,0,1, 2,-2,-1,2,0,1,
            2,3,0,-1,2,0, 2,1,-2,1,-1,0, 2,5,0,-1,1,0, 2,-4,0,4,0,0, 2,-3,2,1,-1,0, 2,-2,1,1,0,0, 2,4,0,-2,1,0,
            2,0,0,2,1,0, 2,-5,4,1,0,0, 2,0,2,0,2,0, 2,-1,2,1,0,0, 2,5,-2,-1,0,0, 2,1,-1,0,0,0, 2,2,-2,0,0,2,
            2,-5,2,3,0,0, 2,-1,-2,1,-2,0, 2,-3,5,-1,0,-1, 2,-1,0,0,0,1, 2,-2,0,0,-2,0, 2,0,-1,1,0,0, 2,-3,1,1,0,1,
            2,3,0,-1,-1,0, 2,1,0,1,-1,0, 2,-1,2,1,1,0, 2,0,-3,2,0,1, 2,1,-1,-1,0,1, 2,-3,0,3,-1,0, 2,0,-2,2,-1,0,
            2,-4,3,2,0,-1, 2,-1,0,1,-2,0, 2,5,0,-1,2,0, 2,-4,5,0,0,-1, 2,-2,4,0,0,-2, 2,-1,0,1,0,2, 2,-2,-2,4,0,0,
            2,3,-2,-1,-1,0, 2,-2,5,-2,0,-1, 2,0,-1,0,-1,1, 2,5,-2,-1,1,0, 1,1,0,0,0,0, 1,-1,0,0,0,0, 1,1,-2,0,0,0,
            1,-2,0,1,0,0, 1,1,0,0,1,0, 1,-1,0,0,-1,0, 1,2,0,-1,0,0, 1,0,0,1,0,0, 1,3,0,0,0,0, 1,-2,2,-1,0,0,
            1,-2,0,1,-1,0, 1,-3,2,0,0,0, 1,0,0,-1,0,0, 1,1,0,0,-1,0, 1,3,0,0,1,0, 1,1,-3,0,0,1, 1,-3,0,2,0,0,
            1,1,2,0,0,0, 1,0,0,1,1,0, 1,2,0,-1,1,0, 1,0,2,-1,0,0, 1,2,-2,1,0,0, 1,3,-2,0,0,0, 1,-1,2,0,0,0,
            1,1,1,0,0,-1, 1,1,-1,0,0,1, 1,4,0,-1,0,0, 1,-4,2,1,0,0, 1,0,-2,1,0,0, 1,-2,2,-1,-1,0, 1,3,0,-2,0,0,
            1,-1,0,2,0,0, 1,-1,0,0,-2,0, 1,3,0,0,2,0, 1,-3,2,0,-1,0, 1,4,0,-1,1,0, 1,0,0,-1,-1,0, 1,1,-2,0,-1,0,
            1,-3,0,2,-1,0, 1,1,0,0,2,0, 1,1,-1,0,0,-1, 1,-1,-1,0,0,1, 1,0,2,-1,1,0, 1,-1,1,0,0,-1, 1,-1,-2,2,0,0,
            1,2,-2,1,1,0, 1,-4,0,3,0,0, 1,-1,2,0,1,0, 1,3,-2,0,1,0, 1,2,0,-1,-1,0, 1,0,0,1,-1,0, 1,-2,2,1,0,0,
            1,4,-2,-1,0,0, 1,-3,3,0,0,-1, 1,-2,1,1,0,-1, 1,-2,3,-1,0,-1, 1,0,-2,1,-1,0, 1,-2,-1,1,0,1, 1,4,-2,1,0,0,
            1,-4,4,-1,0,0, 1,-4,2,1,-1,0, 1,5,-2,0,0,0, 1,3,0,-2,1,0, 1,-5,2,2,0,0, 1,2,0,1,0,0, 1,1,3,0,0,-1,
            1,-2,0,1,-2,0, 1,4,0,-1,2,0, 1,1,-4,0,0,2, 1,5,0,-2,0,0, 1,-1,0,2,1,0, 1,-2,1,0,0,0, 1,4,-2,1,1,0,
            1,-3,4,-2,0,0, 1,-1,3,0,0,-1, 1,3,-3,0,0,1, 1,5,-2,0,1,0, 1,1,2,0,1,0, 1,2,0,1,1,0, 1,-5,4,0,0,0,
            1,-2,0,-1,-2,0, 1,5,0,-2,1,0, 1,1,2,-2,0,0, 1,1,-2,2,0,0, 1,-2,2,1,1,0, 1,0,3,-1,0,-1, 1,2,-3,1,0,1,
            1,-2,-2,3,0,0, 1,-1,2,-2,0,0, 1,-4,3,1,0,-1, 1,-4,0,3,-1,0, 1,-1,-2,2,-1,0, 1,-2,0,3,0,0, 1,4,0,-3,0,0,
            1,0,1,1,0,-1, 1,2,-1,-1,0,1, 1,2,-2,1,-1,0, 1,0,0,-1,-2,0, 1,2,0,1,2,0, 1,2,-2,-1,-1,0, 1,0,0,1,2,0,
            1,0,1,0,0,0, 1,2,-1,0,0,0, 1,0,2,-1,-1,0, 1,-1,-2,0,-2,0, 1,-3,1,0,0,1, 1,3,-2,0,-1,0, 1,-1,-1,0,-1,1,
            1,4,-2,-1,1,0, 1,2,1,-1,0,-1, 1,0,-1,1,0,1, 1,-2,4,-1,0,0, 1,4,-4,1,0,0, 1,-3,1,2,0,-1, 1,-3,3,0,-1,-1,
            1,1,2,0,2,0, 1,1,-2,0,-2,0, 1,3,0,0,3,0, 1,-1,2,0,-1,0, 1,-2,1,-1,0,1, 1,0,-3,1,0,1, 1,-3,-1,2,0,1,
            1,2,0,-1,2,0, 1,6,-2,-1,0,0, 1,2,2,-1,0,0, 1,-1,1,0,-1,-1, 1,-2,3,-1,-1,-1, 1,-1,0,0,0,2, 1,-5,0,4,0,0,
            1,1,0,0,0,-2, 1,-2,1,1,-1,-1, 1,1,-1,0,1,1, 1,1,2,0,0,-2, 1,-3,1,1,0,0, 1,-4,4,-1,-1,0, 1,1,0,-2,-1,0,
            1,-2,-1,1,-1,1, 1,-3,2,2,0,0, 1,5,-2,-2,0,0, 1,3,-4,2,0,0, 1,1,-2,0,0,2, 1,-1,4,-2,0,0, 1,2,2,-1,1,0,
            1,-5,2,2,-1,0, 1,1,-3,0,-1,1, 1,1,1,0,1,-1, 1,6,-2,-1,1,0, 1,-2,2,-1,-2,0, 1,4,-2,1,2,0, 1,-6,4,1,0,0,
            1,5,-4,0,0,0, 1,-3,4,0,0,0, 1,1,2,-2,1,0, 1,-2,1,0,-1,0, 0,2,0,0,0,0, 0,1,0,-1,0,0, 0,0,2,0,0,0,
            0,0,0,0,1,0, 0,2,0,0,1,0, 0,3,0,-1,0,0, 0,1,-2,1,0,0, 0,2,-2,0,0,0, 0,3,0,-1,1,0, 0,0,1,0,0,-1,
            0,2,0,-2,0,0, 0,2,0,0,2,0, 0,3,-2,1,0,0, 0,1,0,-1,-1,0, 0,1,0,-1,1,0, 0,4,-2,0,0,0, 0,1,0,1,0,0,
            0,0,3,0,0,-1, 0,4,0,-2,0,0, 0,3,-2,1,1,0, 0,3,-2,-1,0,0, 0,4,-2,0,1,0, 0,0,2,0,1,0, 0,1,0,1,1,0,
            0,4,0,-2,1,0, 0,3,0,-1,2,0, 0,5,-2,-1,0,0, 0,1,2,-1,0,0, 0,1,-2,1,-1,0, 0,1,-2,1,1,0, 0,2,-2,0,-1,0,
            0,2,-3,0,0,1, 0,2,-2,0,1,0, 0,0,2,-2,0,0, 0,1,-3,1,0,1, 0,0,0,0,2,0, 0,0,1,0,0,1, 0,1,2,-1,1,0,
            0,3,0,-3,0,0, 0,2,1,0,0,-1, 0,1,-1,-1,0,1, 0,1,0,1,2,0, 0,5,-2,-1,1,0, 0,2,-1,0,0,1, 0,2,2,-2,0,0,
            0,1,-1,0,0,0, 0,5,0,-3,0,0, 0,2,0,-2,1,0, 0,1,1,-1,0,-1, 0,3,-4,1,0,0, 0,0,2,0,2,0, 0,2,0,-2,-1,0,
            0,4,-3,0,0,1, 0,3,-1,-1,0,1, 0,0,2,0,0,-2, 0,3,-3,1,0,1, 0,2,-4,2,0,0, 0,4,-2,-2,0,0, 0,3,1,-1,0,-1,
            0,5,-4,1,0,0, 0,3,-2,-1,-1,0, 0,3,-2,1,2,0, 0,4,-4,0,0,0, 0,6,-2,-2,0,0, 0,5,0,-3,1,0, 0,4,-2,0,2,0,
            0,2,2,-2,1,0, 0,0,4,0,0,-2, 0,3,-1,0,0,0, 0,3,-3,-1,0,1, 0,4,0,-2,2,0, 0,1,-2,-1,-1,0, 0,2,-1,0,0,-1,
            0,4,-4,2,0,0, 0,2,1,0,1,-1, 0,3,-2,-1,1,0, 0,4,-3,0,1,1, 0,2,0,0,3,0, 0,6,-4,0,0,0
        ])
        self.idd_342 = self.idd_342_flat.reshape((342, 6))

    def _get_freq_phase_vectorized(self, doodson_matrix, mjd_tt, mjd_ut):
        """
        Kalkulasi vektor untuk frekuensi & fase seluruh konstituen.
        Berdasarkan TDFRPH.F IERS.
        """
        t_cy = (mjd_tt - 51544.5) / 36525.0

        f1 = (134.9634025100 + t_cy * (477198.8675605000 + t_cy * (0.0088553333 + t_cy * (0.0000143431 + t_cy * (-0.0000000680)))))
        f2 = (357.5291091806 + t_cy * (35999.0502911389 + t_cy * (-0.0001536667 + t_cy * (0.0000000378 + t_cy * (-0.0000000032)))))
        f3 = (93.2720906200 + t_cy * (483202.0174577222 + t_cy * (-0.0035420000 + t_cy * (-0.0000002881 + t_cy * (0.0000000012)))))
        f4 = (297.8501954694 + t_cy * (445267.1114469445 + t_cy * (-0.0017696111 + t_cy * (0.0000018314 + t_cy * (-0.0000000088)))))
        f5 = (125.0445550100 + t_cy * (-1934.1362619722 + t_cy * (0.0020756111 + t_cy * (0.0000021394 + t_cy * (-0.0000000165)))))

        day_frac_ut = mjd_ut - np.floor(mjd_ut)
        tau = 360.0 * day_frac_ut - f4
        
        args = np.array([
            tau,
            f3 + f5,
            f3 + f5 - f4,
            f3 + f5 - f1,
            -f5,
            f3 + f5 - f4 - f2
        ])

        phases = np.dot(doodson_matrix, args) % 360.0
        phases = np.where(phases < 0, phases + 360.0, phases)

        fd1 = 0.0362916471 + 0.0000000013 * t_cy
        fd2 = 0.0027377786
        fd3 = 0.0367481951 - 0.0000000005 * t_cy
        fd4 = 0.0338631920 - 0.0000000003 * t_cy
        fd5 = -0.0001470938 + 0.0000000003 * t_cy

        freq_dood = np.array([
            1.0 - fd4,
            fd3 + fd5,
            fd3 + fd5 - fd4,
            fd3 + fd5 - fd1,
            -fd5,
            fd3 + fd5 - fd4 - fd2
        ])
        freqs = np.dot(doodson_matrix, freq_dood)

        return freqs, phases

    def _cubic_spline(self, x_anchors, y_anchors, x_targets):
        """
        Penyelesaian spline kubik natural algorithmic tanpa Scipy.
        Porting absolut dari EVAL.F.
        """
        idx = np.argsort(x_anchors)
        x = x_anchors[idx]
        y = y_anchors[idx]
        
        n = len(x)
        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n-1):
            alpha[i] = (3/h[i])*(y[i+1]-y[i]) - (3/h[i-1])*(y[i]-y[i-1])
        
        l = np.ones(n); mu = np.zeros(n); z = np.zeros(n)
        for i in range(1, n-1):
            l[i] = 2*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i]-h[i-1]*z[i-1])/l[i]
            
        b = np.zeros(n); c = np.zeros(n); d = np.zeros(n)
        for j in range(n-2, -1, -1):
            c[j] = z[j] - mu[j]*c[j+1]
            b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1]+2*c[j])/3
            d[j] = (c[j+1]-c[j])/(3*h[j])
            
        results = np.zeros_like(x_targets)
        for k, xt in enumerate(x_targets):
            if xt <= x[0]:
                results[k] = y[0]
            elif xt >= x[-1]:
                results[k] = y[-1]
            else:
                i = np.searchsorted(x, xt) - 1
                i = max(0, min(i, n-2))
                dx = xt - x[i]
                results[k] = y[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
                
        return results

    def _process_admittance(self, comp_idx, freqs_342, mask, anchors_mask, anchor_freqs):
        blq_keys = np.array(self.main_waves)[anchors_mask]
        anchor_f = anchor_freqs[anchors_mask]
        
        real_anchors = []
        imag_anchors = []
        for key in blq_keys:
            dood = self.main_doodson[self.main_waves.index(key)]
            match_idx = np.where((self.idd_342 == dood).all(axis=1))[0][0]
            tamp_val = abs(self.tamp_342[match_idx])
            
            # ---- PERUBAHAN: gunakan self.modulated_blq, bukan self.blq ----
            amp = self.modulated_blq[key][comp_idx*2]
            ph  = np.deg2rad(-self.modulated_blq[key][comp_idx*2+1])
            # ---------------------------------------------------------------
            
            admit_real = (amp / tamp_val) * np.cos(ph)
            admit_imag = (amp / tamp_val) * np.sin(ph)
            real_anchors.append(admit_real)
            imag_anchors.append(admit_imag)
            
        target_f = freqs_342[mask]
        
        if len(anchor_f) > 0 and len(target_f) > 0:
            real_interp = self._cubic_spline(anchor_f, np.array(real_anchors), target_f)
            imag_interp = self._cubic_spline(anchor_f, np.array(imag_anchors), target_f)
            return real_interp, imag_interp
        else:
            return np.zeros(len(target_f)), np.zeros(len(target_f))

    def _apply_nodal_modulation(self, tide_name: str, tt_jd: float) -> Tuple[float, float]:
        """
        Menghitung faktor f (amplitudo) dan u (fase dalam radian) untuk modulasi nodal 18.6 tahun.
        Referensi: IERS Conventions 2010, Bab 7.
        """
        t_cy = (tt_jd - 2451545.0) / 36525.0
        args = compute_fundamental_arguments(t_cy)
        Om = args['Om']  # dalam radian

        tide_upper = tide_name.upper()
        if tide_upper == 'M2':
            f = 1.0 + 0.037 * math.cos(Om)
            u = -0.037 * math.sin(Om)
        elif tide_upper in ['K1', 'K2']:
            f = 1.0 + 0.036 * math.cos(Om)
            u = -0.036 * math.sin(Om)
        elif tide_upper == 'O1':
            f = 1.0 + 0.038 * math.cos(Om)
            u = -0.038 * math.sin(Om)
        elif tide_upper == 'N2':
            f = 1.0 + 0.018 * math.cos(Om)
            u = -0.018 * math.sin(Om)
        elif tide_upper == 'P1':
            f = 1.0 + 0.017 * math.cos(Om)
            u = -0.017 * math.sin(Om)
        else:
            f = 1.0
            u = 0.0
        return f, u

    def compute_displacement(self, mjd_tt: float, delta_t: float = 0.0) -> Tuple[float, float, float]:
        """
        Menghitung displacement (dU, dW, dS) dari ocean tide loading.
        """
        # ---- KODE EXISTING (JANGAN DIUBAH) ----
        mjd_ut = mjd_tt - delta_t / 86400.0
        
        freqs_342, phases_342 = self._get_freq_phase_vectorized(self.idd_342, mjd_tt, mjd_ut)
        
        n1 = self.idd_342[:, 0]
        phases_342 = np.where(n1 == 0, phases_342 + 180.0, phases_342)
        phases_342 = np.where(n1 == 1, phases_342 + 90.0, phases_342)
        phases_rad = np.deg2rad(phases_342)

        anchor_freqs, _ = self._get_freq_phase_vectorized(self.main_doodson, mjd_tt, mjd_ut)

        mask_lp = n1 == 0
        mask_di = n1 == 1
        mask_sd = n1 == 2
        
        anchor_n1 = self.main_doodson[:, 0]
        anchors_lp = anchor_n1 == 0
        anchors_di = anchor_n1 == 1
        anchors_sd = anchor_n1 == 2

        # ================================================================
        #  BLOK TAMBAHAN: TERAPKAN MODULASI NODAL 18.6 TAHUN
        #  Letakkan di sini, SETELAH anchor_freqs dihitung, SEBELUM loop komponen.
        # ================================================================
        tt_jd = mjd_tt + 2400000.5  # konversi MJD ke JD (TT)
        self.modulated_blq = {}
        for tide in self.main_waves:
            f, u = self._apply_nodal_modulation(tide, tt_jd)
            raw = self.blq[tide]
            # raw = [amp_rad, ph_rad, amp_ew, ph_ew, amp_ns, ph_ns]
            self.modulated_blq[tide] = [
                raw[0] * f,
                raw[1] + math.degrees(u),
                raw[2] * f,
                raw[3] + math.degrees(u),
                raw[4] * f,
                raw[5] + math.degrees(u),
            ]
        # ================================================================

        displacements = []
        for comp_idx in range(3):  # 0=radial, 1=west, 2=south
            # Di dalam _process_admittance, nanti gunakan self.modulated_blq
            re_lp, im_lp = self._process_admittance(comp_idx, freqs_342, mask_lp, anchors_lp, anchor_freqs)
            re_di, im_di = self._process_admittance(comp_idx, freqs_342, mask_di, anchors_di, anchor_freqs)
            re_sd, im_sd = self._process_admittance(comp_idx, freqs_342, mask_sd, anchors_sd, anchor_freqs)
            
            re_full = np.zeros(342)
            im_full = np.zeros(342)
            
            re_full[mask_lp] = re_lp; im_full[mask_lp] = im_lp
            re_full[mask_di] = re_di; im_full[mask_di] = im_di
            re_full[mask_sd] = re_sd; im_full[mask_sd] = im_sd
            
            amp_342 = self.tamp_342 * np.sqrt(re_full**2 + im_full**2)
            phase_admit = np.arctan2(im_full, re_full)
            
            final_phase = phases_rad + phase_admit
            
            disp = np.sum(amp_342 * np.cos(final_phase))
            displacements.append(disp)
            
        return displacements[0], displacements[1], displacements[2] # dU, dW, dS

# ==============================================================================
# DATA KONSTANTA OTL STASIUN JOLOTUNDO
# ==============================================================================
# Data BLQ Stasiun Jolotundo (FES2014b) 
# [amp_rad, ph_rad, amp_ew, ph_ew, amp_ns, ph_ns]
JOLOTUNDO_FES2014_BLQ = {
    'M2':  [0.01007, -164.0, 0.00252, -95.0,  0.00334,  43.4],
    'S2':  [0.00442, -89.8,  0.00115, -54.0,  0.00191, 110.0],
    'N2':  [0.00219, 165.3,  0.00050, -109.0, 0.00062,   5.6],
    'K2':  [0.00121, -95.3,  0.00030, -54.0,  0.00053, 109.8],
    'K1':  [0.01276,  0.7,   0.00118,  7.3,   0.00145,  89.1],
    'O1':  [0.00878, -22.1,  0.00075, -15.7,  0.00011,  49.6],
    'P1':  [0.00390, -1.4,   0.00035,  4.1,   0.00044,  86.4],
    'Q1':  [0.00186, -34.0,  0.00017, -13.0,  0.00007, 166.7],
    'Mf':  [0.00103, -169.5, 0.00006, -41.9,  0.00002,-147.9],
    'Mm':  [0.00061, -173.9, 0.00002, -71.4,  0.00003,-173.8],
    'Ssa': [0.00050, 178.7,  0.00002, -159.4, 0.00002, 169.3],
}

# ==============================================================================
# DATA KONSTANTA OTL GRAVITY & TILT STASIUN JOLOTUNDO (FES2014b)
# ==============================================================================
# Format: [amp_grav, ph_grav, amp_tilt_ew, ph_tilt_ew, amp_tilt_ns, ph_tilt_ns]
# Satuan: Amplitude Grav = nm/s^2, Amplitude Tilt = nrad, Phase = derajat
JOLOTUNDO_FES2014_GRAV_BLQ = {
    'M2':  [29.58, -150.7, 33.58,  117.8, 54.66, -142.1],
    'S2':  [14.88,  -85.3, 15.02,  132.0, 26.74,  -83.9],
    'N2':  [ 6.03,  179.1,  7.28,  103.9, 10.67, -171.9],
    'K2':  [ 4.03,  -91.4,  3.64,  128.5,  7.36,  -83.1],
    'K1':  [35.26,   -0.9, 25.31, -172.6, 23.37, -113.6],
    'O1':  [23.72,  -23.9, 15.80,  163.7,  5.56,  154.4],
    'P1':  [10.76,   -2.9,  7.55, -173.9,  6.96, -114.3],
    'Q1':  [ 4.96,  -35.2,  3.45,  160.0,  0.58,   53.1],
    'Mf':  [ 2.47, -168.0,  0.40,   21.2,  0.29,  132.0],
    'Mm':  [ 1.52, -172.1,  0.20,   12.6,  0.20,   19.1],
    'Ssa': [ 1.24,  178.1,  0.25,    6.0,  0.08,  -53.4],
}

# --------------------------------------------------------------------------
# Main StationDisplacement API
# --------------------------------------------------------------------------
class StationDisplacement:
    def __init__(self, itrf_xyz: np.ndarray, blq_data: Optional[Dict[str, list]] = None):
        self.itrf_xyz = np.asarray(itrf_xyz, dtype=float)
        self.lat_deg, self.lon_deg, self.height_m = itrf_to_geodetic(self.itrf_xyz)
        self.lat_rad, self.lon_rad = math.radians(self.lat_deg), math.radians(self.lon_deg)
        
        self._otl_engine = None
        if blq_data and Asterid342Engine_FES2014:
            self._otl_engine = Asterid342Engine_FES2014(blq_data)

    def total_displacement(self, tt_jd: float, ut1_jd: float,
                           xp_rad: float, yp_rad: float,
                           sun_itrf: np.ndarray, moon_itrf: np.ndarray,
                           include_atm: bool = True) -> np.ndarray:
        """
        Compute total site displacement (ITRS) due to all conventional IERS 2010 effects.

        Returns
        -------
        disp : np.ndarray (3,)
            Displacement vector in ITRS (meters).
        """
        disp = np.zeros(3, dtype=np.float64)
        cal = jd_to_cal(*split_jd(ut1_jd), scale='utc')
        fhour = cal['hour'] + cal['minute'] / 60.0 + cal['second'] / 3600.0

        # ---- 1. Solid Earth Tides ----
        tide_rad = solid_earth_tide(
            self.itrf_xyz, cal['year'], cal['month'], cal['day'],
            fhour, sun_itrf, moon_itrf
        )
        disp += np.array(tide_rad)

        # ---- 2. Pole Tide (rotational deformation) ----
        xp_mean, yp_mean = mean_pole_iers2010(tt_jd)
        dX_pt, dY_pt, dZ_pt = pole_tide(
            xp_rad - xp_mean,
            yp_rad - yp_mean,
            self.lat_rad, self.lon_rad, self.height_m
        )
        disp += np.array([dX_pt, dY_pt, dZ_pt])

        # ---- 3. Ocean Tide Loading (OTL) ----
        if self._otl_engine is not None:
            dU, dW, dS = self._otl_engine.compute_displacement(
                tt_jd - 2400000.5, delta_t_from_jd(tt_jd)
            )
            cos_lat = math.cos(self.lat_rad)
            sin_lat = math.sin(self.lat_rad)
            cos_lon = math.cos(self.lon_rad)
            sin_lon = math.sin(self.lon_rad)

            # OTL returns: dU (radial, up), dW (west), dS (south)
            # Convert to ENU: east = -dW, north = -dS
            de_otl = -dW
            dn_otl = -dS
            du_otl = dU

            # Rotate ENU → XYZ (same as used in Site_Geophysic)
            disp[0] += du_otl * cos_lat * cos_lon - dn_otl * sin_lat * cos_lon - de_otl * sin_lon
            disp[1] += du_otl * cos_lat * sin_lon - dn_otl * sin_lat * sin_lon + de_otl * cos_lon
            disp[2] += du_otl * sin_lat + dn_otl * cos_lat

        # ---- 4. Ocean Pole Tide Loading (Desai 2002) ----
        de_opt, dn_opt, du_opt = ocean_pole_tide_loading(
            self.lat_rad, self.lon_rad,
            xp_rad - xp_mean,
            -(yp_rad - yp_mean)   # sign convention: m2 = -(yp - yp_mean)
        )

        # Ocean pole tide returns (east, north, up) in meters.
        # Rotate ENU → XYZ using the same rotation as for OTL.
        cos_lat = math.cos(self.lat_rad)
        sin_lat = math.sin(self.lat_rad)
        cos_lon = math.cos(self.lon_rad)
        sin_lon = math.sin(self.lon_rad)

        disp[0] += du_opt * cos_lat * cos_lon - dn_opt * sin_lat * cos_lon - de_opt * sin_lon
        disp[1] += du_opt * cos_lat * sin_lon - dn_opt * sin_lat * sin_lon + de_opt * cos_lon
        disp[2] += du_opt * sin_lat + dn_opt * cos_lat

        # ---- 5. Atmospheric Pressure Loading (non-tidal) ----
        if include_atm:
            dX_atm, dY_atm, dZ_atm = atm_loading_displacement(
                ut1_jd, self.lat_rad, self.lon_rad
            )
            disp += np.array([dX_atm, dY_atm, dZ_atm])

        return disp

    def instantaneous_position(self, tt_jd: float, ut1_jd: float,
                               xp_rad: float, yp_rad: float,
                               sun_itrf: np.ndarray, moon_itrf: np.ndarray,
                               include_atm: bool = True) -> np.ndarray:
        
        return self.itrf_xyz + self.total_displacement(tt_jd, ut1_jd, xp_rad, yp_rad, 
                                                       sun_itrf, moon_itrf, include_atm)


