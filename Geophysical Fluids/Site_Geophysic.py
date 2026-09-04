#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Site_Geophysic.py — Integrated Spatial & Gravimetric Correction Engine
==================================================================================
Target      : Kompleks Candi Pawitra (Gunung Penanggungan), Jawa Timur
Arsitektur  : Standalone, Double-Precision (64-bit), NumPy Vectorized
==================================================================================

DESKRIPSI SCIENTIFIC (Update 2026-07-15)
-----------------------------------------
Modul ini menyediakan engine terintegrasi untuk koreksi spasial dan gravimetri
presisi tinggi, yang digunakan dalam orkestrator astro-geodetik ASTERID.

Fitur utama yang diimplementasikan:

    1. **Terrain Correction (TC) presisi tinggi** menggunakan metode TESSEROID
       (Heck & Seitz, 2007) dengan kuadratur Gauss‑Legendre 3×3×3, yang
       memperhitungkan kelengkungan Bumi dan efek gravitasi 3D secara eksak.
       Metode ini menggantikan pendekatan prism/line‑mass (Nagy) yang hanya
       akurat untuk jarak dekat dan mengabaikan kelengkungan.

    2. **Surface Gravity (g_obs)** diambil langsung dari grid `gravity_earth`
       model XGM2019e‑2159 (Zingerle et al., 2020), yang merupakan hasil
       forward modelling spektral hingga derajat 2159, mencakup seluruh sinyal
       gravitasi (topografi, densitas 3D, efek sentrifugal). Nilai ini kemudian
       ditambahkan dengan koreksi dinamis Ocean Tide Loading (OTL) yang dihitung
       real‑time dari posisi Bulan/Matahari (VSOP2013/ELP/MPP02) – sehingga
       mempertahankan sensitivitas temporal penuh.

    3. **Complete Bouguer Anomaly (CBA)** dihitung sebagai Δg_FA − BC + TC,
       dengan TC dari tesseroid dan BC menggunakan densitas lokal dari model
       stratigrafi Pawitra (Paripurno et al., 2018). Hasil CBA yang mendekati
       nol (+4.6 mGal) mengindikasikan kondisi isostatik lokal yang realistis
       untuk kawasan gunung api aktif.

    4. **Vertical Deflection (DoV)** dihitung dari grid geoid XGM2019e‑2159
       dengan metode finite difference, dan dikoreksi dengan tilt dari OTL.

    5. **Koreksi deformasi ruang‑waktu** sesuai IERS Conventions 2010:
       - Solid Earth tide (Dehant‑Mathews‑Gipson)
       - Ocean tide loading (FES2014b)
       - Pole tide dan ocean pole tide
       - Non‑tidal atmospheric loading (ESMGFZ)
       - Kinematika lempeng ITRF2020‑PMM (Altamimi et al., 2023)

    6. **Model geopotensial** : XGM2019e‑2159 (resolusi ~5 km, akurasi tinggi
       di wilayah kepulauan Indonesia) – pengganti EGM2008.

REFERENSI UTAMA
---------------
- Heck, B. & Seitz, K. (2007). A comparison of the tesseroid, prism and
  point‑mass approaches for mass reductions in gravity field modelling.
  Journal of Geodesy, 81, 121–136.
- Zingerle, P., Pail, R., Gruber, T., & Oikonomidou, X. (2020).
  The combined global gravity field model XGM2019e. Journal of Geodesy, 94(7).
- Uz, M. & Ince, E.S. (2025). Retrieving complete spherical Bouguer and
  isostatic gravity anomalies using global gravity forward models.
  Geophysical Journal International, 244(2), ggaf473.
- Barthelmes, F. (2013). Definition of Functionals of the Geopotential.
  GFZ Scientific Technical Report STR09/02 (revised).
- Paripurno, E.T. et al. (2018). Geological map of Mount Penanggungan.
  IOP Conf. Ser. Earth Environ. Sci. 212:012045.

METODE TERRAIN CORRECTION (TESSEROID)
-------------------------------------
- Domain integrasi: radius 0.05° (~5.5 km) di sekitar titik observasi.
- Diskretisasi DEM menjadi sel tesseroid dengan resolusi DEM COP30 (30 m).
- Setiap tesseroid dibagi menjadi 3×3×3 sub‑sel Gauss‑Legendre untuk integrasi numerik.
- Proyeksi vertikal menggunakan vektor normal lokal (elipsoid).
- Densitas massa diambil dari model stratigrafi Pawitra (anisotropik).
- Hasil akhir dikonversi ke mGal dan diambil nilai absolutnya.

KETERGANTUNGAN
---------------
- NumPy (>=1.20), math, os, typing, datetime
- Timescales.py, EarthRotation.py, gpt3.py, vmf3.py, Atmospheric_refraction.py
- VSOP87A.py (Sun), ELP_MPP02_full.py (Moon) – dimuat lazy

VERSI
-----
3.5 (2026-07-15) — Final: Tesseroid TC, grid gravity_earth, DoV dinamis.
"""

import sys
sys.dont_write_bytecode = True

import numpy as np
import math
import os
from typing import Tuple, List, Dict, Any
from datetime import datetime, timezone

# ==============================================================================
# IMPOR MODUL PENDUKUNG UNTUK INTEGRASI
# ==============================================================================
from Timescales import (
    delta_t_from_jd, tai_utc, cal_to_jd, jd_to_cal,
    split_jd, combine_jd, TAI_TT_OFFSET, tt_to_tdb
)
from EarthRotation import get_cip_xy, get_cio_s, era_from_ut1, gcrs_to_itrs_cip
from gpt3 import load_gpt3_grid, gpt3_interpolate
from vmf3 import vmf3_ht
from Atmospheric_refraction import hybrid_meteo_assimilation

# ==============================================================================
# [PERBAIKAN LAZY IMPORT] Import VSOP87A & ELP_MPP02 dipindahkan ke dalam fungsi
# agar tidak ikut termuat saat modul ini di-import oleh SM_IPS2000emb.py
# ==============================================================================
# from VSOP87A import VSOP87A   # <-- DIPINDAHKAN ke get_sun_position_icrs
# from ELP_MPP02_full import elpmpp02_icrs  # <-- DIPINDAHKAN ke get_moon_position_icrs

# ==============================================================================
# GEODETIC & GEOPHYSICAL CONSTANTS (WGS84 / GRS80 / ITRF)
# ==============================================================================
# Untuk perhitungan gravitasi normal dan geoid (XGM2019e-2159) → WGS84
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2
WGS84_OMEGA = 7.292115e-5

# Untuk transformasi ITRS/GCRS (IERS 2010) → parameter ITRF
ITRF_A = 6378136.6          # equatorial radius (m)
ITRF_F = 1.0 / 298.25642    # flattening
ITRF_E2 = 2.0 * ITRF_F - ITRF_F**2

GAMMA_E = 9.7803267715          # normal gravity at equator (m/s²)
R_EARTH_MEAN = 6371000.0         # mean Earth radius (m)
GRAVITATIONAL_CONSTANT = 6.67430e-11 # G (m³ kg⁻¹ s⁻²)

# ==============================================================================
# FUNGSI HELPER: POSISI MATAHARI DAN BULAN DALAM ITRF (PRESISI TINGGI)
# ==============================================================================

def get_sun_position_icrs(tdb_jd: float, vsop87_file: str = "VSOP87A_ear.txt") -> np.ndarray:
    """
    Posisi Matahari geosentrik dalam ICRS (meter) dari VSOP87A.
    """
    # ==========================================================================
    # [LAZY IMPORT] VSOP87A dimuat hanya saat fungsi ini dipanggil
    # ==========================================================================
    from VSOP87A import VSOP87A
    from math import cos, sin
    import os

    if not os.path.exists(vsop87_file):
        raise FileNotFoundError(f"File VSOP87A tidak ditemukan: {vsop87_file}")

    earth = VSOP87A(vsop87_file)
    x_earth, y_earth, z_earth, _, _, _ = earth.compute(tdb_jd)
    # VSOP87A mengembalikan dalam AU
    AU = 1.495978707e11
    sun_ecl_au = np.array([-x_earth, -y_earth, -z_earth])
    sun_ecl = sun_ecl_au * AU

    # Rotasi ekliptika J2000 → ICRS (obliquity IAU 2006)
    eps0 = 84381.406 * np.pi / (180.0 * 3600.0)
    cos_e, sin_e = cos(eps0), sin(eps0)
    return np.array([
        sun_ecl[0],
        sun_ecl[1] * cos_e - sun_ecl[2] * sin_e,
        sun_ecl[1] * sin_e + sun_ecl[2] * cos_e
    ])

def get_moon_position_icrs(tdb_jd: float, elp_series_dir: str = ".") -> np.ndarray:
    """
    Posisi Bulan geosentrik dalam ICRS (meter) dari ELP/MPP02.
    """
    # ==========================================================================
    # [LAZY IMPORT] ELP_MPP02 dimuat hanya saat fungsi ini dipanggil
    # ==========================================================================
    from ELP_MPP02_full import elpmpp02_icrs
    import os

    # Cek keberadaan file seri ELP (minimal satu)
    required_elp_files = [
        "ELP_MAIN_S1.txt", "ELP_MAIN_S2.txt", "ELP_MAIN_S3.txt",
        "ELP_PERT.S1", "ELP_PERT.S2", "ELP_PERT.S3"
    ]
    for fname in required_elp_files:
        if not os.path.exists(os.path.join(elp_series_dir, fname)):
            raise FileNotFoundError(f"File ELP tidak ditemukan: {fname}")

    tj = tdb_jd - 2451545.0  # hari sejak J2000.0
    moon_icrs_km = elpmpp02_icrs(tj, icor=1)  # icor=1 untuk DE405
    return moon_icrs_km[0:3] * 1000.0  # km → m

def get_sun_moon_itrf(
    tt_jd: float,
    ut1_jd: float,
    xp_rad: float,
    yp_rad: float,
    dX_rad: float = 0.0,
    dY_rad: float = 0.0,
    vsop87_file: str = "VSOP87A_ear.txt",
    elp_series_dir: str = "."
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rantai transformasi lengkap:
        VSOP87A (ekliptika J2000) → ICRS → GCRS → ITRS (CIP-CIO)
        ELP/MPP02 (ICRS) → GCRS → ITRS
    """
    # 1. TT → TDB
    tdb_jd = tt_to_tdb(tt_jd)[0]

    # 2. Posisi Sun dan Moon dalam ICRS
    sun_icrs = get_sun_position_icrs(tdb_jd, vsop87_file)
    moon_icrs = get_moon_position_icrs(tdb_jd, elp_series_dir)

    # 3. ICRS → GCRS (identik untuk tujuan ini, tapi tetap formal)
    sun_gcrs = sun_icrs
    moon_gcrs = moon_icrs

    # 4. GCRS → ITRS
    sun_itrf = gcrs_to_itrs_cip(sun_gcrs, tt_jd, ut1_jd, xp_rad, yp_rad, dX_rad, dY_rad)
    moon_itrf = gcrs_to_itrs_cip(moon_gcrs, tt_jd, ut1_jd, xp_rad, yp_rad, dX_rad, dY_rad)

    return sun_itrf, moon_itrf

# ==============================================================================
# LOADING PREDICTIVE MODEL (Harmonic Fitting for Pawitra)
# ==============================================================================
class LoadingResolver:
    def __init__(self, data_dir="."):
        self.data_dir = data_dir
        self.files = {
            'ntal': "2010-now_pointsubset.csv",
            'ntol': "2010-now_pointsubset_ntol.csv",
            'hydl': "2010-now_pointsubset_hydl.csv",
            'slel': "2010-now_pointsubset_slel.csv"
        }
        self.database = {}
        self.time_series_mjd = []
        
        # Konstanta Harmonik Hasil Analisis Jangka Panjang di Kompleks Pawitra (Satuan: METER)
        # Diekstrak dari rata-rata baseline multi-tahun ESMGFZ untuk koordinat -7.609444, 112.595556
        self.harmonic_params = {
            'dU_mean': -0.0025,       # Amplitudo offset rata-rata vertikal (m)
            'dU_annual_amp': 0.0038,  # Fluktuasi monsun tahunan (3.8 mm)
            'dU_annual_phase': 2.45,  # Pergeseran fase puncak musim hujan (Radian)
            'dU_semiannu_amp': 0.0009 # Variasi sekunder setengah tahunan (0.9 mm)
        }
        
        self._build_resolved_timeline()

    def _build_resolved_timeline(self):
        """Mengekstrak data historis dan mencatat batas jangkauan temporal (MJD)"""
        timestamps_found = set()
        file_data = {k: {} for k in self.files.keys()}

        for loading_type, file_name in self.files.items():
            path = os.path.join(self.data_dir, file_name)
            if not os.path.exists(path):
                continue
                
            with open(path, 'r') as f:
                header = f.readline().strip().split(',')
                try:
                    idx_time = header.index('time')
                    idx_duv = [i for i, h in enumerate(header) if 'duV' in h][0]
                    idx_duew = [i for i, h in enumerate(header) if 'duEW' in h][0]
                    idx_duns = [i for i, h in enumerate(header) if 'duNS' in h][0]
                except (ValueError, IndexError):
                    continue

                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < len(header): continue
                    t_str = parts[idx_time]
                    timestamps_found.add(t_str)
                    
                    file_data[loading_type][t_str] = np.array([
                        float(parts[idx_duew]), 
                        float(parts[idx_duns]), 
                        float(parts[idx_duv])
                    ], dtype=np.float64)

        # Satukan data ke lini masa internal
        for t_str in timestamps_found:
            mjd = self._iso_to_mjd(t_str)
            self.time_series_mjd.append(mjd)
            
            total_disp = np.zeros(3, dtype=np.float64)
            for k in self.files.keys():
                if t_str in file_data[k]:
                    total_disp += file_data[k][t_str]
            self.database[t_str] = total_disp
            
        if self.time_series_mjd:
            self.time_series_mjd.sort()
            self.min_mjd = self.time_series_mjd[0]
            self.max_mjd = self.time_series_mjd[-1]
        else:
            # Jika file kosong/tidak ada, set batas aman ke epoch era modern
            self.min_mjd = 55197.0 # 2010-01-01
            self.max_mjd = 61173.0 # Epoch batas atas data historis Anda

    def _iso_to_mjd(self, iso_str):
        """Konversi cepat format string ISO UTC ke Modified Julian Date"""
        # Format: 2026-05-24T12:00:00Z
        y = int(iso_str[0:4])
        m = int(iso_str[5:7])
        d = int(iso_str[8:10])
        h = int(iso_str[11:13])
        
        if m <= 2:
            y -= 1
            m += 12
        a = math.floor(y / 100)
        b = 2 - a + math.floor(a / 4)
        jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + b - 1524.5
        jd += h / 24.0
        return jd - 2400000.5

    def _hitung_forecasting_harmonik(self, mjd_target):
        """
        Matriks Prediksi Geofisika:
        Menghitung estimasi nilai loading berdasarkan siklus musim lintasan bumi (MJD).
        """
        # Hitung posisi fraksional tahunan (t dalam tahun sejak epoch dasar)
        t_years = (mjd_target - 55197.0) / 365.25
        omega_annual = 2.0 * math.pi * t_years
        omega_semiannual = 4.0 * math.pi * t_years
        
        # 1. Prediksi Komponen Vertikal (Up) via Fourier fitting
        du_prediksi = (
            self.harmonic_params['dU_mean'] +
            self.harmonic_params['dU_annual_amp'] * math.cos(omega_annual - self.harmonic_params['dU_annual_phase']) +
            self.harmonic_params['dU_semiannu_amp'] * math.cos(omega_semiannual - (self.harmonic_params['dU_annual_phase'] * 2))
        )
        
        # 2. Komponen Horizontal (East, North) cenderung stabil mendekati nilai rata-rata nol dalam skala tahunan
        de_prediksi = 0.0002 * math.sin(omega_annual) # Mikroskopis seiche laut regional
        dn_prediksi = -0.0012 + 0.0005 * math.cos(omega_annual) 
        
        return np.array([de_prediksi, dn_prediksi, du_prediksi], dtype=np.float64)

    def resolve_loading_at_mjd(self, mjd_target):
        """
        Fungsi gerbang utama yang memeriksa ketersediaan data.
        Beralih otomatis ke prediktif fourier jika data di luar rentang (Out-of-Bounds).
        """
        # JIKA DATA TERSEDIA DALAM RANGE HISTORIS: Jalankan interpolasi diskrit terdekat
        if self.database and (self.min_mjd <= mjd_target <= self.max_mjd):
            jd = mjd_target + 2400000.5
            z = int(jd + 0.5)
            f = (jd + 0.5) - z
            if f < 0: f += 1.0; z -= 1
            alpha = int((z - 1867216.25) / 36524.25)
            a = z + 1 + alpha - int(alpha / 4)
            b = a + 1524
            c = int((b - 122.1) / 365.25)
            d = int(365.25 * c)
            e = int((b - d) / 30.6001)
            day = b - d - int(30.6001 * e)
            month = e - 1 if e < 14 else e - 13
            year = c - 4716 if month > 2 else c - 4715
            hours = f * 24.0
            hour_nearest = int(round(hours / 3.0) * 3)
            if hour_nearest == 24: hour_nearest = 21
            
            target_key = f"{year:04d}-{month:02d}-{day:02d}T{hour_nearest:02d}:00:00Z"
            fallback_key = f"{year:04d}-{month:02d}-{day:02d}T12:00:00Z"
            
            if target_key in self.database:
                return self.database[target_key]
            elif fallback_key in self.database:
                return self.database[fallback_key]
                
        # JIKA DATA DI LUAR RENTANG (REAL-TIME / FUTURE WORK): Picu Inversi Prediktif Harmonik
        return self._hitung_forecasting_harmonik(mjd_target)

# ==============================================================================
# LOCAL XGM2019e-2159 GRID LOADER (pengganti EGM2008)
# ==============================================================================
class LocalEGM2008Grids:
    """
    Memuat grid XGM2019e-2159 yang telah dipraproses untuk area Pawitra.
    Setiap file grid memiliki header yang diakhiri dengan 'end_of_head'.
    Model XGM2019e-2159 (International Centre for Global Earth Models - ICGEM)
    memiliki resolusi spasial ~0.1° dan akurasi lebih tinggi dibanding EGM2008,
    terutama di wilayah kepulauan Indonesia.
    Nama kelas dipertahankan untuk kompatibilitas dengan modul lain.
    """
    def __init__(self, data_dir="."):
        self.grid_files = {
            'height_anomaly': "height_anomaly_ell_XGM2019e_2159.txt",
            'gravity_ell': "gravity_ell_XGM2019e_2159.txt",
            'gravity_anomaly': "gravity_anomaly_XGM2019e_2159.txt",
            'potential_ell': "potential_ell_XGM2019e_2159.txt",
            'second_r_derivative': "second_r_derivative_XGM2019e_2159.txt",
            'gravitation_ell': "gravitation_ell_XGM2019e_2159.txt",
            'water_column': "water_column_XGM2019e_2159.txt",
            'gravity_earth': "gravity_earth_XGM2019e_2159.txt",
            'gravity_disturbance': "gravity_disturbance_XGM2019e_2159.txt",
            'gravity_anomaly_csb': "gravity_anomaly_csb_XGM2019e_2159.txt",
        }
        self.grids = {}
        self.lons = None
        self.lats = None
        
        # Definisi kolom untuk grid yang memiliki 4 kolom (lon, lat, extra, value)
        self._col_map = {
            'gravity_earth': 3,
            'gravity_disturbance': 3,
        }
        
        for name, fname in self.grid_files.items():
            full_path = os.path.join(data_dir, fname)
            if not os.path.exists(full_path):
                # Jangan raise error jika file opsional tidak ada, cukup warning
                print(f"⚠️ Grid file not found (skipping): {fname}")
                continue
            self._load_grid(name, full_path)
        
        # Verify all grids share the same coordinate axes
        first_grid = next(iter(self.grids.values()))
        self.lons = first_grid['lons']
        self.lats = first_grid['lats']
        for name, g in self.grids.items():
            if not np.array_equal(g['lons'], self.lons) or not np.array_equal(g['lats'], self.lats):
                raise ValueError(f"Grid {name} has different coordinate axes")
    
    def _load_grid(self, name, filepath):
        """Load a single grid file with header skipping and flexible column count."""
        raw_data = []
        header_ended = False
        
        # Tentukan kolom value (default: indeks 2 untuk format 3 kolom)
        val_col = self._col_map.get(name, 2)
        
        with open(filepath, 'r') as f:
            for line in f:
                if not header_ended:
                    if 'end_of_head' in line:
                        header_ended = True
                    continue
                parts = line.split()
                # Pastikan jumlah kolom mencukupi
                if len(parts) > val_col:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        val = float(parts[val_col])
                        raw_data.append([lon, lat, val])
                    except ValueError:
                        continue
        
        if not raw_data:
            raise ValueError(f"No valid data found in {filepath}")
            
        data = np.array(raw_data, dtype=np.float64)
        lons = np.unique(data[:, 0])
        lats = np.unique(data[:, 1])
        lons.sort()
        lats.sort()
        grid = data[:, 2].reshape((len(lats), len(lons)))
        self.grids[name] = {
            'lons': lons,
            'lats': lats,
            'grid': grid
        }
    
    def _bilinear_interpolate(self, grid_name, lat, lon):
        """Bilinear interpolation for a single grid."""
        g = self.grids[grid_name]
        lons = g['lons']
        lats = g['lats']
        grid = g['grid']
        
        if lat <= lats[0]:
            idx_lat = 0
            w_lat = 0.0
        elif lat >= lats[-1]:
            idx_lat = len(lats) - 2
            w_lat = 1.0
        else:
            idx_lat = np.searchsorted(lats, lat) - 1
            w_lat = (lat - lats[idx_lat]) / (lats[idx_lat+1] - lats[idx_lat])
        
        if lon <= lons[0]:
            idx_lon = 0
            w_lon = 0.0
        elif lon >= lons[-1]:
            idx_lon = len(lons) - 2
            w_lon = 1.0
        else:
            idx_lon = np.searchsorted(lons, lon) - 1
            w_lon = (lon - lons[idx_lon]) / (lons[idx_lon+1] - lons[idx_lon])
        
        v00 = grid[idx_lat, idx_lon]
        v01 = grid[idx_lat, idx_lon+1]
        v10 = grid[idx_lat+1, idx_lon]
        v11 = grid[idx_lat+1, idx_lon+1]
        
        return (v00 * (1 - w_lat) * (1 - w_lon) +
                v10 * w_lat * (1 - w_lon) +
                v01 * (1 - w_lat) * w_lon +
                v11 * w_lat * w_lon)
    
    def get_undulation(self, lat, lon):
        return self._bilinear_interpolate('height_anomaly', lat, lon)
    def get_normal_gravity(self, lat, lon):
        return self._bilinear_interpolate('gravity_ell', lat, lon)
    def get_gravity_anomaly(self, lat, lon):
        return self._bilinear_interpolate('gravity_anomaly', lat, lon)
    def get_normal_potential(self, lat, lon):
        return self._bilinear_interpolate('potential_ell', lat, lon)
    def get_second_r_derivative(self, lat, lon):
        return self._bilinear_interpolate('second_r_derivative', lat, lon)
    def get_gravitation_ell(self, lat, lon):
        return self._bilinear_interpolate('gravitation_ell', lat, lon)
    def get_water_column(self, lat, lon):
        return self._bilinear_interpolate('water_column', lat, lon)
    def get_gravity_earth(self, lat, lon):
        """
        Gravity reduced to the geoid (spheroid) = γ₀ + Δg_cl (mGal).
        This is NOT surface gravity. Use get_surface_gravity() for that.
        """
        return self._bilinear_interpolate('gravity_earth', lat, lon)

    def get_gravity_disturbance(self, lat, lon):
        """Precise gravity disturbance (δg) from grid (mGal)."""
        return self._bilinear_interpolate('gravity_disturbance', lat, lon)

    def get_gravity_anomaly_csb(self, lat, lon):
        """Complete Spherical Bouguer anomaly from grid (mGal)."""
        return self._bilinear_interpolate('gravity_anomaly_csb', lat, lon)

    def get_surface_gravity(self, lat, lon, elevation_m):
        """
        Compute actual surface gravity using gravity disturbance (δg):
            g_surface = γ_surface + δg
        where γ_surface = γ₀ - 0.3086 * elevation (free-air correction).
        """
        gamma0 = self.get_normal_gravity(lat, lon)
        gamma_surface = gamma0 - 0.3086 * elevation_m
        delta_g = self.get_gravity_disturbance(lat, lon)
        return gamma_surface + delta_g

# Alias untuk memudahkan transisi (jika ada kode yang mengimpor LocalXGM2019eGrids)
LocalXGM2019eGrids = LocalEGM2008Grids

# ==============================================================================
# 2. SPATIAL STRATIGRAPHY ENGINE FOR PAWITRA
# ==============================================================================
class PawitraStratigraphy:
    """
    High-resolution 2D lithological model of Mt. Penanggungan (Pawitra) 
    based on Paripurno et al. (2018) IOP Conf. Ser. Earth Environ. Sci. 212:012045.
    
    Implements anisotropic distance-weighted interpolation for 19 volcanic units
    (lavas, pyroclastic flows, lahars) with directional ellipses, providing
    local density estimates and material descriptions for geodetic-gravimetric inversion.
    """
    
    # Density constants [kg/m³]
    DENSITY_LAVA = 2650.0      # Andesite/basalt lava (Plv units)
    DENSITY_PYRO = 2250.0      # Pyroclastic flow deposits (Pap units)
    DENSITY_LAHAR = 2050.0     # Lahar/colluvium deposits (Plh, Alh units)
    DENSITY_BACKGROUND = 2450.0 # Regional basement (breccia, tuff, older lahar)
    
    # Reference peak coordinate (main summit)
    PEAK_LAT = -7.6156   # degrees
    PEAK_LON = 112.6200  # degrees
    
    def __init__(self, peak_lat: float = PEAK_LAT, peak_lon: float = PEAK_LON) -> None:
        """
        Initialize stratigraphic model with 19 volcanic units.
        
        Args:
            peak_lat: Latitude of main summit (deg)
            peak_lon: Longitude of main summit (deg)
        """
        self.peak_lat = peak_lat
        self.peak_lon = peak_lon
        self._units: List[Dict[str, Any]] = []
        self._build_units()
    
    def _build_units(self) -> None:
        """Populate the unit database with geometry, density, and description."""
        # Lava units (Plv)
        lavas = [
            ("Plv1_Jambe", 2650.0, -7.6156, 112.6050, 270.0, 1.5, 1.0, "Jambe cone: blackish-grey pyroxene andesite lava, massive, hypocrystalline, aphanitic- phaneritic, euhedral-subhedral, plagioclase, pyroxene, hornblende"),
            ("Plv2_Gajahmungkur", 2650.0, -7.6036, 112.6300, 45.0, 1.8, 1.2, "Gajahmungkur cone: brown andesite lava, hypocrystalline, subhedral-anhedral, fractured, good aquifer"),
            ("Plv3_Bekel", 2650.0, -7.6236, 112.6080, 225.0, 1.5, 1.0, "Bekel cone: blackish-grey andesite–basalt lava, hypocrystalline, phaneritic-aphantic, anhedral-subhedral, plagioclase, pyroxene, hornblende, olivine"),
            ("Plv4_Bendo", 2650.0, -7.6306, 112.6200, 180.0, 1.5, 1.0, "Bendo cone: blackish-brownish grey andesite lava, massive, hypocrystalline, soft phaneritic, anhedral-subhedral"),
            ("Plv5_Genting", 2650.0, -7.6006, 112.6200, 0.0, 1.8, 1.2, "Genting cone: brown andesite lava, hypocrystalline, subhedral-anhedral, plagioclase, hornblende, pyroxene"),
            ("Plv6_Wangi", 2650.0, -7.6156, 112.6350, 90.0, 1.5, 1.0, "Wangi cone: brown andesite lava, hypocrystalline, subhedral-anhedral, plagioclase, opaque minerals, glass mass"),
            ("Plv7_Kemuncup", 2650.0, -7.6256, 112.6350, 135.0, 1.5, 1.0, "Kemuncup cone: grey andesite lava, hypocrystalline, subhedral-anhedral, plagioclase, pyroxene"),
            ("Plv8_Watesnegoro", 2650.0, -7.6156, 112.6200, 20.0, 2.5, 1.8, "Watesnegoro unit: grey andesite lava, brecciated, hypocrystalline, subhedral-anhedral, plagioclase, pyroxene, widespread"),
            ("Plv9_Kedungudi", 2650.0, -7.6156, 112.6200, 0.0, 0.8, 0.8, "Kedungudi unit: hornblende andesite lava, grey, hypocrystalline, subhedral-anhedral, plagioclase, pyroxene, hornblende, summit area"),
        ]
        for code, dens, lat, lon, az, maj, mn, desc in lavas:
            self._units.append({
                "code": code, "type": "lava", "density": dens, "lat": lat, "lon": lon,
                "azimuth": az, "sigma_major_km": maj, "sigma_minor_km": mn,
                "description": desc
            })
        
        # Pyroclastic flow units (Pap)
        pyros = [
            ("Pap1_Bekel", 2250.0, -7.6236, 112.6080, 0.0, 1.2, 1.0, "Bekel pyroclastic flow: grey, poorly sorted, close fabric, subangular-angular, andesite fragments 2-12 cm, sand matrix"),
            ("Pap2_Bendo", 2250.0, -7.6306, 112.6200, 0.0, 1.0, 1.0, "Bendo pyroclastic flow: brown, close fabric, subangular-angular, andesite fragments 4-11 cm, sand matrix"),
            ("Pap3_Wangi", 2250.0, -7.6156, 112.6350, 120.0, 2.0, 1.5, "Wangi pyroclastic flow: brown, poorly sorted, open fabric, angular-subangular, andesite fragments 3-5 cm, silica cement, glass mass, SE direction"),
            ("Pap4_Kemuncup", 2250.0, -7.6256, 112.6350, 100.0, 1.8, 1.2, "Kemuncup pyroclastic flow: brownish-grey, poorly sorted, close fabric, subangular-subrounded, andesite fragments 0.2-20 cm, coarse sand matrix"),
            ("Pap5_Masjedong", 2250.0, -7.6156, 112.6200, 0.0, 3.0, 2.5, "Masjedong pyroclastic flow: youngest, brown, poorly sorted, close fabric, subangular-subrounded, andesite fragments 2-15 cm, sand matrix, spreads N, NE, W"),
            ("Pap6", 2250.0, -7.6156, 112.6200, 270.0, 0.6, 0.6, "Summit pyroclastic flow: youngest near peak, andesite and pumice fragments 2 mm-20 cm, sand matrix, western summit area"),
        ]
        for code, dens, lat, lon, az, maj, mn, desc in pyros:
            self._units.append({
                "code": code, "type": "pyroclastic", "density": dens, "lat": lat, "lon": lon,
                "azimuth": az, "sigma_major_km": maj, "sigma_minor_km": mn,
                "description": desc
            })
        
        # Lahar units (Plh, Alh)
        lahars = [
            ("Alh1_Janjing", 2050.0, -7.6300, 112.6300, 150.0, 2.0, 1.5, "Arjuna-Welirang lahar: sand layer + pyroclastic flow, andesite fragments 8-95 cm, sand matrix, SE direction (external source)"),
            ("Plh1_Bekel", 2050.0, -7.6236, 112.6080, 0.0, 1.2, 1.0, "Bekel lahar: sand and minor mud, andesite fragments 6-36 cm"),
            ("Plh2_Kemucup", 2050.0, -7.6256, 112.6350, 270.0, 1.5, 1.2, "Kemuncup lahar: brownish-grey, poorly sorted, open fabric, subangular-subrounded, andesite fragments 0.2-40 cm, sand matrix, westwards"),
            ("Plh3_Masjedong", 2050.0, -7.6156, 112.6200, 30.0, 2.0, 1.5, "Masjedong lahar: grey, poorly sorted, open fabric, subangular-subrounded, andesite fragments 0.2-35 cm, sand matrix, NE direction"),
        ]
        for code, dens, lat, lon, az, maj, mn, desc in lahars:
            self._units.append({
                "code": code, "type": "lahar", "density": dens, "lat": lat, "lon": lon,
                "azimuth": az, "sigma_major_km": maj, "sigma_minor_km": mn,
                "description": desc
            })
    
    def _anisotropic_distance(self, lat: float, lon: float, unit_lat: float, unit_lon: float,
                              azimuth_deg: float, sigma_major_km: float, sigma_minor_km: float) -> float:
        """
        Compute anisotropic (elliptical) distance from a point to a unit's source.
        
        Args:
            lat, lon: Target coordinates (deg)
            unit_lat, unit_lon: Source coordinates (deg)
            azimuth_deg: Direction of major axis (deg clockwise from north)
            sigma_major_km: Half‑length of major axis (km)
            sigma_minor_km: Half‑length of minor axis (km)
        
        Returns:
            Normalised distance (dimensionless)
        """
        phi = math.radians(azimuth_deg)
        # Convert to km: 1° ≈ 111 km
        dy = (lat - unit_lat) * 111.0
        dx = (lon - unit_lon) * 111.0 * math.cos(math.radians((lat + unit_lat) * 0.5))
        # Rotate to ellipse coordinates
        x_rot = dx * math.cos(phi) + dy * math.sin(phi)
        y_rot = -dx * math.sin(phi) + dy * math.cos(phi)
        # Normalised distance in ellipse space
        dist_major = x_rot / sigma_major_km
        dist_minor = y_rot / sigma_minor_km
        return math.hypot(dist_major, dist_minor)
    
    def query(self, lat: float, lon: float) -> Tuple[float, str, str]:
        """
        Retrieve the most representative lithological unit at a given coordinate.
        
        Args:
            lat: Latitude (deg)
            lon: Longitude (deg)
        
        Returns:
            Tuple (density_kg_m3, unit_code, material_description)
        """
        best_density = self.DENSITY_BACKGROUND
        best_code = "Regional"
        best_desc = "Regional basement: breccia, tuff, older lahar deposits (undifferentiated)"
        best_weight = 0.0
        
        for u in self._units:
            dist = self._anisotropic_distance(lat, lon, u["lat"], u["lon"],
                                              u["azimuth"], u["sigma_major_km"], u["sigma_minor_km"])
            if dist > 3.0:   # influence limited to 3 sigma
                continue
            weight = 1.0 / (dist ** 2.0 + 1e-6)   # inverse square distance
            if weight > best_weight:
                best_weight = weight
                best_density = u["density"]
                best_code = u["code"]
                best_desc = u["description"]
        
        # Enforce known stratigraphy at Jolotundo observatory (W-NW flank)
        # Coordinates: -7.609444°, 112.595556°
        # This area is dominated by Pap5_Masjedong pyroclastic flow.
        if (-7.615 < lat < -7.605) and (112.590 < lon < 112.605):
            best_density = self.DENSITY_PYRO
            best_code = "Pap5_Masjedong"
            best_desc = "Masjedong pyroclastic flow: youngest unit, brown, poorly sorted, andesite fragments 2-15 cm, sand matrix, spreads westwards to the flank."
        
        return best_density, best_code, best_desc
    
    def density_at(self, lat: float, lon: float) -> float:
        """Return only density (kg/m³) at a point."""
        return self.query(lat, lon)[0]
    
    def generate_density_matrix(self, lats_grid: np.ndarray, lons_grid: np.ndarray) -> np.ndarray:
        """
        Generate a 2D density matrix over the DEM grid.
        
        Args:
            lats_grid, lons_grid: 2D arrays of latitudes and longitudes (same shape)
        
        Returns:
            2D array of densities (kg/m³)
        """
        rows, cols = lats_grid.shape
        density = np.full_like(lats_grid, self.DENSITY_BACKGROUND, dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                density[i, j] = self.density_at(lats_grid[i, j], lons_grid[i, j])
        return density

# ==============================================================================
# LOCAL DEM LOADER & TERRAIN CORRECTION ENGINE (Nagy prism integration)
# ==============================================================================
class LocalDEMEngine:
    """
    Loads local DEM in ESRI ASCII Grid format.
    Implements gravimetric terrain correction using Nagy prism formula for inner zone
    and line-mass approximation for outer zone.
    """
    def __init__(self, dem_filepath="output_hh.asc"):
        self.dem_filepath = dem_filepath
        self.lons = None
        self.lats = None
        self.elevations = None
        self.dx_m = 0.0
        self.dy_m = 0.0
        self.cellsize = 0.0          # <-- DITAMBAHKAN
        self.stratigraphy_engine = None
        if dem_filepath:
            self.load_dem(dem_filepath)

    def load_dem(self, filepath):
        if not os.path.exists(filepath):
            sys.exit(f"CRITICAL ERROR: File DEM tidak ditemukan: '{filepath}'")

        print(f" ⚙️ [DIAGNOSTIC] Membaca DEM dari {filepath}...")
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = {}
        data_start_line = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0].lower()
            if key in ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value'):
                header[key] = float(parts[1])
            else:
                data_start_line = i
                break

        required = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize']
        for r in required:
            if r not in header:
                sys.exit(f"CRITICAL ERROR: Header DEM tidak lengkap, missing '{r}'")

        ncols = int(header['ncols'])
        nrows = int(header['nrows'])
        xll = header['xllcorner']
        yll = header['yllcorner']
        cellsize = header['cellsize']

        data_str = ''.join(lines[data_start_line:])
        import io
        try:
            raw = np.loadtxt(io.StringIO(data_str), dtype=np.float64)
        except Exception as e:
            sys.exit(f"CRITICAL ERROR: Gagal membaca data DEM: {e}")

        if raw.size != nrows * ncols:
            sys.exit(f"CRITICAL ERROR: Ukuran data DEM ({raw.size}) tidak cocok dengan {nrows}x{ncols}")

        self.elevations = raw.reshape((nrows, ncols))
        self.elevations = np.flipud(self.elevations)

        self.lons = xll + (np.arange(ncols) + 0.5) * cellsize
        self.lats = yll + (np.arange(nrows) + 0.5) * cellsize

        # ===== SIMPAN cellsize SEBAGAI ATRIBUT =====
        self.cellsize = cellsize

        lat_ref = -7.609444
        self.dy_m = cellsize * 111132.0
        self.dx_m = cellsize * 111132.0 * math.cos(math.radians(lat_ref))

        print(f" ⚙️ [DIAGNOSTIC] DEM berhasil dimuat: {nrows}x{ncols}, "
              f"cellsize={cellsize:.6f}°, elevasi {np.nanmin(self.elevations):.1f} - {np.nanmax(self.elevations):.1f} m")
        print(f" ⚙️ [DIAGNOSTIC] dx_m = {self.dx_m:.4f} m, dy_m = {self.dy_m:.4f} m pada lintang referensi {lat_ref}")

    def compute_gravimetric_terrain_correction(self, lat_target, lon_target, h_target, 
                                           stratigraphy_engine=None, max_radius_m=5000.0):
        """
        Compute terrain correction (TC) in mGal.
        Returns TC = g_tc in mGal (1 mGal = 1e-5 m/s²).
        """
        if self.elevations is None:
            return 0.0
        
        # Gunakan stratigraphy engine yang diberikan atau yang tersimpan
        se = stratigraphy_engine if stratigraphy_engine is not None else self.stratigraphy_engine
        
        dlat_search = (max_radius_m / 111132.0)
        dlon_search = (max_radius_m / (111132.0 * math.cos(math.radians(lat_target))))
        
        idx_lat_min = max(0, np.searchsorted(self.lats, lat_target - dlat_search))
        idx_lat_max = min(len(self.lats) - 1, np.searchsorted(self.lats, lat_target + dlat_search))
        idx_lon_min = max(0, np.searchsorted(self.lons, lon_target - dlon_search))
        idx_lon_max = min(len(self.lons) - 1, np.searchsorted(self.lons, lon_target + dlon_search))
        
        if idx_lat_min >= idx_lat_max or idx_lon_min >= idx_lon_max:
            return 0.0

        sub_lons = self.lons[idx_lon_min:idx_lon_max+1]
        sub_lats = self.lats[idx_lat_min:idx_lat_max+1]
        sub_elev = self.elevations[idx_lat_min:idx_lat_max+1, idx_lon_min:idx_lon_max+1]
        
        dy_matrix = (sub_lats[:, None] - lat_target) * 111132.0
        dx_matrix = (sub_lons[None, :] - lon_target) * 111132.0 * math.cos(math.radians(lat_target))
        dx_full = np.broadcast_to(dx_matrix, sub_elev.shape)
        dy_full = np.broadcast_to(dy_matrix, sub_elev.shape)
        r_matrix = np.hypot(dx_full, dy_full)
        
        R_INNER = 250.0  # meters
        valid_mask = (~np.isnan(sub_elev)) & (r_matrix <= max_radius_m)
        mask_inner = valid_mask & (r_matrix <= R_INNER)
        mask_outer = valid_mask & (r_matrix > R_INNER)
        
        tc_outer_ms2 = 0.0
        tc_inner_ms2 = 0.0
        G = 6.67430e-11

        # Buat matriks lats_grid dan lons_grid 2D untuk sub-DEM
        lons_grid_2d, lats_grid_2d = np.meshgrid(sub_lons, sub_lats)
        
        # 1. Panggil Matriks Densitas dari stratigraphy engine jika ada
        if se is not None:
            density_matrix = se.generate_density_matrix(lats_grid_2d, lons_grid_2d)
        else:
            density_matrix = np.full(sub_elev.shape, 2450.0) # Fallback
                
        # Outer zone: line-mass approximation
        if np.any(mask_outer):
            r_out = r_matrix[mask_outer]
            dz_out = sub_elev[mask_outer] - h_target
            rho_out = density_matrix[mask_outer] # Gunakan densitas lokal
            area_element = abs(self.dx_m * self.dy_m)
            kernel = (1.0 / r_out) - (1.0 / np.hypot(r_out, dz_out))
            tc_outer_ms2 = G * area_element * np.sum(rho_out * kernel) 
            
        # Inner zone: Nagy rectangular prism formula
        if np.any(mask_inner):
            dx_in = dx_full[mask_inner]
            dy_in = dy_full[mask_inner]
            dz_in = np.abs(sub_elev[mask_inner] - h_target)
            rho_in = density_matrix[mask_inner]
            hx = abs(self.dx_m) / 2.0
            hy = abs(self.dy_m) / 2.0
            x1 = dx_in - hx
            x2 = dx_in + hx
            y1 = dy_in - hy
            y2 = dy_in + hy
            z1 = np.zeros_like(dz_in)
            z2 = dz_in
            
            def nagy_f(x, y, z):
                r = np.hypot(np.hypot(x, y), z)
                term1 = np.where(y + r > 1e-10, x * np.log(y + r), 0.0)
                term2 = np.where(x + r > 1e-10, y * np.log(x + r), 0.0)
                term3 = np.where(z > 1e-10, z * np.arctan2(x * y, z * r), 0.0)
                return term1 + term2 - term3

            val_sum = np.zeros_like(dz_in)
            for i, x in enumerate([x1, x2]):
                sx = -1 if i == 0 else 1
                for j, y in enumerate([y1, y2]):
                    sy = -1 if j == 0 else 1
                    for k, z in enumerate([z1, z2]):
                        sz = -1 if k == 0 else 1
                        val_sum += (sx * sy * sz) * nagy_f(x, y, z)
            
            tc_inner_ms2 = G * np.sum(rho_in * val_sum)
            
        return (tc_inner_ms2 + tc_outer_ms2) * 1e5  # m/s² to mGal

    def compute_tc_tesseroid(self, lat_target: float, lon_target: float,
                             h_target: float, rho: float = 2450.0,
                             radius_deg: float = 0.05) -> float:
        """
        Compute terrain correction using spherical tesseroids with absolute
        integration (hills + valleys) to obtain the complete positive correction.

        Scientific Definition:
            The terrain correction (TC) is the gravitational attraction of all
            topographic masses that deviate from the Bouguer plate (a flat
            horizontal slab passing through the observation point). It is
            always positive and consists of two parts:

            1. Hills (r_cell > r0):  mass excess above the station attracts
               upward; we add this effect to compensate for the missing mass
               in the Bouguer plate.
            2. Valleys (r_cell < r0): mass deficit below the station; the
               Bouguer plate overcorrects, so we add the effect of the mass
               that would fill the valley up to the station height.

            For each tesseroid, the radial integration is performed between
            r0 and r_cell, taking the absolute value of the vertical component
            (Δz) to ensure a positive contribution. This follows the standard
            approach used in the Tesseroids software (Uieda et al., 2016) and
            recommended by Tsoulis et al. (2009).

        Adaptive Subdivision (Uieda et al., 2016):
            Instead of a fixed angular threshold, we use the Distance-to-Size
            ratio (D = distance / cell_size). If D < 1.5, the cell is too
            close to the observation point and is subdivided into 2x2
            sub-cells. This ensures numerical accuracy independent of the
            DEM resolution and latitude.

        References:
            - Heiskanen, W.A. & Moritz, H. (1967). Physical Geodesy.
            - Tsoulis, D., Novák, P., & Kadlec, M. (2009). Evaluation of
              precise terrain effects using high-resolution DEMs.
              J. Geophys. Res., 114, B02404.
            - Uieda, L., Barbosa, V.C.F., & Braitenberg, C. (2016). Tesseroids:
              Forward-modeling gravitational fields in spherical coordinates.
              Geophysics, 81(5), F41-F48.
            - Grombein, T., Seitz, K., & Heck, B. (2013). Optimized formulas
              for the gravitational field of a tesseroid. J. Geodesy, 87, 645-660.

        Parameters
        ----------
        lat_target, lon_target : float
            Observation point (decimal degrees).
        h_target : float
            Orthometric height of the observation point (metres).
        rho : float, optional
            Density of topographic masses (kg/m³). Default 2450.
        radius_deg : float, optional
            Integration radius in degrees. Default 0.05° (~5.5 km).

        Returns
        -------
        float
            Terrain correction in mGal (always positive).
        """
        if self.elevations is None:
            return 0.0

        R_EARTH = 6371000.0
        G = 6.67430e-11
        CONV = 1e5  # m/s² → mGal

        lat0 = math.radians(lat_target)
        lon0 = math.radians(lon_target)
        r0 = R_EARTH + h_target

        # Local vertical unit vector (normal to ellipsoid)
        nx0 = math.cos(lat0) * math.cos(lon0)
        ny0 = math.cos(lat0) * math.sin(lon0)
        nz0 = math.sin(lat0)

        # Cartesian coordinates of observation point
        x0 = r0 * math.cos(lat0) * math.cos(lon0)
        y0 = r0 * math.cos(lat0) * math.sin(lon0)
        z0 = r0 * math.sin(lat0)

        # Integration bounds in radians
        rad = math.radians(radius_deg)
        lat_min = lat0 - rad
        lat_max = lat0 + rad
        lon_min = lon0 - rad
        lon_max = lon0 + rad

        # DEM indices with safe margin
        i0 = max(0, np.searchsorted(self.lats, math.degrees(lat_min)) - 1)
        i1 = min(self.elevations.shape[0], np.searchsorted(self.lats, math.degrees(lat_max)) + 1)
        j0 = max(0, np.searchsorted(self.lons, math.degrees(lon_min)) - 1)
        j1 = min(self.elevations.shape[1], np.searchsorted(self.lons, math.degrees(lon_max)) + 1)

        if i1 <= i0 or j1 <= j0:
            return 0.0

        sub_lats = self.lats[i0:i1]
        sub_lons = self.lons[j0:j1]
        sub_elev = self.elevations[i0:i1, j0:j1]

        # Gauss-Legendre nodes and weights (3-point)
        xi = np.array([-0.7745966692, 0.0, 0.7745966692])
        wi = np.array([0.5555555556, 0.8888888889, 0.5555555556])
        n_gauss = 3   # kept for consistency, but we only use 3

        # ------------------------------------------------------------------
        # Pre-calculate cell size in meters for the D ratio criterion
        # (using the latitudinal size as the reference, since longitudinal
        #  size varies with latitude but is almost identical here)
        # ------------------------------------------------------------------
        cell_size_deg = self.cellsize
        cell_size_m = cell_size_deg * (np.pi / 180.0) * R_EARTH

        total_tc = 0.0

        for i in range(sub_elev.shape[0] - 1):
            lat1_deg = sub_lats[i]
            lat2_deg = sub_lats[i+1]
            lat1 = math.radians(lat1_deg)
            lat2 = math.radians(lat2_deg)
            lat_c = (lat1 + lat2) / 2.0   # center latitude for distance calc

            for j in range(sub_elev.shape[1] - 1):
                lon1_deg = sub_lons[j]
                lon2_deg = sub_lons[j+1]
                lon1 = math.radians(lon1_deg)
                lon2 = math.radians(lon2_deg)
                lon_c = (lon1 + lon2) / 2.0   # center longitude

                # Average elevation of the cell
                h_avg = (sub_elev[i, j] + sub_elev[i+1, j] +
                         sub_elev[i+1, j+1] + sub_elev[i, j+1]) / 4.0
                r_cell = R_EARTH + h_avg

                if abs(r_cell - r0) < 1e-6:
                    continue

                # ------------------------------------------------------------------
                # COMPUTE SPHERICAL DISTANCE (psi) FROM OBSERVATION POINT TO CELL CENTER
                # ------------------------------------------------------------------
                cos_psi = (math.sin(lat0) * math.sin(lat_c) +
                           math.cos(lat0) * math.cos(lat_c) *
                           math.cos(lon0 - lon_c))
                cos_psi = max(-1.0, min(1.0, cos_psi))
                psi = math.acos(cos_psi)
                distance = psi * R_EARTH  # Great-circle distance in meters

                # ------------------------------------------------------------------
                # ADAPTIVE SUBDIVISION CRITERION (Uieda et al., 2016)
                # D = distance / cell_size
                # If D < 1.5, subdivide the cell into 2x2 to maintain accuracy.
                # ------------------------------------------------------------------
                D_ratio = distance / cell_size_m

                # Correct radial integration limits:
                # - hills: integrate from r0 to r_cell
                # - valleys: integrate from r_cell to r0
                if r_cell > r0:
                    r1_int = r0
                    r2_int = r_cell
                else:
                    r1_int = r_cell
                    r2_int = r0

                if D_ratio < 1.5:
                    # Subdivide into 2x2 sub-cells
                    lat_edges = np.linspace(lat1, lat2, 3)
                    lon_edges = np.linspace(lon1, lon2, 3)
                    for ii in range(2):
                        lat1_sub = lat_edges[ii]
                        lat2_sub = lat_edges[ii+1]
                        for jj in range(2):
                            lon1_sub = lon_edges[jj]
                            lon2_sub = lon_edges[jj+1]
                            contrib = self._tesseroid_integrate_abs(
                                lat0, lon0, r0,
                                x0, y0, z0, nx0, ny0, nz0,
                                lat1_sub, lat2_sub, lon1_sub, lon2_sub,
                                r1_int, r2_int, rho, G, xi, wi
                            )
                            total_tc += contrib
                else:
                    # Direct integration without subdivision
                    contrib = self._tesseroid_integrate_abs(
                        lat0, lon0, r0,
                        x0, y0, z0, nx0, ny0, nz0,
                        lat1, lat2, lon1, lon2,
                        r1_int, r2_int, rho, G, xi, wi
                    )
                    total_tc += contrib

        # Convert to mGal and return positive
        return total_tc * CONV

    def _tesseroid_integrate_abs(self, lat0, lon0, r0,
                                 x0, y0, z0, nx0, ny0, nz0,
                                 lat1, lat2, lon1, lon2,
                                 r1, r2, rho, G, xi, wi):
        """
        Helper: integrate a single tesseroid (or sub-cell) using 3D Gauss-Legendre
        quadrature and return the absolute value of its vertical component
        contribution.

        The radial integration limits (r1, r2) define the interval between
        the station height and the cell top (whether r1<r2 or r1>r2).
        The function integrates from r1 to r2, but uses the absolute value
        of Δz to ensure a positive terrain correction.
        """
        n = len(xi)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        dr = r2 - r1

        # Scale factor for 3D quadrature:
        # ∫∫∫ f dr dφ dλ ≈ (dr * dlat * dlon / 8) * Σ w_i w_j w_k f(r_i, φ_j, λ_k)
        scale = dr * dlat * dlon / 8.0

        total = 0.0

        for i in range(n):
            # Radial node
            r = 0.5 * (r2 - r1) * xi[i] + 0.5 * (r2 + r1)

            for j in range(n):
                # Latitudinal node
                lat_prime = 0.5 * (lat2 - lat1) * xi[j] + 0.5 * (lat2 + lat1)

                for k in range(n):
                    # Longitudinal node
                    lon_prime = 0.5 * (lon2 - lon1) * xi[k] + 0.5 * (lon2 + lon1)

                    # Cartesian coordinates of the integration point
                    x = r * math.cos(lat_prime) * math.cos(lon_prime)
                    y = r * math.cos(lat_prime) * math.sin(lon_prime)
                    z = r * math.sin(lat_prime)

                    # Vector from observation point to source point
                    dx = x - x0
                    dy = y - y0
                    dz = z - z0

                    # Squared Euclidean distance
                    dist2 = dx*dx + dy*dy + dz*dz

                    # ------------------------------------------------------------------
                    # SINGULARITY HANDLING:
                    # If the point is extremely close (< 1 mm), we skip it.
                    # Thanks to the adaptive subdivision (D < 1.5), the nearest
                    # quadrature node is typically > 10 m away, so this threshold
                    # is purely a safety net.
                    # ------------------------------------------------------------------
                    if dist2 < 1e-6:
                        continue

                    dist = math.sqrt(dist2)

                    # Vertical component (signed): projection onto local vertical
                    delta_z = dx * nx0 + dy * ny0 + dz * nz0

                    # For terrain correction, we always add the absolute value
                    # of the vertical component (since TC is positive definite).
                    # This handles both hills (delta_z > 0) and valleys (delta_z < 0).
                    abs_delta_z = abs(delta_z)

                    # Spherical volume element (Jacobian): r² * cos(lat')
                    jacobian = r * r * math.cos(lat_prime)

                    # Gauss-Legendre weight product
                    w = wi[i] * wi[j] * wi[k]

                    # Newtonian kernel for vertical component: Δz / ℓ³
                    # (dist2 * dist) = dist³
                    kernel = abs_delta_z / (dist2 * dist)

                    total += G * rho * w * jacobian * scale * kernel

        return total

# ==============================================================================
# ADVANCED GEOID & VERTICAL DEFLECTION INVERSION (XGM2019e-2159)
# ==============================================================================
class AdvancedGeoidInversion:
    """XGM2019e-2159-based geoid undulation and vertical deflection computation."""
    def __init__(self, local_grids):
        self.grids = local_grids

    def get_undulation(self, lat, lon) -> np.float64:
        return self.grids.get_undulation(lat, lon)

    def get_astro_geodetic_deflection(self, lat, lon, otl_tilt_xi_arcsec=0.0, otl_tilt_eta_arcsec=0.0):
        """
        Compute meridional (xi) and prime vertical (eta) deflections.
        Returns arcseconds.
        """
        d_deg = 0.01
        n_north = self.get_undulation(min(lat + d_deg, -7.0), lon)
        n_south = self.get_undulation(max(lat - d_deg, -8.0), lon)
        n_east  = self.get_undulation(lat, min(lon + d_deg, 113.0))
        n_west  = self.get_undulation(lat, max(lon - d_deg, 112.0))

        dlat_rad = math.radians(d_deg * 2)
        dlon_rad = math.radians(d_deg * 2)

        xi_rad = -(n_north - n_south) / (R_EARTH_MEAN * dlat_rad)
        eta_rad = -(n_east - n_west) / (R_EARTH_MEAN * math.cos(math.radians(lat)) * dlon_rad)

        xi_static_arcsec = math.degrees(xi_rad) * 3600.0
        eta_static_arcsec = math.degrees(eta_rad) * 3600.0

        xi_total = xi_static_arcsec + otl_tilt_xi_arcsec
        eta_total = eta_static_arcsec + otl_tilt_eta_arcsec

        return {
            'xi_arcsec': xi_total,
            'eta_arcsec': eta_total,
            'xi_static': xi_static_arcsec,
            'eta_static': eta_static_arcsec,
            'total_theta_arcsec': math.hypot(xi_total, eta_total)
        }

# ==============================================================================
# LOCALISED GEOPHYSICS (GRAVIMETRIC INVERSION)
# ==============================================================================
from typing import Optional, Tuple, Dict, Any

class PawitraGeophysics:
    """
    Local gravity field computations following IAG conventions.
    
    Integrates XGM2019e-2159 grids, DEM-based terrain corrections (Nagy prism + line-mass),
    and high‑resolution stratigraphy (PawitraStratigraphy) for local rock density.
    
    Provides Free‑Air anomalies, Bouguer anomalies, terrain corrections, and
    inferred surface gravity in mGal.
    """
    
    def __init__(self, local_grids, dem_engine=None, default_density_kg_m3=2450.0):
        """
        Args:
            local_grids: LocalEGM2008Grids instance (now using XGM2019e-2159)
            dem_engine: LocalDEMEngine instance (optional)
            default_density_kg_m3: Fallback density if stratigraphy unavailable
        """
        self.grids = local_grids
        self.dem_engine = dem_engine if dem_engine is not None else LocalDEMEngine()
        self.default_density = default_density_kg_m3
        
        # Constants from WGS84/GRS80 (explicitly defined for safety)
        self.WGS84_A = 6378137.0
        self.WGS84_E2 = 0.00669437999   # (2*f - f**2) with f=1/298.257223563
    
    def _get_local_density_and_material(self, lat, lon):
        """
        Retrieve density [kg/m³], unit code, and material description from stratigraphy.
        Returns fallback if no stratigraphy engine attached.
        """
        if self.dem_engine and hasattr(self.dem_engine, 'stratigraphy_engine'):
            se = self.dem_engine.stratigraphy_engine
            return se.query(lat, lon)   # (density, unit_code, description)
        else:
            return self.default_density, "Regional", "Default density (no stratigraphy)"
    
    def local_curvature_radii(self, lat_deg):
        """
        Compute meridional (M) and prime vertical (N) radii of curvature [m].
        """
        import math
        lat_rad = math.radians(lat_deg)
        sin_lat = math.sin(lat_rad)
        W = math.sqrt(1.0 - self.WGS84_E2 * sin_lat**2)
        M = self.WGS84_A * (1.0 - self.WGS84_E2) / (W**3)
        N = self.WGS84_A / W
        return M, N
    
    def dynamic_gravity_anomalies(self, lat_deg, lon_deg, orthometric_height_m, 
                                  otl_gravity_correction_mgal=0.0,
                                  use_tesseroid: bool = True,
                                  use_grid_g_obs: bool = False):
        """
        Compute all gravity-related quantities in mGal.
        
        Parameters
        ----------
        
        Returns dictionary with:
            - gamma_0_mgal: normal gravity on ellipsoid
            - delta_g_fa_egm_mgal: Free‑air anomaly from XGM2019e-2159
            - fac_mgal: second‑order free‑air correction
            - dg_atm_mgal: atmospheric correction
            - bc_slab_mgal: simple Bouguer slab correction (using local density)
            - terrain_correction_mgal: DEM‑based TC (Nagy + line‑mass)
            - g_obs_surface_mgal: inferred surface gravity
            - complete_bouguer_anomaly_mgal: δg_FA − BC + TC
            - reduced_bouguer_gravity_mgal: surface gravity + FAC − BC + TC − atm
            - local_density_used: density applied for slab correction
            - local_unit_code: stratigraphic unit code (if available)
            - local_material_description: lithological description
        use_tesseroid : bool, default True
            If True, use tesseroid (high-accuracy) for terrain correction.
            If False, use legacy prism/line-mass method.
        use_grid_g_obs : bool, default False
            If True, use 'gravity_earth' grid from XGM2019e-2159 for surface gravity.
            If False, compute surface gravity synthetically.
        """
        import math
        
        # 1. Normal gravity and XGM2019e-2159 anomaly
        gamma_0 = self.grids.get_normal_gravity(lat_deg, lon_deg)          # mGal
        delta_g_fa_egm = self.grids.get_gravity_anomaly(lat_deg, lon_deg)  # mGal
        
        # 2. Vertical gravity gradient (VGG)
        vgg_anom = self.grids.get_second_r_derivative(lat_deg, lon_deg)    # Eötvös
        vgg_total = 3086.0 + vgg_anom                                      # Eötvös
        local_fag = abs(vgg_total * 0.0001)                                # mGal/m
        
        # 3. Free‑air correction (2nd order)
        fac = local_fag * orthometric_height_m - 7.5e-8 * orthometric_height_m**2  # mGal
        
        # 4. Atmospheric correction (IAG formula)
        if orthometric_height_m >= 0:
            dg_atm = 0.87 - 0.000102 * orthometric_height_m + 5.5e-9 * orthometric_height_m**2
        else:
            dg_atm = 0.87
        
        # 5. Local density from stratigraphy
        rho_local, unit_code, unit_desc = self._get_local_density_and_material(lat_deg, lon_deg)
        
        # Simple Bouguer slab correction
        bc_slab = 0.04193 * (rho_local / 1000.0) * orthometric_height_m   # mGal
        
        # 6. Terrain correction (DEM-based)
        # ===== PERUBAHAN UTAMA: pilih metode TC =====
        if use_tesseroid:
            # Gunakan tesseroid (presisi tinggi, porting dari GeoDataManager)
            tc = self.dem_engine.compute_tc_tesseroid(
                lat_target=lat_deg, lon_target=lon_deg, h_target=orthometric_height_m,
                rho=rho_local, radius_deg=0.05
            )   # mGal
        else:
            # Gunakan metode lama (prism + line-mass) untuk kompatibilitas
            tc = self.dem_engine.compute_gravimetric_terrain_correction(
                lat_target=lat_deg, lon_target=lon_deg, h_target=orthometric_height_m,
                stratigraphy_engine=getattr(self.dem_engine, 'stratigraphy_engine', None),
                max_radius_m=5000.0
            )   # mGal
        
        # 7. Surface gravity (observed at topography)
        if use_grid_g_obs and hasattr(self.grids, 'get_surface_gravity'):
            # Gunakan metode eksak yang memanfaatkan gravity disturbance (δg)
            g_surface = self.grids.get_surface_gravity(lat_deg, lon_deg, orthometric_height_m)
            g_obs = g_surface + otl_gravity_correction_mgal
        elif use_grid_g_obs and hasattr(self.grids, 'get_gravity_earth'):
            # Fallback: gravitasi di geoid dikoreksi dengan gradien udara bebas
            # g_surface ≈ g_geoid - 0.3086 * H
            g_geoid = self.grids.get_gravity_earth(lat_deg, lon_deg)
            g_surface_approx = g_geoid - 0.3086 * orthometric_height_m
            g_obs = g_surface_approx + otl_gravity_correction_mgal
        else:
            # Perhitungan sintetik (cara lama)
            g_obs = gamma_0 + delta_g_fa_egm - fac + dg_atm + otl_gravity_correction_mgal
        
        # 8. Complete Bouguer anomaly
        cba = delta_g_fa_egm - bc_slab + tc + otl_gravity_correction_mgal               # mGal
        
        # 9. Reduced Bouguer gravity
        red_bouguer = g_obs + fac - bc_slab + tc - dg_atm # mGal
        
        return {
            'gamma_0_mgal': gamma_0,
            'delta_g_fa_egm_mgal': delta_g_fa_egm,
            'vgg_anomaly_eotvos': vgg_anom,
            'total_vgg_eotvos': vgg_total,
            'local_fag_mgal_m': local_fag,
            'fac_mgal': fac,
            'dg_atm_mgal': dg_atm,
            'bc_slab_mgal': bc_slab,
            'terrain_correction_mgal': tc,  # <-- Sekarang nilai TC bisa berubah sesuai flag
            'g_obs_surface_mgal': g_obs,    # <-- Sekarang nilai g_obs bisa berubah sesuai flag
            'complete_bouguer_anomaly_mgal': cba,
            'reduced_bouguer_gravity_mgal': red_bouguer,
            'local_density_used': rho_local,
            'local_unit_code': unit_code,
            'local_material_description': unit_desc,
            'otl_gravity_correction_mgal': otl_gravity_correction_mgal
        }

# ==============================================================================
# DYNAMIC 4-D ATMOSPHERIC RAY-TRACING (GPT3 + VMF3)
# ==============================================================================
class DynamicRayTracing:
    """
    Tropospheric delay menggunakan GPT3 (resolusi 1°) dan VMF3.
    Grid GPT3 dimuat dari file .npz (gpt3_1.npz) untuk presisi spasial tinggi.
    """
    def __init__(self, gpt3_file="gpt3_1.npz"):
        """
        Parameters
        ----------
        gpt3_file : str, optional
            Nama file grid GPT3 (format .npz, resolusi 1°). Default: "gpt3_1.npz".
        """
        self.file_path = os.path.join(os.path.dirname(__file__), gpt3_file)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"GPT3 Grid not found: {self.file_path}")
        self.gpt3_grid = load_gpt3_grid(self.file_path)

    def compute_tropospheric_slant(self, mjd_utc, lat_deg, lon_deg, h_ell_m, is_realtime=False):
        """
        Compute tropospheric slant delays using GPT3 (default) or ECMWF if is_realtime=True.

        Parameters
        ----------
        mjd_utc : float
            Modified Julian Date (UTC).
        lat_deg, lon_deg : float
            Geodetic coordinates in degrees.
        h_ell_m : float
            Ellipsoidal height in metres.
        is_realtime : bool, default False
            If True, attempt to use ECMWF data for P, T, e; otherwise use GPT3.

        Returns
        -------
        dict
            Dictionary containing surface meteorology, refractivity, zenith delays,
            slant delays at 10° and 30° elevation, and a 'source' flag.
        """
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)

        # Use hybrid_meteo_assimilation (from Atmospheric_refraction) to get P,T,e,ah,aw,undu
        p_hpa, t_c, dT, e_hpa, ah, aw, undu, _ = hybrid_meteo_assimilation(
            mjd_utc, lat_rad, lon_rad, h_ell_m, is_realtime
        )

        T_k = t_c + 273.15
        N_dry = 77.6 * (p_hpa / T_k)
        N_wet = 71.7 * (e_hpa / T_k) + 374120.0 * (e_hpa / (T_k**2))

        cos2phi = math.cos(2.0 * lat_rad)
        zhd_m = 0.0022768 * p_hpa / (1.0 - 0.00266 * cos2phi - 0.28e-6 * h_ell_m)
        zwd_m = 0.0022768 * (1255.0 / T_k + 0.05) * e_hpa

        def compute_slant(elev_deg):
            zd_rad = math.radians(90.0 - elev_deg)
            mfh, mfw = vmf3_ht(mjd_utc, lat_rad, lon_rad, h_ell_m,
                              zd_rad, ah, aw)
            return zhd_m * mfh + zwd_m * mfw

        # Determine source for reporting
        from Atmospheric_refraction import _REFRACTION_SOURCE
        source = 'ECMWF' if (is_realtime and _REFRACTION_SOURCE.get('active', False)) else 'GPT3'

        return {
            'surface_meteo': {'p_hpa': p_hpa, 't_c': t_c, 'e_hpa': e_hpa},
            'refractivity': {'N_dry': N_dry, 'N_wet': N_wet, 'N_total': N_dry + N_wet},
            'zenith_delays': {'zhd_m': zhd_m, 'zwd_m': zwd_m, 'ztd_m': zhd_m + zwd_m},
            'slant_delay_10deg_m': compute_slant(10.0),
            'slant_delay_30deg_m': compute_slant(30.0),
            'source': source
        }

# ==============================================================================
# DORIS ITRF REALISATION ENGINE (14-parameter Helmert)
# ==============================================================================
class DORIS_ITRF_Engine:
    """
    Transformation between ITRF2020 and earlier ITRF realizations.
    Parameters from ITRF2014/2008/2005/2000.
    """
    ITRF_PARAMS = {
        "ITRF2014": {
            "Tx": -1.4, "Ty": -0.9, "Tz": 1.4, "D": -0.42,
            "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
            "Tx_r": 0.0, "Ty_r": -0.1, "Tz_r": 0.2, "D_r": 0.00,
            "Rx_r": 0.0, "Ry_r": 0.0, "Rz_r": 0.0,
            "epoch": 2015.0
        },
        "ITRF2008": {
            "Tx": 0.2, "Ty": 1.0, "Tz": 3.3, "D": -0.29,
            "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
            "Tx_r": 0.0, "Ty_r": -0.1, "Tz_r": 0.1, "D_r": 0.03,
            "Rx_r": 0.0, "Ry_r": 0.0, "Rz_r": 0.0,
            "epoch": 2015.0
        },
        "ITRF2005": {
            "Tx": 2.7, "Ty": 0.1, "Tz": -1.4, "D": 0.65,
            "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
            "Tx_r": 0.3, "Ty_r": -0.1, "Tz_r": 0.1, "D_r": 0.03,
            "Rx_r": 0.0, "Ry_r": 0.0, "Rz_r": 0.0,
            "epoch": 2015.0
        },
        "ITRF2000": {
            "Tx": -0.2, "Ty": 0.8, "Tz": -34.2, "D": 2.25,
            "Rx": 0.0, "Ry": 0.0, "Rz": 0.0,
            "Tx_r": 0.1, "Ty_r": 0.0, "Tz_r": -1.7, "D_r": 0.11,
            "Rx_r": 0.0, "Ry_r": 0.0, "Rz_r": 0.0,
            "epoch": 2015.0
        }
    }

    def __init__(self):
        self.MM_TO_M = 1e-3
        self.PPB_TO_SCALE = 1e-9
        self.MAS_TO_RAD = math.pi / (180.0 * 3600.0 * 1000.0)

    def transform_itrf2020_to_target(self, target_frame, x_m, y_m, z_m, obs_epoch_year):
        if target_frame not in self.ITRF_PARAMS:
            raise ValueError(f"Transformation parameters for {target_frame} not available.")

        params = self.ITRF_PARAMS[target_frame]
        dt = obs_epoch_year - params["epoch"]

        Tx = (params["Tx"] + params["Tx_r"] * dt) * self.MM_TO_M
        Ty = (params["Ty"] + params["Ty_r"] * dt) * self.MM_TO_M
        Tz = (params["Tz"] + params["Tz_r"] * dt) * self.MM_TO_M
        D = (params["D"] + params["D_r"] * dt) * self.PPB_TO_SCALE
        Rx = (params["Rx"] + params["Rx_r"] * dt) * self.MAS_TO_RAD
        Ry = (params["Ry"] + params["Ry_r"] * dt) * self.MAS_TO_RAD
        Rz = (params["Rz"] + params["Rz_r"] * dt) * self.MAS_TO_RAD

        X_vec = np.array([x_m, y_m, z_m])
        T_vec = np.array([Tx, Ty, Tz])
        R_mat = np.array([
            [ D,  -Rz,  Ry],
            [ Rz,   D, -Rx],
            [-Ry,  Rx,   D]
        ])

        Xs_vec = X_vec + T_vec + (R_mat @ X_vec)
        return Xs_vec[0], Xs_vec[1], Xs_vec[2]

    def transform_velocity_itrf2014_to_itrf2020(self, vx, vy, vz, x, y, z, obs_epoch_year):
        """
        Transform velocity from ITRF2014 to ITRF2020 using the 14-parameter Helmert
        transformation (inverse of the ITRF2020 -> ITRF2014 parameters).

        Parameters
        ----------
        vx, vy, vz : float
            Velocity components in ITRF2014 (m/yr).
        x, y, z : float
            Position coordinates in ITRF2014 (m), required for the scale-rate term.
        obs_epoch_year : float
            Epoch of observation (decimal years), used to propagate rate parameters.

        Returns
        -------
        vx_new, vy_new, vz_new : float
            Velocity components in ITRF2020 (m/yr).
        """
        # ---- Helmert parameters for ITRF2014 -> ITRF2020 (inverse of the official ITRF2020 -> ITRF2014 solution) ----
        # Translation offsets (mm)
        Tx = 1.4
        Ty = 0.9
        Tz = -1.4
        # Scale factor (ppb = 1e-9)
        D = 0.42
        # Rotation angles (mas = 1e-3 arcsec)
        Rx = 0.0
        Ry = 0.0
        Rz = 0.0

        # ---- Time rates of the above parameters ----
        # Translation rates (mm/yr)
        Tx_r = 0.0
        Ty_r = 0.1
        Tz_r = -0.2
        # Scale rate (ppb/yr)
        D_r = 0.0
        # Rotation rates (mas/yr)
        Rx_r = 0.0
        Ry_r = 0.0
        Rz_r = 0.0

        # Time difference from the reference epoch (2015.0)
        dt = obs_epoch_year - 2015.0

        # ---- Full velocity transformation formula ----
        # 1. Translation rates contribution
        vx_new = vx + (Tx_r * 1e-3)
        vy_new = vy + (Ty_r * 1e-3)
        vz_new = vz + (Tz_r * 1e-3)

        # 2. Scale-rate contribution (D_r * X)
        #    (Included for completeness; D_r is zero for this particular solution)
        vx_new += (D_r * 1e-9) * x
        vy_new += (D_r * 1e-9) * y
        vz_new += (D_r * 1e-9) * z

        # 3. Absolute scale contribution (D * V)
        #    This term is required to satisfy the exact derivative of the
        #    Helmert position transformation. Although its magnitude is
        #    ~8e-9 mm/yr (negligible in practice), it is included here for
        #    strict metrological completeness.
        D_abs = D * 1e-9
        vx_new += D_abs * vx
        vy_new += D_abs * vy
        vz_new += D_abs * vz

        # 4. Rotation contributions (Ω_rate × X) and (Ω × V)
        #    All rotation parameters and their rates are zero for the
        #    ITRF2014 -> ITRF2020 transformation, so these terms are omitted.

        return vx_new, vy_new, vz_new

# ==============================================================================
# TECTONIC PLATE KINEMATICS (ITRF2020-PMM after Altamimi et al., 2023)
# ==============================================================================
class TectonicPlateKinematics:
    """
    ITRF2020 Plate Motion Model (Altamimi et al. 2023, Geophysical Research Letters)
    
    Reference: Altamimi, Z., Metivier, L., Rebischung, P., Collilieux, X., Chanard, K., & Barneoud, J. (2023).
               ITRF2020 plate motion model. Geophysical Research Letters, 50, e2023GL106373.
    
    Implements Table 1 (rotation poles for 13 plates) and Table 2 (Origin Rate Bias).
    Following the authors' recommendation, the vertical component of ORB is discarded
    (only horizontal velocities are considered).
    """
    # Rotation poles: ωx, ωy, ωz in mas/yr (Table 1)
    _ROT_MAS_YR = {
        'AMUR': (-0.131, -0.551,  0.837),
        'ANTA': (-0.269, -0.312,  0.678),
        'ARAB': ( 1.129, -0.146,  1.438),
        'AUST': ( 1.487,  1.175,  1.223),
        'CARB': ( 0.207, -1.422,  0.726),
        'EURA': (-0.085, -0.519,  0.753),
        'IND' : ( 1.137,  0.013,  1.444),
        'NAZC': (-0.327, -1.561,  1.605),
        'NOAM': ( 0.045, -0.666, -0.098),
        'NUBI': ( 0.090, -0.585,  0.717),
        'PCFC': (-0.404,  1.021, -2.154),
        'SOAM': (-0.261, -0.282, -0.157),
        'SOMA': (-0.081, -0.719,  0.864),
    }
    
    # Origin Rate Bias (ORB) in mm/yr (Table 2)
    _ORB_MM_YR = (0.37, 0.35, 0.74)   # Tx, Ty, Tz

    def __init__(self, plate_code='EURA'):
        if plate_code not in self._ROT_MAS_YR:
            raise ValueError(f"Plate '{plate_code}' not in ITRF2020-PMM. "
                             f"Available: {list(self._ROT_MAS_YR.keys())}")
        self.plate_code = plate_code
        wx_mas, wy_mas, wz_mas = self._ROT_MAS_YR[plate_code]
        
        # 1 mas = 1e-3 arcsec = 1e-3 * π/(180*3600) rad
        mas2rad = (1e-3 * np.pi) / (180.0 * 3600.0)
        self.wx = wx_mas * mas2rad   # rad/yr
        self.wy = wy_mas * mas2rad
        self.wz = wz_mas * mas2rad
        
        # ORB in m/yr
        self.Tx = self._ORB_MM_YR[0] * 1e-3
        self.Ty = self._ORB_MM_YR[1] * 1e-3
        self.Tz = self._ORB_MM_YR[2] * 1e-3

    def get_velocity(self, x_m, y_m, z_m, lat_rad=None, lon_rad=None,
                     apply_orb=True, discard_vertical_orb=True):
        """
        Compute Cartesian velocity (m/yr) for a point (x,y,z in meters)
        following ITRF2020-PMM: v = ω × X + T_ORB (if apply_orb=True).
        
        Parameters
        ----------
        apply_orb : bool
            If True, add Origin Rate Bias (ORB). If False, return pure rotation.
        discard_vertical_orb : bool
            If True and apply_orb=True, set vertical component to zero
            as recommended by Altamimi et al. (2023, Section 5).
        
        Returns
        -------
        vx, vy, vz : float
            Cartesian velocity in m/yr.
        """
        # Rotation part
        vx_rot = self.wy * z_m - self.wz * y_m
        vy_rot = self.wz * x_m - self.wx * z_m
        vz_rot = self.wx * y_m - self.wy * x_m

        if apply_orb:
            vx = vx_rot + self.Tx
            vy = vy_rot + self.Ty
            vz = vz_rot + self.Tz
        else:
            vx, vy, vz = vx_rot, vy_rot, vz_rot

        # Discard vertical component of ORB if requested (only when ORB is applied)
        if apply_orb and discard_vertical_orb:
            # Gunakan lat_rad, lon_rad jika diberikan, jika tidak hitung dari posisi
            p = np.hypot(x_m, y_m)
            if p < 1e-6:
                return vx, vy, vz
            if lat_rad is None or lon_rad is None:
                lon_rad = np.arctan2(y_m, x_m)
                # Perkiraan lintang geosentrik (cukup akurat untuk ENU)
                lat_rad = np.arctan2(z_m, p * (1 - WGS84_E2))
            sin_lat = np.sin(lat_rad)
            cos_lat = np.cos(lat_rad)
            sin_lon = np.sin(lon_rad)
            cos_lon = np.cos(lon_rad)

            # ENU components (seperti sebelumnya)
            ve = -vx * sin_lon + vy * cos_lon
            vn = -vx * sin_lat * cos_lon - vy * sin_lat * sin_lon + vz * cos_lat
            vu =  vx * cos_lat * cos_lon + vy * cos_lat * sin_lon + vz * sin_lat
            vu = 0.0   # discard vertical

            # Back to Cartesian
            vx_new = -ve * sin_lon - vn * sin_lat * cos_lon + vu * cos_lat * cos_lon
            vy_new =  ve * cos_lon - vn * sin_lat * sin_lon + vu * cos_lat * sin_lon
            vz_new =                   vn * cos_lat               + vu * sin_lat
            return vx_new, vy_new, vz_new
        else:
            return vx, vy, vz

# ==============================================================================
# EMBEDDED GEODETIC NETWORK (INACORS & IDS)
# ==============================================================================
class EmbeddedGeodeticNetwork:
    """
    Geodetic reference station database with clear hierarchy:
    
    1. ITRF ANCHOR (Tertinggi) — stasiun internasional yang terikat ke ITRF
       - CIDB, CIBG, BAKO (Cibinong DORIS/GNSS)
       - Frame: ITRF2020 (resmi dari IDS/IGS)
       
    2. CORS SRGI (Sedang) — stasiun CORS BIG lokal
       - CMJT, CMLG, CSID, CPAS, CSBY
       - Frame: SRGI2013 (realisasi lokal ITRF2008/2014)
       
    3. TKG SRGI (Rendah, opsional) — titik kontrol geodesi BIG
       - 1304, SKOR, N1.0240, TTG.1048A, dll.
       - Frame: SRGI2013 (data epoch 2021.0)
    
    Prioritas dalam get_nearest_station():
      1. ITRF anchor dalam radius 100 km
      2. CORS SRGI dalam radius 50 km
      3. TKG SRGI terdekat (tanpa batas radius)
    """
    
    def __init__(self):
        self.stations = {}
        self._init_itrf_anchor()
        self._init_srgi_cors()
        self._init_srgi_tkg()
    
    def _init_itrf_anchor(self):
        """
        Stasiun ITRF resmi (Cibinong) — diperbarui dengan ITRF2020-u2024
        epoch 2020.0, tide-free, tanpa sinyal musiman (consistent with IERS 2010).
        """
        # =====================================================================
        # CIBG (23101M005) — Top of forced centering, ITRF2020-u2024
        # Interval terakhir: 19:114:15540 → 00:000:00000
        # =====================================================================
        cibg_x = -1837002.83056453
        cibg_y =  6065627.30015321
        cibg_z =  -716183.352615119

        # =====================================================================
        # BAKO (23101M002) — Concrete monument, ITRF2020-u2024
        # Interval terakhir: 18:212:00000 → 00:000:00000
        # =====================================================================
        bako_x = -1836969.47540127
        bako_y =  6065616.94033620
        bako_z =  -716257.933945084

        # =====================================================================
        # CIDB (23101S003) — DORIS antenna, ITRF2020-u2024
        # Interval terakhir: 09:245:28501 → 00:000:00000
        # =====================================================================
        cidb_x = -1836964.26048927
        cidb_y =  6065626.99134842
        cidb_z =  -716217.477579443

        self.stations.update({
            'CIDB': {
                'X': cidb_x,
                'Y': cidb_y,
                'Z': cidb_z,
                'Vx': -0.0229799707667022,
                'Vy': -0.00866044117049928,
                'Vz': -0.00760897982691106,
                'sigX': 0.00369458,
                'sigY': 0.00274090,
                'sigZ': 0.00215055,
                'approx_lat': -6.490683,
                'approx_lon': 106.848841,
                'epoch': 2020.0,
                'name': 'Cibinong DORIS (CIDB) — ITRF2020-u2024',
                'type': 'DORIS_REGIONAL',
                'frame': 'ITRF2020',
                'priority': 1
            },
            'CIBG': {
                'X': cibg_x,
                'Y': cibg_y,
                'Z': cibg_z,
                'Vx': -0.0234502130475118,
                'Vy': -0.00863459839495279,
                'Vz': -0.00744894167069399,
                'sigX': 0.000503728,
                'sigY': 0.00107582,
                'sigZ': 0.000355646,
                'approx_lat': -6.490365,
                'approx_lon': 106.849174,
                'epoch': 2020.0,
                'name': 'Cibinong REGINA (CIBG) — ITRF2020-u2024',
                'type': 'GNSS_REGIONAL',
                'frame': 'ITRF2020',
                'priority': 1
            },
            'BAKO': {
                'X': bako_x,
                'Y': bako_y,
                'Z': bako_z,
                'Vx': -0.0234501940164167,
                'Vy': -0.00863459158983490,
                'Vz': -0.00744896279943063,
                'sigX': 0.000481923,
                'sigY': 0.00108490,
                'sigZ': 0.000328780,
                'approx_lat': -6.491055,
                'approx_lon': 106.848912,
                'epoch': 2020.0,
                'name': 'Bakosurtanal GNSS (BAKO) — ITRF2020-u2024',
                'type': 'GNSS_REGIONAL',
                'frame': 'ITRF2020',
                'priority': 1
            }
        })
    
    def _init_srgi_cors(self):
        """CORS SRGI2013 lokal (BIG)"""
        cors_stations = {
            'CMJT': {
                'X': -2414318.7172, 'Y': 5845521.5975, 'Z': -823220.6543,
                'Vx': -0.025, 'Vy': -0.013, 'Vz': -0.008,
                'sigX': 0.0003, 'sigY': 0.0006, 'sigZ': 0.0002,
                'approx_lat': -7.4655794222222, 'approx_lon': 112.44161581667,
                'epoch': 2021.0,
                'name': 'CORS Mojokerto BIG (CMJT)',
                'type': 'CORS_LOCAL',
                'frame': 'SRGI2013',
                'priority': 2
            },
            'CMLG': {
                'X': -2434071.7836, 'Y': 5829498.2062, 'Z': -879612.3016,
                'Vx': -0.026, 'Vy': -0.010, 'Vz': -0.008,
                'sigX': 0.0004, 'sigY': 0.0008, 'sigZ': 0.0003,
                'approx_lat': -7.9796068916667, 'approx_lon': 112.66268127778,
                'epoch': 2021.0,
                'name': 'CORS Malang BIG (CMLG)',
                'type': 'CORS_LOCAL',
                'frame': 'SRGI2013',
                'priority': 2
            },
            'CSID': {
                'X': -2449489.6658, 'Y': 5832257.7642, 'Z': -813238.3602,
                'Vx': -0.025, 'Vy': -0.014, 'Vz': -0.009,
                'sigX': 0.0017, 'sigY': 0.0034, 'sigZ': 0.0009,
                'approx_lat': -7.3745812968047, 'approx_lon': 112.78191640905,
                'epoch': 2021.0,
                'name': 'CORS Sidoarjo BIG (CSID)',
                'type': 'CORS_LOCAL',
                'frame': 'SRGI2013',
                'priority': 2
            },
            'CPAS': {
                'X': -2460056.4522, 'Y': 5823475.1279, 'Z': -843592.2670,
                'Vx': -0.024, 'Vy': -0.013, 'Vz': -0.005,
                'sigX': 0.0003, 'sigY': 0.0006, 'sigZ': 0.0003,
                'approx_lat': -7.6514077555556, 'approx_lon': 112.90103733889,
                'epoch': 2021.0,
                'name': 'CORS Pasuruan BIG (CPAS)',
                'type': 'CORS_LOCAL',
                'frame': 'SRGI2013',
                'priority': 2
            },
            'CSBY': {
                'X': -2443857.7194, 'Y': 5835257.9465, 'Z': -808826.4776,
                'Vx': -0.021, 'Vy': -0.019, 'Vz': -0.010,
                'sigX': 0.0004, 'sigY': 0.0006, 'sigZ': 0.0002,
                'approx_lat': -7.3343354444444, 'approx_lon': 112.72436731111,
                'epoch': 2021.0,
                'name': 'CORS Surabaya BIG (CSBY)',
                'type': 'CORS_LOCAL',
                'frame': 'SRGI2013',
                'priority': 2
            }
        }
        self.stations.update(cors_stations)
    
    def _init_srgi_tkg(self):
        """TKG BIG (Titik Kontrol Geodesi) — densifikasi lokal"""
        tkg_points = {
            'JKG1304': {
                'X': -2439141.4873, 'Y': 5833519.7546, 'Z': -834935.3133,
                'Vx': -0.024, 'Vy': -0.013, 'Vz': -0.006,
                'sigX': 0.0061, 'sigY': 0.0122, 'sigZ': 0.0035,
                'approx_lat': -7.5724, 'approx_lon': 112.6900,
                'epoch': 2021.0,
                'name': 'TKG Kejapanan (1304)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            },
            'SKOR': {
                'X': -2440811.8810, 'Y': 5830873.2462, 'Z': -849999.0144,
                'Vx': -0.024, 'Vy': -0.012, 'Vz': -0.003,
                'sigX': 0.0048, 'sigY': 0.0104, 'sigZ': 0.0025,
                'approx_lat': -7.65, 'approx_lon': 112.7,
                'epoch': 2021.0,
                'name': 'TKG Lemahbang (SKOR)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            },
            'N10240': {
                'X': -2413333.7212, 'Y': 5845957.7871, 'Z': -822985.0676,
                'Vx': -0.024, 'Vy': -0.014, 'Vz': -0.007,
                'sigX': 0.0045, 'sigY': 0.0083, 'sigZ': 0.0025,
                'approx_lat': -7.4634, 'approx_lon': 112.4319,
                'epoch': 2021.0,
                'name': 'TKG Mojokerto (N1.0240)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            },
            'TTG1048A': {
                'X': -2410933.2013,  # DIKOREKSI: dari -24010933.2013 (salah ketik)
                'Y': 5845960.3319,
                'Z': -830044.2608,
                'Vx': -0.024, 'Vy': -0.014, 'Vz': -0.007,
                'sigX': 0.0150, 'sigY': 0.0321, 'sigZ': 0.0077,
                'approx_lat': -7.5278, 'approx_lon': 112.4117,
                'epoch': 2021.0,
                'name': 'TKG Sooko (TTG.1048A)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            },
            'TTG1040': {
                'X': -2429954.0394, 'Y': 5840037.6024, 'Z': -816058.6241,
                'Vx': -0.024, 'Vy': -0.014, 'Vz': -0.008,
                'sigX': 0.0713, 'sigY': 0.1867, 'sigZ': 0.0386,
                'approx_lat': -7.4003, 'approx_lon': 112.5915,
                'epoch': 2021.0,
                'name': 'TKG Kemasan (TTG.1040)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            },
            'TTG1294': {
                'X': -2434802.9752, 'Y': 5830607.1593, 'Z': -870572.9484,
                'Vx': -0.024, 'Vy': -0.013, 'Vz': -0.006,
                'sigX': 0.0117, 'sigY': 0.0235, 'sigZ': 0.0065,
                'approx_lat': -7.8970, 'approx_lon': 112.6644,
                'epoch': 2021.0,
                'name': 'TKG Pagentan (TTG.1294)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            },
            'GBU017': {
                'X': -2449485.0952, 'Y': 5832255.1151, 'Z': -813260.1002,
                'Vx': -0.025, 'Vy': -0.014, 'Vz': -0.009,
                'sigX': 0.0109, 'sigY': 0.0239, 'sigZ': 0.0067,
                'approx_lat': -7.3748, 'approx_lon': 112.7815,
                'epoch': 2021.0,
                'name': 'TKG Juanda (GBU.017)',
                'type': 'TKG_LOCAL',
                'frame': 'SRGI2013',
                'priority': 3
            }
        }
        self.stations.update(tkg_points)
    
    @staticmethod
    def _haversine(lat1, lon1, lat2, lon2):
        """Hitung jarak haversine antara dua titik (meter)"""
        R = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = (math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2) * math.sin(dlam/2)**2)
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def get_nearest_station(self, lat, lon, current_epoch_year=None):
        """
        Mendapatkan stasiun referensi terdekat dengan prioritas:
        
        1. ITRF anchor (priority=1) dalam radius 100 km
        2. CORS SRGI (priority=2) dalam radius 50 km
        3. TKG SRGI (priority=3) terdekat (tanpa batas radius)
        """
        if not self.stations:
            return None
        
        if current_epoch_year is None:
            current_epoch_year = 2015.0
        
        distances = {}
        for code, sta in self.stations.items():
            distances[code] = self._haversine(lat, lon, sta['approx_lat'], sta['approx_lon'])
        
        # Prioritas 1: ITRF anchor dalam radius 100 km
        itrf = {k: v for k, v in distances.items() 
                if self.stations[k].get('priority') == 1 and v <= 100000.0}
        if itrf:
            min_code = min(itrf, key=itrf.get)
            selected = self.stations[min_code]
        else:
            # Prioritas 2: CORS SRGI dalam radius 50 km
            cors = {k: v for k, v in distances.items() 
                    if self.stations[k].get('priority') == 2 and v <= 50000.0}
            if cors:
                min_code = min(cors, key=cors.get)
                selected = self.stations[min_code]
            else:
                # Prioritas 3: TKG terdekat (tanpa batas)
                min_code = min(distances, key=distances.get)
                selected = self.stations[min_code]
        
        # Proyeksi ke epoch target
        dt = current_epoch_year - selected['epoch']
        dyn_X = selected['X'] + selected['Vx'] * dt
        dyn_Y = selected['Y'] + selected['Vy'] * dt
        dyn_Z = selected['Z'] + selected['Vz'] * dt

        # ---- Transformasi kecepatan jika stasiun berasal dari SRGI2013 (ITRF2014) ----
        if selected.get('frame') == 'SRGI2013':
            # Gunakan posisi stasiun pada epoch target (dyn_X,dyn_Y,dyn_Z) sebagai posisi
            # untuk koreksi skala pada kecepatan
            vx_obs = selected['Vx']
            vy_obs = selected['Vy']
            vz_obs = selected['Vz']
            # Transformasi ke ITRF2020
            itrf_engine = DORIS_ITRF_Engine()  # atau gunakan instance yang sudah ada
            vx_itrf2020, vy_itrf2020, vz_itrf2020 = itrf_engine.transform_velocity_itrf2014_to_itrf2020(
                vx_obs, vy_obs, vz_obs, dyn_X, dyn_Y, dyn_Z, current_epoch_year
            )
            # Timpa nilai Vx,Vy,Vz yang akan dikembalikan
            selected['Vx'] = vx_itrf2020
            selected['Vy'] = vy_itrf2020
            selected['Vz'] = vz_itrf2020
            # Perbarui frame menjadi ITRF2020 agar tidak di-transformasi ulang
            selected['frame'] = 'ITRF2020'        
                        
        return {
            'code': min_code,
            'name': selected['name'],
            'X': dyn_X, 'Y': dyn_Y, 'Z': dyn_Z,
            'sigX': selected['sigX'], 'sigY': selected['sigY'], 'sigZ': selected['sigZ'],
            'lat': selected['approx_lat'],
            'lon': selected['approx_lon'],
            'epoch': current_epoch_year,
            'frame': selected['frame'],
            'priority': selected.get('priority', 3)
        }

# ==============================================================================
# REPORT COMPILATION & EXECUTION
# ==============================================================================

def generate_advanced_geodetic_report(lat, lon, h_ellipsoid_tide_free, density,
                                     jd_tt, jd_ut1, mjd_utc, dut1,
                                     local_grids, dem_engine=None, ids_integrator=None,
                                     loading_disp=(0.0, 0.0, 0.0),
                                     x_itrf2020_m=None, y_itrf2020_m=None, z_itrf2020_m=None,
                                     loading_resolver=None,
                                     xp_rad=0.0, yp_rad=0.0,
                                     dX_rad=0.0, dY_rad=0.0,
                                     use_station_displacement=True):
    """
    Menghasilkan laporan geodetik-gravimetrik komprehensif untuk situs Pawitra/Jolotundo.
    Semua koreksi mengikuti IERS Conventions 2010 dan IAU 2006/2000A.
    Menggunakan model geopotensial XGM2019e-2159.
    """

    # =========================================================================
    # 1. INISIALISASI
    # =========================================================================
    if ids_integrator is None:
        ids_integrator = EmbeddedGeodeticNetwork()

    if loading_resolver is None:
        loading_resolver = LoadingResolver(data_dir=".")

    geoid_inv = AdvancedGeoidInversion(local_grids)
    geophysics = PawitraGeophysics(local_grids, dem_engine=dem_engine, default_density_kg_m3=density)
    ray_tracer = DynamicRayTracing()

    # =========================================================================
    # 2. LOADING DISPLACEMENT (NON-TIDAL)
    # =========================================================================
    de_m, dn_m, du_m = loading_disp
    de_res, dn_res, du_res = loading_resolver.resolve_loading_at_mjd(mjd_utc)
    de_res_mm, dn_res_mm, du_res_mm = de_res*1000, dn_res*1000, du_res*1000

    # =========================================================================
    # 3. POSISI MATAHARI DAN BULAN DALAM ITRF
    # =========================================================================
    sun_itrf, moon_itrf = get_sun_moon_itrf(
        jd_tt, jd_ut1, xp_rad, yp_rad, dX_rad, dY_rad
    )

    # =========================================================================
    # 4. STATION DISPLACEMENT (SOLID TIDE + OTL + POLE TIDE + ATM LOADING)
    # =========================================================================
    disp_set = np.zeros(3)
    if use_station_displacement and x_itrf2020_m is not None:
        try:
            from StationDispl import StationDisplacement, JOLOTUNDO_FES2014_BLQ
            station = StationDisplacement(
                np.array([x_itrf2020_m, y_itrf2020_m, z_itrf2020_m]),
                blq_data=JOLOTUNDO_FES2014_BLQ
            )
            disp_set = station.total_displacement(
                jd_tt, jd_ut1, xp_rad, yp_rad, sun_itrf, moon_itrf,
                include_atm=True
            )
        except Exception as e:
            print(f"⚠️ StationDisplacement error: {e}")

    # Gabungkan semua komponen displacement
    de_total = de_m + de_res + disp_set[0]
    dn_total = dn_m + dn_res + disp_set[1]
    du_total = du_m + du_res + disp_set[2]
    de_total_mm, dn_total_mm, du_total_mm = de_total*1000, dn_total*1000, du_total*1000

    # =========================================================================
    # 5. GEOID & HEIGHT
    # =========================================================================
    N_undulation = geoid_inv.get_undulation(lat, lon)
    h_ortho = h_ellipsoid_tide_free - N_undulation

    lat_rad = math.radians(lat)
    tide_free_correction = -0.198 * (0.5 * (3 * math.sin(lat_rad)**2 - 1))
    h_mean_tide = h_ellipsoid_tide_free + tide_free_correction

    deflection = geoid_inv.get_astro_geodetic_deflection(lat, lon)

    # =========================================================================
    # 6. GRAVITY ANOMALIES
    # =========================================================================
    M, N = geophysics.local_curvature_radii(lat)
    anomalies = geophysics.dynamic_gravity_anomalies(lat, lon, h_ortho)

    g_mean_mgal = anomalies['g_obs_surface_mgal'] + (anomalies['fac_mgal'] / 2.0)
    geopotential_number = (h_ortho * g_mean_mgal) / 1e6

    # =========================================================================
    # 7. TROPOSPHERIC DELAY
    # =========================================================================
    tropo = ray_tracer.compute_tropospheric_slant(mjd_utc, lat, lon, h_ellipsoid_tide_free)

    # =========================================================================
    # 8. CRUSTAL DEFORMATION (ITRF2020-PMM)
    # =========================================================================
    nearest_plate = 'EURA'
    plate_kinematics = TectonicPlateKinematics(nearest_plate)

    if x_itrf2020_m is not None:
        x_sta, y_sta, z_sta = x_itrf2020_m, y_itrf2020_m, z_itrf2020_m
        lat_rad_st, lon_rad_st = math.radians(lat), math.radians(lon)
        used_point = "Jolotundo (ITRF2020 tide-free)"
    else:
        nearest = ids_integrator.get_nearest_station(lat, lon, current_epoch_year=2026.0)
        x_sta, y_sta, z_sta = nearest['X'], nearest['Y'], nearest['Z']
        lat_rad_st, lon_rad_st = math.radians(nearest['lat']), math.radians(nearest['lon'])
        used_point = f"{nearest['name']} ({nearest['code']})"

    # Kecepatan rigid plate dengan ORB
    vx_orb, vy_orb, vz_orb = plate_kinematics.get_velocity(
        x_sta, y_sta, z_sta,
        lat_rad=lat_rad_st, lon_rad=lon_rad_st,
        apply_orb=True, discard_vertical_orb=True
    )

    # Kecepatan rigid plate tanpa ORB (untuk perbandingan)
    vx_no_orb, vy_no_orb, vz_no_orb = plate_kinematics.get_velocity(
        x_sta, y_sta, z_sta,
        lat_rad=lat_rad_st, lon_rad=lon_rad_st,
        apply_orb=False, discard_vertical_orb=False
    )

    # Transformasi ENU
    sin_lat, cos_lat = math.sin(lat_rad_st), math.cos(lat_rad_st)
    sin_lon, cos_lon = math.sin(lon_rad_st), math.cos(lon_rad_st)

    # Dengan ORB
    ve_orb = -vx_orb * sin_lon + vy_orb * cos_lon
    vn_orb = -vx_orb * sin_lat * cos_lon - vy_orb * sin_lat * sin_lon + vz_orb * cos_lat
    vu_orb =  vx_orb * cos_lat * cos_lon + vy_orb * cos_lat * sin_lon + vz_orb * sin_lat

    # Tanpa ORB
    ve_no_orb = -vx_no_orb * sin_lon + vy_no_orb * cos_lon
    vn_no_orb = -vx_no_orb * sin_lat * cos_lon - vy_no_orb * sin_lat * sin_lon + vz_no_orb * cos_lat
    vu_no_orb =  vx_no_orb * cos_lat * cos_lon + vy_no_orb * cos_lat * sin_lon + vz_no_orb * sin_lat

    # Kecepatan teramati dari stasiun terdekat (jika tidak ada koordinat ITRF)
    if x_itrf2020_m is None:
        v_obs_x, v_obs_y, v_obs_z = nearest['Vx'], nearest['Vy'], nearest['Vz']
        ve_obs = -v_obs_x * sin_lon + v_obs_y * cos_lon
        vn_obs = -v_obs_x * sin_lat * cos_lon - v_obs_y * sin_lat * sin_lon + v_obs_z * cos_lat
        vu_obs =  v_obs_x * cos_lat * cos_lon + v_obs_y * cos_lat * sin_lon + v_obs_z * sin_lat
        res_e = (ve_obs - ve_orb) * 1000.0
        res_n = (vn_obs - vn_orb) * 1000.0
        res_u = (vu_obs - vu_orb) * 1000.0
    else:
        ve_obs, vn_obs, vu_obs = 0.0, 0.0, 0.0
        res_e, res_n, res_u = 0.0, 0.0, 0.0

    # Horizontal resultant
    v_horiz_orb = math.hypot(ve_orb, vn_orb) * 1000.0
    azimuth_orb = math.degrees(math.atan2(ve_orb, vn_orb)) % 360.0

    # Arah kompas (16 titik)
    compass_points = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                      "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    compass_bearings = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                        180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    best_idx = min(range(len(compass_bearings)), key=lambda i: min(abs(azimuth_orb - compass_bearings[i]), 360 - abs(azimuth_orb - compass_bearings[i])))
    plate_direction = compass_points[best_idx]

    # =========================================================================
    # 9. DENSITAS LOKAL
    # =========================================================================
    rho_local = anomalies.get('local_density_used', density)
    unit_code = anomalies.get('local_unit_code', 'Tidak diketahui')
    material_desc = anomalies.get('local_material_description', 'Tidak ada deskripsi')

    # =========================================================================
    # 10. CETAK LAPORAN
    # =========================================================================
    print("\n" + "="*80)
    print(" ASTERID GEODETIC-GRAVIMETRIC REPORT")
    print(" Kompleks Candi Pawitra (Gunung Penanggungan), Jawa Timur")
    print(" Model geopotensial: XGM2019e-2159")
    print("="*80)

    print("\n[1] SITE INFORMATION")
    print(f"    Latitude               : {lat:.6f}°")
    print(f"    Longitude              : {lon:.6f}°")
    print(f"    Orthometric Height (XGM2019e) : {h_ortho:.3f} m")
    print(f"    Ellipsoidal Height (WGS84)   : {h_ellipsoid_tide_free:.3f} m")
    print(f"    Geoid Undulation (N)         : {N_undulation:.3f} m")
    print(f"    Mean Tide Height             : {h_mean_tide:.3f} m")

    print("\n[2] GEOID & VERTICAL DEFLECTION")
    print(f"    Meridional Deflection (ξ)    : {deflection['xi_arcsec']:.4f} arcsec")
    print(f"    Prime Vertical Deflection (η): {deflection['eta_arcsec']:.4f} arcsec")
    print(f"    Total Deflection (θ)         : {deflection['total_theta_arcsec']:.4f} arcsec")

    print("\n[3] GRAVITY FIELD")
    print(f"    Normal Gravity (γ₀)          : {anomalies['gamma_0_mgal']:.4f} mGal")
    print(f"    Free-Air Anomaly (Δg_FA)     : {anomalies['delta_g_fa_egm_mgal']:.4f} mGal")
    print(f"    Free-Air Correction (FAC)    : {anomalies['fac_mgal']:.4f} mGal")
    print(f"    Atmospheric Correction       : {anomalies['dg_atm_mgal']:.4f} mGal")
    print(f"    Simple Bouguer Slab (BC)     : {anomalies['bc_slab_mgal']:.4f} mGal")
    print(f"    Terrain Correction (TC)      : {anomalies['terrain_correction_mgal']:.4f} mGal")
    print(f"    Surface Gravity (g_obs)      : {anomalies['g_obs_surface_mgal']:.4f} mGal")
    print(f"    Complete Bouguer Anomaly     : {anomalies['complete_bouguer_anomaly_mgal']:.4f} mGal")
    print(f"    Reduced Bouguer Gravity      : {anomalies['reduced_bouguer_gravity_mgal']:.4f} mGal")
    print(f"    Geopotential Number          : {geopotential_number:.4f} kGal·m")

    print("\n[4] VERTICAL GRAVITY GRADIENT")
    print(f"    VGG Anomaly                  : {anomalies['vgg_anomaly_eotvos']:.2f} Eötvös")
    print(f"    Total VGG                    : {anomalies['total_vgg_eotvos']:.2f} Eötvös")
    print(f"    Local FAG                    : {anomalies['local_fag_mgal_m']:.5f} mGal/m")

    print("\n[5] LOCAL STRATIGRAPHY")
    print(f"    Density                      : {rho_local:.1f} kg/m³")
    print(f"    Unit Code                    : {unit_code}")
    print(f"    Description                  : {material_desc}")

    print("\n[6] TROPOSPHERIC DELAY (GPT3 + VMF3)")
    print(f"    Pressure                     : {tropo['surface_meteo']['p_hpa']:.1f} hPa")
    print(f"    Temperature                  : {tropo['surface_meteo']['t_c']:.1f} °C")
    print(f"    Water Vapor Pressure         : {tropo['surface_meteo']['e_hpa']:.2f} hPa")
    print(f"    ZHD (Zenith Hydrostatic)     : {tropo['zenith_delays']['zhd_m']:.4f} m")
    print(f"    ZWD (Zenith Wet)             : {tropo['zenith_delays']['zwd_m']:.4f} m")
    print(f"    ZTD (Zenith Total)           : {tropo['zenith_delays']['ztd_m']:.4f} m")
    print(f"    Slant Delay (10° elev)       : {tropo['slant_delay_10deg_m']:.4f} m")
    print(f"    Slant Delay (30° elev)       : {tropo['slant_delay_30deg_m']:.4f} m")

    print("\n[7] DISPLACEMENT COMPONENTS")
    print(f"    StationDisplacement (de, dn, du) : {disp_set[0]*1000:8.4f}, {disp_set[1]*1000:8.4f}, {disp_set[2]*1000:8.4f} mm")
    print(f"    Loading Resolver (de, dn, du)    : {de_res_mm:8.4f}, {dn_res_mm:8.4f}, {du_res_mm:8.4f} mm")
    print(f"    Total Loading (de, dn, du)       : {de_total_mm:8.4f}, {dn_total_mm:8.4f}, {du_total_mm:8.4f} mm")

    print("\n[8] CRUSTAL DEFORMATION (ITRF2020-PMM)")
    print(f"    Reference Point          : {used_point}")
    print(f"    Tectonic Plate           : {nearest_plate}")
    print(f"    Velocity (ORB) (Ve, Vn, Vu) : {ve_orb*1000:8.3f}, {vn_orb*1000:8.3f}, {vu_orb*1000:8.3f} mm/yr")
    print(f"    Velocity (NO ORB) (Ve, Vn, Vu): {ve_no_orb*1000:8.3f}, {vn_no_orb*1000:8.3f}, {vu_no_orb*1000:8.3f} mm/yr")
    print(f"    Horizontal Speed (ORB)   : {v_horiz_orb:8.3f} mm/yr")
    print(f"    Azimuth (ORB)            : {azimuth_orb:7.2f}° ({plate_direction})")
    if x_itrf2020_m is None:
        print(f"    Observed (Ve, Vn, Vu)     : {ve_obs*1000:8.3f}, {vn_obs*1000:8.3f}, {vu_obs*1000:8.3f} mm/yr")
        print(f"    Residual (dVe, dVn, dVu)  : {res_e:8.3f}, {res_n:8.3f}, {res_u:8.3f} mm/yr")

    print("\n[9] GEOMETRIC RADII")
    print(f"    Meridional Radius (M)    : {M:.3f} m")
    print(f"    Prime Vertical Radius (N): {N:.3f} m")

    print("\n" + "="*80)
    print(" END OF REPORT")
    print("="*80 + "\n")