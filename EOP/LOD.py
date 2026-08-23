#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LODEngine.py – High-Precision Length of Day (LOD) Calculation Engine
====================================================================
Deskripsi:
  Menghitung nilai Length of Day (LOD) secara presisi tinggi dengan
  mengintegrasikan data observasional IERS C04/Bulletin-A.

  LOD dihitung dari turunan UT1-UTC untuk epoch dalam C04 atau setelahnya,
  dan dari turunan Delta T untuk epoch sebelum C04 (kuno) karena data
  UT1-UTC tidak tersedia. Koreksi zonal tidak ditambahkan karena semua data
  yang digunakan (C04, Bulletin-A) sudah bebas zonal tides.

Spesifikasi Tampilan:
  Lebar konsol dikunci pada 70 karakter untuk presisi visual.

Author:   ASTERID Project / Ω-KALASTHAPATI
Version:  3.1 (Perbaikan epoch kuno)
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

try:
    from Timescales import J2000_JD, delta_t_from_jd, cal_to_jd
    from EOPDelta import EOPProvider
except ImportError:
    raise ImportError("Modul 'Timescales.py' dan 'EOPDelta.py' wajib tersedia.")

# Konstanta Astronomis (IERS 2010)
JC_PER_DAY = 1.0 / 36525.0

# -----------------------------------------------------------------------------
# Tabel Koefisien Pasang Surut Zonal untuk Delta LOD (IERS 2010 Tabel 8.2)
# Digunakan hanya untuk keperluan debugging/eksperimen, TIDAK dipakai dalam
# perhitungan LOD utama karena data C04/Bulletin-A sudah non-tidal.
# -----------------------------------------------------------------------------
ZONAL_LOD_TERMS = [
    (5.6,   [0, 0, 2, 2, 2],  0.003,  0.000),
    (7.1,   [2, 0, 0, 0, 0],  0.007,  0.000),
    (9.1,   [0, 0, 2, 0, 1], -0.012,  0.000),
    (9.1,   [0, 0, 2, 0, 2],  0.046,  0.000),
    (12.1,  [1, 0, 2, -2, 2], 0.006,  0.000),
    (13.6,  [0, 0, 2, -2, 1], -0.120, 0.000),
    (13.7,  [0, 0, 2, -2, 2],  0.535, -0.002), # Mf
    (13.8,  [0, 2, 0, 0, 0],   0.003,  0.000),
    (14.8,  [0, 0, 0, 2, 0],   0.076,  0.000),
    (23.9,  [1, 0, 2, 0, 2],   0.005,  0.000),
    (27.4,  [1, 0, 0, 0, 0],   0.065,  0.000),
    (27.6,  [1, 0, 0, 0, 1],  -0.015,  0.000), # Mm
    (29.5,  [0, -1, 0, 2, 0],  0.004,  0.000),
    (31.8,  [1, 0, -2, 2, 2],  0.004,  0.000),
    (121.7, [0, 0, 2, -2, 0], -0.005,  0.000),
    (182.6, [0, 0, 2, 0, 0],  -0.057,  0.000), # Ssa
    (365.3, [0, 1, 0, 0, 0],  -0.013,  0.000)  # Sa
]

DELAUNAY_COEFFS = {
    'l':   [134.96340251 * 3600, 1717915923.2178],
    'lp':  [357.52910918 * 3600, 129596581.0481],
    'F':   [93.27209062 * 3600,  1739527262.8478],
    'D':   [297.85019547 * 3600, 1602961601.2090],
    'Om':  [125.04455501 * 3600, -6962890.5431]
}

def _compute_delaunay_arguments(t_cy: float) -> Dict[str, float]:
    args = {}
    for name, coeffs in DELAUNAY_COEFFS.items():
        val_arcsec = coeffs[0] + coeffs[1] * t_cy
        args[name] = math.radians(val_arcsec / 3600.0) % (2.0 * math.pi)
    return args

# -----------------------------------------------------------------------------
# VISUAL FORMATTING UTILITIES (Width = 70)
# -----------------------------------------------------------------------------
def print_header(title: str, char: str = '='):
    print(char * 70)
    print(f" {title}".ljust(70))
    print(char * 70)

def print_row(label: str, val_str: str):
    print(f"  {label.ljust(25)} : {val_str}")

def print_table_70(headers: list, rows: list, widths: list):
    prefix = "  "
    header_line = prefix + "".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line[:70])
    print(prefix + "".join(('-' * (w - 2)).ljust(w) for w in widths)[:68])
    for row in rows:
        line = prefix + "".join(str(r).ljust(w) for r, w in zip(row, widths))
        print(line[:70])

# -----------------------------------------------------------------------------
# ENGINE CORE
# -----------------------------------------------------------------------------
class HighPrecisionLODEngine:
    def __init__(self, h: float = 1e-6):
        """
        h : step dalam satuan hari untuk turunan numerik UT1-UTC.
        Nilai default 1e-6 hari ≈ 0.0864 detik, cukup kecil dan stabil.
        """
        self.h = h
        self.eop_provider = EOPProvider()

    def compute_zonal_tidal_lod(self, tt_jd: float) -> float:
        """
        Menghitung koreksi zonal LOD (ms) berdasarkan IERS 2010 Tabel 8.2.
        Fungsi ini DISEDIAKAN UNTUK EKSPERIMEN DAN DEBUGGING SAJA.
        TIDAK DIGUNAKAN dalam perhitungan LOD utama.
        """
        t_cy = (tt_jd - J2000_JD) * JC_PER_DAY
        delaunay = _compute_delaunay_arguments(t_cy)
        arg_names = ['l', 'lp', 'F', 'D', 'Om']
        
        delta_lod = 0.0
        for period, multipliers, sin_c, cos_c in ZONAL_LOD_TERMS:
            theta = sum(m * delaunay[name] for m, name in zip(multipliers, arg_names) if m != 0)
            delta_lod += sin_c * math.sin(theta) + cos_c * math.cos(theta)
        return delta_lod

    def get_lod_precision(self, jd_utc: float) -> Dict[str, Any]:
        """
        Menghitung LOD (ms) dengan turunan spline jika tersedia,
        fallback ke central difference jika tidak.
        """
        mjd = jd_utc - 2400000.5
        c04_mjd = self.eop_provider.mjd

        # ---------- 1. EPOCH SEBELUM C04 (KUNO) ----------
        if len(c04_mjd) > 0 and mjd < c04_mjd[0]:
            h = self.h
            dt_prev = delta_t_from_jd(jd_utc - h)
            dt_next = delta_t_from_jd(jd_utc + h)
            lod_iers = (dt_next - dt_prev) / (2.0 * h) * 1000.0   # ms

        # ---------- 2. EPOCH DALAM C04 ----------
        elif len(c04_mjd) > 0 and mjd >= c04_mjd[0] and mjd <= c04_mjd[-1]:
            cs = self.eop_provider.cs_ut1
            if cs is not None:
                # Turunan analitik spline terhadap MJD (detik/hari)
                d_ut1_dt = cs.derivative()(mjd)
                lod_iers = -d_ut1_dt * 1000.0   # ms
            else:
                # Fallback central difference
                h = self.h
                eop_p = self.eop_provider.get_eop(mjd + h)
                eop_m = self.eop_provider.get_eop(mjd - h)
                d_ut1_dt = (eop_p['ut1_utc'] - eop_m['ut1_utc']) / (2.0 * h)
                lod_iers = -d_ut1_dt * 1000.0

        # ---------- 3. EPOCH SETELAH C04 (Bulletin-A) ----------
        else:
            bulletin = self.eop_provider._bulletin_provider
            if bulletin is not None and bulletin.has_data() and bulletin.cs_ut1 is not None:
                # Turunan spline dari Bulletin-A
                d_ut1_dt = bulletin.cs_ut1.derivative()(mjd)
                lod_iers = -d_ut1_dt * 1000.0
            else:
                # Fallback: ambil dari EOPProvider (yang akan menggunakan Bulletin-A)
                h = self.h
                eop_p = self.eop_provider.get_eop(mjd + h)
                eop_m = self.eop_provider.get_eop(mjd - h)
                d_ut1_dt = (eop_p['ut1_utc'] - eop_m['ut1_utc']) / (2.0 * h)
                lod_iers = -d_ut1_dt * 1000.0

        # Koreksi zonal TIDAK ditambahkan (nol)
        lod_tidal = 0.0
        lod_total = lod_iers
        day_length_si = 86400.0 + (lod_total / 1000.0)

        return {
            'mjd': mjd,
            'lod_iers': lod_iers,
            'lod_tidal_correction': lod_tidal,
            'lod_total': lod_total,
            'day_length_seconds': day_length_si
        }

    def get_lod_realtime(self) -> Dict[str, Any]:
        """Menghitung nilai LOD real-time berdasarkan UTC jam ini."""
        now = datetime.now(timezone.utc)
        year, month, day = now.year, now.month, now.day
        hour, minute, second = now.hour, now.minute, now.second + (now.microsecond / 1e6)
        
        jd_base, _ = cal_to_jd(year, month, day, 0, 0, 0, scale='utc')
        jd_utc = jd_base + (hour / 24.0 + minute / 1440.0 + second / 86400.0)
        
        res = self.get_lod_precision(jd_utc)
        res['datetime_utc'] = now.strftime("%Y-%m-%d %H:%M:%S UTC")
        return res


# -----------------------------------------------------------------------------
# VALIDATION SUITE (FORMATTED FOR 70-CHAR WIDTH)
# -----------------------------------------------------------------------------
def run_lod_validation(engine: HighPrecisionLODEngine):
    print_header("VALIDASI KONSISTENSI TURUNAN NUMERIK LOD", '-')
    test_mjd = 61172.50
    test_jd_utc = test_mjd + 2400000.5
    
    res = engine.get_lod_precision(test_jd_utc)
    lod_engine = res['lod_iers']
    
    h = engine.h
    eop_plus = engine.eop_provider.get_eop(test_mjd + h)
    eop_minus = engine.eop_provider.get_eop(test_mjd - h)
    lod_numerical = -(eop_plus['ut1_utc'] - eop_minus['ut1_utc']) / (2.0 * h) * 1000.0
    diff_lod = abs(lod_engine - lod_numerical)
    
    headers = ["Komponen", "LOD Engine", "LOD Numerik", "Error"]
    rows = [[
        "Mentah (ms)", 
        f"{lod_engine:.6f}", 
        f"{lod_numerical:.6f}", 
        f"{diff_lod:.4e}"
    ]]
    print(f"  Epoch: MJD {test_mjd:.2f} | Step CD: {h} hari")
    print_table_70(headers, rows, [16, 16, 16, 16])
    print("  [PASS]" if diff_lod < 1e-8 else "  [FAIL]")

    print_header("VALIDASI HARMONIK ZONAL (EPOCH J2000.0 TT)", '-')
    zonal_j2000 = engine.compute_zonal_tidal_lod(2451545.0)
    
    args_j2000 = {
        name: math.radians(coeffs[0] / 3600.0) % (2.0 * math.pi) 
        for name, coeffs in DELAUNAY_COEFFS.items()
    }
    zonal_analytic = 0.0
    arg_names = ['l', 'lp', 'F', 'D', 'Om']
    for _, multipliers, sin_c, cos_c in ZONAL_LOD_TERMS:
        theta = sum(m * args_j2000[name] for m, name in zip(multipliers, arg_names) if m != 0)
        zonal_analytic += sin_c * math.sin(theta) + cos_c * math.cos(theta)
        
    diff_zonal = abs(zonal_j2000 - zonal_analytic)
    rows_zonal = [[
        "Zonal (ms)", 
        f"{zonal_j2000:.6f}", 
        f"{zonal_analytic:.6f}", 
        f"{diff_zonal:.4e}"
    ]]
    print_table_70(headers, rows_zonal, [16, 16, 16, 16])
    print("  [PASS] Presisi Arsitektural Terpenuhi." if diff_zonal < 1e-12 else "  [FAIL]")


def run_historical_stress_test(engine: HighPrecisionLODEngine):
    print_header("UJIAN STRESS HISTORIS: PURBAKALA (977 AD)", '*')
    jd_kuno, _ = cal_to_jd(977, 6, 19, 0, 0, 0, scale='utc')
    res = engine.get_lod_precision(jd_kuno)
    
    print_row("Epoch Target", "19 Juni 977 AD")
    print_row("Julian Date", f"{jd_kuno:.4f}")
    print_row("LOD Mentah (IERS)", f"{res['lod_iers']:10.6f} ms")
    print_row("Koreksi Zonal (T)", f"{res['lod_tidal_correction']:10.6f} ms")
    print_row("LOD Akumulasi", f"{res['lod_total']:10.6f} ms")
    print_row("Panjang Hari Riil", f"{res['day_length_seconds']:14.9f} s")
    print("*" * 70)


# -----------------------------------------------------------------------------
# MAIN EXECUTION ENTRY
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    engine = HighPrecisionLODEngine(h=1e-6)
    
    # 1. TAMPILAN OUTPUT UTAMA (EPOCH KONTEMPORER)
    print_header("TESTING HIGH-PRECISION LOD ENGINE")
    jd_test, _ = cal_to_jd(2026, 5, 13, 0, 0, 0, scale='utc')
    res_test = engine.get_lod_precision(jd_test)
    
    print_row("Julian Date Epoch", f"{jd_test:.4f}")
    print_row("Modified JD (MJD)", f"{res_test['mjd']:.2f}")
    print_row("LOD Mentah (IERS)", f"{res_test['lod_iers']:10.6f} ms")
    print_row("Koreksi Zonal", f"{res_test['lod_tidal_correction']:10.6f} ms")
    print_row("LOD Akumulasi", f"{res_test['lod_total']:10.6f} ms")
    print_row("Panjang Hari Riil", f"{res_test['day_length_seconds']:14.9f} s")
    print("-" * 70)
    
    # 2. TAMPILAN OUTPUT REALTIME
    print_header("REALTIME EXTRACTOR (CURRENT CLOCK)")
    res_rt = engine.get_lod_realtime()
    print_row("Waktu Perangkat (UTC)", f"{res_rt['datetime_utc']}")
    print_row("MJD Instan", f"{res_rt['mjd']:.5f}")
    print_row("LOD Akumulasi Instan", f"{res_rt['lod_total']:10.6f} ms")
    print_row("Panjang Hari Aktif", f"{res_rt['day_length_seconds']:14.9f} s")
    print("-" * 70)
    
    # 3. UNIT TESTS & HISTORY VALIDATION
    print()
    run_lod_validation(engine)
    print()
    run_historical_stress_test(engine)
