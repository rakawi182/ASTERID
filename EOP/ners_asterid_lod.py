#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare LOD with NASA NERS
==========================
Membandingkan LOD dari sistem kita (EOPDelta + LODEngine) dengan nilai
yang diberikan oleh NASA Network Earth Rotation Service (NERS) pada
epoch yang sama.
"""

import sys
sys.dont_write_bytecode = True

from datetime import datetime, timezone
from Timescales import cal_to_jd
from EOPDelta import EOPProvider
from LODEngine import HighPrecisionLODEngine

def compare_with_ners(ners_time_str: str, ners_lod_s: float, ners_ut1_utc_s: float = None):
    """
    Bandingkan LOD sistem kita dengan data NERS.
    
    Parameters:
        ners_time_str : string waktu NERS dalam format "YYYY-MM-DD HH:MM:SS" atau "YYYY.MM.DD-HH:MM:SS"
        ners_lod_s    : nilai LOD dari NERS (dalam detik)
        ners_ut1_utc_s: (opsional) nilai UT1-UTC dari NERS (dalam detik) untuk verifikasi tambahan
    """
    # Parse waktu NERS
    if '-' in ners_time_str and '.' in ners_time_str:
        # Format "2026.08.07-17:48:10"
        date_part, time_part = ners_time_str.split('-')
        year, month, day = map(int, date_part.split('.'))
        hour, minute, second = map(int, time_part.split(':'))
    else:
        # Format "YYYY-MM-DD HH:MM:SS"
        dt = datetime.strptime(ners_time_str, "%Y-%m-%d %H:%M:%S")
        year, month, day = dt.year, dt.month, dt.day
        hour, minute, second = dt.hour, dt.minute, dt.second

    # Hitung JD UTC
    jd_base, _ = cal_to_jd(year, month, day, 0, 0, 0, scale='utc')
    jd_utc = jd_base + (hour / 24.0 + minute / 1440.0 + second / 86400.0)
    mjd = jd_utc - 2400000.5

    # Inisialisasi engine
    engine = HighPrecisionLODEngine(h=1e-6)
    eop = EOPProvider()

    # Dapatkan LOD dari sistem kita
    res = engine.get_lod_precision(jd_utc)
    lod_ours_s = res['lod_total'] / 1000.0  # konversi ms -> detik

    # Dapatkan UT1-UTC dari sistem kita (jika tersedia)
    eop_data = eop.get_eop(mjd)
    ut1_utc_ours = eop_data.get('ut1_utc', None)

    # Tampilkan hasil
    print("=" * 70)
    print(" PERBANDINGAN LOD: SISTEM KITA vs NASA NERS")
    print("=" * 70)
    print(f"  Waktu target    : {ners_time_str} (UTC)")
    print(f"  MJD             : {mjd:.6f}")
    print(f"  JD              : {jd_utc:.6f}")
    print("-" * 70)
    print("  Parameter          Sistem Kita        NASA NERS         Selisih")
    print("-" * 70)
    print(f"  LOD (s)          {lod_ours_s:15.9f}  {ners_lod_s:15.9f}  {lod_ours_s - ners_lod_s:15.9f}")
    if ut1_utc_ours is not None and ners_ut1_utc_s is not None:
        print(f"  UT1-UTC (s)      {ut1_utc_ours:15.9f}  {ners_ut1_utc_s:15.9f}  {ut1_utc_ours - ners_ut1_utc_s:15.9f}")
    print("-" * 70)
    
    # Interpretasi
    diff_lod_ms = (lod_ours_s - ners_lod_s) * 1000.0
    print(f"  Selisih LOD     : {diff_lod_ms:.3f} ms")
    if abs(diff_lod_ms) < 0.1:
        print("  ⚠️  Perbedaan < 0.1 ms (masih dalam toleransi prediksi Bulletin-A)")
    else:
        print("  ⚠️  Perbedaan > 0.1 ms, periksa sumber data atau metode")
    print("=" * 70)

if __name__ == "__main__":
    # Data dari NERS pada 2026-08-07 17:48:10 UTC
    # Dari tabel: lod = -2.228501146D-04 s
    # ut1mtai = -36.9893167 s -> UT1-UTC = ut1mtai + 37 = 0.0106833 s
    compare_with_ners(
        ners_time_str="2026.08.07-17:48:10",
        ners_lod_s=-2.228501146e-4,
        ners_ut1_utc_s=0.0106833   # -36.9893167 + 37
    )