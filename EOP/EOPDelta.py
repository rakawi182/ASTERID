#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EOPDelta.py – Earth Orientation Parameters Provider
=====================================================

Menyediakan data Earth Orientation Parameters (EOP) dari:
1. IERS C04 series (1962–sekarang) – sumber utama untuk data historis.
2. IERS Bulletin‑A (file bulletina-*.txt) – untuk data terbaru dan prediksi
   setelah batas akhir C04.

Strategi:
  - Sebelum 1962 (MJD < data pertama C04): semua parameter = 0.0
    (mengandalkan ΔT murni dari Timescales.py untuk rotasi Bumi).
  - 1962–2026-05-18 (dalam C04): interpolasi Cubic Spline dari C04.
  - Setelah 2026-05-18: gunakan Bulletin-A (dengan Cubic Spline) jika tersedia,
    fallback clamp ke nilai terakhir C04.

Satuan keluaran:
  - x_pole, y_pole, dX, dY  : miliarcsecond (mas)
  - ut1_utc                  : detik
  - lod                      : detik (offset dari 86400 s)

Dependencies:
  - numpy, scipy (untuk CubicSpline)
  - datetime, re, glob, os (standar library)

Author:   ASTERID Project
Version:  5.0 (LOD dari Bulletin-A via turunan UT1-UTC)
"""

import numpy as np
import os
import glob
import re
from typing import Dict, List, Tuple
from datetime import datetime, timedelta, timezone
from scipy.interpolate import CubicSpline

# =============================================================================
#  EOPBulletinProvider : membaca semua file Bulletin-A dalam direktori
# =============================================================================
class EOPBulletinProvider:
    """
    Provider untuk data Earth Orientation Parameters dari file IERS Bulletin-A.
    Mengekstrak:
      - Rapid Service (observasi) untuk x, y, UT1-UTC
      - Predictions untuk x, y, UT1-UTC
      - IAU2000A Celestial Pole Offset Series untuk dX, dY (nutasi)
    Satuan:
      - x, y  : arcsec → dikonversi ke mas (×1000)
      - UT1-UTC : detik
      - dX, dY : langsung dalam mas (msec of arc)
      - LOD   : dihitung dari turunan UT1-UTC harian (dalam detik)

    Interpolasi menggunakan Cubic Spline (dengan bc_type='natural') untuk
    kurva yang mulus. Fallback ke linear jika data < 4 titik.
    """
    def __init__(self, directory: str = "."):
        self.directory = directory
        self._loaded = False
        self.mjd = np.array([])
        self.x = np.array([])          # mas
        self.y = np.array([])          # mas
        self.ut1_utc = np.array([])     # detik
        self.lod = np.array([])         # detik (offset LOD)
        self.mjd_nut = np.array([])
        self.dX = np.array([])          # mas
        self.dY = np.array([])          # mas

        # Atribut untuk spline
        self.cs_x = None
        self.cs_y = None
        self.cs_ut1 = None
        self.cs_lod = None
        self.cs_dX = None
        self.cs_dY = None

        self._load_all()

    def _load_all(self):
        pattern = os.path.join(self.directory, "bulletina-*.txt")
        files = glob.glob(pattern)
        if not files:
            pattern2 = os.path.join(self.directory, "bulletina*")
            files = glob.glob(pattern2)
        if not files:
            print("[EOPBulletin] Tidak ditemukan file Bulletin-A.")
            return

        files = sorted(files)
        print(f"[EOPBulletin] Ditemukan {len(files)} file: {[os.path.basename(f) for f in files]}")

        raw_eop = []  # (mjd, x_arcsec, y_arcsec, ut1_sec, source, file_idx)
        raw_nut = []  # (mjd, dX_mas, dY_mas)

        for idx, fpath in enumerate(files):
            eop_list, nut_list = self._parse_file(fpath)
            for mjd, x, y, ut1, src in eop_list:
                raw_eop.append((mjd, x, y, ut1, src, idx))
            raw_nut.extend(nut_list)

        print(f"[EOPBulletin] Hasil parse: {len(raw_eop)} data EOP, {len(raw_nut)} data nutasi.")

        if not raw_eop and not raw_nut:
            print("[EOPBulletin] Tidak ada data yang diekstrak.")
            return

        # Gabungkan data EOP dengan prioritas: rapid > prediksi terbaru
        if raw_eop:
            merged = {}
            for mjd, x, y, ut1, src, idx in raw_eop:
                if mjd in merged:
                    existing_src, existing_idx = merged[mjd][3], merged[mjd][4]
                    if existing_src == 'rapid' and src == 'rapid':
                        if idx > existing_idx:
                            merged[mjd] = (x, y, ut1, src, idx)
                    elif existing_src == 'rapid' and src == 'pred':
                        pass
                    elif existing_src == 'pred' and src == 'rapid':
                        merged[mjd] = (x, y, ut1, src, idx)
                    elif existing_src == 'pred' and src == 'pred':
                        if idx > existing_idx:
                            merged[mjd] = (x, y, ut1, src, idx)
                else:
                    merged[mjd] = (x, y, ut1, src, idx)

            sorted_eop = sorted(merged.items())
            self.mjd = np.array([item[0] for item in sorted_eop])
            self.x = np.array([item[1][0] * 1000.0 for item in sorted_eop])
            self.y = np.array([item[1][1] * 1000.0 for item in sorted_eop])
            self.ut1_utc = np.array([item[1][2] for item in sorted_eop])
            print(f"[EOPBulletin] Data EOP: MJD {self.mjd[0]:.1f} – {self.mjd[-1]:.1f}")

            # ---------- HITUNG LOD DARI UT1-UTC HARIAN ----------
            if len(self.mjd) >= 2:
                lod_vals = np.zeros(len(self.mjd))
                for i in range(1, len(self.mjd)):
                    delta_ut1 = self.ut1_utc[i] - self.ut1_utc[i-1]
                    delta_mjd = self.mjd[i] - self.mjd[i-1]
                    if delta_mjd != 0:
                        # LOD offset dalam detik: -ΔUT1/ΔMJD
                        lod_vals[i] = -delta_ut1 / delta_mjd
                    else:
                        lod_vals[i] = 0.0
                # Untuk hari pertama, gunakan nilai hari kedua
                if len(lod_vals) > 1:
                    lod_vals[0] = lod_vals[1]
                self.lod = lod_vals
                if len(self.mjd) >= 4:
                    self.cs_lod = CubicSpline(self.mjd, self.lod, bc_type='natural')
            else:
                self.lod = np.array([])
                self.cs_lod = None

        if raw_nut:
            nut_dict = {}
            for mjd, dx, dy in raw_nut:
                nut_dict[mjd] = (dx, dy)
            sorted_nut = sorted(nut_dict.items())
            self.mjd_nut = np.array([item[0] for item in sorted_nut])
            self.dX = np.array([item[1][0] for item in sorted_nut])
            self.dY = np.array([item[1][1] for item in sorted_nut])
            print(f"[EOPBulletin] Data nutasi: MJD {self.mjd_nut[0]:.1f} – {self.mjd_nut[-1]:.1f}")

        # Buat spline untuk EOP (x, y, ut1) jika data mencukupi
        if len(self.mjd) >= 4:
            self.cs_x = CubicSpline(self.mjd, self.x, bc_type='natural')
            self.cs_y = CubicSpline(self.mjd, self.y, bc_type='natural')
            self.cs_ut1 = CubicSpline(self.mjd, self.ut1_utc, bc_type='natural')
        if len(self.mjd_nut) >= 4:
            self.cs_dX = CubicSpline(self.mjd_nut, self.dX, bc_type='natural')
            self.cs_dY = CubicSpline(self.mjd_nut, self.dY, bc_type='natural')

        self._loaded = True

    def _parse_file(self, filepath: str) -> Tuple[List, List]:
        eop_data = []
        nut_data = []
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"[EOPBulletin] Gagal baca {filepath}: {e}")
            return eop_data, nut_data

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if 'COMBINED EARTH ORIENTATION PARAMETERS:' in line:
                i += 1
                while i < len(lines) and not re.match(r'^\s*\d', lines[i]):
                    i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 10 and re.match(r'^\d+\.?\d*$', parts[3]):
                        try:
                            mjd = float(parts[3])
                            x = float(parts[4])
                            y = float(parts[6])
                            ut1 = float(parts[8])
                            eop_data.append((mjd, x, y, ut1, 'rapid'))
                        except ValueError:
                            break
                        i += 1
                    else:
                        break
                continue

            elif 'PREDICTIONS:' in line:
                i += 1
                while i < len(lines) and not re.match(r'^\s*\d', lines[i]):
                    i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 7 and re.match(r'^\d+\.?\d*$', parts[3]):
                        try:
                            mjd = float(parts[3])
                            x = float(parts[4])
                            y = float(parts[5])
                            ut1 = float(parts[6])
                            eop_data.append((mjd, x, y, ut1, 'pred'))
                        except ValueError:
                            break
                        i += 1
                    else:
                        break
                continue

            elif 'IAU2000A Celestial Pole Offset Series' in line:
                i += 1
                while i < len(lines) and not re.match(r'^\s*\d', lines[i]):
                    i += 1
                while i < len(lines):
                    parts = lines[i].split()
                    if len(parts) >= 5 and re.match(r'^\d+\.?\d*$', parts[0]):
                        try:
                            mjd = float(parts[0])
                            dX = float(parts[1])
                            dY = float(parts[3])
                            nut_data.append((mjd, dX, dY))
                        except ValueError:
                            break
                        i += 1
                    else:
                        break
                continue

            i += 1
        return eop_data, nut_data

    def has_data(self) -> bool:
        return self._loaded and (len(self.mjd) > 0 or len(self.mjd_nut) > 0)

    def get_eop(self, mjd: float) -> Dict[str, float]:
        """
        Mengembalikan parameter EOP terinterpolasi untuk MJD tertentu.
        Menggunakan Cubic Spline jika tersedia (≥4 titik), fallback ke linear.
        Clamp ke nilai ujung jika di luar rentang.
        """
        result = {'x_pole': 0.0, 'y_pole': 0.0, 'ut1_utc': 0.0, 'dX': 0.0, 'dY': 0.0, 'lod': 0.0}

        # EOP (x, y, ut1)
        if self._loaded and len(self.mjd) > 0:
            if mjd <= self.mjd[0]:
                result['x_pole'] = float(self.x[0])
                result['y_pole'] = float(self.y[0])
                result['ut1_utc'] = float(self.ut1_utc[0])
            elif mjd >= self.mjd[-1]:
                result['x_pole'] = float(self.x[-1])
                result['y_pole'] = float(self.y[-1])
                result['ut1_utc'] = float(self.ut1_utc[-1])
            else:
                if self.cs_x is not None:   # spline tersedia
                    result['x_pole'] = float(self.cs_x(mjd))
                    result['y_pole'] = float(self.cs_y(mjd))
                    result['ut1_utc'] = float(self.cs_ut1(mjd))
                else:                       # fallback linear
                    result['x_pole'] = float(np.interp(mjd, self.mjd, self.x))
                    result['y_pole'] = float(np.interp(mjd, self.mjd, self.y))
                    result['ut1_utc'] = float(np.interp(mjd, self.mjd, self.ut1_utc))

        # Nutasi (dX, dY)
        if self._loaded and len(self.mjd_nut) > 0:
            if mjd <= self.mjd_nut[0]:
                result['dX'] = float(self.dX[0])
                result['dY'] = float(self.dY[0])
            elif mjd >= self.mjd_nut[-1]:
                result['dX'] = float(self.dX[-1])
                result['dY'] = float(self.dY[-1])
            else:
                if self.cs_dX is not None:
                    result['dX'] = float(self.cs_dX(mjd))
                    result['dY'] = float(self.cs_dY(mjd))
                else:
                    result['dX'] = float(np.interp(mjd, self.mjd_nut, self.dX))
                    result['dY'] = float(np.interp(mjd, self.mjd_nut, self.dY))

        # LOD (dihitung dari UT1-UTC)
        if self._loaded and len(self.mjd) > 0:
            if mjd <= self.mjd[0]:
                result['lod'] = float(self.lod[0]) if len(self.lod) > 0 else 0.0
            elif mjd >= self.mjd[-1]:
                result['lod'] = float(self.lod[-1]) if len(self.lod) > 0 else 0.0
            else:
                if self.cs_lod is not None:
                    result['lod'] = float(self.cs_lod(mjd))
                else:
                    result['lod'] = float(np.interp(mjd, self.mjd, self.lod)) if len(self.lod) > 0 else 0.0

        return result


# =============================================================================
#  EOPProvider : Singleton dengan deteksi otomatis Bulletin-A
# =============================================================================
class EOPProvider:
    """
    Singleton provider untuk data Earth Orientation Parameters (EOP)
    dari file IERS C04 (format satu kolom per parameter) dan
    IERS Bulletin-A untuk tanggal di luar rentang C04.

    Strategi:
      - Sebelum 1962 (MJD < data pertama C04): semua parameter = 0.0
        (mengandalkan ΔT murni untuk rotasi Bumi).
      - 1962–2026-05-18 (dalam C04): interpolasi Cubic Spline dari C04.
      - Setelah 2026-05-18: gunakan Bulletin-A (dengan spline) jika tersedia,
        fallback clamp ke nilai terakhir C04.

    Penggunaan:
      eop = EOPProvider()
      params = eop.get_eop(mjd)
    """
    _instance = None
    _loaded = False
    _bulletin_provider = None

    # Atribut untuk C04
    mjd = np.array([])
    x_pole = np.array([])
    y_pole = np.array([])
    ut1_utc = np.array([])
    dX = np.array([])
    dY = np.array([])
    lod = np.array([])          
    
    cs_x = None
    cs_y = None
    cs_ut1 = None
    cs_dX = None
    cs_dY = None
    cs_lod = None         

    def __new__(cls, filename="EOP_20u24_C04_one_file_1962-now.txt", bulletin_dir=None):
        if cls._instance is None:
            cls._instance = super(EOPProvider, cls).__new__(cls)
            if not os.path.exists(filename) and os.path.exists("EOP_20u24_C04_one_file_1962-now.txt"):
                filename = "EOP_20u24_C04_one_file_1962-now.txt"
            cls._instance._load_c04(filename)

            # Deteksi otomatis Bulletin-A jika tidak diberikan
            if bulletin_dir is None:
                c04_dir = os.path.dirname(filename)
                if c04_dir == '':
                    c04_dir = '.'
                if glob.glob(os.path.join(c04_dir, "bulletina-*.txt")):
                    bulletin_dir = c04_dir
                elif glob.glob("bulletina-*.txt"):
                    bulletin_dir = "."
                else:
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    if glob.glob(os.path.join(script_dir, "bulletina-*.txt")):
                        bulletin_dir = script_dir

            if bulletin_dir is not None:
                cls._bulletin_provider = EOPBulletinProvider(bulletin_dir)
        return cls._instance

    def _load_c04(self, filename: str):
        """
        Memuat data dari file teks C04.
        Kolom yang digunakan: MJD (indeks 4), x (5), y (6), UT1-UTC (7), dX (8), dY (9).
        Satuan: x,y,dX,dY dalam arcsec → dikonversi ke mas (×1000)
        UT1-UTC tetap dalam detik.
        Setelah dimuat, buat Cubic Spline untuk tiap parameter.
        """
        if os.path.exists(filename):
            try:
                # Kolom yang dibaca: MJD(4), x(5), y(6), UT1-UTC(7), dX(8), dY(9), LOD(12)
                data = np.loadtxt(filename, comments='#', usecols=(4, 5, 6, 7, 8, 9, 12))
                self.mjd = data[:, 0]
                self.x_pole = data[:, 1] * 1000.0
                self.y_pole = data[:, 2] * 1000.0
                self.ut1_utc = data[:, 3]
                self.dX = data[:, 4] * 1000.0
                self.dY = data[:, 5] * 1000.0
                self.lod = data[:, 6]             # Nilai LOD dalam satuan detik (s)

                # Buat spline jika data >= 4 titik
                if len(self.mjd) >= 4:
                    self.cs_x = CubicSpline(self.mjd, self.x_pole, bc_type='natural')
                    self.cs_y = CubicSpline(self.mjd, self.y_pole, bc_type='natural')
                    self.cs_ut1 = CubicSpline(self.mjd, self.ut1_utc, bc_type='natural')
                    self.cs_dX = CubicSpline(self.mjd, self.dX, bc_type='natural')
                    self.cs_dY = CubicSpline(self.mjd, self.dY, bc_type='natural')
                    self.cs_lod = CubicSpline(self.mjd, self.lod, bc_type='natural')
                else:
                    self.cs_x = None

                EOPProvider._loaded = True
                print(f"[EOPDelta] C04 dimuat: {len(self.mjd)} titik dari {filename}")
            except Exception as e:
                print(f"[EOPDelta] Gagal memuat data {filename}: {e}")
                EOPProvider._loaded = False

    def get_eop(self, mjd: float) -> Dict[str, float]:
        """
        Mengembalikan parameter EOP untuk MJD tertentu.

        Args:
            mjd : Modified Julian Date (MJD = JD - 2400000.5)

        Returns:
            Dict dengan kunci: 'x_pole', 'y_pole', 'ut1_utc', 'dX', 'dY', 'lod'
            Semua dalam satuan mas (kecuali ut1_utc dan lod dalam detik).
        """
        # 1. MASA KUNO (sebelum data C04 pertama): Kembalikan 0.0
        if EOPProvider._loaded and mjd < self.mjd[0]:
            return {'x_pole': 0.0, 'y_pole': 0.0, 'ut1_utc': 0.0, 'dX': 0.0, 'dY': 0.0, 'lod': 0.0}

        # 2. MASA MODERN (dalam rentang C04): Interpolasi Cubic Spline
        if EOPProvider._loaded and mjd >= self.mjd[0] and mjd <= self.mjd[-1]:
            if self.cs_x is not None:
                return {
                    'x_pole': float(self.cs_x(mjd)),
                    'y_pole': float(self.cs_y(mjd)),
                    'ut1_utc': float(self.cs_ut1(mjd)),
                    'dX': float(self.cs_dX(mjd)),
                    'dY': float(self.cs_dY(mjd)),
                    'lod': float(self.cs_lod(mjd))
                }
            else:
                return {
                    'x_pole': float(np.interp(mjd, self.mjd, self.x_pole)),
                    'y_pole': float(np.interp(mjd, self.mjd, self.y_pole)),
                    'ut1_utc': float(np.interp(mjd, self.mjd, self.ut1_utc)),
                    'dX': float(np.interp(mjd, self.mjd, self.dX)),
                    'dY': float(np.interp(mjd, self.mjd, self.dY)),
                    'lod': float(np.interp(mjd, self.mjd, self.lod))
                }

        # 3. MASA DEPAN (setelah C04): Gunakan Bulletin-A jika tersedia
        if EOPProvider._bulletin_provider is not None and EOPProvider._bulletin_provider.has_data():
            res = EOPProvider._bulletin_provider.get_eop(mjd)
            # res sudah memiliki 'lod' hasil perhitungan dari Bulletin-A
            return res

        # 4. FALLBACK (tidak ada Bulletin): Clamp ke nilai terakhir C04
        if not EOPProvider._loaded:
            return {'x_pole': 0.0, 'y_pole': 0.0, 'ut1_utc': 0.0, 'dX': 0.0, 'dY': 0.0, 'lod': 0.0}
        return {
            'x_pole': float(self.x_pole[-1]),
            'y_pole': float(self.y_pole[-1]),
            'ut1_utc': float(self.ut1_utc[-1]),
            'dX': float(self.dX[-1]),
            'dY': float(self.dY[-1]),
            'lod': float(self.lod[-1])
        }

    def get_last_available_date(self) -> str:
        """
        Mengembalikan string tanggal kalender dari baris terakhir data C04.
        """
        if not EOPProvider._loaded or len(self.mjd) == 0:
            return "Unknown"
        epoch = datetime(1858, 11, 17)
        return (epoch + timedelta(days=float(self.mjd[-1]))).strftime('%Y-%m-%d')


# =============================================================================
#  Pengujian
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" EOP PROVIDER – DATA MENTAH (tanpa koreksi sub-diurnal)")
    print("=" * 80)
    print("Nilai yang ditampilkan adalah EOP harian rata-rata dari")
    print("C04 (untuk epoch ≤ 2026-05-18) atau Bulletin-A (untuk epoch lebih baru).")
    print("Masa lalu (sebelum 1962) → semua EOP = 0.0 (mengandalkan ΔT murni).")
    print("Interpolasi menggunakan Cubic Spline (natural) untuk semua data.\n")

    eop = EOPProvider()

    # Daftar kasus uji
    test_cases = [
        (58849.0, "2020-01-01 (dalam C04)"),
        (61210.0, "2026-06-19 (hari ini)"),
        (35000.0, "1954-09-15 (sebelum C04)"),
        (61567.0, "2027-06-11 (prediksi terakhir)"),
        (10000.0, "1886-04-04 (kuno)"),
    ]

    # Header tabel dengan tambahan LOD
    print("┌────────────┬─────────────┬────────────┬────────────┬────────────┬────────────┬────────────┐")
    print("│    MJD     │  Tanggal    │  x (mas)   │  y (mas)   │ UT1-UTC (s)│  dX (mas)  │  LOD (s)   │")
    print("├────────────┼─────────────┼────────────┼────────────┼────────────┼────────────┼────────────┤")

    for mjd, desc in test_cases:
        res = eop.get_eop(mjd)
        epoch = datetime(1858, 11, 17)
        date_obj = epoch + timedelta(days=mjd)
        date_str = date_obj.strftime("%Y-%m-%d")

        print(f"│ {mjd:10.1f} │ {date_str} │ {res['x_pole']:10.3f} │ {res['y_pole']:10.3f} │ {res['ut1_utc']:10.6f} │ {res['dX']:10.3f} │ {res['lod']:10.6f} │")

    print("└────────────┴─────────────┴────────────┴────────────┴────────────┴────────────┴────────────┘")

    print(f"\nTanggal terakhir data C04: {eop.get_last_available_date()}")
    print("\nCatatan: Untuk tanggal setelah C04, data diambil dari IERS Bulletin-A.")

    # =========================================================================
    #  EOP REALTIME (saat ini)
    # =========================================================================
    print("\n" + "=" * 80)
    print(" EOP REALTIME (saat ini)")
    print("=" * 80)

    now = datetime.now(timezone.utc)
    epoch = datetime(1858, 11, 17, 0, 0, 0, tzinfo=timezone.utc)
    mjd_now = (now - epoch).total_seconds() / 86400.0
    res_now = eop.get_eop(mjd_now)
    date_now = epoch + timedelta(days=mjd_now)

    print(f"  Waktu UTC  : {date_now.strftime('%Y-%m-%d %H:%M:%S')} (MJD {mjd_now:.3f})")
    print(f"  x_pole     : {res_now['x_pole']:.3f} mas")
    print(f"  y_pole     : {res_now['y_pole']:.3f} mas")
    print(f"  UT1-UTC    : {res_now['ut1_utc']:.6f} s")
    print(f"  dX         : {res_now['dX']:.3f} mas")
    print(f"  dY         : {res_now['dY']:.3f} mas")
    print(f"  LOD        : {res_now['lod']:.6f} s")
    print("=" * 80)

    print("\nUji selesai.")
