import numpy as np
import os
from typing import Dict

class EOPProvider:
    """
    Singleton provider untuk data Earth Orientation Parameters (EOP)
    dari file IERS C04 (format satu kolom per parameter).
    Data yang dimuat: MJD, x_pole, y_pole, UT1-UTC, dX, dY.
    
    Untuk epoch di luar rentang data (sebelum 1962 atau setelah data terakhir),
    semua parameter dikembalikan sebagai 0.0 agar tidak merusak matriks rotasi
    pada perhitungan astronomi historis (kuno).
    """
    _instance = None
    _loaded = False

    def __new__(cls, filename="EOP_20u24_C04_one_file_1962-now.txt"):
        if cls._instance is None:
            cls._instance = super(EOPProvider, cls).__new__(cls)
            # Adaptif terhadap nama file default lama
            if not os.path.exists(filename) and os.path.exists("EOP_20u24_C04_one_file_1962-now.txt"):
                filename = "EOP_20u24_C04_one_file_1962-now.txt"
            cls._instance._load_data(filename)
        return cls._instance

    def _load_data(self, filename: str):
        """
        Memuat data dari file teks C04.
        Kolom yang digunakan: MJD (indeks 4), x (5), y (6), UT1-UTC (7), dX (8), dY (9).
        Satuan: x,y,dX,dY dalam arcsec → dikonversi ke mas (×1000) agar sesuai dengan
        modul EarthRotation.py yang mengharapkan mas.
        UT1-UTC tetap dalam detik.
        """
        if os.path.exists(filename):
            try:
                data = np.loadtxt(filename, comments='#', usecols=(4, 5, 6, 7, 8, 9))
                self.mjd = data[:, 0]
                self.x_pole = data[:, 1] * 1000.0   # arcsec → mas
                self.y_pole = data[:, 2] * 1000.0
                self.ut1_utc = data[:, 3]           # detik
                self.dX = data[:, 4] * 1000.0
                self.dY = data[:, 5] * 1000.0
                self.mjd_list = self.mjd             # kompatibilitas
                EOPProvider._loaded = True
            except Exception as e:
                print(f"[EOPDelta] Gagal memuat data {filename}: {e}")
                EOPProvider._loaded = False
        else:
            print(f"[EOPDelta] Peringatan: File {filename} tidak ditemukan.")
            EOPProvider._loaded = False

    def get_eop(self, mjd: float) -> Dict[str, float]:
        """
        Mengembalikan parameter EOP untuk MJD tertentu.
        Masa lalu: 0.0 (mengandalkan Delta T murni).
        Masa depan: Hold/Clamp ke nilai observasi terakhir.
        """
        if not EOPProvider._loaded:
            return {'x_pole': 0.0, 'y_pole': 0.0, 'ut1_utc': 0.0, 'dX': 0.0, 'dY': 0.0}

        # 1. Masa kuno (sebelum 1962): Kembalikan 0.0 secara ketat
        if mjd < self.mjd[0]:
            return {'x_pole': 0.0, 'y_pole': 0.0, 'ut1_utc': 0.0, 'dX': 0.0, 'dY': 0.0}

        # 2. Masa depan (melewati batas akhir file data): Gunakan nilai observasi terakhir
        # Mencegah nilai anjlok tiba-tiba ke 0 saat menguji tanggal saat ini (misal 2026)
        if mjd > self.mjd[-1]:
            return {
                'x_pole': float(self.x_pole[-1]),
                'y_pole': float(self.y_pole[-1]),
                'ut1_utc': float(self.ut1_utc[-1]),
                'dX': float(self.dX[-1]),
                'dY': float(self.dY[-1])
            }

        # 3. Masa modern (dalam rentang data): Interpolasi linier
        return {
            'x_pole': float(np.interp(mjd, self.mjd, self.x_pole)),
            'y_pole': float(np.interp(mjd, self.mjd, self.y_pole)),
            'ut1_utc': float(np.interp(mjd, self.mjd, self.ut1_utc)),
            'dX': float(np.interp(mjd, self.mjd, self.dX)),
            'dY': float(np.interp(mjd, self.mjd, self.dY))
        }

    def get_last_available_date(self) -> str:
        """
        Mengembalikan string tanggal kalender dari baris terakhir data EOP.
        """
        if not EOPProvider._loaded or len(self.mjd) == 0:
            return "Unknown"
        
        last_mjd = float(self.mjd[-1])
        # Konversi MJD ke Gregorian (Epoch MJD: 17 November 1858)
        from datetime import datetime, timedelta
        epoch = datetime(1858, 11, 17)
        last_date = epoch + timedelta(days=last_mjd)
        return last_date.strftime('%Y-%m-%d')

if __name__ == "__main__":
    # Pengujian singkat
    eop = EOPProvider("EOP_20u24_C04_one_file_1962-now.txt")
    print("Testing MJD 61072.0 (2026-02-01):", eop.get_eop(61072.0))
    print("Testing MJD 10000.0 (kuno, di luar data):", eop.get_eop(10000.0))