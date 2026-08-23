import numpy as np
from datetime import datetime, timedelta

def cek_batas_eop(filename="EOP_20u24_C04_one_file_1962-now.txt"):
    try:
        # Membaca hanya kolom MJD (indeks ke-4 berdasarkan format loadtxt milikmu)
        mjd_data = np.loadtxt(filename, comments='#', usecols=(4,))
        
        mjd_awal = mjd_data[0]
        mjd_akhir = mjd_data[-1]

        # Konversi MJD ke Gregorian
        # Titik nol MJD adalah 17 November 1858 tengah malam
        epoch = datetime(1858, 11, 17)
        tgl_awal = epoch + timedelta(days=mjd_awal)
        tgl_akhir = epoch + timedelta(days=mjd_akhir)

        print("-" * 50)
        print(f"📄 File: {filename}")
        print("-" * 50)
        print(f"Data Pertama : MJD {mjd_awal}  ->  {tgl_awal.strftime('%Y-%m-%d')}")
        print(f"Data Terakhir: MJD {mjd_akhir}  ->  {tgl_akhir.strftime('%Y-%m-%d')}")
        print("-" * 50)
        
    except Exception as e:
        print(f"Gagal membaca file: {e}")

if __name__ == "__main__":
    cek_batas_eop()
