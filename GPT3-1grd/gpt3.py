import numpy as np
import math
import os

# -----------------------------------------------------------------------------
# 1. MEMBACA GRID GPT3 DARI FORMAT .NPZ (Resolusi Tinggi 1°x1°)
# -----------------------------------------------------------------------------
def load_gpt3_grid(filepath):
    """
    Membaca file grid gpt3_1.npz hasil perbaikan dan mengembalikan 
    dictionary berisi array untuk setiap parameter dalam bentuk asli.
    """
    data_load = np.load(filepath)
    
    # Langsung mengambil matriks 2D tanpa perlu melakukan np.column_stack lagi
    grid = {
        'p':    data_load['p'],
        'T':    data_load['T'],
        'Q':    data_load['Q'] / 1000.0,
        'dT':   data_load['dT'] / 1000.0,
        'undu': data_load['undu'],
        'Hs':   data_load['Hs'],
        'ah':   data_load['ah'] / 1000.0,
        'aw':   data_load['aw'] / 1000.0,
        'la':   data_load['la'],
        'Tm':   data_load['Tm'],
        'Gn_h': data_load['Gn_h'] / 100000.0,
        'Ge_h': data_load['Ge_h'] / 100000.0,
        'Gn_w': data_load['Gn_w'] / 100000.0,
        'Ge_w': data_load['Ge_w'] / 100000.0,
        'lat':  data_load['lat'],
        'lon':  data_load['lon']
    }
    return grid

# -----------------------------------------------------------------------------
# 2. KONVERSI MJD → DOY 
# -----------------------------------------------------------------------------
def mjd_to_doy(mjd):
    """Mengubah Modified Julian Date menjadi day‑of‑year (dengan pecahan)."""
    hour = int((mjd % 1) * 24)
    minute = int((((mjd % 1) * 24) - hour) * 60)
    sec = ((((mjd % 1) * 24) - hour) * 60 - minute) * 60

    if sec >= 60:
        minute += 1
        sec -= 60
    if minute >= 60:
        hour += 1
        minute -= 60

    jd = mjd + 2400000.5
    if hour >= 24:
        jd += 1
        hour = 0

    jd_int = int(jd + 0.5)
    a = jd_int + 32044
    b = int((4 * a + 3) // 146097)
    c = a - int((b * 146097) // 4)
    d = int((4 * c + 3) // 1461)
    e = c - int((1461 * d) // 4)
    m = int((5 * e + 2) // 153)

    day = e - int((153 * m + 2) // 5) + 1
    month = m + 3 - 12 * int(m // 10)
    year = b * 100 + d - 4800 + int(m // 10)

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        leap = 1
    else:
        leap = 0
    doy = sum(days_in_month[:month-1]) + day
    if leap == 1 and month > 2:
        doy += 1
    doy += mjd - int(mjd)  
    return doy

# -----------------------------------------------------------------------------
# 3. INTERPOLASI GPT3 RESOLUSI 1°
# -----------------------------------------------------------------------------
def gpt3_interpolate(mjd, lat_rad, lon_rad, h_ell_m, grid, it=0):
    """
    Menghitung parameter troposfer dari model GPT3 Resolusi Tinggi 1°x1°.
    """
    PI = math.pi
    doy = mjd_to_doy(mjd)

    if it == 1:
        cosfy = 0.0; coshy = 0.0; sinfy = 0.0; sinhy = 0.0
    else:
        cosfy = math.cos(doy / 365.25 * 2 * PI)
        coshy = math.cos(doy / 365.25 * 4 * PI)
        sinfy = math.sin(doy / 365.25 * 2 * PI)
        sinhy = math.sin(doy / 365.25 * 4 * PI)

    lon_deg = math.degrees(lon_rad)
    if lon_deg < 0:
        lon_deg += 360.0
    lat_deg = math.degrees(lat_rad)
    ppod = (90.0 - lat_deg)   # polar distance dalam derajat
    plon = lon_deg

    # Penyesuaian Indeks Grid untuk Resolusi 1° (pusat di 0.5, 1.5, dst.)
    ipod = int((ppod + 1.0) / 1.0)
    ilon = int((plon + 1.0) / 1.0)
    
    if ipod == 181:
        ipod = 180
    if ilon == 361:
        ilon = 1
    if ilon == 0:
        ilon = 360

    diffpod = (ppod - (ipod * 1.0 - 0.5)) / 1.0
    difflon = (plon - (ilon * 1.0 - 0.5)) / 1.0

    # Bilinear jika berada di zona aman interpolasi 1 derajat
    bilinear = 0
    if 0.5 < ppod < 179.5:
        bilinear = 1

    def seasonal(coeffs):
        return coeffs[0] + coeffs[1]*cosfy + coeffs[2]*sinfy + coeffs[3]*coshy + coeffs[4]*sinhy

    gm = 9.80665
    dMtr = 28.965e-3
    Rg = 8.3143

    if bilinear == 0:
        # Nearest neighbour (Ukuran baris sekarang adalah 360 grid bujur)
        ix = (ipod - 1) * 360 + (ilon - 1)   
        undu = float(grid['undu'][ix])
        hgt = h_ell_m - undu

        T0 = seasonal(grid['T'][ix])
        p0 = seasonal(grid['p'][ix])
        Q = seasonal(grid['Q'][ix])
        dT_val = seasonal(grid['dT'][ix])
        redh = hgt - float(grid['Hs'][ix])

        T_out = (T0 + dT_val * redh) - 273.15
        dT_out = dT_val * 1000.0
        Tv = T0 * (1.0 + 0.6077 * Q)
        c = gm * dMtr / (Rg * Tv) if Tv > 0 else 0.0
        p_out = (p0 * math.exp(-c * redh)) / 100.0   

        ah_val = seasonal(grid['ah'][ix])
        aw_val = seasonal(grid['aw'][ix])
        la_val = seasonal(grid['la'][ix])
        Tm_val = seasonal(grid['Tm'][ix])
        Gn_h_val = seasonal(grid['Gn_h'][ix])
        Ge_h_val = seasonal(grid['Ge_h'][ix])
        Gn_w_val = seasonal(grid['Gn_w'][ix])
        Ge_w_val = seasonal(grid['Ge_w'][ix])

        e0 = Q * p0 / (0.622 + 0.378 * Q) / 100.0   
        e_out = e0 * (100.0 * p_out / p0) ** (la_val + 1.0)

    else:
        # Bilinear interpolation (4 titik grid tetangga)
        ipod1 = ipod + int(math.copysign(1, diffpod))
        ilon1 = ilon + int(math.copysign(1, difflon))
        if ilon1 == 361: ilon1 = 1
        if ilon1 == 0:  ilon1 = 360

        ix1 = (ipod - 1) * 360 + (ilon - 1)
        ix2 = (ipod1 - 1) * 360 + (ilon - 1)
        ix3 = (ipod - 1) * 360 + (ilon1 - 1)
        ix4 = (ipod1 - 1) * 360 + (ilon1 - 1)
        idxs = [ix1, ix2, ix3, ix4]

        pl, Tl, dTl, el, ahl, awl, lal, Tml, Gn_hl, Ge_hl, Gn_wl, Ge_wl, undul = [], [], [], [], [], [], [], [], [], [], [], [], []

        for ix in idxs:
            undu_i = float(grid['undu'][ix])
            undul.append(undu_i)
            hgt = h_ell_m - undu_i
            T0 = seasonal(grid['T'][ix])
            p0 = seasonal(grid['p'][ix])
            Q = seasonal(grid['Q'][ix])
            dT_i = seasonal(grid['dT'][ix])
            redh = hgt - float(grid['Hs'][ix])

            T_i = (T0 + dT_i * redh) - 273.15
            Tl.append(T_i)
            dTl.append(dT_i * 1000.0)
            Tv = T0 * (1.0 + 0.6077 * Q)
            c = gm * dMtr / (Rg * Tv) if Tv > 0 else 0.0
            p_i = (p0 * math.exp(-c * redh)) / 100.0
            pl.append(p_i)

            ahl.append(seasonal(grid['ah'][ix]))
            awl.append(seasonal(grid['aw'][ix]))
            lal.append(seasonal(grid['la'][ix]))
            Tml.append(seasonal(grid['Tm'][ix]))
            Gn_hl.append(seasonal(grid['Gn_h'][ix]))
            Ge_hl.append(seasonal(grid['Ge_h'][ix]))
            Gn_wl.append(seasonal(grid['Gn_w'][ix]))
            Ge_wl.append(seasonal(grid['Ge_w'][ix]))

            e0 = Q * p0 / (0.622 + 0.378 * Q) / 100.0
            ei = e0 * (100.0 * p_i / p0) ** (lal[-1] + 1.0)
            el.append(ei)

        dnpod1 = abs(diffpod)
        dnpod2 = 1.0 - dnpod1
        dnlon1 = abs(difflon)
        dnlon2 = 1.0 - dnlon1

        def interp(arr):
            r1 = dnpod2 * arr[0] + dnpod1 * arr[1]
            r2 = dnpod2 * arr[2] + dnpod1 * arr[3]
            return dnlon2 * r1 + dnlon1 * r2

        p_out = interp(pl)
        T_out = interp(Tl)
        dT_out = interp(dTl)
        e_out = interp(el)
        ah_val = interp(ahl)
        aw_val = interp(awl)
        la_val = interp(lal)
        Tm_val = interp(Tml)
        undu = interp(undul)
        Gn_h_val = interp(Gn_hl)
        Ge_h_val = interp(Ge_hl)
        Gn_w_val = interp(Gn_wl)
        Ge_w_val = interp(Ge_wl)

    return {
        'p': p_out,                # hPa
        'T': T_out,                # deg C
        'dT': dT_out,              # deg/km
        'Tm': Tm_val,              # K
        'e': e_out,                # hPa
        'ah': ah_val,              # dimensionless
        'aw': aw_val,              # dimensionless
        'la': la_val,              # dimensionless
        'undu': undu,              # m
        'Gn_h': Gn_h_val,          # m
        'Ge_h': Ge_h_val,          # m
        'Gn_w': Gn_w_val,          # m
        'Ge_w': Ge_w_val           # m
    }

# -----------------------------------------------------------------------------
# Pengujian Blok Eksekusi Utama
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    grid_file = "gpt3_1.npz"
    if os.path.exists(grid_file):
        print("Memuat berkas biner .npz...")
        gpt3_grid = load_gpt3_grid(grid_file)
        print("Selesai memuat grid.")
        
        # Koordinat Pengujian (Jolotundo)
        mjd_test = 60000.0
        lat_test = math.radians(-7.609444444444444)
        lon_test = math.radians(112.59555666666666)
        h_ell_test = 561.0   
        
        result = gpt3_interpolate(mjd_test, lat_test, lon_test, h_ell_test, gpt3_grid, it=0)
        print("\nHasil Interpolasi Resolusi Tinggi 1°:")
        for k, v in result.items():
            print(f"{k:6s}: {v:.6f}")
    else:
        print(f"File {grid_file} tidak ditemukan.")
