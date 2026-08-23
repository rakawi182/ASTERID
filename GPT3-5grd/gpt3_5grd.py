import numpy as np
import math
import os

# -----------------------------------------------------------------------------
# 1. MEMBACA GRID GPT3_5
# -----------------------------------------------------------------------------
def load_gpt3_grid(filepath):
    """
    Baca file grid gpt3_5_grd.csv dan kembalikan dictionary berisi array
    untuk setiap parameter dalam bentuk asli (sebelum scaling).
    """
    # File memiliki 1 baris komentar '% ...', 1 baris header, lalu data numerik
    # Kolom: lat, lon, kemudian 5 nilai untuk setiap parameter (total 64 kolom)
    data = np.loadtxt(filepath, skiprows=2)
    
    # Ekstrak kolom sesuai posisi (indeks 0‑based)
    # Parameter yang memiliki 5 koefisien musiman
    p_grid    = data[:, 2:7]          # pressure (Pa) – akan dibagi 100 nanti
    T_grid    = data[:, 7:12]         # temperature (K)
    Q_grid    = data[:, 12:17] / 1000.0    # specific humidity (kg/kg)
    dT_grid   = data[:, 17:22] / 1000.0   # temp lapse rate (K/m)
    u_grid    = data[:, 22]           # geoid undulation (m)
    Hs_grid   = data[:, 23]           # orthometric grid height (m)
    ah_grid   = data[:, 24:29] / 1000.0   # hydrostatic mapping coeff (dimensionless)
    aw_grid   = data[:, 29:34] / 1000.0   # wet mapping coeff (dimensionless)
    la_grid   = data[:, 34:39]        # water vapor decrease factor (dimensionless)
    Tm_grid   = data[:, 39:44]        # mean temperature (K)
    Gn_h_grid = data[:, 44:49] / 100000.0  # hydro north gradient (m)
    Ge_h_grid = data[:, 49:54] / 100000.0  # hydro east gradient (m)
    Gn_w_grid = data[:, 54:59] / 100000.0  # wet north gradient (m)
    Ge_w_grid = data[:, 59:64] / 100000.0  # wet east gradient (m)
    
    # Koordinat grid (tidak diperlukan untuk interpolasi karena kita gunakan indexing)
    lat_grid = data[:, 0]
    lon_grid = data[:, 1]
    
    grid = {
        'p': p_grid, 'T': T_grid, 'Q': Q_grid, 'dT': dT_grid,
        'undu': u_grid, 'Hs': Hs_grid, 'ah': ah_grid, 'aw': aw_grid,
        'la': la_grid, 'Tm': Tm_grid,
        'Gn_h': Gn_h_grid, 'Ge_h': Ge_h_grid,
        'Gn_w': Gn_w_grid, 'Ge_w': Ge_w_grid,
        'lat': lat_grid, 'lon': lon_grid
    }
    return grid

# -----------------------------------------------------------------------------
# 2. KONVERSI MJD → DOY (seperti di GPT2/GPT3)
# -----------------------------------------------------------------------------
def mjd_to_doy(mjd):
    """Mengubah Modified Julian Date menjadi day‑of‑year (dengan pecahan)."""
    hour = int((mjd % 1) * 24)
    minute = int((((mjd % 1) * 24) - hour) * 60)
    sec = ((((mjd % 1) * 24) - hour) * 60 - minute) * 60

    # koreksi jika detik/menit/jam == 60
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
    # Tahun kabisat
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        leap = 1
    else:
        leap = 0
    doy = sum(days_in_month[:month-1]) + day
    if leap == 1 and month > 2:
        doy += 1
    doy += mjd - int(mjd)  # tambahkan pecahan hari
    return doy

# -----------------------------------------------------------------------------
# 3. INTERPOLASI GPT3_5
# -----------------------------------------------------------------------------
def gpt3_interpolate(mjd, lat_rad, lon_rad, h_ell_m, grid, it=0):
    """
    Menghitung parameter troposfer dari model GPT3_5.
    
    Parameters
    ----------
    mjd : float
        Modified Julian Date.
    lat_rad : float
        Lintang dalam radian (utara positif).
    lon_rad : float
        Bujur dalam radian (timur positif).
    h_ell_m : float
        Tinggi ellipsoidal dalam meter.
    grid : dict
        Hasil dari load_gpt3_grid().
    it : int
        0 = variasi musiman dihitung, 1 = parameter rata‑rata tahunan saja.
    
    Returns
    -------
    dict dengan kunci:
        p (hPa), T (deg C), dT (deg/km), Tm (K), e (hPa),
        ah (dimensionless), aw (dimensionless), la (dimensionless),
        undu (m), Gn_h, Ge_h, Gn_w, Ge_w (m)
    """
    PI = math.pi
    # Konversi mjd ke doy
    doy = mjd_to_doy(mjd)

    # Faktor musiman
    if it == 1:
        cosfy = 0.0; coshy = 0.0; sinfy = 0.0; sinhy = 0.0
    else:
        cosfy = math.cos(doy / 365.25 * 2 * PI)
        coshy = math.cos(doy / 365.25 * 4 * PI)
        sinfy = math.sin(doy / 365.25 * 2 * PI)
        sinhy = math.sin(doy / 365.25 * 4 * PI)

    # Koordinat dalam derajat
    lon_deg = math.degrees(lon_rad)
    if lon_deg < 0:
        lon_deg += 360.0
    lat_deg = math.degrees(lat_rad)
    ppod = (90.0 - lat_deg)   # polar distance dalam derajat
    plon = lon_deg

    # Indeks grid (5°x5°, pusat grid di 2.5, 7.5, ...)
    ipod = int((ppod + 5.0) / 5.0)
    ilon = int((plon + 5.0) / 5.0)
    
    # Koreksi agar indeks tidak keluar
    if ipod == 37:
        ipod = 36
    if ilon == 73:
        ilon = 1
    if ilon == 0:
        ilon = 72

    diffpod = (ppod - (ipod * 5 - 2.5)) / 5.0
    difflon = (plon - (ilon * 5 - 2.5)) / 5.0

    # Bilinear atau nearest?
    bilinear = 0
    if 2.5 < ppod < 177.5:
        bilinear = 1

    # Fungsi bantu untuk menghitung nilai musiman dari 5 koefisien
    def seasonal(coeffs):
        return coeffs[0] + coeffs[1]*cosfy + coeffs[2]*sinfy + coeffs[3]*coshy + coeffs[4]*sinhy

    # Konstanta
    gm = 9.80665
    dMtr = 28.965e-3
    Rg = 8.3143

    if bilinear == 0:
        # Nearest neighbour
        ix = (ipod - 1) * 72 + (ilon - 1)   # indeks baris (0‑based)
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
        p_out = (p0 * math.exp(-c * redh)) / 100.0   # Pascal -> hPa

        ah_val = seasonal(grid['ah'][ix])
        aw_val = seasonal(grid['aw'][ix])
        la_val = seasonal(grid['la'][ix])
        Tm_val = seasonal(grid['Tm'][ix])
        Gn_h_val = seasonal(grid['Gn_h'][ix])
        Ge_h_val = seasonal(grid['Ge_h'][ix])
        Gn_w_val = seasonal(grid['Gn_w'][ix])
        Ge_w_val = seasonal(grid['Ge_w'][ix])

        # Tekanan uap air (menggunakan la)
        e0 = Q * p0 / (0.622 + 0.378 * Q) / 100.0   # pada grid
        e_out = e0 * (100.0 * p_out / (p0/100.0)) ** (la_val + 1.0)   # hati‑hati p0 di sini dalam Pa, p_out dalam hPa -> p0 harus Pa -> gunakan p0 (Pa) / 100 = tekanan di grid dalam hPa = p0/100
        # Rumus dari Fortran: e0 = Q*p0/(0.622+0.378*Q)/100   (di grid, e0 dalam hPa)
        # e = e0 * (100*p(k)/p0)**(la+1)   --> p(k) hasil  p_out (hPa), p0 adalah tekanan grid dalam Pa -> 100*p_out / p0
        e_out = e0 * (100.0 * p_out / p0) ** (la_val + 1.0)

    else:
        # Bilinear interpolation (4 titik)
        ipod1 = ipod + int(math.copysign(1, diffpod))
        ilon1 = ilon + int(math.copysign(1, difflon))
        if ilon1 == 73: ilon1 = 1
        if ilon1 == 0:  ilon1 = 72

        ix1 = (ipod - 1) * 72 + (ilon - 1)
        ix2 = (ipod1 - 1) * 72 + (ilon - 1)
        ix3 = (ipod - 1) * 72 + (ilon1 - 1)
        ix4 = (ipod1 - 1) * 72 + (ilon1 - 1)
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
# Contoh penggunaan & integrasi ke pipeline Anda
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Path ke file grid (letakkan di direktori yang sama dengan script atau sesuaikan)
    grid_file = "gpt3_5_grd.csv"
    if os.path.exists(grid_file):
        gpt3_grid = load_gpt3_grid(grid_file)
        # Contoh pemanggilan untuk Jolotundo (mjd = 60000 misalnya)
        mjd_test = 60000.0
        lat_test = math.radians(-7.909444)
        lon_test = math.radians(112.595556)
        h_ell_test = 583.5   # ellipsoidal height (WGS84)
        result = gpt3_interpolate(mjd_test, lat_test, lon_test, h_ell_test, gpt3_grid, it=0)
        for k, v in result.items():
            print(f"{k:6s}: {v:.6f}")
    else:
        print(f"File {grid_file} tidak ditemukan. Pastikan grid GPT3_5 tersedia.")
