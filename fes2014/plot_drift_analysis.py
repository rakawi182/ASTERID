import matplotlib.pyplot as plt

# Data yang diekstrak dari tabel analisis (Seri VSOP2013 dan ELP/MPP02)
tahun = [-2000.0, -1500.0, -1000.0, -500.0, 0.0, 500.0, 1000.0, 1500.0, 
         1900.0, 2000.0, 2050.0, 2500.0, 3000.0, 3500.0, 4000.0, 4500.0, 
         5000.0, 5500.0, 6000.0]

err_emb = [1109.911, 896.029, 667.029, 502.539, 355.984, 233.876, 117.722, 57.209, 
           43.813, 44.253, 44.721, 65.831, 126.509, 217.778, 379.837, 607.605, 
           931.033, 1298.882, 1639.532]

err_earth = [1108.667, 899.041, 667.448, 501.474, 356.073, 234.088, 117.711, 57.207, 
             43.814, 44.254, 44.721, 65.822, 126.559, 217.991, 379.298, 606.897, 
             933.151, 1300.718, 1634.256]

lon_err = [1.5528, 1.2541, 0.9337, 0.7032, 0.4974, 0.3254, 0.1599, 0.0692, 
           0.0464, 0.0475, 0.0486, 0.0836, 0.1731, 0.3026, 0.5303, 0.8488, 
           1.3002, 1.8125, 2.2858]

radial_err = [12.787, 10.513, 7.656, 5.107, 2.813, 1.201, 0.156, -0.089, 
              -0.016, 0.012, 0.029, 0.097, -0.269, -1.242, -3.106, -5.863, 
              -9.380, -12.925, -15.390]

# Membuat figure dan subplot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# 1. Grafik Error Jarak (EMB dan Earth)
ax1.plot(tahun, err_emb, marker='o', label='Err EMB (km)', color='blue')
ax1.plot(tahun, err_earth, marker='s', label='Err Earth (km)', color='green', linestyle='--')
ax1.set_ylabel('Error Jarak (km)')
ax1.set_title('Error EMB & Earth terhadap Waktu (Tahun)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()
# Memberi garis penanda pada tahun 2000
ax1.axvline(x=2000, color='red', linestyle='-', alpha=0.5) 

# 2. Grafik Longitude Error (Drift)
ax2.plot(tahun, lon_err, marker='^', label='Lon Err (")', color='orange')
ax2.set_ylabel('Error Bujur (arcsec)')
ax2.set_title('Ringkasan Drift / Error Bujur terhadap Waktu')
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()
ax2.axvline(x=2000, color='red', linestyle='-', alpha=0.5)

# 3. Grafik Radial Error
ax3.plot(tahun, radial_err, marker='d', label='Radial Err (km)', color='purple')
ax3.set_xlabel('Tahun')
ax3.set_ylabel('Radial Error (km)')
ax3.set_title('Radial Error terhadap Waktu (Tahun)')
ax3.grid(True, linestyle=':', alpha=0.7)
ax3.legend()
ax3.axvline(x=2000, color='red', linestyle='-', alpha=0.5)

# Menyesuaikan tata letak
plt.tight_layout()

# Menampilkan grafik
plt.show()
