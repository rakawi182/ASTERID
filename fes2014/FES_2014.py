import sys
sys.dont_write_bytecode = True

import numpy as np
import os

# Mencoba mengimpor data stasiun BLQ dari file yang sudah ada
try:
    from BLQ_Jolotundo_fes2014b import blq_data
    # --- UPDATE DATUM FINAL ---
    blq_data['metadata']['station'] = "JOLOTUNDO OBSV"
    blq_data['metadata']['latitude'] = -7.609444
    blq_data['metadata']['longitude'] = 112.595556
    blq_data['metadata']['height'] = 583.355
except ImportError:
    blq_data = None
    print("[Peringatan] File BLQ_Jolotundo_fes2014b.py tidak ditemukan.")

class FES2014Loading:
    def __init__(self, pot_file='data_pot.npz', wh_file='data_wh.npz', love_file='load_Love_numbers_from_Gegout.250.txt'):
        self.pot_file = pot_file
        self.wh_file = wh_file
        self.love_file = love_file
        self.pot_data = None
        self.wh_data = None
        self.love_numbers = {}
        self.station_data = None

    def load_harmonic_coefficients(self):
        if os.path.exists(self.pot_file) and os.path.exists(self.wh_file):
            self.pot_data = np.load(self.pot_file)
            self.wh_data = np.load(self.wh_file)
            print(f"[OK] Harmonik model FES2014 dimuat.")
        else:
            print(f"[Error] File .npz tidak ditemukan.")

    def load_love_numbers(self):
        if not os.path.exists(self.love_file): return
        n_list, h_list, l_list, k_list = [], [], [], []
        with open(self.love_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        n_list.append(int(parts[0]))
                        h_list.append(float(parts[1].replace('D', 'e')))
                        l_list.append(float(parts[2].replace('D', 'e')))
                        k_list.append(float(parts[3].replace('D', 'e')))
                    except ValueError: continue
        self.love_numbers = {'n': np.array(n_list), 'h_n': np.array(h_list), 'l_n': np.array(l_list), 'k_n': np.array(k_list)}
        print(f"[OK] Love Numbers dimuat.")

    def load_station_blq(self, blq_dict):
        self.station_data = blq_dict
        print(f"[OK] Stasiun {self.station_data['metadata']['station']} diintegrasikan.")

    def get_station_summary(self):
        meta = self.station_data['metadata']
        order = meta['description']['constituents_order']
        amp = self.station_data['radial']['amplitudes']
        
        summary = f"\n--- Ringkasan {meta['station']} ---\n"
        summary += f"Posisi: {meta['latitude']}, {meta['longitude']} | Elev: {meta['height']}m\n"
        summary += "Komponen Radial:\n"
        for i, c in enumerate(order):
            summary += f"  - {c}: {amp[i]:.5f} m\n"
        return summary

# ==========================================
# EKSEKUSI
# ==========================================
if __name__ == "__main__":
    loading_model = FES2014Loading()
    loading_model.load_harmonic_coefficients()
    loading_model.load_love_numbers()
    if blq_data:
        loading_model.load_station_blq(blq_data)
        print(loading_model.get_station_summary())
