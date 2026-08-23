#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_geoid_models.py - Bandingkan undulasi geoid dari EGM2008, EIGEN-6C4, dan XGM2019e
"""

import sys
import os
import math

def parse_grid_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    head_end = None
    for i, line in enumerate(lines):
        if 'end_of_head' in line:
            head_end = i
            break
    if head_end is None:
        raise ValueError("end_of_head not found")
    data = []
    for line in lines[head_end+1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                lon = float(parts[0])
                lat = float(parts[1])
                val = float(parts[2])
                data.append((lon, lat, val))
            except:
                continue
    return data

def bilinear_interp(data, lon, lat):
    lons = sorted(set(d[0] for d in data))
    lats = sorted(set(d[1] for d in data), reverse=True)
    if lon <= lons[0]:
        i = 0
    elif lon >= lons[-1]:
        i = len(lons)-2
    else:
        i = 0
        while i < len(lons)-1 and lons[i+1] < lon:
            i += 1
    if lat >= lats[0]:
        j = 0
    elif lat <= lats[-1]:
        j = len(lats)-2
    else:
        j = 0
        while j < len(lats)-1 and lats[j+1] > lat:
            j += 1
    lon1, lon2 = lons[i], lons[i+1]
    lat1, lat2 = lats[j], lats[j+1]
    def get_val(lon, lat):
        for d in data:
            if abs(d[0]-lon) < 1e-9 and abs(d[1]-lat) < 1e-9:
                return d[2]
        return float('nan')
    f11 = get_val(lon1, lat1)
    f12 = get_val(lon2, lat1)
    f21 = get_val(lon1, lat2)
    f22 = get_val(lon2, lat2)
    denom = (lon2 - lon1) * (lat1 - lat2)
    if abs(denom) < 1e-12:
        return (f11+f12+f21+f22)/4.0
    w1 = (lon2 - lon) * (lat1 - lat) / denom
    w2 = (lon - lon1) * (lat1 - lat) / denom
    w3 = (lon2 - lon) * (lat - lat2) / denom
    w4 = (lon - lon1) * (lat - lat2) / denom
    return f11*w1 + f21*w2 + f12*w3 + f22*w4

def main():
    # Dua titik: Jolotundo dan Rumah
    points = {
        'Jolotundo': (-7.609444, 112.595556),
        'Rumah': (-7.521951, 112.566089),
    }
    
    files = {
        'EGM2008': 'height_anomaly_ell_EGM2008_Pawitra.txt',
        'EIGEN-6C4': 'height_anomaly_ell_EIGEN-6C4.txt',
        'XGM2019e_2159': 'height_anomaly_ell_XGM2019e_2159.txt'
    }
    
    # Baca semua data
    all_data = {}
    for name, fname in files.items():
        if not os.path.exists(fname):
            print(f"File {fname} tidak ditemukan, lewati.")
            continue
        all_data[name] = parse_grid_file(fname)
    
    if not all_data:
        print("Tidak ada data yang bisa dibaca.")
        return
    
    # Cetak hasil per titik
    for point_name, (lat, lon) in points.items():
        print(f"\n{'='*60}")
        print(f"Titik: {point_name}  ({lat:.6f}, {lon:.6f})")
        print(f"{'='*60}")
        results = {}
        for model, data in all_data.items():
            val = bilinear_interp(data, lon, lat)
            results[model] = val
            print(f"  {model:15s} : {val:8.4f} m")
        # Bandingkan dengan EGM2008
        if 'EGM2008' in results:
            base = results['EGM2008']
            print(f"\n  Selisih vs EGM2008:")
            for model, val in results.items():
                diff = val - base
                print(f"    {model:15s} : {diff:+.4f} m")
    
    # Perbedaan antara Jolotundo dan Rumah
    print(f"\n{'='*60}")
    print("Perbedaan antara Jolotundo dan Rumah")
    print(f"{'='*60}")
    # Ambil nilai Jolotundo dan Rumah
    jolotundo_vals = {}
    rumah_vals = {}
    for model, data in all_data.items():
        jolotundo_vals[model] = bilinear_interp(data, 112.595556, -7.609444)
        rumah_vals[model] = bilinear_interp(data, 112.566089, -7.521951)
    
    for model in all_data.keys():
        diff = jolotundo_vals[model] - rumah_vals[model]
        print(f"  {model:15s} : {diff:+.4f} m  (Jolotundo - Rumah)")

if __name__ == "__main__":
    main()