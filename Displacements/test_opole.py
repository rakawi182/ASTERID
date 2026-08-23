#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ocean_pole_tide.py
Validate StationDispl.ocean_pole_tide_loading against IERS test file.
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from typing import Tuple, List

# Assume StationDispl.py is in the same directory or PYTHONPATH
try:
    from StationDispl import ocean_pole_tide_loading
except ImportError:
    print("Error: Could not import ocean_pole_tide_loading from StationDispl.py")
    sys.exit(1)

ARCSEC_TO_RAD = math.pi / (180.0 * 3600.0)
GAMMA_R = 0.6870
GAMMA_I = 0.0036
K = 5.3394043696e3  # meters/radian

def parse_test_file(filepath: str):
    """
    Parse opoleloadcmcor.txt and extract test cases.
    Returns lat_rad, lon_rad, lists of m1_rad, m2_rad, exp_du, exp_dn, exp_de.
    """
    lat_deg = -43.75
    lon_deg = 232.25
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    m1_rad_list = []
    m2_rad_list = []
    exp_du = []
    exp_dn = []
    exp_de = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Cari baris header yang mengandung "MJD"
    start_idx = -1
    for i, line in enumerate(lines):
        if "MJD" in line and "xbar_p" in line and "ybar_p" in line:
            start_idx = i + 1
            break

    if start_idx == -1:
        print("Error: Header not found in test file.")
        return lat_rad, lon_rad, [], [], [], [], []

    # Parsing data
    for line in lines[start_idx:]:
        if not line.strip() or line.startswith("------"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        try:
            # Cek apakah kolom pertama adalah angka (MJDN)
            float(parts[0])
        except ValueError:
            continue

        try:
            m1_arcsec = float(parts[5])
            m2_arcsec = float(parts[6])
            u_rad = float(parts[7])
            u_north = float(parts[8])
            u_east = float(parts[9])

            m1_rad = m1_arcsec * ARCSEC_TO_RAD
            m2_rad = m2_arcsec * ARCSEC_TO_RAD

            m1_rad_list.append(m1_rad)
            m2_rad_list.append(m2_rad)
            exp_du.append(u_rad)
            exp_dn.append(u_north)
            exp_de.append(u_east)

        except (ValueError, IndexError) as e:
            print(f"Warning: Skipping line due to error: {e}")
            continue

    return lat_rad, lon_rad, m1_rad_list, m2_rad_list, exp_du, exp_dn, exp_de


def main():
    # 1. Parse test file
    test_file = "opoleloadcmcor.txt"
    try:
        lat_rad, lon_rad, m1_list, m2_list, exp_du, exp_dn, exp_de = parse_test_file(test_file)
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found.")
        return

    if not m1_list:
        print("Error: No test data parsed. Check file format.")
        return

    print(f"Found {len(m1_list)} test cases.")
    print(f"Test site: Lat = {math.degrees(lat_rad):.2f}, Lon = {math.degrees(lon_rad):.2f}\n")

    # 2. Run tests
    passed = 0
    failed = 0
    tolerance = 1e-8  # meters

    for i, (m1, m2, exp_u, exp_n, exp_e) in enumerate(zip(m1_list, m2_list, exp_du, exp_dn, exp_de)):
        try:
            result = ocean_pole_tide_loading(lat_rad, lon_rad, m1, m2)
            if result is None or len(result) != 3:
                print(f"Test {i+1}: Function returned invalid result.")
                failed += 1
                continue

            calc_e = result[0]  # east
            calc_n = result[1]  # north
            calc_u = result[2]  # up

            diff_e = abs(calc_e - exp_e)
            diff_n = abs(calc_n - exp_n)
            diff_u = abs(calc_u - exp_u)

            if diff_u < tolerance and diff_n < tolerance and diff_e < tolerance:
                status = "PASS"
                passed += 1
            else:
                status = "FAIL"
                failed += 1
                print(f"\n[FAIL] Test {i+1}:")
                print(f"  Input: m1={m1/ARCSEC_TO_RAD:.3f} as, m2={m2/ARCSEC_TO_RAD:.3f} as")
                print(f"  Expected (U,N,E): {exp_u:.6e}, {exp_n:.6e}, {exp_e:.6e} m")
                print(f"  Computed (U,N,E): {calc_u:.6e}, {calc_n:.6e}, {calc_e:.6e} m")
                print(f"  Diff (U,N,E):    {diff_u:.2e}, {diff_n:.2e}, {diff_e:.2e} m")
        except Exception as e:
            print(f"Test {i+1}: Function raised exception: {e}")
            failed += 1
            continue

    print("\n" + "=" * 50)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    if failed == 0:
        print("✅ All tests passed! Your current implementation is correct.")
    else:
        print("❌ Some tests failed. You need to update the implementation.")
        print("\n[FIX] Your code is missing K and gamma factors.")
        print("The correct formula should be:")
        print("  term1 = m1 * 0.6870 + m2 * 0.0036")
        print("  term2 = m2 * 0.6870 - m1 * 0.0036")
        print("  du = K * (term1 * urR + term2 * urI)")
        print("  dn = K * (term1 * unR + term2 * unI)")
        print("  de = K * (term1 * ueR + term2 * ueI)")
        print(f"  where K = {K} meters/radian")
        print("  and m1, m2 are in RADIANS.")

if __name__ == "__main__":
    main()