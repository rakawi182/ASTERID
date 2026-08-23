#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TimeDisplay.py — High‑Precision Time Scale Converter & Display
================================================================
This module provides a comprehensive tool for converting and displaying
time across all major astronomical time scales with full IERS 2010 compliance.

Supported time scales:
    UTC, TAI, TT, TDB, UT1, TCG, TCB, WIB (UTC+7), LMST (Local Mean Solar Time)

Features:
    - Real‑time display of all scales with full epoch information
    - Conversion from any input scale (ISO‑8601 or JD)
    - Detailed breakdown of offsets: TAI−UTC, TT−TAI, TDB−TT, TCG−TT, TCB−TDB
    - Earth Rotation Angle (ERA) in degrees and radians
    - ΔT = TT−UT1 and DUT1 = UT1−UTC
    - Local Mean Solar Time at the observatory longitude (112.595556°E)
    - WIB (UTC+7) local time for Indonesia

All calculations are based on:
    - IAU 2000/2006 resolutions
    - IERS Conventions (2010)
    - SOFA time scale functions

Author:   ASTERID Project
Version:  3.3 (2026-08-08) — Added observatory location header
"""

import sys
import math
from datetime import datetime, timezone
from typing import Dict, Optional

sys.dont_write_bytecode = True

# -----------------------------------------------------------------------------
# Core imports
# -----------------------------------------------------------------------------
from Timescales import (
    J2000_JD,
    MJD_ZERO,
    TAI_TT_OFFSET,
    T0_JD,
    LG,
    LB,
    TDB0,
    split_jd,
    combine_jd,
    cal_to_jd,
    jd_to_cal,
    tai_utc,
    delta_t_from_jd,
    utc_to_tai,
    tai_to_tt,
    tt_to_tcg,
    tcg_to_tt,
    tcb_to_tdb,
    tdb_to_tcb,
    tt_to_tdb,
    tdb_to_tt,
    era,
    TimeScaleConverter,
)
from EOPDelta import EOPProvider
from Atmospheric_refraction import SITE_LAT_DEG, SITE_LON_DEG, SITE_ELEV_M

# -----------------------------------------------------------------------------
# ANSI escape sequences for terminal formatting
# -----------------------------------------------------------------------------
BOLD = '\033[1m'
RESET = '\033[0m'
DIM = '\033[2m'
UNDERLINE = '\033[4m'

def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"

def header_line(text: str, width: int = 70) -> str:
    return bold(text.center(width))

def separator(char: str = '═', width: int = 70) -> str:
    return char * width


# -----------------------------------------------------------------------------
# Core converter class
# -----------------------------------------------------------------------------
class TimeConverter:
    """
    High‑precision time scale converter providing full traceability
    to IERS standards.
    """

    def __init__(self, eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt"):
        self.eop_provider = EOPProvider(eop_file)
        self.tsc = TimeScaleConverter(eop_file)
        self.lon_deg = SITE_LON_DEG   # 112.595556

    def dut1(self, mjd: float) -> float:
        """UT1−UTC in seconds at the given MJD (UTC)."""
        try:
            return self.eop_provider.get_eop(mjd)['ut1_utc']
        except Exception:
            return 0.0

    def delta_t(self, jd_utc: float) -> float:
        """ΔT = TT−UT1 in seconds at given JD (UTC)."""
        return delta_t_from_jd(jd_utc)

    def convert(self, jd: float, scale_in: str = 'utc') -> Dict[str, float]:
        """
        Convert from any input scale to all supported time scales.

        Parameters
        ----------
        jd : float
            Julian Date in the input scale.
        scale_in : str
            Input scale: 'utc', 'tai', 'tt', 'tdb', 'ut1', 'tcg', 'tcb'.

        Returns
        -------
        dict
            Dictionary with keys (lowercase) 'utc', 'tai', 'tt', 'tdb', 'ut1', 'tcg', 'tcb'
            and corresponding JD values.
        """
        scale = scale_in.lower()
        valid = ('utc', 'tai', 'tt', 'tdb', 'ut1', 'tcg', 'tcb')
        if scale not in valid:
            raise ValueError(f"Unknown time scale: {scale_in}")

        result = {k: None for k in valid}
        result[scale] = jd

        # Route through TT as the hub
        if scale == 'utc':
            jd1, jd2 = split_jd(jd)
            tai1, tai2 = utc_to_tai(jd1, jd2)
            tai = combine_jd(tai1, tai2)
            tt1, tt2 = tai_to_tt(tai1, tai2)
            tt = combine_jd(tt1, tt2)
            result['tai'] = tai
            result['tt'] = tt
            dut1 = self.dut1(jd - MJD_ZERO)
            result['ut1'] = jd + dut1 / 86400.0

        elif scale == 'tai':
            tai = jd
            tt1, tt2 = tai_to_tt(*split_jd(tai))
            tt = combine_jd(tt1, tt2)
            result['tt'] = tt
            utc1, utc2 = tai_to_utc(*split_jd(tai))
            utc = combine_jd(utc1, utc2)
            result['utc'] = utc
            dut1 = self.dut1(utc - MJD_ZERO)
            result['ut1'] = utc + dut1 / 86400.0

        elif scale == 'tt':
            tt = jd
            tai1, tai2 = tt_to_tai(*split_jd(tt))
            tai = combine_jd(tai1, tai2)
            result['tai'] = tai
            utc1, utc2 = tai_to_utc(*split_jd(tai))
            utc = combine_jd(utc1, utc2)
            result['utc'] = utc
            dut1 = self.dut1(utc - MJD_ZERO)
            result['ut1'] = utc + dut1 / 86400.0

        elif scale == 'tdb':
            tdb = jd
            tt1, tt2 = tdb_to_tt(*split_jd(tdb))
            tt = combine_jd(tt1, tt2)
            result['tt'] = tt
            tai1, tai2 = tt_to_tai(*split_jd(tt))
            tai = combine_jd(tai1, tai2)
            result['tai'] = tai
            utc1, utc2 = tai_to_utc(*split_jd(tai))
            utc = combine_jd(utc1, utc2)
            result['utc'] = utc
            dut1 = self.dut1(utc - MJD_ZERO)
            result['ut1'] = utc + dut1 / 86400.0

        elif scale == 'ut1':
            ut1 = jd
            # Iterate once to refine ΔT
            dt = delta_t_from_jd(ut1)
            tt = ut1 + dt / 86400.0
            dt2 = delta_t_from_jd(tt)
            tt = ut1 + dt2 / 86400.0
            result['tt'] = tt
            tai1, tai2 = tt_to_tai(*split_jd(tt))
            tai = combine_jd(tai1, tai2)
            result['tai'] = tai
            utc1, utc2 = tai_to_utc(*split_jd(tai))
            utc = combine_jd(utc1, utc2)
            result['utc'] = utc

        elif scale == 'tcg':
            tcg = jd
            tt1, tt2 = tcg_to_tt(*split_jd(tcg))
            tt = combine_jd(tt1, tt2)
            result['tt'] = tt
            tai1, tai2 = tt_to_tai(*split_jd(tt))
            tai = combine_jd(tai1, tai2)
            result['tai'] = tai
            utc1, utc2 = tai_to_utc(*split_jd(tai))
            utc = combine_jd(utc1, utc2)
            result['utc'] = utc
            dut1 = self.dut1(utc - MJD_ZERO)
            result['ut1'] = utc + dut1 / 86400.0

        elif scale == 'tcb':
            tcb = jd
            tdb1, tdb2 = tcb_to_tdb(*split_jd(tcb))
            tdb = combine_jd(tdb1, tdb2)
            result['tdb'] = tdb
            tt1, tt2 = tdb_to_tt(*split_jd(tdb))
            tt = combine_jd(tt1, tt2)
            result['tt'] = tt
            tai1, tai2 = tt_to_tai(*split_jd(tt))
            tai = combine_jd(tai1, tai2)
            result['tai'] = tai
            utc1, utc2 = tai_to_utc(*split_jd(tai))
            utc = combine_jd(utc1, utc2)
            result['utc'] = utc
            dut1 = self.dut1(utc - MJD_ZERO)
            result['ut1'] = utc + dut1 / 86400.0

        # ---- Ensure TDB, TCG, TCB are always derived from TT ----
        if result['tt'] is not None:
            if result['tdb'] is None:
                tdb1, tdb2 = tt_to_tdb(*split_jd(result['tt']))
                result['tdb'] = combine_jd(tdb1, tdb2)
            if result['tcg'] is None:
                tcg1, tcg2 = tt_to_tcg(*split_jd(result['tt']))
                result['tcg'] = combine_jd(tcg1, tcg2)
        if result['tdb'] is not None and result['tcb'] is None:
            tcb1, tcb2 = tdb_to_tcb(*split_jd(result['tdb']))
            result['tcb'] = combine_jd(tcb1, tcb2)

        return result

    def jd_to_iso(self, jd: float, scale: str = 'utc') -> str:
        """Convert JD to ISO 8601 string for a given time scale."""
        cal = jd_to_cal(*split_jd(jd), scale=scale)
        frac = cal['second'] - int(cal['second'])
        sec_int = int(cal['second'])
        if frac:
            return f"{cal['year']:04d}-{cal['month']:02d}-{cal['day']:02d} " \
                   f"{cal['hour']:02d}:{cal['minute']:02d}:{sec_int:02d}.{int(frac*1e6):06d}"
        else:
            return f"{cal['year']:04d}-{cal['month']:02d}-{cal['day']:02d} " \
                   f"{cal['hour']:02d}:{cal['minute']:02d}:{sec_int:02d}"

    def detailed_info(self, jd: float, scale_in: str = 'utc') -> Dict:
        """
        Return comprehensive time information including ISO, JD, MJD,
        offsets, ΔT, DUT1, ERA, and local times (WIB, LMST).
        """
        jd_dict = self.convert(jd, scale_in)
        iso_dict = {}
        # Use uppercase keys for consistency with display
        for sc, val in jd_dict.items():
            if val is not None:
                iso_dict[sc.upper()] = self.jd_to_iso(val, sc)

        utc = jd_dict['utc']
        tai = jd_dict['tai']
        tt  = jd_dict['tt']
        tdb = jd_dict['tdb']
        ut1 = jd_dict['ut1']
        tcg = jd_dict['tcg']
        tcb = jd_dict['tcb']

        # ---- Additional local times ----
        # WIB = UTC + 7 hours
        if utc is not None:
            wib_jd = utc + 7.0 / 24.0
            iso_dict['WIB'] = self.jd_to_iso(wib_jd, 'utc')

        # Local Mean Solar Time = UT1 + longitude/15 hours
        if ut1 is not None:
            lon_offset_hours = self.lon_deg / 15.0
            lmst_jd = ut1 + lon_offset_hours / 24.0
            iso_dict['LMST'] = self.jd_to_iso(lmst_jd, 'utc')

        # Compute offsets
        def diff(a, b):
            return (a - b) * 86400.0 if (a is not None and b is not None) else None

        offsets = {
            'TAI−UTC':   diff(tai, utc),
            'TT−TAI':    diff(tt, tai),
            'TDB−TT':    diff(tdb, tt),
            'TCG−TT':    diff(tcg, tt),
            'TCB−TDB':   diff(tcb, tdb),
            'UT1−UTC':   diff(ut1, utc),
            'TT−UT1':    diff(tt, ut1),
        }

        era_rad = era(ut1) if ut1 is not None else None

        return {
            'iso': iso_dict,
            'jd': jd_dict,
            'mjd': {k: v - MJD_ZERO for k, v in jd_dict.items() if v is not None},
            'offsets': offsets,
            'era_rad': era_rad,
            'era_deg': math.degrees(era_rad) if era_rad is not None else None,
        }

    def realtime_detailed(self) -> Dict:
        """Detailed information for the current UTC time."""
        now_utc = datetime.now(timezone.utc)
        jd_utc = combine_jd(*cal_to_jd(
            now_utc.year, now_utc.month, now_utc.day,
            now_utc.hour, now_utc.minute,
            now_utc.second + now_utc.microsecond / 1e6,
            scale='utc'
        ))
        return self.detailed_info(jd_utc, 'utc')

    def from_iso_detailed(self, iso_str: str, scale_in: str = 'utc') -> Dict:
        """Detailed information from an ISO 8601 string."""
        iso_str = iso_str.strip().replace('T', ' ')
        # Try several formats
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(iso_str, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized ISO format: {iso_str}")
        jd1, jd2 = cal_to_jd(
            dt.year, dt.month, dt.day,
            dt.hour, dt.minute,
            dt.second + dt.microsecond / 1e6,
            scale=scale_in
        )
        jd = combine_jd(jd1, jd2)
        return self.detailed_info(jd, scale_in)


# -----------------------------------------------------------------------------
# Display functions (professional formatting)
# -----------------------------------------------------------------------------
def print_detailed(info: Dict, title: str = "RELATIVISTIC TIME SCALE REALIZATION") -> None:
    """
    Print a beautifully formatted table of all time scales and parameters.
    """
    width = 70
    sep = separator('═', width)
    thin = separator('─', width)

    print(f"\n{sep}")
    print(f"{bold(title.center(width))}")
    print(f"{bold('IERS Conventions (2010) — IAU 2000/2006 Resolutions'.center(width))}")
    # --- Tambahkan baris lokasi observatorium ---
    loc_str = f"Jolotundo Obsv, Mt. Pawitra  Lat: {SITE_LAT_DEG:+.6f}°  Lon: {SITE_LON_DEG:+.6f}°"
    print(f"{bold(loc_str.center(width))}")
    print(sep)

    # ---- ISO 8601 ----
    print(f"\n{bold('ISO 8601')}")
    print(thin)
    # Display in a specific order: UTC, TAI, TT, TDB, UT1, TCG, TCB, WIB, LMST
    order = ['UTC', 'TAI', 'TT', 'TDB', 'UT1', 'TCG', 'TCB', 'WIB', 'LMST']
    for scale in order:
        iso = info['iso'].get(scale)
        if iso:
            print(f"  {scale.upper():>6} : {iso}")

    # ---- JD ----
    print(f"\n{bold('Julian Dates (JD)')}")
    print(thin)
    for scale, value in info['jd'].items():
        if value is not None:
            print(f"  {scale.upper():>6} : {value:.9f}")

    # ---- MJD ----
    print(f"\n{bold('Modified Julian Dates (MJD)')}")
    print(thin)
    for scale, value in info['mjd'].items():
        if value is not None:
            print(f"  {scale.upper():>6} : {value:.9f}")

    # ---- Offsets ----
    print(f"\n{bold('Time Scale Offsets (seconds)')}")
    print(thin)
    offs = info['offsets']
    for label, val in offs.items():
        if val is not None:
            print(f"  {label:>10} : {val:+.9f}")

    # ---- Earth Rotation Angle ----
    if info['era_deg'] is not None:
        print(f"\n{bold('Earth Rotation Angle (ERA)')}")
        print(thin)
        print(f"  {bold('ERA')} : {info['era_deg']:.9f}°  ({info['era_rad']:.9f} rad)")

    # ---- ΔT and DUT1 highlighted ----
    print(f"\n{bold('Earth Rotation Parameters')}")
    print(thin)
    dut1 = offs.get('UT1−UTC')
    dt   = offs.get('TT−UT1')
    if dut1 is not None:
        print(f"  {bold('DUT1 = UT1−UTC')} : {dut1:+.9f} s")
    if dt is not None:
        print(f"  {bold('ΔT   = TT−UT1')}  : {dt:+.9f} s")

    print(f"\n{sep}")


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
def realtime_detailed(eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt") -> Dict:
    return TimeConverter(eop_file).realtime_detailed()

def from_iso_detailed(iso_str: str, scale_in: str = 'utc',
                      eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt") -> Dict:
    return TimeConverter(eop_file).from_iso_detailed(iso_str, scale_in)


# -----------------------------------------------------------------------------
# Interactive command-line interface
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(separator('═', 70))
    print(bold("  TIME SCALE CONVERTER & DISPLAY  ".center(70)))
    print(bold("  IERS Conventions (2010) — IAU 2000/2006  ".center(70)))
    print(separator('═', 70))
    print("Select mode:")
    print("  [1]  Real‑time display")
    print("  [2]  Convert from ISO 8601")
    print("  [3]  Convert from JD")
    print("  [q]  Quit")
    print(separator('─', 70))

    tc = TimeConverter()

    while True:
        mode = input(f"\n{bold('Mode')}: ").strip().lower()
        if mode == 'q' or mode == 'quit':
            break
        elif mode == '1':
            info = tc.realtime_detailed()
            print_detailed(info, "RELATIVISTIC TIME SCALE REALIZATION")
        elif mode == '2':
            iso = input("Enter ISO 8601 time: ").strip()
            if not iso:
                continue
            scale = input("Input scale (utc/tai/tt/tdb/ut1) [utc]: ").strip() or 'utc'
            try:
                info = tc.from_iso_detailed(iso, scale)
                print_detailed(info, f"CONVERSION FROM: {iso}  ({scale.upper()})")
            except Exception as e:
                print(f"❌ Error: {e}")
        elif mode == '3':
            try:
                jd = float(input("Enter JD: ").strip())
            except ValueError:
                print("❌ Invalid JD.")
                continue
            scale = input("Input scale (utc/tai/tt/tdb/ut1/tcg/tcb) [utc]: ").strip() or 'utc'
            try:
                info = tc.detailed_info(jd, scale)
                print_detailed(info, f"CONVERSION FROM JD = {jd:.9f}  ({scale.upper()})")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Invalid option. Choose 1, 2, 3, or q.")

    print("\nDone.\n")