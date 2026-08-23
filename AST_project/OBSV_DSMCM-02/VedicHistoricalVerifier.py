#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VedicHistoricalVerifier.py – Standalone Module for Verifying Historical Textual Data
====================================================================================

PURPOSE
-------
This module provides a lightweight, command‑line‑friendly interface to compute
the Vedic components (Tithi, Nakṣatra, Yoga, Karaṇa, Rasi) and the modern
solar/lunar ephemeris (ecliptic, equatorial, topocentric) for any given date
and time, including negative years (e.g., -3101 for the Kali Yuga epoch).

It is intended SOLELY for verifying historical textual records (e.g., ancient
observations, inscriptions, or chronicles) by comparing the computed positions
with those recorded in manuscripts. It is NOT part of the main high‑precision
ephemeris pipeline (DT18, AstroCalc, etc.) and is kept separate to avoid
interference with production‑grade astrometric calculations.

All calculations are performed using the same underlying VSOP2013 + ELP/MPP02
engine (via SM_VSOP2013) and include full IAU 2006/2000A precession‑nutation,
ensuring that the results are consistent with modern astronomical standards.

FEATURES
--------
- Accepts dates in ISO 8601 format: YYYY-MM-DDTHH:MM:SS (or with space)
- Supports negative years (e.g., -4713-01-01T12:00:00)
- Calendar selection: Gregorian (default), Julian, or auto (Julian if year < 1582)
- Computes both Sayana (tropical) and Nirayana (sidereal, Lahiri ayanamsa)
- Displays solar and lunar ecliptic longitudes, latitudes, distances
- Displays equatorial (GCRS) coordinates of the Sun and Moon
- Displays topocentric apparent coordinates (CIRS, Equinox, Az/Alt)
- Outputs all Vedic components with clear labelling
- Includes Wuku system (KA, cycle, triple wara, TU-PA-A status)
- Supports timezone offset for local date calculation (default WIB = UTC+7)
- Can be used interactively or via command‑line arguments

DEPENDENCIES
------------
- VedicMathVSOP2013.py
- SM_VSOP2013.py (AstronomicalEphemeris)
- Timescales.py, EarthRotation.py, Coord_Transform.py
- Atmospheric_refraction.py (for topocentric corrections)
- DateTime.py (for now_utc)
- wuku_system.py (for Wuku calculations)

AUTHOR
------
ASTERID Project

VERSION
-------
1.3 (2026-07-07) – Added timezone support for local date (WIB)
"""

import sys
sys.dont_write_bytecode = True

import math
import argparse
import re
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone

# --------------------------------------------------------------------------
# Core modules
# --------------------------------------------------------------------------
from VedicMathVSOP2013 import (
    get_vedic_from_ephemeris,
    get_solar_ephemeris,
    normalize_angle,
    normalize_angle_rad,
    ZODIAC,
    NAKSHATRAS,
    YOGAS,
    KARANA_60
)
from SM_VSOP2013 import AstronomicalEphemeris
from Timescales import (
    J2000_JD,
    TimeScaleConverter,
    delta_t_from_jd,
    tai_utc,
    split_jd,
    combine_jd,
    jd_to_cal          # <-- tambahan
)
from DateTime import DateTime, now_utc
from wuku_system import WukuMechanicalEngine

# --------------------------------------------------------------------------
# Calendar conversion – identical to AstroCalc_VSOP2013.py
# (supports negative years, Julian/Gregorian/auto)
# --------------------------------------------------------------------------

def _julian_to_jd_noon(year: int, month: int, day: int) -> int:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - 32083

def _gregorian_to_jd_noon(year: int, month: int, day: int) -> int:
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    return day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045

def julian_to_jd(year: int, month: int, day: int,
                 hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    jd_noon = _julian_to_jd_noon(year, month, day)
    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd_noon + day_frac - 0.5

def gregorian_to_jd(year: int, month: int, day: int,
                    hour: int = 0, minute: int = 0, second: float = 0.0) -> float:
    jd_noon = _gregorian_to_jd_noon(year, month, day)
    day_frac = (hour + minute / 60.0 + second / 3600.0) / 24.0
    return jd_noon + day_frac - 0.5

def parse_iso_and_calc_jd(iso_str: str, calendar: str) -> Tuple[float, str]:
    """
    Parse ISO 8601 date and return (JD, calendar_used).
    Identical to AstroCalc_VSOP2013.parse_iso_and_calc_jd.
    """
    iso_str = iso_str.strip()
    pattern = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})(?:T(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d+(?:\.\d+)?))?)?$'
    m = re.match(pattern, iso_str)
    if not m:
        pattern2 = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})\s+(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d+(?:\.\d+)?))?$'
        m = re.match(pattern2, iso_str)
    if not m:
        pattern3 = r'^(?P<sign>-?)(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})(?:T|\s+)(?P<hour>\d{2}):(?P<minute>\d{2})$'
        m = re.match(pattern3, iso_str)
    if not m:
        raise ValueError(f"Invalid ISO 8601: {repr(iso_str)}")

    sign = m.group('sign')
    year = int(sign + m.group('year'))
    month = int(m.group('month'))
    day = int(m.group('day'))
    hour = int(m.group('hour') or 0)
    minute = int(m.group('minute') or 0)
    second_str = m.group('second')
    second = float(second_str) if second_str is not None else 0.0

    if calendar == 'julian':
        jd = julian_to_jd(year, month, day, hour, minute, second)
        cal_used = 'julian'
    elif calendar == 'gregorian':
        jd = gregorian_to_jd(year, month, day, hour, minute, second)
        cal_used = 'gregorian'
    else:  # auto
        if year < 1582:
            jd = julian_to_jd(year, month, day, hour, minute, second)
            cal_used = 'julian'
        else:
            jd = gregorian_to_jd(year, month, day, hour, minute, second)
            cal_used = 'gregorian'
    return jd, cal_used

# --------------------------------------------------------------------------
# Main computation function
# --------------------------------------------------------------------------
def compute_historical_report(
    iso_date: str,
    calendar: str = 'auto',
    eop_file: str = "EOP_20u24_C04_one_file_1962-now.txt",
    lat_deg: float = -7.609444,   # Jolotundo (example)
    lon_deg: float = 112.595556,
    height_m: float = 532.021,
    apply_refraction: bool = False,
    tz_offset: float = 7.0        # WIB = UTC+7
) -> Dict:
    """
    Compute a comprehensive historical verification report for a given date.
    tz_offset: hours to add to UTC for local date calculation (default 7.0 for WIB)
    """
    # 1. Parse and convert to JD (UTC)
    jd_utc, cal_used = parse_iso_and_calc_jd(iso_date, calendar)

    # 2. Time scales (ephemeris tetap menggunakan UTC)
    tsc = TimeScaleConverter()
    tt_jd = tsc.utc_to_tt(jd_utc)
    tdb_jd = tsc.utc_to_tdb(jd_utc)
    delta_t = delta_t_from_jd(tt_jd)
    ut1_jd = tt_jd - delta_t / 86400.0

    # 3. Ephemeris engine
    ephem = AstronomicalEphemeris(eop_file=eop_file)
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    # 4. Vedic components (ephemeris tetap menggunakan TT dari UTC)
    vedic = get_vedic_from_ephemeris(tt_jd, ephem, use_ayanamsa=True)

    # 5. Solar ephemeris
    sun_eph = get_solar_ephemeris(tt_jd, lat_rad, lon_rad, height_m, ephem, apply_refraction)
    # 6. Lunar ephemeris (using ephem directly)
    moon_gcrs = ephem.moon_geocentric_gcrs(tt_jd, unit=False)  # km
    from Coord_Transform import cartesian_to_spherical
    moon_ra, moon_dec, moon_dist = cartesian_to_spherical(moon_gcrs)
    moon_ecl_lon, moon_ecl_lat, moon_ecl_r = ephem.moon_ecliptic_of_date(tt_jd, unit=False, degrees=True)
    moon_top = ephem.moon_apparent_topocentric(tt_jd, lat_rad, lon_rad, height_m, apply_refraction)

    # ---- Wuku System (menggunakan waktu lokal) ----
    wuku_engine = WukuMechanicalEngine()
    # Hitung JD lokal: UTC + tz_offset jam
    jd_local = jd_utc + tz_offset / 24.0
    # Dapatkan tanggal dan waktu lokal menggunakan jd_to_cal dari Timescales
    cal_local = jd_to_cal(*split_jd(jd_local), scale='utc')
    local_year = cal_local['year']
    local_month = cal_local['month']
    local_day = cal_local['day']
    local_hour = cal_local['hour']
    local_minute = cal_local['minute']
    local_second = cal_local['second']

    # Untuk Wuku, kita butuh tanggal lokal (hari)
    ka = wuku_engine.date_to_ka(local_year, local_month, local_day, cal_used)
    wuku_info = wuku_engine.get_wuku_by_ka(ka)
    epoch_info = wuku_engine.get_detailed_wuku_epoch_info(ka)

    # 7. Compile report
    report = {
        'jd_utc': jd_utc,
        'jd_tt': tt_jd,
        'jd_tdb': tdb_jd,
        'delta_t': delta_t,
        'ayanamsa': vedic['ayanamsa'],
        'sayana': vedic['sayana'],
        'nirayana': vedic['nirayana'],
        'sun_ecliptic': {
            'lon_deg': sun_eph['ecliptic']['lon_deg'],
            'lat_deg': sun_eph['ecliptic']['lat_deg'],
            'dist_km': sun_eph['ecliptic']['dist_km'],
            'dist_au': sun_eph['ecliptic']['dist_au']
        },
        'sun_equatorial': sun_eph['equatorial_gcrs'],
        'sun_topocentric': sun_eph['topocentric'],
        'moon_ecliptic': {
            'lon_deg': moon_ecl_lon,
            'lat_deg': moon_ecl_lat,
            'dist_km': moon_ecl_r,
            'dist_au': moon_ecl_r / 149597870.7
        },
        'moon_equatorial': {
            'ra_deg': math.degrees(normalize_angle_rad(moon_ra)),
            'dec_deg': math.degrees(moon_dec),
            'dist_km': moon_dist
        },
        'moon_topocentric': {
            'ra_cirs_deg': moon_top['ra_cirs_deg'],
            'dec_cirs_deg': moon_top['dec_cirs_deg'],
            'ra_eqx_deg': moon_top['ra_eqx_deg'],
            'dec_eqx_deg': moon_top['dec_eqx_deg'],
            'az_deg': moon_top['az_deg'],
            'alt_geom_deg': moon_top['alt_geom_deg'],
            'alt_app_deg': moon_top['alt_app_deg'],
            'dist_km': moon_top['dist_km']
        },
        'metadata': {
            'iso_date': iso_date,
            'calendar_used': cal_used,
            'observer_lat_deg': lat_deg,
            'observer_lon_deg': lon_deg,
            'observer_height_m': height_m,
            'refraction_applied': apply_refraction,
            'tz_offset': tz_offset,
            'local_date_components': (local_year, local_month, local_day),
            'local_time_components': (local_hour, local_minute, local_second),
            'local_jd': jd_local
        },
        'wuku': {
            'ka': ka,
            'wuku_info': wuku_info,
            'epoch_info': epoch_info,
            'calendar_used': cal_used,
            'date_components': (local_year, local_month, local_day),
            'jd_local': jd_local,
            'tz_offset': tz_offset
        }
    }
    return report

# --------------------------------------------------------------------------
# Pretty printer (Strict W=72 Alignment)
# --------------------------------------------------------------------------
def print_historical_report(report: Dict) -> None:
    W = 72
    # Karakter Box-Drawing untuk garis solid tebal/tipis
    thick_sep = "═" * W
    thin_sep  = "─" * W
    
    # ANSI Escape codes untuk efek tebal (bold)
    BOLD = '\033[1m'
    RESET = '\033[0m'

    meta = report['metadata']
    tz = meta['tz_offset']
    tz_str = f"UTC{'+' if tz >= 0 else ''}{tz:.1f}"
    
    # Konversi ke int agar format d berfungsi
    ly, lm, ld = meta['local_date_components']
    lh, lmn, ls = meta['local_time_components']
    local_year, local_month, local_day = int(ly), int(lm), int(ld)
    local_h, local_minute, local_second = int(lh), int(lmn), int(ls)
    local_date_str = f"{local_year:04d}-{local_month:02d}-{local_day:02d} {local_h:02d}:{local_minute:02d}:{local_second:02d}"

    print(f"\n{BOLD}{thick_sep}{RESET}")
    print(f"{BOLD}{'HISTORICAL VERIFICATION REPORT'.center(W)}{RESET}")
    print(f"{BOLD}{thick_sep}{RESET}")
    
    # Header Utama - Lebar label disamakan secara konsisten (16 karakter)
    print(f"  {'Date (UTC)':<16}: {report['metadata']['iso_date']} (Calendar: {report['metadata']['calendar_used']})")
    print(f"  {'Local date':<16}: {local_date_str} ({tz_str})")
    print(f"  {'JD (UTC)':<16}: {report['jd_utc']:.6f}")
    print(f"  {'JD (TT)':<16}: {report['jd_tt']:.6f}")
    print(f"  {'JD (TDB)':<16}: {report['jd_tdb']:.6f}")
    print(f"  {'ΔT':<16}: {report['delta_t']:.3f} s")
    print(f"{BOLD}{thick_sep}{RESET}")

    say = report['sayana']
    nir = report['nirayana']
    aya = report['ayanamsa']

    def section_header(title):
        print(f"\n{BOLD}{title}{RESET}")
        print(f"{BOLD}{thin_sep}{RESET}")

    # [1] SUN EPHEMERIS - Lebar label 20
    section_header("[1] SUN EPHEMERIS")
    print(f"{BOLD}  Ecliptic of date:{RESET}")
    print(f"    {'Longitude':<20}: {report['sun_ecliptic']['lon_deg']:.6f}°")
    print(f"    {'Latitude':<20}: {report['sun_ecliptic']['lat_deg']:.6f}°")
    print(f"    {'Distance':<20}: {report['sun_ecliptic']['dist_km']:.3f} km")
    print(f"{BOLD}  Equatorial (GCRS):{RESET}")
    print(f"    {'RA':<20}: {report['sun_equatorial']['ra_deg']:.6f}°")
    print(f"    {'Dec':<20}: {report['sun_equatorial']['dec_deg']:.6f}°")
    print(f"    {'Distance':<20}: {report['sun_equatorial']['dist_km']:.3f} km")
    print(f"{BOLD}  Topocentric apparent:{RESET}")
    print(f"    {'RA (CIRS)':<20}: {report['sun_topocentric']['ra_cirs_deg']:.6f}°")
    print(f"    {'Dec (CIRS)':<20}: {report['sun_topocentric']['dec_cirs_deg']:.6f}°")
    print(f"    {'RA (Equinox)':<20}: {report['sun_topocentric']['ra_eqx_deg']:.6f}°")
    print(f"    {'Dec (Equinox)':<20}: {report['sun_topocentric']['dec_eqx_deg']:.6f}°")
    print(f"    {'Azimuth':<20}: {report['sun_topocentric']['az_deg']:.6f}°")
    print(f"    {'Elevation (geom.)':<20}: {report['sun_topocentric']['alt_geom_deg']:.6f}°")
    print(f"    {'Elevation (app.)':<20}: {report['sun_topocentric']['alt_app_deg']:.6f}°")
    print(f"    {'Distance':<20}: {report['sun_topocentric']['dist_km']:.3f} km")

    # [2] MOON EPHEMERIS - Lebar label 20
    section_header("[2] MOON EPHEMERIS")
    print(f"{BOLD}  Ecliptic of date:{RESET}")
    print(f"    {'Longitude':<20}: {report['moon_ecliptic']['lon_deg']:.6f}°")
    print(f"    {'Latitude':<20}: {report['moon_ecliptic']['lat_deg']:.6f}°")
    print(f"    {'Distance':<20}: {report['moon_ecliptic']['dist_km']:.3f} km")
    print(f"{BOLD}  Equatorial (GCRS):{RESET}")
    print(f"    {'RA':<20}: {report['moon_equatorial']['ra_deg']:.6f}°")
    print(f"    {'Dec':<20}: {report['moon_equatorial']['dec_deg']:.6f}°")
    print(f"    {'Distance':<20}: {report['moon_equatorial']['dist_km']:.3f} km")
    print(f"{BOLD}  Topocentric apparent:{RESET}")
    print(f"    {'RA (CIRS)':<20}: {report['moon_topocentric']['ra_cirs_deg']:.6f}°")
    print(f"    {'Dec (CIRS)':<20}: {report['moon_topocentric']['dec_cirs_deg']:.6f}°")
    print(f"    {'RA (Equinox)':<20}: {report['moon_topocentric']['ra_eqx_deg']:.6f}°")
    print(f"    {'Dec (Equinox)':<20}: {report['moon_topocentric']['dec_eqx_deg']:.6f}°")
    print(f"    {'Azimuth':<20}: {report['moon_topocentric']['az_deg']:.6f}°")
    print(f"    {'Elevation (geom.)':<20}: {report['moon_topocentric']['alt_geom_deg']:.6f}°")
    print(f"    {'Elevation (app.)':<20}: {report['moon_topocentric']['alt_app_deg']:.6f}°")
    print(f"    {'Distance':<20}: {report['moon_topocentric']['dist_km']:.3f} km")

    # [3] WUKU SYSTEM - Lebar label 26
    wuku = report.get('wuku')
    if wuku:
        wi = wuku['wuku_info']
        ei = wuku['epoch_info']
        section_header("[3] WUKU SYSTEM (210-HARI SIKLUS)")
        print(f"    {'Kali Ahargana (KA)':<26}: {wuku['ka']:,}")
        print(f"    {'Siklus Wuku':<26}: {ei['cycle_number']}")
        print(f"    {'Posisi dalam siklus':<26}: {ei['position_in_cycle']}/210")
        print(f"    {'Progres siklus':<26}: {ei['progress_percent']:.1f}%")
        print(f"    {'Wuku saat ini':<26}: {wi['wuku_name']} (#{wi['wuku_number']})")
        print(f"    {'Hari dalam Wuku':<26}: {wi['day_in_wuku']}/7")
        print(f"    {'Triple Wara':<26}: {wi['wara_triple_full']}")
        print(f"    {'Kode Triple':<26}: {wi['wara_triple']}")
        print(f"    {'TU-PA-A':<26}: {'YA' if wi['is_tu_pa_a'] else 'TIDAK'}")
        if not wi['is_tu_pa_a']:
            print(f"    {'Hari ke TU-PA-A berikutnya':<26}: {ei['days_to_next_tu_pa_a']}")
        print(f"    {'Hari sejak epoch wuku':<26}: {ei['days_since_epoch']:,} hari")
        print(f"    {'Arah waktu':<26}: {ei['direction']}")

    # [4] VEDIC COMPONENTS SAYANA - Lebar label 20
    section_header("[4] VEDIC COMPONENTS – SAYANA (TROPICAL)")
    print(f"    {'Sun longitude':<20}: {say['sun_longitude']:.6f}°")
    print(f"    {'Sun Rasi':<20}: {say['sun_rasi']['name']} ({say['sun_rasi']['degrees_in']:.2f}°)")
    print(f"    {'Moon longitude':<20}: {say['moon_longitude']:.6f}°")
    print(f"    {'Moon Rasi':<20}: {say['moon_rasi']['name']} ({say['moon_rasi']['degrees_in']:.2f}°)")
    print(f"    {'Tithi':<20}: {say['tithi']['tithi']} {say['tithi']['paksa']} ({say['tithi']['percent']:.1f}%)")
    print(f"    {'Nakṣatra':<20}: {say['nakshatra']['name']} (pada {say['nakshatra']['pada']})")
    print(f"    {'Yoga':<20}: {say['yoga']['name']} ({say['yoga']['percent']:.1f}%)")
    print(f"    {'Karaṇa':<20}: {say['karana']['name']} (ke-{say['karana']['karana_num']})")

    # [5] VEDIC COMPONENTS NIRAYANA - Lebar label 20
    section_header("[5] VEDIC COMPONENTS – NIRAYANA (SIDEREAL, LAHIRI)")
    print(f"    {'Ayanamsa':<20}: {aya:.6f}°")
    print(f"    {'Sun longitude':<20}: {nir['sun_longitude']:.6f}°")
    print(f"    {'Sun Rasi':<20}: {nir['sun_rasi']['name']} ({nir['sun_rasi']['degrees_in']:.2f}°)")
    print(f"    {'Moon longitude':<20}: {nir['moon_longitude']:.6f}°")
    print(f"    {'Moon Rasi':<20}: {nir['moon_rasi']['name']} ({nir['moon_rasi']['degrees_in']:.2f}°)")
    print(f"    {'Tithi':<20}: {nir['tithi']['tithi']} {nir['tithi']['paksa']} ({nir['tithi']['percent']:.1f}%)")
    print(f"    {'Nakṣatra':<20}: {nir['nakshatra']['name']} (pada {nir['nakshatra']['pada']})")
    print(f"    {'Yoga':<20}: {nir['yoga']['name']} ({nir['yoga']['percent']:.1f}%)")
    print(f"    {'Karaṇa':<20}: {nir['karana']['name']} (ke-{nir['karana']['karana_num']})")

    # [6] NOTES - Lebar label 20
    section_header("[6] NOTES")
    print(f"    {'Observer':<20}: Lat {meta['observer_lat_deg']:.6f}°, Lon {meta['observer_lon_deg']:.6f}°, H {meta['observer_height_m']:.3f} m")
    print(f"    {'Timezone for local':<20}: UTC{'+' if tz >= 0 else ''}{tz:.1f}")
    print(f"    {'Refraction applied':<20}: {meta['refraction_applied']}")
    print(f"    {BOLD}All calculations include full IAU 2006/2000A precession‑nutation.{RESET}")
    print(f"{BOLD}{thick_sep}{RESET}\n")

# --------------------------------------------------------------------------
# Command‑line interface
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Historical verifier for Vedic components and ephemeris positions.",
        epilog="Examples:\n"
               "  python VedicHistoricalVerifier.py 2026-07-06T12:30:45\n"
               "  python VedicHistoricalVerifier.py -3101-01-01T12:00:00 --calendar auto\n"
               "  python VedicHistoricalVerifier.py --now\n"
               "  python VedicHistoricalVerifier.py 1041-11-06 --calendar julian"
    )
    parser.add_argument(
        "date", nargs='?', default=None,
        help="ISO 8601 date/time. If omitted, use --now or interactive prompt."
    )
    parser.add_argument(
        "--now", action='store_true',
        help="Use the current UTC time."
    )
    parser.add_argument(
        "--calendar", choices=['gregorian', 'julian', 'auto'], default='auto',
        help="Calendar system: 'gregorian', 'julian', or 'auto' (Julian if year < 1582)."
    )
    parser.add_argument(
        "--eop", default="EOP_20u24_C04_one_file_1962-now.txt",
        help="Path to EOP file (optional for ancient dates)."
    )
    parser.add_argument(
        "--lat", type=float, default=-7.609444,
        help="Observer latitude (deg)."
    )
    parser.add_argument(
        "--lon", type=float, default=112.595556,
        help="Observer longitude (deg)."
    )
    parser.add_argument(
        "--height", type=float, default=554.509,
        help="Observer height (m)."
    )
    parser.add_argument(
        "--refraction", action='store_true',
        help="Apply atmospheric refraction to topocentric altitudes."
    )
    parser.add_argument(
        "--tz", type=float, default=7.0,
        help="Timezone offset (hours) from UTC for local date. Default 7.0 (WIB)."
    )

    args = parser.parse_args()

    # Determine date string and refraction behavior
    is_realtime = False
    if args.now:
        dt = now_utc()
        iso_str = dt.to_iso()
        print(f"Using current UTC time: {iso_str}")
        is_realtime = True
    elif args.date is None:
        print("Enter date/time in ISO 8601 format (YYYY-MM-DDTHH:MM:SS)")
        print("  Examples: 2026-07-06T12:30:45  or  -4713-01-01T12:00:00")
        print("  Press Enter to use current UTC time.")
        user_input = input("> ").strip()
        if user_input == "":
            dt = now_utc()
            iso_str = dt.to_iso()
            print(f"Using current UTC time: {iso_str}")
            is_realtime = True
        else:
            iso_str = user_input
    else:
        iso_str = args.date

    # Apply refraction automatically for realtime, otherwise respect --refraction flag
    if is_realtime:
        apply_refraction = True
    else:
        apply_refraction = args.refraction

    try:
        report = compute_historical_report(
            iso_str,
            calendar=args.calendar,
            eop_file=args.eop,
            lat_deg=args.lat,
            lon_deg=args.lon,
            height_m=args.height,
            apply_refraction=apply_refraction,
            tz_offset=args.tz
        )
        print_historical_report(report)
    except Exception as e:
        print(f"\n❌ ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()