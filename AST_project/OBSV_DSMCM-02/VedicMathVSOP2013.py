#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VedicMathVSOP2013.py – High‑Precision Vedic Calendar Calculations
==================================================================

This module computes the five fundamental elements of the Vedic calendar
(Tithi, Nakṣatra, Yoga, Karaṇa) using the high‑precision VSOP2013
planetary theory and ELP/MPP02 lunar ephemeris, as provided by the
`SM_VSOP2013` orchestrator.

All positions are referred to the **true ecliptic and equinox of date**,
thus fully incorporating the IAU 2006/2000A precession‑nutation model
(IERS Conventions 2010). The module also offers convenience functions
to retrieve the modern solar ephemeris in ecliptic, equatorial (GCRS),
and topocentric frames.

Dependencies
------------
- SM_VSOP2013.py (AstronomicalEphemeris)
- EarthRotation.py, Coord_Transform.py (for rotations)
- Timescales.py (for time conversions)
- Atmospheric_refraction.py (for topocentric corrections)

Author:   ASTERID Project
Version:  2.0 (Full precision with nutation)
Date:     2026-07-06
"""

import sys
sys.dont_write_bytecode = True

import math
import numpy as np
from typing import Dict, Optional, Tuple, Union

# --------------------------------------------------------------------------
# Nomenclatures (consistent with Old_Java_Astronomy)
# --------------------------------------------------------------------------
NAKSHATRAS = [
    "Aswini", "Bharani", "Krittika", "Rohini", "Mrigasira", "Ardra",
    "Punarvasu", "Pushya", "Aslesha", "Magha", "Purva Phalguni",
    "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Visakha",
    "Anuradha", "Jyestha", "Mula", "Purva Ashadha", "Uttara Ashadha",
    "Sravana", "Dhanistha", "Satabhisha", "Purva Bhadrapada",
    "Uttara Bhadrapada", "Revati"
]

YOGAS = [
    "Vishkumbha", "Priti", "Ayushman", "Saubhagya", "Sobhana", "Atiganda",
    "Sukarma", "Dhriti", "Shula", "Ganda", "Vriddhi", "Dhruva",
    "Vyaghata", "Harshana", "Vajra", "Siddhi", "Vyatipata", "Variyan",
    "Parigha", "Shiva", "Siddha", "Sadhya", "Shubha", "Shukla",
    "Brahma", "Indra", "Vaidhriti"
]

# 60 Karaṇas: first one is Kimstughna, then 7 basic ones repeated 8 times,
# and the last three are Sakuni, Catuspada, Naga.
KARANA_BASE = ["Bava", "Balava", "Kaulava", "Taitila", "Gara", "Vanija", "Vishti"]
KARANA_60 = ["Kimstughna"] + (KARANA_BASE * 8) + ["Sakuni", "Catuspada", "Naga"]

# Zodiac signs (for Sayana and Nirayana Rasi)
ZODIAC = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"]

# --------------------------------------------------------------------------
# Angle normalisation
# --------------------------------------------------------------------------
def normalize_angle(deg: float) -> float:
    """Normalise an angle to the range [0, 360) degrees."""
    return deg % 360.0

def normalize_angle_rad(rad: float) -> float:
    """Normalise an angle to the range [0, 2π) radians."""
    return rad % (2.0 * math.pi)

# --------------------------------------------------------------------------
# Core Vedic calculations (pure logic, independent of ephemeris)
# --------------------------------------------------------------------------
def calculate_rasi(lon_deg: float) -> Dict:
    """
    Compute the zodiac sign (Rasi) from ecliptic longitude.

    Parameters
    ----------
    lon_deg : float
        Ecliptic longitude (degrees, 0–360).

    Returns
    -------
    dict
        Contains index (0‑based), name, and degrees within the sign.
    """
    idx = int(lon_deg // 30) % 12
    deg_in = lon_deg % 30.0
    return {
        "index": idx,
        "name": ZODIAC[idx],
        "degrees_in": deg_in
    }

def calculate_tithi(sun_lon_deg: float, moon_lon_deg: float) -> Dict:
    """
    Compute Tithi and Pakṣa from the longitudinal difference.

    Parameters
    ----------
    sun_lon_deg, moon_lon_deg : float
        Ecliptic longitudes (degrees, 0–360).

    Returns
    -------
    dict
        Contains tithi number (1–30), pakṣa (Sukla/Krsna),
        elongation (degrees), degrees within current tithi,
        and percentage progress.
    """
    delta = (moon_lon_deg - sun_lon_deg) % 360.0
    tithi_idx = int(delta // 12) + 1          # 1..30
    paksa = "Sukla" if delta < 180.0 else "Krsna"
    deg_in = delta % 12.0
    return {
        "tithi": tithi_idx,
        "paksa": paksa,
        "elongation": delta,
        "degrees_in_tithi": deg_in,
        "percent": (deg_in / 12.0) * 100.0
    }

def calculate_nakshatra(moon_lon_deg: float) -> Dict:
    """
    Compute Nakṣatra (27‑fold division) and its pada.

    Parameters
    ----------
    moon_lon_deg : float
        Ecliptic longitude of the Moon (degrees).

    Returns
    -------
    dict
        Index (0‑based), name, degrees within the nakṣatra, and pada (1‑4).
    """
    span = 360.0 / 27.0
    idx = int(moon_lon_deg // span) % 27
    name = NAKSHATRAS[idx]
    deg_in = moon_lon_deg % span
    pada = int(deg_in // (span / 4.0)) + 1
    return {
        "index": idx,
        "name": name,
        "degrees_in": deg_in,
        "pada": pada
    }

def calculate_yoga(sun_lon_deg: float, moon_lon_deg: float) -> Dict:
    """
    Compute Yoga as the sum of solar and lunar longitudes modulo 360°,
    divided into 27 equal parts.

    Parameters
    ----------
    sun_lon_deg, moon_lon_deg : float
        Ecliptic longitudes (degrees).

    Returns
    -------
    dict
        Index, name, total longitude, degrees within the yoga, progress %.
    """
    total = (sun_lon_deg + moon_lon_deg) % 360.0
    span = 360.0 / 27.0
    idx = int(total // span) % 27
    name = YOGAS[idx]
    deg_in = total % span
    return {
        "index": idx,
        "name": name,
        "total_longitude": total,
        "degrees_in": deg_in,
        "percent": (deg_in / span) * 100.0
    }

def calculate_karana(sun_lon_deg: float, moon_lon_deg: float) -> Dict:
    """
    Compute Karaṇa (60‑fold division) from the elongation.

    Parameters
    ----------
    sun_lon_deg, moon_lon_deg : float
        Ecliptic longitudes (degrees).

    Returns
    -------
    dict
        Karaṇa number (1‑60), name, and degrees within the karaṇa.
    """
    delta = (moon_lon_deg - sun_lon_deg) % 360.0
    karana_num = int(delta // 6) + 1
    if karana_num > 60:
        karana_num = 60
    name = KARANA_60[karana_num - 1]
    deg_in = delta % 6.0
    return {
        "karana_num": karana_num,
        "name": name,
        "degrees_in": deg_in
    }

def compute_vedic_components(
    sun_lon_tropical: float,
    moon_lon_tropical: float,
    ayanamsa_deg: Optional[float] = None
) -> Dict:
    """
    High‑level function that returns all Vedic components in either
    tropical or nirayana (sidereal) mode.

    Parameters
    ----------
    sun_lon_tropical, moon_lon_tropical : float
        True ecliptic longitudes of the Sun and Moon (degrees, 0–360).
    ayanamsa_deg : float, optional
        If provided, the longitudes are converted to nirayana by
        subtracting this value. If None, tropical results are returned.

    Returns
    -------
    dict
        Contains:
            - mode : 'tropical' or 'nirayana'
            - sun_longitude, moon_longitude (final used values)
            - tithi, nakshatra, yoga, karana, rasi (each as a dict)
    """
    if ayanamsa_deg is not None:
        sun_lon = normalize_angle(sun_lon_tropical - ayanamsa_deg)
        moon_lon = normalize_angle(moon_lon_tropical - ayanamsa_deg)
        mode = "nirayana"
    else:
        sun_lon = normalize_angle(sun_lon_tropical)
        moon_lon = normalize_angle(moon_lon_tropical)
        mode = "tropical"

    return {
        "mode": mode,
        "sun_longitude": sun_lon,
        "moon_longitude": moon_lon,
        "tithi": calculate_tithi(sun_lon, moon_lon),
        "nakshatra": calculate_nakshatra(moon_lon),
        "yoga": calculate_yoga(sun_lon, moon_lon),
        "karana": calculate_karana(sun_lon, moon_lon),
        "sun_rasi": calculate_rasi(sun_lon),
        "moon_rasi": calculate_rasi(moon_lon)
    }

def compute_vedic_both_modes(
    sun_lon_tropical: float,
    moon_lon_tropical: float,
    ayanamsa_deg: float
) -> Dict:
    """
    Compute Vedic components for both Sayana (tropical) and Nirayana (sidereal)
    modes simultaneously.

    Parameters
    ----------
    sun_lon_tropical, moon_lon_tropical : float
        True ecliptic longitudes of the Sun and Moon (degrees, 0–360).
    ayanamsa_deg : float
        Ayanamsa value in degrees (Lahiri or other).

    Returns
    -------
    dict
        Contains:
            - sayana : dict with all components in tropical mode
            - nirayana : dict with all components in sidereal mode
            - ayanamsa : the ayanamsa value used
            - sun_longitude_tropical, moon_longitude_tropical (raw inputs)
    """
    sayana = compute_vedic_components(sun_lon_tropical, moon_lon_tropical, ayanamsa_deg=None)
    nirayana = compute_vedic_components(sun_lon_tropical, moon_lon_tropical, ayanamsa_deg)

    return {
        "sayana": sayana,
        "nirayana": nirayana,
        "ayanamsa": ayanamsa_deg,
        "sun_longitude_tropical": sun_lon_tropical,
        "moon_longitude_tropical": moon_lon_tropical
    }

# --------------------------------------------------------------------------
# Ephemeris extraction from SM_VSOP2013 (with nutation)
# --------------------------------------------------------------------------
def ayanamsa_lahiri(jd_tt: float) -> float:
    """
    Lahiri ayanamsa (degrees) as a function of TT Julian date.
    This formula is derived from the IAU 2006 precession and is consistent
    with the VSOP2013 reference frame.
    """
    T = (jd_tt - 2451545.0) / 36525.0
    # Mean ayanamsa (J2000.0 to the equinox of date)
    # The 23.856858° is the value at J2000.0; the rate is 0.0139697128° per century.
    mean = 23.856858 + 0.013969712777777778 * T * 100.0
    return mean % 360.0

def get_vedic_from_ephemeris(
    tt_jd: float,
    ephem: Optional['AstronomicalEphemeris'] = None,
    use_ayanamsa: bool = True,
    ayanamsa_func = None
) -> Dict:
    """
    Retrieve the current true ecliptic longitudes of the Sun and Moon
    from the AstronomicalEphemeris (which already includes precession
    and nutation), then compute the Vedic components.

    If use_ayanamsa is True, returns a dictionary with both Sayana and
    Nirayana modes. If False, returns only Sayana (tropical).

    Parameters
    ----------
    tt_jd : float
        Julian date in TT (Terrestrial Time).
    ephem : AstronomicalEphemeris, optional
        An instance from `SM_VSOP2013`. If None, a new one is created.
    use_ayanamsa : bool
        If True, returns both Sayana and Nirayana (with Lahiri ayanamsa).
        If False, returns only Sayana (tropical).
    ayanamsa_func : callable, optional
        Function to compute ayanamsa from TT_JD. Defaults to `ayanamsa_lahiri`.

    Returns
    -------
    dict
        If use_ayanamsa is True:
            - sayana, nirayana, ayanamsa, sun_longitude_tropical, moon_longitude_tropical
        If use_ayanamsa is False:
            - mode: 'tropical', sun_longitude, moon_longitude, tithi, nakshatra, yoga, karana, rasi
    """
    if ephem is None:
        from SM_VSOP2013 import AstronomicalEphemeris
        ephem = AstronomicalEphemeris()

    # True ecliptic coordinates of date (including nutation)
    sun_lon, _, _ = ephem.sun_ecliptic_of_date(tt_jd, unit=False, degrees=True)
    moon_lon, _, _ = ephem.moon_ecliptic_of_date(tt_jd, unit=False, degrees=True)

    if use_ayanamsa:
        if ayanamsa_func is None:
            ayanamsa = ayanamsa_lahiri(tt_jd)
        else:
            ayanamsa = ayanamsa_func(tt_jd)

        result = compute_vedic_both_modes(sun_lon, moon_lon, ayanamsa)
        result["tt_jd"] = tt_jd
        return result
    else:
        result = compute_vedic_components(sun_lon, moon_lon, ayanamsa_deg=None)
        result["tt_jd"] = tt_jd
        result["ayanamsa"] = None
        return result

# --------------------------------------------------------------------------
# Modern solar ephemeris (ecliptic, equatorial, topocentric)
# --------------------------------------------------------------------------
def get_solar_ephemeris(
    tt_jd: float,
    lat_rad: float = 0.0,
    lon_rad: float = 0.0,
    height_m: float = 0.0,
    ephem: Optional['AstronomicalEphemeris'] = None,
    apply_refraction: bool = False
) -> Dict:
    """
    Compute the modern solar ephemeris in three frames:

        1. Ecliptic of date (true ecliptic and equinox of date)
        2. Equatorial / GCRS (astrometric, ICRS)
        3. Topocentric apparent (CIRS and Equinox‑based) with optional refraction

    Parameters
    ----------
    tt_jd : float
        Julian date in TT.
    lat_rad, lon_rad : float
        Observer geodetic latitude and longitude (radians).
    height_m : float
        Observer height above ellipsoid (meters).
    ephem : AstronomicalEphemeris, optional
    apply_refraction : bool
        If True, the topocentric altitude is corrected for atmospheric
        refraction using the VMF3+GPT3 model.

    Returns
    -------
    dict
        Contains:
            - ecliptic : dict with lon_deg, lat_deg, dist_km, dist_au
            - equatorial_gcrs : dict with ra_deg, dec_deg, dist_km
            - topocentric : dict with ra_cirs_deg, dec_cirs_deg,
                           ra_eqx_deg, dec_eqx_deg, az_deg, alt_geom_deg,
                           alt_app_deg, dist_km
    """
    if ephem is None:
        from SM_VSOP2013 import AstronomicalEphemeris
        ephem = AstronomicalEphemeris()

    # 1. Ecliptic of date
    sun_lon, sun_lat, sun_r = ephem.sun_ecliptic_of_date(tt_jd, unit=False, degrees=True)
    ecliptic = {
        "lon_deg": sun_lon,
        "lat_deg": sun_lat,
        "dist_km": sun_r,
        "dist_au": sun_r / 149597870.7
    }

    # 2. Equatorial GCRS (geocentric astrometric, without topocentric corrections)
    sun_gcrs = ephem.sun_geocentric_gcrs(tt_jd, unit=False)  # km
    from Coord_Transform import cartesian_to_spherical
    ra_gcrs, dec_gcrs, dist_km = cartesian_to_spherical(sun_gcrs)
    equatorial_gcrs = {
        "ra_deg": math.degrees(normalize_angle_rad(ra_gcrs)),
        "dec_deg": math.degrees(dec_gcrs),
        "dist_km": dist_km
    }

    # 3. Topocentric apparent (with light‑time, aberration, deflection, optional refraction)
    sun_top = ephem.sun_apparent_topocentric(
        tt_jd, lat_rad, lon_rad, height_m,
        apply_refraction=apply_refraction,
        refraction_model='vmf3' if apply_refraction else 'bennett'
    )
    topocentric = {
        "ra_cirs_deg": sun_top["ra_cirs_deg"],
        "dec_cirs_deg": sun_top["dec_cirs_deg"],
        "ra_eqx_deg": sun_top["ra_eqx_deg"],
        "dec_eqx_deg": sun_top["dec_eqx_deg"],
        "az_deg": sun_top["az_deg"],
        "alt_geom_deg": sun_top["alt_geom_deg"],
        "alt_app_deg": sun_top["alt_app_deg"],
        "dist_km": sun_top["dist_km"]
    }

    return {
        "ecliptic": ecliptic,
        "equatorial_gcrs": equatorial_gcrs,
        "topocentric": topocentric
    }

# --------------------------------------------------------------------------
# Example usage / self‑test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    from Timescales import J2000_JD
    from SM_VSOP2013 import AstronomicalEphemeris
    import math

    # Observing site (Jolotundo, as an example)
    lat_deg = -7.609444
    lon_deg = 112.595556
    height_m = 554.509
    lat_rad = math.radians(lat_deg)
    lon_rad = math.radians(lon_deg)

    tt = J2000_JD + 100.0   # some epoch

    print("=" * 80)
    print(" VEDIC CALCULATIONS WITH VSOP2013 + ELP/MPP02")
    print(f" TT JD = {tt:.6f}")
    print("=" * 80)

    # Create ephemeris object
    ephem = AstronomicalEphemeris()

    # --- Vedic components (both Sayana and Nirayana) ---
    vedic = get_vedic_from_ephemeris(tt, ephem, use_ayanamsa=True)
    print("\n[1] VEDIC COMPONENTS (SAYANA / TROPICAL)")
    say = vedic["sayana"]
    print(f"    Sun longitude        : {say['sun_longitude']:.6f}°")
    print(f"    Sun Rasi              : {say['sun_rasi']['name']} ({say['sun_rasi']['degrees_in']:.2f}°)")
    print(f"    Moon longitude        : {say['moon_longitude']:.6f}°")
    print(f"    Moon Rasi             : {say['moon_rasi']['name']} ({say['moon_rasi']['degrees_in']:.2f}°)")
    print(f"    Tithi                : {say['tithi']['tithi']} {say['tithi']['paksa']} ({say['tithi']['percent']:.1f}%)")
    print(f"    Nakṣatra             : {say['nakshatra']['name']} (pada {say['nakshatra']['pada']})")
    print(f"    Yoga                 : {say['yoga']['name']} ({say['yoga']['percent']:.1f}%)")
    print(f"    Karaṇa               : {say['karana']['name']} (ke-{say['karana']['karana_num']})")

    print("\n[2] VEDIC COMPONENTS (NIRAYANA / SIDEREAL)")
    nir = vedic["nirayana"]
    print(f"    Ayanamsa (Lahiri)    : {vedic['ayanamsa']:.6f}°")
    print(f"    Sun longitude        : {nir['sun_longitude']:.6f}°")
    print(f"    Sun Rasi              : {nir['sun_rasi']['name']} ({nir['sun_rasi']['degrees_in']:.2f}°)")
    print(f"    Moon longitude        : {nir['moon_longitude']:.6f}°")
    print(f"    Moon Rasi             : {nir['moon_rasi']['name']} ({nir['moon_rasi']['degrees_in']:.2f}°)")
    print(f"    Tithi                : {nir['tithi']['tithi']} {nir['tithi']['paksa']} ({nir['tithi']['percent']:.1f}%)")
    print(f"    Nakṣatra             : {nir['nakshatra']['name']} (pada {nir['nakshatra']['pada']})")
    print(f"    Yoga                 : {nir['yoga']['name']} ({nir['yoga']['percent']:.1f}%)")
    print(f"    Karaṇa               : {nir['karana']['name']} (ke-{nir['karana']['karana_num']})")

    # --- Modern solar ephemeris ---
    solar = get_solar_ephemeris(tt, lat_rad, lon_rad, height_m, ephem, apply_refraction=True)
    print("\n[3] MODERN SOLAR EPHEMERIS")
    print("    Ecliptic of date:")
    print(f"        Longitude : {solar['ecliptic']['lon_deg']:.6f}°")
    print(f"        Latitude  : {solar['ecliptic']['lat_deg']:.6f}°")
    print(f"        Distance  : {solar['ecliptic']['dist_km']:.3f} km")
    print("    Equatorial (GCRS, astrometric):")
    print(f"        RA        : {solar['equatorial_gcrs']['ra_deg']:.6f}°")
    print(f"        Dec       : {solar['equatorial_gcrs']['dec_deg']:.6f}°")
    print(f"        Distance  : {solar['equatorial_gcrs']['dist_km']:.3f} km")
    print("    Topocentric apparent (CIRS & Equinox):")
    print(f"        RA (CIRS)      : {solar['topocentric']['ra_cirs_deg']:.6f}°")
    print(f"        Dec (CIRS)     : {solar['topocentric']['dec_cirs_deg']:.6f}°")
    print(f"        RA (Equinox)   : {solar['topocentric']['ra_eqx_deg']:.6f}°")
    print(f"        Dec (Equinox)  : {solar['topocentric']['dec_eqx_deg']:.6f}°")
    print(f"        Azimuth        : {solar['topocentric']['az_deg']:.6f}°")
    print(f"        Elevation (geom.): {solar['topocentric']['alt_geom_deg']:.6f}°")
    print(f"        Elevation (app.): {solar['topocentric']['alt_app_deg']:.6f}°")
    print(f"        Distance       : {solar['topocentric']['dist_km']:.3f} km")
    print("=" * 80)