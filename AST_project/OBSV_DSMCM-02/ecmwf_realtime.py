#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecmwf_realtime.py — High-Resolution Real-Time ECMWF IFS HRES Atmospheric Retrieval
==================================================================================
This module implements a robust, production‑grade interface to the ECMWF
IFS HRES (9 km native resolution) via the Open‑Meteo /v1/ecmwf endpoint.

It is specifically optimised for a fixed geographic location (Jolotundo/
Pawitra Observatory, East Java) and incorporates:
    • Smart hourly‑based persistent caching to eliminate redundant network
      requests within the same forecast hour.
    • Exponential backoff retry logic (4 attempts) to mitigate intermittent
      connectivity in remote or mountainous terrain.
    • Automatic cache lifecycle management — old cache files are purged
      only after a new successful retrieval, ensuring that prior valid data
      remains available if a network request fails at the hour boundary.
    • Cache expiry of 6 hours, respecting the 6‑hour ECMWF update cycle.
    • Physical extrapolation of surface parameters (P, T, e, Td) to the
      target orthometric height using the ICAO standard atmosphere and a
      water‑vapour scale height of 2000 m.

Data Source:
    Open‑Meteo ECMWF API (IFS HRES, O1280 reduced Gaussian grid, ~9 km)
    https://open-meteo.com/en/docs/ecmwf-api

Dependencies:
    urllib, json, ssl, os, time, pandas, numpy, math

Author:
    ASTERID Consortium — Jolotundo Research Observatory

Version:
    2.1 (2026-07-09) — Added 6‑hour cache expiry and offline fallback
"""

import urllib.request
import urllib.parse
import json
import math
import ssl
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any

# =============================================================================
# SSL CONTEXT FOR ANDROID / PYDROID3 COMPATIBILITY
# =============================================================================
# An unverified SSL context is required to circumvent handshake timeouts
# encountered in the Android/Pydroid3 execution environment.
_SSL_CONTEXT = ssl._create_unverified_context()

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
G = 9.80665          # Standard gravity (m·s⁻²)
R_DRY = 287.05       # Specific gas constant for dry air (J·kg⁻¹·K⁻¹)
LAPSE_RATE = 0.0065  # ICAO standard lapse rate (K·m⁻¹)

# =============================================================================
# AUXILIARY FUNCTIONS
# =============================================================================

def _vapor_pressure_from_dewpoint(Td_c: float) -> float:
    """
    Compute saturation vapour pressure (hPa) from dew‑point temperature (°C)
    using the Magnus formula.

    Parameters
    ----------
    Td_c : float
        Dew‑point temperature in degrees Celsius.

    Returns
    -------
    float
        Saturation vapour pressure in hPa.
    """
    return 6.112 * math.exp(17.67 * Td_c / (Td_c + 243.5))


def _interpolate_to_height(df: pd.DataFrame, target_height_m: float,
                           h_surface_m: float, utc_time: Optional[datetime] = None) -> Dict[str, float]:
    """
    Extrapolate surface‑level meteorological parameters to the target
    orthometric height using physically based vertical profiles.

    The extrapolation follows:
        1. Temperature: ICAO standard lapse rate (6.5 K·km⁻¹).
        2. Pressure: hydrostatic equation with mean temperature.
        3. Vapour pressure: exponential decay with a water‑vapour scale
           height of 2000 m.
        4. Dew point: derived from the extrapolated vapour pressure via
           the inverse Magnus formula.

    Parameters
    ----------
    df : pandas.DataFrame
        Hourly data from the Open‑Meteo response, containing at least the
        columns 'time', 'temperature_2m', 'dewpoint_2m', and
        'surface_pressure'.
    target_height_m : float
        Orthometric height (m) of the observation site.
    h_surface_m : float
        Model orography (m) at the grid point.
    utc_time : datetime, optional
        Reference UTC time; if provided, the nearest hour is selected.

    Returns
    -------
    dict
        Extrapolated parameters: p, T, Td, e, rh_2m, cloud_cover,
        wind_speed_10m, wind_direction_10m, precipitation, h_surface,
        p_surface, T_surface.
    """
    if not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])

    if utc_time is not None:
        target = utc_time.replace(minute=0, second=0, microsecond=0)
        if target.tzinfo is not None:
            target = target.replace(tzinfo=None)
        df['diff'] = abs(df['time'] - target)
        row = df.loc[df['diff'].idxmin()]
    else:
        row = df.mean(numeric_only=True)

    T_surface_c = row['temperature_2m']
    Td_surface_c = row['dewpoint_2m']
    p_surface_hpa = row['surface_pressure']

    # Optional parameters (may be absent in some responses)
    cloud_cover = row.get('cloud_cover', 0.0)
    wind_speed = row.get('wind_speed_10m', 0.0)
    wind_dir = row.get('wind_direction_10m', 0.0)
    precip = row.get('precipitation', 0.0)
    rh_2m = row.get('relative_humidity_2m', 0.0)

    # ---- 1. Temperature (ICAO lapse rate) ----
    T_surface_K = T_surface_c + 273.15
    delta_h = target_height_m - h_surface_m
    T_target_K = T_surface_K - LAPSE_RATE * delta_h
    T_target_c = T_target_K - 273.15
    T_mean_K = (T_surface_K + T_target_K) / 2.0

    # ---- 2. Pressure (hydrostatic equation) ----
    if abs(delta_h) < 1e-6:
        p_target_hpa = p_surface_hpa
    else:
        exponent = -G * delta_h / (R_DRY * T_mean_K)
        p_target_hpa = p_surface_hpa * math.exp(exponent)

    # ---- 3. Vapour pressure and dew point (exponential decay) ----
    H_w = 2000.0  # water‑vapour scale height (m)
    e_surface = _vapor_pressure_from_dewpoint(Td_surface_c)
    e_target_hpa = e_surface * math.exp(-delta_h / H_w)

    try:
        ln_e = math.log(e_target_hpa / 6.112)
        Td_target_c = (243.5 * ln_e) / (17.67 - ln_e)
    except Exception:
        # Fallback: linear dew‑point gradient (~1.8 K·km⁻¹)
        Td_target_c = Td_surface_c - (0.0018 * delta_h)

    return {
        'p': p_target_hpa,
        'T': T_target_c,
        'Td': Td_target_c,
        'e': e_target_hpa,
        'rh_2m': rh_2m,
        'cloud_cover': cloud_cover,
        'wind_speed_10m': wind_speed,
        'wind_direction_10m': wind_dir,
        'precipitation': precip,
        'h_surface': h_surface_m,
        'p_surface': p_surface_hpa,
        'T_surface': T_surface_c,
    }


def _parse_ecmwf_response(raw_data: dict, target_height_m: float,
                          time_requested: datetime) -> Optional[Dict[str, Any]]:
    """
    Parse the raw JSON response from Open‑Meteo and apply vertical
    extrapolation to the target height.

    Parameters
    ----------
    raw_data : dict
        The JSON dictionary returned by the Open‑Meteo API.
    target_height_m : float
        Orthometric height (m) of the observation site.
    time_requested : datetime
        The UTC time for which the data was requested.

    Returns
    -------
    dict or None
        A dictionary containing the extrapolated parameters and metadata,
        or None if the response lacks the 'hourly' key.
    """
    if 'hourly' not in raw_data:
        return None

    h_surface_m = raw_data.get('elevation', 0.0)
    df = pd.DataFrame(raw_data['hourly'])
    df['time'] = pd.to_datetime(df['time'])

    # Extrapolate to the target height
    # FIX: Pass time_requested instead of None to avoid daily mean aggregation
    result = _interpolate_to_height(df, target_height_m, h_surface_m, time_requested)

    # Attach metadata
    result['metadata'] = {
        'source': 'ECMWF IFS HRES (9 km)',
        'time_requested': time_requested.isoformat(),
        'latitude': raw_data.get('latitude', 0.0),
        'longitude': raw_data.get('longitude', 0.0),
        'target_height_m': target_height_m,
        'data_points': len(df),
        'model_resolution': '9 km (O1280)',
        'time_used': df['time'].iloc[0].isoformat() if len(df) > 0 else None,
    }
    result['raw_df'] = df

    return result


# =============================================================================
# PRIMARY RETRIEVAL FUNCTION WITH ROBUST CACHING
# =============================================================================

def get_ecmwf_at_point(
    lat: float,
    lon: float,
    height_m: float,
    utc_time: Optional[datetime] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve ECMWF IFS HRES 9 km atmospheric data for a fixed geographic point
    using a smart caching layer with exponential backoff retries.

    The caching strategy is designed for operational resilience in
    connectivity‑challenged environments:
        • The request time is rounded to the current hour to form a unique
          cache identifier (e.g., 20260709_1200).
        • If a valid cache file for that exact hour already exists and is
          younger than 6 hours, it is returned immediately — no network request.
        • If no valid cache exists, the module performs a request with exponential
          backoff (2, 4, 8, 16 s) up to 4 attempts.
        • Cache files older than 6 hours are considered expired and discarded,
          aligning with the 6‑hour ECMWF model update cycle.
        • If network fails, the module scans for any remaining cache file
          and uses the most recent one if it is still fresh (≤ 6 hours).
        • Old cache files are purged after a successful new retrieval.

    Parameters
    ----------
    lat, lon : float
        Geodetic coordinates in decimal degrees.
    height_m : float
        Orthometric height (m) of the observation site.
    utc_time : datetime, optional
        UTC time for the retrieval. If None, the current UTC time is used.

    Returns
    -------
    dict or None
        A dictionary containing extrapolated atmospheric parameters and
        metadata, or None if all retrieval attempts fail.
    """
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)

    # ---- 1. Hourly cache identifier ----
    epoch_id = utc_time.strftime("%Y%m%d_%H00")
    cache_file = f"ecmwf_cache_{epoch_id}.json"

    # ---- 2. Check existing cache for this exact hour ----
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            if "hourly" in cached_data:
                cache_mtime = os.path.getmtime(cache_file)
                cache_age_hours = (time.time() - cache_mtime) / 3600.0
                # Only use if younger than 6 hours (1 ECMWF cycle)
                if cache_age_hours <= 6.0:
                    return _parse_ecmwf_response(cached_data, height_m, utc_time)
        except Exception:
            pass  # Cache corrupted or expired; proceed to network request

    # ---- 3. Network request with exponential backoff ----
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "dewpoint_2m",
            "surface_pressure",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
            "precipitation",
            "relative_humidity_2m"
        ],
        "start_date": utc_time.strftime("%Y-%m-%d"),
        "end_date": utc_time.strftime("%Y-%m-%d"),
        "timezone": "UTC"
        # past_days deliberately omitted: we only need the current epoch
    }

    url = "https://api.open-meteo.com/v1/ecmwf"
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}"

    max_retries = 4
    raw_data = None

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(full_url, headers={'User-Agent': 'AsteridGeodetic/3.0'})
            with urllib.request.urlopen(req, timeout=20, context=_SSL_CONTEXT) as resp:
                raw_data = json.loads(resp.read().decode())
            break
        except Exception:
            if attempt == max_retries - 1:
                break
            time.sleep(2 ** (attempt + 1))

    # ---- 4. If network succeeded, save cache and return ----
    if raw_data is not None:
        try:
            with open(cache_file, 'w') as f:
                json.dump(raw_data, f)
            # Clean up old cache files (optional)
            for f in os.listdir("."):
                if f.startswith("ecmwf_cache_") and f.endswith(".json") and f != cache_file:
                    os.remove(f)
        except Exception:
            pass
        return _parse_ecmwf_response(raw_data, height_m, utc_time)

    # ---- 5. OFFLINE FALLBACK: scan for any fresh cache file ----
    import glob
    existing_caches = glob.glob("ecmwf_cache_*.json")
    if existing_caches:
        # Pick the most recently modified cache file
        latest_cache = max(existing_caches, key=os.path.getmtime)
        cache_mtime = os.path.getmtime(latest_cache)
        cache_age_hours = (time.time() - cache_mtime) / 3600.0

        # Enforce the 6‑hour meteorological expiry threshold
        if cache_age_hours <= 6.0:
            try:
                with open(latest_cache, 'r') as f:
                    cached_data = json.load(f)
                if "hourly" in cached_data:
                    return _parse_ecmwf_response(cached_data, height_m, utc_time)
            except Exception:
                pass

    # ---- 6. Total failure: return None (triggers GPT3 fallback upstream) ----
    return None


# =============================================================================
# SELF‑TEST / DIAGNOSTIC REPORT
# =============================================================================

if __name__ == "__main__":
    import sys
    from datetime import datetime

    # Display formatting constants
    W = 72
    THICK_SEP = "═" * W
    THIN_SEP  = "─" * W
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Jolotundo Observatory coordinates
    LAT = -7.609444
    LON = 112.595556
    
    # ------------------------------------------------------------------
    # KETINGGIAN ORTHOMETRIK JOLOTUNDO (VALIDASI TERBARU)
    # ------------------------------------------------------------------
    # Sumber: Copernicus DEM (COP30) resolusi 30 m pada koordinat
    #         -7.609444°, 112.595556° → elevasi ortometrik = 554.509 m.
    # Undulasi geoid dari XGM2019e-2159 (tide-free) = 28.8464 m.
    # Maka elevasi ellipsoidal WGS84 = 583.355 m.
    HEIGHT = 554.509  # orthometric height (m)

    print(f"\n{BOLD}{THICK_SEP}{RESET}")
    print(f"{BOLD}{'JOLOTUNDO OBSV – ECMWF IFS HRES (9 km) METEOROLOGICAL REPORT'.center(W)}{RESET}")
    print(f"{BOLD}{'REAL‑TIME ATMOSPHERIC PROFILE WITH SMART HOURLY CACHE'.center(W)}{RESET}")
    print(f"{BOLD}{THICK_SEP}{RESET}")

    try:
        result = get_ecmwf_at_point(LAT, LON, HEIGHT)

        if result is None:
            print("\n  ❌ ECMWF data retrieval failed — falling back to GPT3 climatology.")
            sys.exit(0)

        md = result['metadata']

        def section(title: str) -> None:
            print(f"\n{BOLD}{title}{RESET}")
            print(f"{BOLD}{THIN_SEP}{RESET}")

        def print_line(label: str, value: str) -> None:
            print(f"  {label:<30}: {value}")

        section("[1] STATION METADATA")
        print_line("Station", "Jolotundo OBSV")
        print_line("Latitude", f"{md['latitude']:+.6f}°")
        print_line("Longitude", f"{md['longitude']:+.6f}°")
        print_line("Target Height", f"{md['target_height_m']:.1f} m")
        print_line("Model Elevation", f"{result['h_surface']:.1f} m")
        print_line("Model Resolution", md.get('model_resolution', '9 km'))
        print_line("Data Points", f"{md['data_points']} hourly")

        section("[2] EXTRAPOLATED PROFILE (TARGET HEIGHT)")
        print_line("Pressure (p)", f"{result['p']:.2f} hPa")
        print_line("Temperature (T)", f"{result['T']:.2f} °C")
        print_line("Dew Point (Td)", f"{result['Td']:.2f} °C")
        print_line("Vapour Pressure (e)", f"{result['e']:.2f} hPa")

        section("[3] SURFACE & BOUNDARY LAYER PARAMETERS")
        print_line("Relative Humidity", f"{result['rh_2m']:.1f} %")
        print_line("Cloud Cover", f"{result['cloud_cover']:.1f} %")
        print_line("Precipitation", f"{result['precipitation']:.2f} mm·h⁻¹")
        print_line("Wind Speed (10 m)", f"{result['wind_speed_10m']:.1f} km·h⁻¹")
        print_line("Wind Direction", f"{result['wind_direction_10m']:.0f}° (True North)")

        section("[4] MODEL SURFACE REFERENCE & CACHE STATUS")
        print_line("Surface Pressure", f"{result['p_surface']:.2f} hPa")
        print_line("Surface Temperature", f"{result['T_surface']:.2f} °C")
        # Derive cache filename from metadata
        time_str = md['time_requested']
        cache_id = datetime.fromisoformat(time_str).strftime("%Y%m%d_%H00")
        print_line("Cache File", f"ecmwf_cache_{cache_id}.json")
        print_line("Data Source", md['source'])

        print(f"\n{BOLD}{THICK_SEP}{RESET}")
        print(f"{BOLD}{'REPORT GENERATED SUCCESSFULLY – EXITING.'.center(W)}{RESET}")
        print(f"{BOLD}{THICK_SEP}{RESET}\n")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

