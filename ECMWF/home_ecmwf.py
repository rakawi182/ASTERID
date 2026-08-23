#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ecmwf_meteo.py — Standalone ECMWF IFS HRES 9 km Weather Reporter
=================================================================
Fully independent module to retrieve real-time meteorological data
from Open-Meteo ECMWF API. Uses verified parameters and handles
optional fields gracefully. Cache valid for 6 hours.

Author: ASTERID Consortium — Home Observatory
Version: 4.1 (2026-08-01)
"""

import sys
import json
import math
import os
import time
import csv
import ssl
import urllib.request
import urllib.parse
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
G_STANDARD = 9.80665
R_DRY_AIR = 287.05
LAPSE_RATE_ICAO = 0.0065
H_SCALE_WATER = 2000.0
MAGNUS_A = 17.67
MAGNUS_B = 243.5

_SSL_CTX = ssl._create_unverified_context()

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def _vapour_pressure_from_dewpoint(Td_c: float) -> float:
    return 6.112 * math.exp(MAGNUS_A * Td_c / (Td_c + MAGNUS_B))

def _dewpoint_from_vapour_pressure(e_hpa: float) -> float:
    if e_hpa <= 0.0:
        return -273.15
    ln_ratio = math.log(e_hpa / 6.112)
    return (MAGNUS_B * ln_ratio) / (MAGNUS_A - ln_ratio)

def _extrapolate_to_target_height(T_surface_c, Td_surface_c, P_surface_hpa,
                                  h_surface_m, h_target_m, rh_surface_pct=0.0):
    delta_h = h_target_m - h_surface_m
    T_surface_K = T_surface_c + 273.15
    T_target_K = T_surface_K - LAPSE_RATE_ICAO * delta_h
    T_target_c = T_target_K - 273.15
    T_mean_K = (T_surface_K + T_target_K) / 2.0

    if abs(delta_h) < 1e-6:
        P_target_hpa = P_surface_hpa
    else:
        exponent = -G_STANDARD * delta_h / (R_DRY_AIR * T_mean_K)
        P_target_hpa = P_surface_hpa * math.exp(exponent)

    e_surface_hpa = _vapour_pressure_from_dewpoint(Td_surface_c)
    e_target_hpa = e_surface_hpa * math.exp(-delta_h / H_SCALE_WATER)
    e_target_hpa = max(0.01, min(120.0, e_target_hpa))
    Td_target_c = _dewpoint_from_vapour_pressure(e_target_hpa)

    if T_target_c > -40.0:
        e_sat_target = _vapour_pressure_from_dewpoint(T_target_c)
        rh_target = (e_target_hpa / e_sat_target) * 100.0 if e_sat_target > 0 else 0.0
    else:
        rh_target = 0.0

    return {
        'T_c': T_target_c,
        'Td_c': Td_target_c,
        'P_hpa': P_target_hpa,
        'e_hpa': e_target_hpa,
        'RH_pct': min(100.0, max(0.0, rh_target)),
        'T_surface_c': T_surface_c,
        'Td_surface_c': Td_surface_c,
        'P_surface_hpa': P_surface_hpa,
        'h_surface_m': h_surface_m,
        'delta_h_m': delta_h,
    }

def _find_nearest_hour_index(time_list: List[str], target_utc: datetime) -> int:
    target_ts = target_utc.timestamp()
    best_idx = 0
    best_diff = float('inf')
    for i, t_str in enumerate(time_list):
        clean_t_str = t_str.replace('Z', '')
        dt = datetime.fromisoformat(clean_t_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        diff = abs(dt.timestamp() - target_ts)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx

def _wind_direction_name(deg: float) -> str:
    """Convert wind direction in degrees to a 16-point compass name."""
    deg = deg % 360.0
    # 16-point compass
    names = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int(round(deg / 22.5)) % 16
    return names[idx]

def _wind_to_direction(deg_from: float) -> float:
    """Return the direction toward which the wind is blowing (deg + 180 mod 360)."""
    return (deg_from + 180.0) % 360.0

# -----------------------------------------------------------------------------
# CORE API REQUEST
# -----------------------------------------------------------------------------
def _fetch_from_openmeteo(lat: float, lon: float, start_date: str, end_date: str) -> Optional[Dict[str, Any]]:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "surface_pressure",
            "pressure_msl",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "precipitation",
            "shortwave_radiation"
        ],
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "UTC",
    }

    url = "https://api.open-meteo.com/v1/ecmwf"
    query = urllib.parse.urlencode(params, doseq=True)
    full_url = f"{url}?{query}"

    max_retries = 4
    backoff = [2, 4, 8, 16]

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(full_url, headers={'User-Agent': 'AsteridMeteo/4.1'})
            with urllib.request.urlopen(req, timeout=25, context=_SSL_CTX) as resp:
                raw = json.loads(resp.read().decode('utf-8'))
                if "hourly" in raw:
                    return raw
                return None
        except urllib.error.HTTPError as e:
            print(f"⚠️ HTTP error {e.code}: {e.reason}")
            try:
                error_body = e.read().decode()
                print(f"   Response: {error_body[:200]}")
            except:
                pass
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff[attempt])
        except Exception as e:
            print(f"⚠️ Request error: {e}")
            if attempt == max_retries - 1:
                return None
            time.sleep(backoff[attempt])
    return None

# -----------------------------------------------------------------------------
# MAIN RETRIEVAL FUNCTION
# -----------------------------------------------------------------------------
def get_ecmwf_meteo(
    lat: float,
    lon: float,
    target_height_m: float,
    utc_time: Optional[datetime] = None,
    cache_dir: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    if cache_dir is None:
        cache_dir = os.getcwd()
    os.makedirs(cache_dir, exist_ok=True)

    hour_id = utc_time.strftime("%Y%m%d_%H00")
    cache_file = os.path.join(cache_dir, f"ecmwf_cache_{hour_id}.json")

    # ---- 1. Check cache ----
    if os.path.isfile(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            if "hourly" in cached:
                mtime = os.path.getmtime(cache_file)
                age_hours = (time.time() - mtime) / 3600.0
                if age_hours <= 6.0:
                    return _parse_and_extrapolate(cached, target_height_m, utc_time)
        except Exception as e:
            print(f"⚠️ Cache read error: {e}")

    # ---- 2. Network request ----
    date_str = utc_time.strftime("%Y-%m-%d")
    raw = _fetch_from_openmeteo(lat, lon, date_str, date_str)

    if raw is not None:
        try:
            with open(cache_file, 'w') as f:
                json.dump(raw, f)
            # Keep only 10 most recent caches
            all_caches = sorted(
                [f for f in os.listdir(cache_dir) if f.startswith("ecmwf_cache_") and f.endswith(".json")],
                key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)),
                reverse=True
            )
            for old_file in all_caches[10:]:
                os.remove(os.path.join(cache_dir, old_file))
        except Exception:
            pass
        return _parse_and_extrapolate(raw, target_height_m, utc_time)

    # ---- 3. Offline fallback ----
    all_caches = sorted(
        [f for f in os.listdir(cache_dir) if f.startswith("ecmwf_cache_") and f.endswith(".json")],
        key=lambda x: os.path.getmtime(os.path.join(cache_dir, x)),
        reverse=True
    )
    for fname in all_caches:
        fpath = os.path.join(cache_dir, fname)
        try:
            mtime = os.path.getmtime(fpath)
            age_hours = (time.time() - mtime) / 3600.0
            if age_hours <= 6.0:
                with open(fpath, 'r') as f:
                    cached = json.load(f)
                if "hourly" in cached:
                    return _parse_and_extrapolate(cached, target_height_m, utc_time)
        except Exception:
            continue

    return None

# -----------------------------------------------------------------------------
# PARSE AND EXTRAPOLATE
# -----------------------------------------------------------------------------
def _parse_and_extrapolate(raw: Dict[str, Any], target_height_m: float, utc_time: datetime) -> Dict[str, Any]:
    hourly = raw['hourly']
    time_list = hourly['time']
    idx = _find_nearest_hour_index(time_list, utc_time)

    T_surface = hourly['temperature_2m'][idx]
    RH_surface = hourly['relative_humidity_2m'][idx]
    Td_surface = hourly['dewpoint_2m'][idx]
    P_surface = hourly['surface_pressure'][idx]
    wind_speed = hourly['wind_speed_10m'][idx]
    wind_dir = hourly['wind_direction_10m'][idx]
    precip = hourly['precipitation'][idx]
    cloud_total = hourly['cloud_cover'][idx]

    P_msl = hourly.get('pressure_msl', [None])[idx] if 'pressure_msl' in hourly else None
    wind_gust = hourly.get('wind_gusts_10m', [None])[idx] if 'wind_gusts_10m' in hourly else None
    cloud_low = hourly.get('cloud_cover_low', [None])[idx] if 'cloud_cover_low' in hourly else None
    cloud_mid = hourly.get('cloud_cover_mid', [None])[idx] if 'cloud_cover_mid' in hourly else None
    cloud_high = hourly.get('cloud_cover_high', [None])[idx] if 'cloud_cover_high' in hourly else None
    sw_rad = hourly.get('shortwave_radiation', [None])[idx] if 'shortwave_radiation' in hourly else None

    h_surface = raw.get('elevation', 0.0)

    extrap = _extrapolate_to_target_height(
        T_surface, Td_surface, P_surface, h_surface, target_height_m, RH_surface
    )

    result = {
        'temperature_c': extrap['T_c'],
        'dewpoint_c': extrap['Td_c'],
        'pressure_hpa': extrap['P_hpa'],
        'vapour_pressure_hpa': extrap['e_hpa'],
        'relative_humidity_pct': extrap['RH_pct'],
        'surface_temperature_c': extrap['T_surface_c'],
        'surface_dewpoint_c': extrap['Td_surface_c'],
        'surface_pressure_hpa': extrap['P_surface_hpa'],
        'model_orography_m': extrap['h_surface_m'],
        'height_delta_m': extrap['delta_h_m'],
        'wind_speed_kmh': wind_speed,
        'wind_direction_deg': wind_dir,      # direction FROM
        'wind_direction_to_deg': _wind_to_direction(wind_dir),  # direction TO
        'precipitation_mm': precip,
        'cloud_cover_total_pct': cloud_total,
        'pressure_msl_hpa': P_msl,
        'wind_gust_kmh': wind_gust,
        'cloud_cover_low_pct': cloud_low,
        'cloud_cover_mid_pct': cloud_mid,
        'cloud_cover_high_pct': cloud_high,
        'shortwave_radiation_wm2': sw_rad,
        'metadata': {
            'source': 'ECMWF IFS HRES (O1280, 9 km)',
            'latitude': raw.get('latitude', 0.0),
            'longitude': raw.get('longitude', 0.0),
            'target_height_m': target_height_m,
            'forecast_hour_utc': time_list[idx],
            'data_points': len(time_list),
            'resolution': '9 km (reduced Gaussian grid)',
        }
    }
    return result

# -----------------------------------------------------------------------------
# REPORT GENERATION (enhanced with wind "to" direction)
# -----------------------------------------------------------------------------
def generate_meteo_report(lat: float, lon: float, height_m: float, cache_dir: Optional[str] = None) -> None:
    data = get_ecmwf_meteo(lat, lon, height_m, cache_dir=cache_dir)
    if data is None:
        print("❌ ECMWF data retrieval failed. Please check network connectivity or cache integrity.", file=sys.stderr)
        return

    md = data['metadata']
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    W = 72
    BOLD = '\033[1m'
    RESET = '\033[0m'
    THICK = "═" * W
    THIN = "─" * W

    # Wind direction information
    wind_from_deg = data['wind_direction_deg']
    wind_to_deg = data['wind_direction_to_deg']
    from_name = _wind_direction_name(wind_from_deg)
    to_name = _wind_direction_name(wind_to_deg)

    print(f"\n{BOLD}{THICK}{RESET}")
    print(f"{BOLD}  ECMWF IFS HRES (9 km) — REAL-TIME ATMOSPHERIC STATE{RESET}")
    print(f"{BOLD}  Global Numerical Weather Prediction — O1280 Gaussian Grid{RESET}")
    print(f"{BOLD}{THICK}{RESET}")
    print(f"  {'Station Coordinates':<30}: {lat:+.6f}°, {lon:+.6f}°")
    print(f"  {'Target Orthometric Height':<30}: {height_m:.1f} m")
    print(f"  {'Model Orography':<30}: {data['model_orography_m']:.1f} m")
    print(f"  {'Report Generated (UTC)':<30}: {now_utc}")
    print(f"  {'Forecast Hour (UTC)':<30}: {md['forecast_hour_utc']}")
    print(f"  {'Source Model':<30}: {md['source']}")
    print(f"{BOLD}{THIN}{RESET}")
    print(f"  {'Surface Temperature (2 m)':<30}: {data['temperature_c']:>7.2f} °C")
    print(f"  {'Dew Point (2 m)':<30}: {data['dewpoint_c']:>7.2f} °C")
    print(f"  {'Relative Humidity (2 m)':<30}: {data['relative_humidity_pct']:>7.1f} %")
    print(f"  {'Vapour Pressure':<30}: {data['vapour_pressure_hpa']:>7.2f} hPa")
    print(f"  {'Surface Pressure (target)':<30}: {data['pressure_hpa']:>7.2f} hPa")
    print(f"  {'Surface Pressure (model)':<30}: {data['surface_pressure_hpa']:>7.2f} hPa")
    if data['pressure_msl_hpa'] is not None:
        print(f"  {'Mean Sea Level Pressure':<30}: {data['pressure_msl_hpa']:>7.2f} hPa")
    print(f"{BOLD}{THIN}{RESET}")
    print(f"  {'Wind Speed (10 m)':<30}: {data['wind_speed_kmh']:>7.1f} km·h⁻¹")
    print(f"  {'Wind Direction (from)':<30}: {wind_from_deg:>7.0f}° (True North)  [{from_name}]")
    print(f"  {'Wind Direction (toward)':<30}: {wind_to_deg:>7.0f}° (True North)  [{to_name}]")
    if data['wind_gust_kmh'] is not None:
        print(f"  {'Wind Gust (10 m)':<30}: {data['wind_gust_kmh']:>7.1f} km·h⁻¹ (max)")
    print(f"{BOLD}{THIN}{RESET}")
    print(f"  {'Total Cloud Cover':<30}: {data['cloud_cover_total_pct']:>7.1f} %")
    if data['cloud_cover_low_pct'] is not None:
        print(f"  {'Low-level Clouds':<30}: {data['cloud_cover_low_pct']:>7.1f} %")
        print(f"  {'Mid-level Clouds':<30}: {data['cloud_cover_mid_pct']:>7.1f} %")
        print(f"  {'High-level Clouds':<30}: {data['cloud_cover_high_pct']:>7.1f} %")
    print(f"{BOLD}{THIN}{RESET}")
    print(f"  {'Precipitation (hourly)':<30}: {data['precipitation_mm']:>7.2f} mm")
    if data['shortwave_radiation_wm2'] is not None:
        print(f"  {'Shortwave Radiation':<30}: {data['shortwave_radiation_wm2']:>7.1f} W·m⁻²")
    print(f"{BOLD}{THICK}{RESET}\n")

# -----------------------------------------------------------------------------
# CSV EXPORT
# -----------------------------------------------------------------------------
def export_meteo_csv(lat: float, lon: float, height_m: float,
                     filename: str = "ecmwf_meteo_log.csv",
                     cache_dir: Optional[str] = None) -> None:
    data = get_ecmwf_meteo(lat, lon, height_m, cache_dir=cache_dir)
    if data is None:
        print("❌ No data to export.", file=sys.stderr)
        return

    fieldnames = [
        'timestamp_utc', 'lat', 'lon', 'target_height_m',
        'temperature_c', 'dewpoint_c', 'relative_humidity_pct',
        'vapour_pressure_hpa', 'pressure_hpa', 'surface_pressure_hpa', 'pressure_msl_hpa',
        'wind_speed_kmh', 'wind_direction_deg', 'wind_direction_to_deg',
        'wind_gust_kmh', 'precipitation_mm', 'cloud_cover_total_pct',
        'cloud_cover_low_pct', 'cloud_cover_mid_pct', 'cloud_cover_high_pct',
        'shortwave_radiation_wm2', 'model_orography_m'
    ]
    row = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'lat': lat,
        'lon': lon,
        'target_height_m': height_m,
        'temperature_c': data['temperature_c'],
        'dewpoint_c': data['dewpoint_c'],
        'relative_humidity_pct': data['relative_humidity_pct'],
        'vapour_pressure_hpa': data['vapour_pressure_hpa'],
        'pressure_hpa': data['pressure_hpa'],
        'surface_pressure_hpa': data['surface_pressure_hpa'],
        'pressure_msl_hpa': data['pressure_msl_hpa'],
        'wind_speed_kmh': data['wind_speed_kmh'],
        'wind_direction_deg': data['wind_direction_deg'],
        'wind_direction_to_deg': data['wind_direction_to_deg'],
        'wind_gust_kmh': data['wind_gust_kmh'],
        'precipitation_mm': data['precipitation_mm'],
        'cloud_cover_total_pct': data['cloud_cover_total_pct'],
        'cloud_cover_low_pct': data['cloud_cover_low_pct'],
        'cloud_cover_mid_pct': data['cloud_cover_mid_pct'],
        'cloud_cover_high_pct': data['cloud_cover_high_pct'],
        'shortwave_radiation_wm2': data['shortwave_radiation_wm2'],
        'model_orography_m': data['model_orography_m'],
    }
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"✅ Data appended to {filename}")

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    HOME_LAT = -7.521951
    HOME_LON = 112.566089
    HOME_ORTHO = 27.07   # elevation in meters

    print("⏳ Initialising ECMWF IFS HRES retrieval engine...")
    try:
        generate_meteo_report(HOME_LAT, HOME_LON, HOME_ORTHO)
        # Optionally export CSV
        # export_meteo_csv(HOME_LAT, HOME_LON, HOME_ORTHO)
    except KeyboardInterrupt:
        print("\n⚠️ Operation interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)