#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoDataManager.py — Standalone Geospatial Data Manager for DEM & Geoid Grids
=================================================================================

**DESCRIPTION**

This module provides self-contained classes for reading and interpolating
geospatial raster data essential for high‑precision geodetic and geophysical
computations. It supports:

    1. Digital Elevation Models (DEM) in Arc ASCII Grid (.asc) format.
    2. XGM2019e-2159 global gravity field grids in ICGEM long‑lat‑value format,
       including grids with additional columns (e.g., gravity_earth includes
       h_over_geoid as an extra column).
    3. Tesseroid‑based terrain corrections (Heck & Seitz, 2007) with
       Gauss‑Legendre 3×3×3 quadrature, accounting for Earth's curvature.
    4. Airy‑Heiskanen isostatic compensation modelling (Heiskanen & Moritz, 1967).
    5. 2D spectral analysis (FFT) to derive dominant wavelengths and power spectra.
    6. Full suite of derived geodetic/geophysical quantities:
        - Geopotential number (C) and dynamic height
        - Spherical shell correction (Barthelmes, 2013, eq. 69-71)
        - Free-air correction and normal gravity at surface
        - Slope, aspect, profile/plan curvature from DEM
        - Terrain correction from grid (CSB - BG)
        - Comparisons between classical (Δg_cl), modern (Δg), spherical (Δg_sa),
          and Bouguer (CSB, BG) anomalies
        - Horizontal gradient of gravity field

**SCIENTIFIC BACKGROUND**

**1. Digital Elevation Models (DEM)**
    - A DEM provides the topographic elevation of the Earth's surface relative
      to a reference ellipsoid or the geoid.
    - The Arc ASCII Grid format used here stores elevation values in a regular
      grid with a header specifying the geographic bounds and cell size.
    - Bilinear interpolation is applied to obtain continuous elevation estimates
      between grid nodes.

**2. XGM2019e-2159 Global Gravity Field Model**
    - Developed by the International Centre for Global Earth Models (ICGEM)
      and the German Research Centre for Geosciences (GFZ‑Potsdam).
    - Published in Zingerle et al. (2020), Journal of Geodesy.
    - Combines satellite gravity data (GOCO06s, GRACE, GOCE) with terrestrial
      gravity data and topographic forward modelling up to degree/order 2159.
    - Spatial resolution: ~0.1° (≈ 5 km at the equator).
    - Tide‑free system, consistent with the WGS84 reference ellipsoid.

**3. Gravity Field Functionals – Theoretical Definitions (Barthelmes, 2013)**

This module implements the following gravity field functionals as defined in
the ICGEM standard (Barthelmes, STR09/02, revised 2013) and the methodology
of Uz & Ince (2025) for complete spherical Bouguer and isostatic anomalies:

    - Geoid undulation (N):  The height of the geoid above the reference
      ellipsoid.  Defined by W(N) = U(0).  Over the oceans N coincides with
      the height anomaly ζ; over land a topographic correction is required.

    - Height anomaly (ζ):  The Molodensky height anomaly, defined as the
      distance from the Earth's surface to the point where the normal potential
      U equals the actual potential W at the surface.  It does not require any
      hypothesis about mass densities inside the Earth.

    - Gravity disturbance (δg):  The difference between the magnitude of
      actual gravity and normal gravity at the same point in space:
          δg(h,λ,φ) = |∇W(h,λ,φ)| - |∇U(h,φ)|
      This is the most rigorous functional for comparing observed gravity with
      a global model because it avoids downward continuation.

    - Classical gravity anomaly (Δg_cl):  The difference between the harmonic
      downward continuation of gravity to the geoid and normal gravity on the
      ellipsoid:
          Δg_cl(λ,φ) = |∇W^c(N)| - |∇U(0)|
      This is the quantity traditionally used in Stokes's integral and in
      terrestrial gravimetry.  It corresponds to the "free‑air gravity anomaly"
      without terrain corrections.

    - Complete spherical Bouguer anomaly (Δg_CSB):  The free‑air anomaly after
      removing the gravitational attraction of all topographic masses above the
      geoid (including terrain corrections), and replacing water masses below
      the geoid with standard crustal density.  This is the most rigorous
      Bouguer reduction, accounting for spherical geometry and full terrain
      effects, as described in Uz & Ince (2025).

    - Simple Bouguer anomaly (Δg_BG):  The free‑air anomaly corrected only for
      the infinite slab (Bouguer plate) effect using a constant crustal density
      (2670 kg/m³) and water density (1025 kg/m³).  Terrain corrections are
      neglected, making it a simpler but less accurate reduction.

    - Isostatic anomaly (Δg_ISO):  The Bouguer anomaly further corrected for
      isostatic compensation (Airy‑Heiskanen model) assuming that topographic
      loads are compensated by crustal roots (or anti‑roots under oceans) at a
      nominal compensation depth (T = 30 km).  This anomaly isolates departures
      from isostatic equilibrium.

    - Spherical approximation anomaly (Δg_SA):  The gravity anomaly computed
      in spherical approximation from spherical harmonic coefficients without
      any topographic or isostatic corrections.  It is simpler but less accurate
      in mountainous regions.

**4. Tesseroid Terrain Correction (Heck & Seitz, 2007)**
    - The gravitational effect of topographic masses is computed using spherical
      prisms (tesseroids) with Gauss‑Legendre quadrature (3×3×3).
    - This accounts for Earth's curvature, providing significantly higher
      accuracy than Cartesian prisms for regional/global applications.
    - The vertical component of gravitational attraction (gz) is computed at the
      observation point and returned in mGal.

**5. Airy‑Heiskanen Isostatic Model (Heiskanen & Moritz, 1967)**
    - Topographic masses are compensated by crustal roots (under continents)
      or anti‑roots (under oceans) relative to a normal crustal thickness T₀.
    - Root depth: t = (ρ_c / (ρ_m - ρ_c)) * H (on land).
    - Moho depth: Moho = T₀ + t.

**6. Spectral Analysis (Cooley & Tukey, 1965)**
    - 2D Fast Fourier Transform is used to derive the dominant wavelengths and
      power spectra of gridded data.

All grids are referenced to the WGS84 ellipsoid and are tide‑free, consistent
with the XGM2019e‑2159 model.

**Grid Products and their Column Formats:**
    Most grids have the format: longitude, latitude, value.
    However, some grids (e.g., gravity_earth, gravity_disturbance) include an
    additional column (h_over_geoid) before the actual value. The reader
    automatically detects the number of columns and extracts the correct value
    column based on the grid name.

**Grid Products Available:**
    - height_anomaly_ell: Geoid height anomaly (ζ) — follows the plumb line.
    - geoid: Geoid undulation (N) — along the ellipsoid normal.
    - gravity_ell: Normal gravity (γ₀) on the ellipsoid (mGal).
    - gravity_anomaly: Free‑air gravity anomaly (Δg) (mGal).
    - gravity_earth: Gravity reduced to the geoid (spheroid) after downward
      continuation, equivalent to normal gravity at ellipsoid plus classical
      free-air anomaly: γ₀ + Δg_cl (mGal). This grid has an extra column:
      h_over_geoid (topographic height above geoid), but the gravity value
      itself is NOT at the Earth's surface.	
    - gravitation_ell: Gravitational acceleration on the ellipsoid (mGal).
    - potential_ell: Normal potential (U₀) on the ellipsoid (m²/s²).
    - second_r_derivative: Vertical gravity gradient (Eötvös).
    - water_column: Equivalent water column height (metres).
    - gravity_disturbance: Precise gravity disturbance (δg) with elevation
      correction (mGal).  This grid has an extra column: h_over_geoid.
    - gravity_disturbance_sa: Gravity disturbance in spherical approximation
      (mGal).
    - gravity_anomaly_cl: Classical (free‑air) gravity anomaly (mGal).
    - gravity_anomaly_csb: Complete spherical Bouguer anomaly (mGal).
    - gravity_anomaly_bg: Simple Bouguer anomaly (mGal).
    - gravity_anomaly_iso: Isostatic anomaly (Airy‑Heiskanen) (mGal).
    - gravity_anomaly_sa: Spherical approximation anomaly (mGal).

**METHODS**
    - get_undulation(lat, lon, use_geoid=False): returns geoid height (ζ or N).
    - get_normal_gravity(lat, lon): returns γ₀ in mGal.
    - get_gravity_anomaly(lat, lon): returns Δg in mGal.
    - get_gravity_earth(lat, lon): returns absolute surface gravity in mGal.
    - get_gravitation(lat, lon): returns gravitation on the ellipsoid.
    - get_potential(lat, lon): returns normal potential U₀.
    - get_second_r_derivative(lat, lon): returns vertical gradient in Eötvös.
    - get_water_column(lat, lon): returns water column height in metres.
    - get_gravity_disturbance(lat, lon): returns precise gravity disturbance
      δg with elevation correction (mGal).
    - get_gravity_disturbance_sa(lat, lon): returns gravity disturbance in
      spherical approximation (mGal).
    - get_gravity_anomaly_cl(lat, lon): returns classical free‑air gravity
      anomaly (mGal).
    - get_gravity_anomaly_csb(lat, lon): returns complete spherical Bouguer
      anomaly (mGal).
    - get_gravity_anomaly_bg(lat, lon): returns simple Bouguer anomaly (mGal).
    - get_gravity_anomaly_iso(lat, lon): returns isostatic anomaly (mGal).
    - get_gravity_anomaly_sa(lat, lon): returns spherical approximation
      anomaly (mGal).
    - get_horizontal_gradient(lat, lon): returns horizontal gradient (∂g/∂x, ∂g/∂y) in mGal/m.
    - get_normal_gravity_at_45(): returns normal gravity at 45° latitude.

**DEPENDENCIES**
    - numpy (>= 1.20.0)
    - Standard library: os, math, typing, io, textwrap, shutil

**REFERENCES**
    - Barthelmes, F. (2013). Definition of Functionals of the Geopotential and
      Their Calculation from Spherical Harmonic Models. Scientific Technical
      Report STR09/02, GFZ Potsdam. (Revised edition)
    - Zingerle, P., Pail, R., Gruber, T., & Oikonomidou, X. (2020).
      The combined global gravity field model XGM2019e.
      Journal of Geodesy, 94(7). DOI: 10.1007/s00190-020-01398-0
    - Uz, M. & Ince, E.S. (2025). Retrieving complete spherical Bouguer and
      isostatic gravity anomalies using global gravity forward models.
      Geophysical Journal International, 244(2), ggaf473.
    - Heck, B. & Seitz, K. (2007). A comparison of the tesseroid, prism and
      point‑mass approaches for mass reductions in gravity field modelling.
      Journal of Geodesy, 81, 121–136.
    - Heiskanen, W.A. & Moritz, H. (1967). Physical Geodesy. W.H. Freeman.
    - Hofmann-Wellenhof, B. & Moritz, H. (2006). Physical Geodesy. Springer.
    - IERS Conventions (2010). IERS Technical Note 36.
    - ESRI Arc ASCII Grid Format Specification.
    - Horn, B.K.P. (1981). Hill shading and the reflectance map.
      Proceedings of the IEEE, 69(1), 14–47.
    - Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine
      calculation of complex Fourier series. Math. Comput., 19, 297–301.

**AUTHOR**
    ASTERID Research Consortium — Jolotundo Archaeological Observatory

**VERSION**
    3.4 (2026-07-15) — Final stable version: separated global/local calculations,
    corrected Airy-Heiskanen root formula, added TC_local and CBA_local.
"""

import os
import math
import io
import textwrap
import shutil
import numpy as np
from typing import Tuple, Dict, Optional, List, Union, Any

# ============================================================================
# GEODETIC CONSTANTS (WGS84 Reference System)
# ============================================================================
WGS84_A = 6378137.0          # Semi‑major axis (equatorial radius) in metres.
WGS84_F = 1.0 / 298.257223563  # Flattening factor.
WGS84_E2 = 2.0 * WGS84_F - WGS84_F**2  # First eccentricity squared.
WGS84_OMEGA = 7.292115e-5    # Earth's angular velocity (rad/s).

# ============================================================================
# PHYSICAL CONSTANTS FOR DERIVED PARAMETERS
# ============================================================================
GRAVITATIONAL_CONSTANT = 6.67430e-11      # m³ kg⁻¹ s⁻²
RHO_CRUST = 2670.0                        # kg/m³ (Bouguer / Airy)
RHO_MANTLE = 3270.0                       # kg/m³ (Airy-Heiskanen)
FREE_AIR_GRADIENT = -0.3086               # mGal/m (normal gravity gradient)
PI = math.pi
BOUGUER_COEFF = 2.0 * PI * GRAVITATIONAL_CONSTANT * RHO_CRUST * 1e5  # ~0.1119 mGal/m
T0 = 30000.0  # nominal crustal thickness (30 km) for Airy-Heiskanen


# ============================================================================
# CLASS: Arc ASCII Grid DEM Reader
# ============================================================================

class ASCDEMReader:
    """
    Reads and interpolates Digital Elevation Models in Arc ASCII Grid (.asc) format.

    The Arc ASCII Grid format is a widely used interchange format for raster
    geospatial data, consisting of a header with metadata and a row‑major array
    of elevation values.

    **Format Specification:**
        ncols        <integer>
        nrows        <integer>
        xllcorner    <float>   (west longitude of lower‑left cell)
        yllcorner    <float>   (south latitude of lower‑left cell)
        cellsize     <float>   (grid spacing in decimal degrees)
        NODATA_value <float>   (optional, default -9999)
        <data row 1> <data row 2> ... <data row n>

    **Methods:**
        - get_elevation(lat, lon): Returns elevation at a single point using
          bilinear interpolation.
        - get_elevation_grid(lats, lons): Returns a 2D array of elevations for
          an arbitrary grid of coordinate arrays.
        - get_slope(lat, lon): Returns slope in degrees using Horn's method.
        - get_aspect(lat, lon): Returns aspect in degrees clockwise from North.
        - get_profile_curvature(lat, lon): Returns profile curvature (1/m).
        - get_plan_curvature(lat, lon): Returns plan curvature (1/m).
        - get_terrain_correction_tesseroid(lat, lon): Tesseroid‑based terrain
          correction using Gauss‑Legendre 3×3×3 integration (Heck & Seitz, 2007).
          Returns in mGal.
        - get_metadata(): Returns a dictionary of DEM metadata.

    **Example:**
        >>> dem = ASCDEMReader('jolotundo_cop30.asc')
        >>> elev = dem.get_elevation(-7.609444, 112.595556)
        >>> tc = dem.get_terrain_correction_tesseroid(-7.609444, 112.595556)
        >>> print(f"Elevation: {elev:.2f} m, TC: {tc:.4f} mGal")
    """

    def __init__(self, filepath: str):
        """
        Initialise the DEM reader and load the grid from disk.

        Parameters
        ----------
        filepath : str
            Path to the Arc ASCII Grid (.asc) file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"DEM file not found: {filepath}")

        self.filepath = filepath
        self.lons: Optional[np.ndarray] = None
        self.lats: Optional[np.ndarray] = None
        self.elevations: Optional[np.ndarray] = None
        self.ncols: int = 0
        self.nrows: int = 0
        self.xllcorner: float = 0.0
        self.yllcorner: float = 0.0
        self.cellsize: float = 0.0
        self.nodata_value: float = -9999.0

        self._load()

    def _load(self) -> None:
        """Load the DEM file, parse the header, and populate the grid."""
        with open(self.filepath, 'r') as f:
            lines = f.readlines()

        header = {}
        data_start_line = 0

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 2:
                continue
            key = parts[0].lower()
            if key in ('ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize', 'nodata_value'):
                header[key] = float(parts[1])
            else:
                data_start_line = i
                break

        required = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize']
        for r in required:
            if r not in header:
                raise ValueError(f"Incomplete DEM header: missing '{r}'")

        self.ncols = int(header['ncols'])
        self.nrows = int(header['nrows'])
        self.xllcorner = header['xllcorner']
        self.yllcorner = header['yllcorner']
        self.cellsize = header['cellsize']
        self.nodata_value = header.get('nodata_value', -9999.0)

        # Read the data block.
        data_str = ''.join(lines[data_start_line:])
        try:
            raw = np.loadtxt(io.StringIO(data_str), dtype=np.float64)
        except Exception as e:
            raise ValueError(f"Failed to parse DEM data: {e}")

        expected_size = self.nrows * self.ncols
        if raw.size != expected_size:
            raise ValueError(
                f"Data size mismatch: got {raw.size}, expected {expected_size}"
            )

        # Reshape and flip vertically (Arc ASCII Grid stores data from top to bottom).
        self.elevations = raw.reshape((self.nrows, self.ncols))
        self.elevations = np.flipud(self.elevations)

        # Generate coordinate axes (cell centres).
        self.lons = self.xllcorner + (np.arange(self.ncols) + 0.5) * self.cellsize
        self.lats = self.yllcorner + (np.arange(self.nrows) + 0.5) * self.cellsize

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return a dictionary containing DEM metadata.

        Returns
        -------
        dict
            Metadata including dimensions, bounds, and elevation statistics.
        """
        return {
            'filepath': self.filepath,
            'ncols': self.ncols,
            'nrows': self.nrows,
            'xllcorner': self.xllcorner,
            'yllcorner': self.yllcorner,
            'cellsize': self.cellsize,
            'nodata_value': self.nodata_value,
            'lon_min': float(self.lons[0]),
            'lon_max': float(self.lons[-1]),
            'lat_min': float(self.lats[0]),
            'lat_max': float(self.lats[-1]),
            'elevation_min': float(np.nanmin(self.elevations)),
            'elevation_max': float(np.nanmax(self.elevations)),
            'elevation_mean': float(np.nanmean(self.elevations)),
        }

    def _bilinear_interpolate(self, lat: float, lon: float) -> float:
        """
        Perform bilinear interpolation on the elevation grid.

        Parameters
        ----------
        lat, lon : float
            Geographic coordinates in decimal degrees.

        Returns
        -------
        float
            Interpolated elevation in metres. Returns NaN if outside the grid
            or if all four surrounding cells are NODATA.
        """
        if self.elevations is None:
            return np.nan

        lons = self.lons
        lats = self.lats
        grid = self.elevations

        # Check if the point is inside the grid bounds.
        if lat <= lats[0] or lat >= lats[-1] or lon <= lons[0] or lon >= lons[-1]:
            return np.nan

        # Find the indices of the four surrounding cells.
        idx_lat = np.searchsorted(lats, lat) - 1
        idx_lon = np.searchsorted(lons, lon) - 1

        if idx_lat < 0 or idx_lat >= len(lats) - 1 or idx_lon < 0 or idx_lon >= len(lons) - 1:
            return np.nan

        # Interpolation weights.
        w_lat = (lat - lats[idx_lat]) / (lats[idx_lat + 1] - lats[idx_lat])
        w_lon = (lon - lons[idx_lon]) / (lons[idx_lon + 1] - lons[idx_lon])

        # Cell values.
        v00 = grid[idx_lat, idx_lon]
        v01 = grid[idx_lat, idx_lon + 1]
        v10 = grid[idx_lat + 1, idx_lon]
        v11 = grid[idx_lat + 1, idx_lon + 1]

        # If any cell is NODATA, return NaN.
        if any(np.isnan([v00, v01, v10, v11])):
            return np.nan

        # Bilinear interpolation formula.
        return (v00 * (1 - w_lat) * (1 - w_lon) +
                v10 * w_lat * (1 - w_lon) +
                v01 * (1 - w_lat) * w_lon +
                v11 * w_lat * w_lon)

    def get_elevation(self, lat: float, lon: float) -> float:
        """
        Get the interpolated elevation at a single geographic point.

        Parameters
        ----------
        lat, lon : float
            Geographic coordinates in decimal degrees.

        Returns
        -------
        float
            Elevation in metres, or NaN if outside the grid or NODATA.
        """
        return self._bilinear_interpolate(lat, lon)

    def get_elevation_grid(self, lats_grid: np.ndarray, lons_grid: np.ndarray) -> np.ndarray:
        """
        Get interpolated elevations over a 2D grid of coordinates.

        Parameters
        ----------
        lats_grid, lons_grid : np.ndarray
            2D arrays of latitudes and longitudes (same shape).

        Returns
        -------
        np.ndarray
            2D array of elevations (same shape as input).
        """
        if lats_grid.shape != lons_grid.shape:
            raise ValueError("lats_grid and lons_grid must have the same shape.")

        rows, cols = lats_grid.shape
        result = np.full_like(lats_grid, np.nan, dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                result[i, j] = self.get_elevation(lats_grid[i, j], lons_grid[i, j])
        return result

    def _get_3x3_window(self, lat: float, lon: float):
        """Get 3x3 elevation window around point for slope/aspect/curvature."""
        if self.elevations is None:
            return None, None, None, None, None

        lons = self.lons
        lats = self.lats
        grid = self.elevations

        idx_lat = np.searchsorted(lats, lat) - 1
        idx_lon = np.searchsorted(lons, lon) - 1

        # Clamp to valid range
        idx_lat = max(1, min(idx_lat, len(lats) - 2))
        idx_lon = max(1, min(idx_lon, len(lons) - 2))

        # Extract 3x3 window
        z = grid[idx_lat-1:idx_lat+2, idx_lon-1:idx_lon+2]
        lat_win = lats[idx_lat-1:idx_lat+2]
        lon_win = lons[idx_lon-1:idx_lon+2]

        # Center coordinates of window
        lat_c = lat_win[1]
        lon_c = lon_win[1]
        # Cell sizes in meters at center
        dy = self.cellsize * 111320.0 
        dx = self.cellsize * 111320.0 * math.cos(math.radians(lat_c))

        return z, dx, dy, lat_c, lon_c

    def get_slope(self, lat: float, lon: float) -> float:
        """
        Compute slope in degrees using 3x3 window (Horn's method).
        """
        win = self._get_3x3_window(lat, lon)
        if win[0] is None:
            return np.nan
        z, dx, dy, _, _ = win

        # Horn's slope (central differences)
        dz_dx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0])) / (8.0 * dx)
        dz_dy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2])) / (8.0 * dy)

        slope_rad = math.atan2(math.sqrt(dz_dx**2 + dz_dy**2), 1.0)
        return math.degrees(slope_rad)

    def get_aspect(self, lat: float, lon: float) -> float:
        """
        Compute aspect in degrees from North, clockwise (0=North, 90=East).
        """
        win = self._get_3x3_window(lat, lon)
        if win[0] is None:
            return np.nan
        z, dx, dy, _, _ = win

        dz_dx = ((z[0,2] + 2*z[1,2] + z[2,2]) - (z[0,0] + 2*z[1,0] + z[2,0])) / (8.0 * dx)
        dz_dy = ((z[2,0] + 2*z[2,1] + z[2,2]) - (z[0,0] + 2*z[0,1] + z[0,2])) / (8.0 * dy)

        aspect = math.atan2(dz_dy, -dz_dx)
        aspect_deg = math.degrees(aspect)
        # Convert to 0-360 clockwise from North
        aspect_deg = 90.0 - aspect_deg
        if aspect_deg < 0:
            aspect_deg += 360.0
        return aspect_deg

    def get_profile_curvature(self, lat: float, lon: float) -> float:
        """Profile curvature (curvature in the direction of slope). Unit: 1/m."""
        win = self._get_3x3_window(lat, lon)
        if win[0] is None:
            return np.nan
        z, dx, dy, _, _ = win

        # Second derivatives (finite differences)
        d2z_dx2 = (z[1,2] - 2*z[1,1] + z[1,0]) / (dx**2)
        d2z_dy2 = (z[2,1] - 2*z[1,1] + z[0,1]) / (dy**2)
        d2z_dxdy = (z[2,2] - z[2,0] - z[0,2] + z[0,0]) / (4.0 * dx * dy)

        # First derivatives (central)
        dz_dx = (z[1,2] - z[1,0]) / (2.0 * dx)
        dz_dy = (z[2,1] - z[0,1]) / (2.0 * dy)

        p = dz_dx
        q = dz_dy
        r = d2z_dx2
        s = d2z_dxdy
        t = d2z_dy2

        # Profile curvature: curvature along the steepest slope direction
        denom = (p**2 + q**2) * math.sqrt(1 + p**2 + q**2)
        if abs(denom) < 1e-12:
            return 0.0
        profile_curv = - (p**2 * r + 2*p*q*s + q**2 * t) / denom
        return profile_curv

    def get_plan_curvature(self, lat: float, lon: float) -> float:
        """Plan curvature (curvature perpendicular to slope). Unit: 1/m."""
        win = self._get_3x3_window(lat, lon)
        if win[0] is None:
            return np.nan
        z, dx, dy, _, _ = win

        d2z_dx2 = (z[1,2] - 2*z[1,1] + z[1,0]) / (dx**2)
        d2z_dy2 = (z[2,1] - 2*z[1,1] + z[0,1]) / (dy**2)
        d2z_dxdy = (z[2,2] - z[2,0] - z[0,2] + z[0,0]) / (4.0 * dx * dy)

        dz_dx = (z[1,2] - z[1,0]) / (2.0 * dx)
        dz_dy = (z[2,1] - z[0,1]) / (2.0 * dy)

        p = dz_dx
        q = dz_dy
        r = d2z_dx2
        s = d2z_dxdy
        t = d2z_dy2

        denom = (p**2 + q**2) * math.sqrt(1 + p**2 + q**2)
        if abs(denom) < 1e-12:
            return 0.0
        plan_curv = - (p**2 * t - 2*p*q*s + q**2 * r) / denom
        return plan_curv

    def get_terrain_correction_tesseroid(self, lat: float, lon: float,
                                         radius_deg: float = 0.05,
                                         rho: float = RHO_CRUST,
                                         n_gauss: int = 3) -> float:
        """
        Compute the complete terrain correction (positive definite) using
        spherical tesseroids with Gauss-Legendre quadrature.

        Scientific Definition
        ---------------------
        The terrain correction (TC) is the gravitational attraction of all
        topographic masses that deviate from the Bouguer plate (a flat
        horizontal slab passing through the observation point). It is
        always positive and consists of two parts:

        1. Hills (r_cell > r0):  mass excess above the station attracts
           upward; we add this effect to compensate for the missing mass
           in the Bouguer plate.
        2. Valleys (r_cell < r0): mass deficit below the station; the
           Bouguer plate overcorrects, so we add the effect of the mass
           that would fill the valley up to the station height.

        Thus, for each tesseroid, the radial integration is performed
        between r0 and r_cell, taking the absolute value of the vertical
        component (Δz) to ensure a positive contribution. This follows
        the standard approach used in the Tesseroids software
        (Uieda et al., 2016) and recommended by Tsoulis et al. (2009).

        Adaptive Subdivision (Uieda et al., 2016)
        -----------------------------------------
        Instead of a fixed angular threshold, we use the Distance-to-Size
        ratio (D = distance / cell_size). If D < 1.5, the cell is too
        close to the observation point and is subdivided into 2x2
        sub-cells. This ensures numerical accuracy independent of the
        DEM resolution and latitude.

        References
        ----------
        - Heiskanen, W.A. & Moritz, H. (1967). Physical Geodesy.
        - Tsoulis, D., Novák, P., & Kadlec, M. (2009). Evaluation of
          precise terrain effects using high-resolution DEMs.
          J. Geophys. Res., 114, B02404.
        - Uieda, L., Barbosa, V.C.F., & Braitenberg, C. (2016). Tesseroids:
          Forward-modeling gravitational fields in spherical coordinates.
          Geophysics, 81(5), F41-F48.
        - Grombein, T., Seitz, K., & Heck, B. (2013). Optimized formulas
          for the gravitational field of a tesseroid. J. Geodesy, 87, 645-660.

        Parameters
        ----------
        lat, lon : float
            Observation point (decimal degrees).
        radius_deg : float
            Integration radius (degrees). Default 0.05° (~5.5 km).
        rho : float
            Density of topographic masses (kg/m³). Default 2670.
        n_gauss : int
            Number of Gauss-Legendre nodes per dimension (1-4). Default 3.

        Returns
        -------
        float
            Terrain correction in mGal (always positive).
        """
        # ------------------------------------------------------------------
        # Constants
        # ------------------------------------------------------------------
        R_EARTH = 6371000.0
        G = 6.67430e-11
        CONV = 1e5

        # ------------------------------------------------------------------
        # Observation point in spherical and Cartesian coordinates
        # ------------------------------------------------------------------
        lat0 = math.radians(lat)
        lon0 = math.radians(lon)
        elev0 = self.get_elevation(lat, lon)
        if np.isnan(elev0):
            return np.nan

        r0 = R_EARTH + elev0

        # Local vertical unit vector (normal to ellipsoid)
        nx0 = math.cos(lat0) * math.cos(lon0)
        ny0 = math.cos(lat0) * math.sin(lon0)
        nz0 = math.sin(lat0)

        # Cartesian coordinates of observation point
        x0 = r0 * math.cos(lat0) * math.cos(lon0)
        y0 = r0 * math.cos(lat0) * math.sin(lon0)
        z0 = r0 * math.sin(lat0)

        # ------------------------------------------------------------------
        # Integration bounds (radians)
        # ------------------------------------------------------------------
        rad = math.radians(radius_deg)
        lat_min = lat0 - rad
        lat_max = lat0 + rad
        lon_min = lon0 - rad
        lon_max = lon0 + rad

        # DEM indices within the radius
        i0 = max(0, np.searchsorted(self.lats, math.degrees(lat_min)) - 1)
        i1 = min(self.nrows, np.searchsorted(self.lats, math.degrees(lat_max)) + 1)
        j0 = max(0, np.searchsorted(self.lons, math.degrees(lon_min)) - 1)
        j1 = min(self.ncols, np.searchsorted(self.lons, math.degrees(lon_max)) + 1)

        if i1 <= i0 or j1 <= j0:
            return 0.0

        # ------------------------------------------------------------------
        # Gauss-Legendre nodes and weights (cached)
        # ------------------------------------------------------------------
        gauss_cache = {
            1: (np.array([0.0]), np.array([2.0])),
            2: (np.array([-0.5773502692, 0.5773502692]), np.array([1.0, 1.0])),
            3: (np.array([-0.7745966692, 0.0, 0.7745966692]),
                np.array([0.5555555556, 0.8888888889, 0.5555555556])),
            4: (np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116]),
                np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])),
        }
        if n_gauss not in gauss_cache:
            raise ValueError("n_gauss must be 1, 2, 3, or 4")
        xi, wi = gauss_cache[n_gauss]

        # ------------------------------------------------------------------
        # Pre-calculate cell size in meters for the D ratio criterion
        # (using the latitudinal size as the reference, since longitudinal
        #  size varies with latitude but is almost identical here)
        # ------------------------------------------------------------------
        cell_size_deg = self.cellsize
        # Approximate cell size in meters at the latitude of the observation point
        cell_size_m = cell_size_deg * (np.pi / 180.0) * R_EARTH

        # ------------------------------------------------------------------
        # Integration loop
        # ------------------------------------------------------------------
        total_tc = 0.0

        for i in range(i0, i1 - 1):
            lat1_deg = self.lats[i]
            lat2_deg = self.lats[i+1]
            lat1 = math.radians(lat1_deg)
            lat2 = math.radians(lat2_deg)
            lat_c = (lat1 + lat2) / 2.0  # Center latitude for distance calc

            for j in range(j0, j1 - 1):
                lon1_deg = self.lons[j]
                lon2_deg = self.lons[j+1]
                lon1 = math.radians(lon1_deg)
                lon2 = math.radians(lon2_deg)
                lon_c = (lon1 + lon2) / 2.0  # Center longitude for distance calc

                # Average elevation of the cell (used to define the top boundary)
                h_avg = (self.elevations[i, j] + self.elevations[i+1, j] +
                         self.elevations[i+1, j+1] + self.elevations[i, j+1]) / 4.0
                r_cell = R_EARTH + h_avg

                # If the cell is exactly at the station height, skip (no mass deviation)
                if abs(r_cell - r0) < 1e-6:
                    continue

                # ------------------------------------------------------------------
                # COMPUTE SPHERICAL DISTANCE (psi) FROM OBSERVATION POINT TO CELL CENTER
                # ------------------------------------------------------------------
                cos_psi = (math.sin(lat0) * math.sin(lat_c) +
                           math.cos(lat0) * math.cos(lat_c) *
                           math.cos(lon0 - lon_c))
                cos_psi = max(-1.0, min(1.0, cos_psi))
                psi = math.acos(cos_psi)
                distance = psi * R_EARTH  # Great-circle distance in meters

                # ------------------------------------------------------------------
                # ADAPTIVE SUBDIVISION CRITERION (Uieda et al., 2016)
                # D = distance / cell_size
                # If D < 1.5, subdivide the cell into 2x2 to maintain accuracy.
                # ------------------------------------------------------------------
                D_ratio = distance / cell_size_m

                if D_ratio < 1.5:
                    # Subdivide into 2x2 sub-cells
                    lat_edges = np.linspace(lat1, lat2, 3)
                    lon_edges = np.linspace(lon1, lon2, 3)
                    for ii in range(2):
                        lat1_sub = lat_edges[ii]
                        lat2_sub = lat_edges[ii+1]
                        for jj in range(2):
                            lon1_sub = lon_edges[jj]
                            lon2_sub = lon_edges[jj+1]

                            # Correct radial integration limits:
                            # - Hills: integrate from r0 to r_cell
                            # - Valleys: integrate from r_cell to r0
                            if r_cell > r0:
                                r1_int = r0
                                r2_int = r_cell
                            else:
                                r1_int = r_cell
                                r2_int = r0

                            contrib = self._tesseroid_integrate_abs(
                                lat0, lon0, r0,
                                x0, y0, z0, nx0, ny0, nz0,
                                lat1_sub, lat2_sub, lon1_sub, lon2_sub,
                                r1_int, r2_int, rho, G, xi, wi
                            )
                            total_tc += contrib
                else:
                    # Direct integration without subdivision
                    if r_cell > r0:
                        r1_int = r0
                        r2_int = r_cell
                    else:
                        r1_int = r_cell
                        r2_int = r0

                    contrib = self._tesseroid_integrate_abs(
                        lat0, lon0, r0,
                        x0, y0, z0, nx0, ny0, nz0,
                        lat1, lat2, lon1, lon2,
                        r1_int, r2_int, rho, G, xi, wi
                    )
                    total_tc += contrib

        # Convert to mGal and return positive
        return total_tc * CONV

    # --------------------------------------------------------------------------
    # Helper: Tesseroid Integration with Absolute Vertical Component
    # --------------------------------------------------------------------------
    def _tesseroid_integrate_abs(self, lat0, lon0, r0,
                                 x0, y0, z0, nx0, ny0, nz0,
                                 lat1, lat2, lon1, lon2,
                                 r1, r2, rho, G, xi, wi):
        """
        Integrate a single tesseroid (or sub-cell) using 3D Gauss-Legendre
        quadrature and return the absolute value of its vertical component
        contribution.

        The radial integration limits (r1, r2) define the interval between
        the station height and the cell top (whether r1<r2 or r1>r2).
        The function integrates from r1 to r2, but uses the absolute value
        of Δz to ensure a positive terrain correction.
        """
        n = len(xi)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        dr = r2 - r1

        # Scale factor for 3D quadrature:
        # ∫∫∫ f dr dφ dλ ≈ (dr * dlat * dlon / 8) * Σ w_i w_j w_k f(r_i, φ_j, λ_k)
        scale = dr * dlat * dlon / 8.0

        total = 0.0

        for i in range(n):
            # Radial node
            r = 0.5 * (r2 - r1) * xi[i] + 0.5 * (r2 + r1)

            for j in range(n):
                # Latitudinal node
                lat_prime = 0.5 * (lat2 - lat1) * xi[j] + 0.5 * (lat2 + lat1)

                for k in range(n):
                    # Longitudinal node
                    lon_prime = 0.5 * (lon2 - lon1) * xi[k] + 0.5 * (lon2 + lon1)

                    # Cartesian coordinates of the integration point
                    x = r * math.cos(lat_prime) * math.cos(lon_prime)
                    y = r * math.cos(lat_prime) * math.sin(lon_prime)
                    z = r * math.sin(lat_prime)

                    # Vector from observation point to source point
                    dx = x - x0
                    dy = y - y0
                    dz = z - z0

                    # Squared Euclidean distance
                    dist2 = dx*dx + dy*dy + dz*dz

                    # ------------------------------------------------------------------
                    # SINGULARITY HANDLING:
                    # If the point is extremely close (< 1 mm), we skip it.
                    # Thanks to the adaptive subdivision (D < 1.5), the nearest
                    # quadrature node is typically > 10 m away, so this threshold
                    # is purely a safety net.
                    # ------------------------------------------------------------------
                    if dist2 < 1e-6:
                        continue

                    dist = math.sqrt(dist2)

                    # Vertical component (signed): projection onto local vertical
                    delta_z = dx * nx0 + dy * ny0 + dz * nz0

                    # For terrain correction, we always add the absolute value
                    # of the vertical component (since TC is positive definite).
                    # This handles both hills (delta_z > 0) and valleys (delta_z < 0).
                    abs_delta_z = abs(delta_z)

                    # Spherical volume element (Jacobian): r² * cos(lat')
                    jacobian = r * r * math.cos(lat_prime)

                    # Gauss-Legendre weight product
                    w = wi[i] * wi[j] * wi[k]

                    # Newtonian kernel for vertical component: Δz / ℓ³
                    # (dist2 * dist) = dist³
                    kernel = abs_delta_z / (dist2 * dist)

                    total += G * rho * w * jacobian * scale * kernel

        return total

# ============================================================================
# CLASS: XGM2019e-2159 Grid Reader
# ============================================================================

class XGMGridReader:
    """
    Reads and interpolates global gravity field grids from the XGM2019e-2159 model.

    The XGM2019e-2159 model (Zingerle et al., 2020) is a combined global
    gravity field model developed by GFZ‑Potsdam and ICGEM. It combines
    satellite data (GOCO06s, GRACE, GOCE), terrestrial gravity data, and
    topographic forward modelling up to degree/order 2159.

    The grids are provided in ICGEM format:
        - Header containing metadata (ending with 'end_of_head').
        - Data in long‑lat‑value format, sorted by latitude and longitude.

    Some grids (e.g., gravity_earth, gravity_disturbance) include an extra
    column (h_over_geoid) before the actual value. The reader automatically
    detects the number of columns and extracts the correct column based on the
    grid name.

    **Products (grid files):**
        - height_anomaly_ell: Geoid height anomaly (ζ) — follows plumb line.
        - geoid: Geoid undulation (N) — along ellipsoid normal.
        - gravity_ell: Normal gravity on ellipsoid (γ₀) in mGal.
        - gravity_anomaly: Free‑air gravity anomaly (Δg) in mGal.
        - gravity_earth: Gravity reduced to the geoid (spheroid), equivalent to
          normal gravity at ellipsoid plus classical free-air anomaly (γ₀ + Δg_cl)
          (mGal). This grid has an extra column: h_over_geoid (topographic height
          above geoid), but the gravity value itself is NOT at the Earth's surface.
        - gravitation_ell: Gravitational acceleration on ellipsoid in mGal.
        - potential_ell: Normal potential (U₀) on ellipsoid.
        - second_r_derivative: Vertical gravity gradient (∂²V/∂r²) in Eötvös.
        - water_column: Equivalent water column height (for ocean loading).
        - gravity_disturbance: Precise gravity disturbance (δg) with elevation
          correction (mGal).  This grid has an extra column: h_over_geoid.
        - gravity_disturbance_sa: Gravity disturbance in spherical approximation
          (mGal).
        - gravity_anomaly_cl: Classical (free‑air) gravity anomaly (mGal).
        - gravity_anomaly_csb: Complete spherical Bouguer anomaly (mGal) — fully
          terrain‑corrected, spherical geometry (Uz & Ince, 2025).
        - gravity_anomaly_bg: Simple Bouguer anomaly (mGal) — plate correction only.
        - gravity_anomaly_iso: Isostatic anomaly (Airy‑Heiskanen) (mGal) —
          compensation depth T ≈ 30 km (GRS80).
        - gravity_anomaly_sa: Spherical approximation anomaly (mGal) — no topography.

    **Methods:**
        - get_undulation(lat, lon): Returns geoid height anomaly (ζ) or undulation (N).
        - get_normal_gravity(lat, lon): Returns normal gravity (γ₀) in mGal.
        - get_gravity_anomaly(lat, lon): Returns gravity anomaly (Δg) in mGal.
        - get_gravity_earth(lat, lon): returns gravity at the geoid (spheroid),   i.e., γ₀ + Δg_cl (mGal). NOT surface gravity.     
        - get_gravitation(lat, lon): Returns gravitation on the ellipsoid.
        - get_potential(lat, lon): Returns normal potential (U₀).
        - get_second_r_derivative(lat, lon): Returns vertical gradient in Eötvös.
        - get_water_column(lat, lon): Returns water column height in metres.
        - get_gravity_disturbance(lat, lon): Returns precise gravity disturbance
          δg (with elevation correction) in mGal.
        - get_gravity_disturbance_sa(lat, lon): Returns gravity disturbance in
          spherical approximation (mGal).
        - get_gravity_anomaly_cl(lat, lon): Returns classical free‑air gravity
          anomaly (mGal).
        - get_gravity_anomaly_csb(lat, lon): Returns complete spherical Bouguer
          anomaly (mGal).
        - get_gravity_anomaly_bg(lat, lon): Returns simple Bouguer anomaly (mGal).
        - get_gravity_anomaly_iso(lat, lon): Returns isostatic anomaly (mGal).
        - get_gravity_anomaly_sa(lat, lon): Returns spherical approximation
          anomaly (mGal).
        - get_horizontal_gradient(lat, lon): Returns horizontal gradient (∂g/∂x, ∂g/∂y) in mGal/m.
        - get_normal_gravity_at_45(): Returns normal gravity at 45° latitude.

    **Example:**
        >>> xgm = XGMGridReader(data_dir='.', grid_prefix='XGM2019e_2159')
        >>> und = xgm.get_undulation(-7.609444, 112.595556, use_geoid=False)
        >>> gamma = xgm.get_normal_gravity(-7.609444, 112.595556)
        >>> csb = xgm.get_gravity_anomaly_csb(-7.609444, 112.595556)
        >>> iso = xgm.get_gravity_anomaly_iso(-7.609444, 112.595556)
        >>> print(f"Undulation: {und:.3f} m, Gamma: {gamma:.4f} mGal, CSB: {csb:.4f} mGal, ISO: {iso:.4f} mGal")
    """

    # Map grid names to the column index that contains the value.
    # Most grids use column 2 (0‑based) as the value.
    # gravity_earth and gravity_disturbance have an extra column (h_over_geoid)
    # at index 2, so value is at index 3.
    _COLUMN_MAP = {
        'height_anomaly': 2,
        'geoid': 2,
        'gravity_ell': 2,
        'gravity_anomaly': 2,
        'gravity_earth': 3,    # extra column: h_over_geoid
        'gravitation_ell': 2,
        'potential_ell': 2,
        'second_r_derivative': 2,
        'water_column': 2,
        'gravity_disturbance': 3,      # extra column: h_over_geoid
        'gravity_disturbance_sa': 2,
        'gravity_anomaly_cl': 2,
        # Gravity anomaly functionals (all have 3 columns)
        'gravity_anomaly_csb': 2,
        'gravity_anomaly_bg': 2,
        'gravity_anomaly_iso': 2,
        'gravity_anomaly_sa': 2,
    }

    def __init__(self, data_dir: str = ".", grid_prefix: str = "XGM2019e_2159"):
        """
        Initialise the XGM grid reader and load all available grid files.

        Parameters
        ----------
        data_dir : str
            Directory containing the grid files.
        grid_prefix : str
            Prefix for the grid filenames. Default: "XGM2019e_2159".
            Expected filenames:
                - height_anomaly_ell_{prefix}.txt
                - geoid_{prefix}.txt
                - gravity_ell_{prefix}.txt
                - gravity_anomaly_{prefix}.txt
                - gravity_earth_{prefix}.txt
                - gravitation_ell_{prefix}.txt
                - potential_ell_{prefix}.txt
                - second_r_derivative_{prefix}.txt
                - water_column_{prefix}.txt
                - gravity_disturbance_{prefix}.txt
                - gravity_disturbance_sa_{prefix}.txt
                - gravity_anomaly_cl_{prefix}.txt
                - gravity_anomaly_csb_{prefix}.txt
                - gravity_anomaly_bg_{prefix}.txt
                - gravity_anomaly_iso_{prefix}.txt
                - gravity_anomaly_sa_{prefix}.txt
        """
        self.data_dir = data_dir
        self.grid_prefix = grid_prefix
        self.grids: Dict[str, Dict] = {}
        self.lons: Optional[np.ndarray] = None
        self.lats: Optional[np.ndarray] = None

        self._load_grids()

    def _load_grids(self) -> None:
        """Load all grid files found in the data directory."""
        grid_files = {
            'height_anomaly': f"height_anomaly_ell_{self.grid_prefix}.txt",
            'geoid': f"geoid_{self.grid_prefix}.txt",
            'gravity_ell': f"gravity_ell_{self.grid_prefix}.txt",
            'gravity_anomaly': f"gravity_anomaly_{self.grid_prefix}.txt",
            'gravity_earth': f"gravity_earth_{self.grid_prefix}.txt",
            'gravitation_ell': f"gravitation_ell_{self.grid_prefix}.txt",
            'potential_ell': f"potential_ell_{self.grid_prefix}.txt",
            'second_r_derivative': f"second_r_derivative_{self.grid_prefix}.txt",
            'water_column': f"water_column_{self.grid_prefix}.txt",
            'gravity_disturbance': f"gravity_disturbance_{self.grid_prefix}.txt",
            'gravity_disturbance_sa': f"gravity_disturbance_sa_{self.grid_prefix}.txt",
            'gravity_anomaly_cl': f"gravity_anomaly_cl_{self.grid_prefix}.txt",
            # Additional gravity anomaly functionals
            'gravity_anomaly_csb': f"gravity_anomaly_csb_{self.grid_prefix}.txt",
            'gravity_anomaly_bg': f"gravity_anomaly_bg_{self.grid_prefix}.txt",
            'gravity_anomaly_iso': f"gravity_anomaly_iso_{self.grid_prefix}.txt",
            'gravity_anomaly_sa': f"gravity_anomaly_sa_{self.grid_prefix}.txt",
        }

        loaded = []
        for name, fname in grid_files.items():
            full_path = os.path.join(self.data_dir, fname)
            if not os.path.exists(full_path):
                print(f"⚠️ Warning: Grid file not found: {fname}")
                continue
            self._load_single_grid(name, full_path)
            loaded.append(name)

        if not self.grids:
            raise FileNotFoundError(
                f"No XGM grid files found in {self.data_dir} with prefix {self.grid_prefix}"
            )

        # Verify that all grids share the same coordinate axes.
        first_grid = next(iter(self.grids.values()))
        self.lons = first_grid['lons']
        self.lats = first_grid['lats']
        for name, g in self.grids.items():
            if not np.array_equal(g['lons'], self.lons) or not np.array_equal(g['lats'], self.lats):
                raise ValueError(f"Grid '{name}' has different coordinate axes")

        print(f"✅ Loaded {len(loaded)} XGM grid(s): {', '.join(loaded)}")

    def _load_single_grid(self, name: str, filepath: str) -> None:
        """
        Load a single ICGEM‑format grid file with header ending in 'end_of_head'.

        The file may have 3 or more columns. The actual value column is determined
        by the _COLUMN_MAP for the grid name.

        Parameters
        ----------
        name : str
            Internal grid name (e.g., 'height_anomaly').
        filepath : str
            Full path to the grid file.
        """
        raw_data = []
        header_ended = False

        with open(filepath, 'r') as f:
            for line in f:
                if not header_ended:
                    if 'end_of_head' in line:
                        header_ended = True
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        lon = float(parts[0])
                        lat = float(parts[1])
                        # Determine which column to use as the value
                        col_idx = self._COLUMN_MAP.get(name, 2)  # default to index 2
                        if col_idx < len(parts):
                            val = float(parts[col_idx])
                        else:
                            continue
                        raw_data.append([lon, lat, val])
                    except ValueError:
                        continue

        if not raw_data:
            raise ValueError(f"No valid data found in {filepath}")

        data = np.array(raw_data, dtype=np.float64)
        lons = np.unique(data[:, 0])
        lats = np.unique(data[:, 1])
        lons.sort()
        lats.sort()

        # Reshape to 2D grid: rows = latitude, columns = longitude.
        grid = data[:, 2].reshape((len(lats), len(lons)))

        self.grids[name] = {
            'lons': lons,
            'lats': lats,
            'grid': grid
        }

    def _bilinear_interpolate(self, grid_name: str, lat: float, lon: float) -> float:
        """
        Bilinear interpolation from a single grid.

        Parameters
        ----------
        grid_name : str
            Name of the grid to interpolate (e.g., 'height_anomaly').
        lat, lon : float
            Geographic coordinates in decimal degrees.

        Returns
        -------
        float
            Interpolated value, or NaN if outside the grid.
        """
        g = self.grids.get(grid_name)
        if g is None:
            return np.nan

        lons = g['lons']
        lats = g['lats']
        grid = g['grid']

        if lat <= lats[0] or lat >= lats[-1] or lon <= lons[0] or lon >= lons[-1]:
            return np.nan

        idx_lat = np.searchsorted(lats, lat) - 1
        idx_lon = np.searchsorted(lons, lon) - 1

        if idx_lat < 0 or idx_lat >= len(lats) - 1 or idx_lon < 0 or idx_lon >= len(lons) - 1:
            return np.nan

        w_lat = (lat - lats[idx_lat]) / (lats[idx_lat + 1] - lats[idx_lat])
        w_lon = (lon - lons[idx_lon]) / (lons[idx_lon + 1] - lons[idx_lon])

        v00 = grid[idx_lat, idx_lon]
        v01 = grid[idx_lat, idx_lon + 1]
        v10 = grid[idx_lat + 1, idx_lon]
        v11 = grid[idx_lat + 1, idx_lon + 1]

        return (v00 * (1 - w_lat) * (1 - w_lon) +
                v10 * w_lat * (1 - w_lon) +
                v01 * (1 - w_lat) * w_lon +
                v11 * w_lat * w_lon)

    # --- Accessor Methods ---

    def get_undulation(self, lat: float, lon: float, use_geoid: bool = False) -> float:
        """
        Get the geoid height at a point.

        Parameters
        ----------
        lat, lon : float
            Geographic coordinates in decimal degrees.
        use_geoid : bool, optional
            If True, use the 'geoid' grid (N, along ellipsoid normal).
            If False (default), use 'height_anomaly_ell' (ζ, along plumb line).
            The difference is typically < 0.05 m but ζ is more rigorous.

        Returns
        -------
        float
            Geoid height (undulation or height anomaly) in metres.
        """
        grid = 'geoid' if use_geoid else 'height_anomaly'
        return self._bilinear_interpolate(grid, lat, lon)

    def get_normal_gravity(self, lat: float, lon: float) -> float:
        """
        Get normal gravity on the reference ellipsoid (WGS84/GRS80).

        Returns
        -------
        float
            Normal gravity (γ₀) in milligal (mGal).
        """
        return self._bilinear_interpolate('gravity_ell', lat, lon)

    def get_gravity_anomaly(self, lat: float, lon: float) -> float:
        """
        Get the free‑air gravity anomaly (Δg).

        Returns
        -------
        float
            Gravity anomaly in milligal (mGal).
        """
        return self._bilinear_interpolate('gravity_anomaly', lat, lon)

    def get_gravity_earth(self, lat: float, lon: float) -> float:
        """
        Get the gravity value reduced to the geoid (spheroid).

        This is equivalent to normal gravity at ellipsoid plus classical
        free-air gravity anomaly: γ₀ + Δg_cl (mGal). It does NOT represent
        gravity at the Earth's surface. For surface gravity, use
        get_surface_gravity().

        Returns
        -------
        float
            Gravity at the geoid in milligal (mGal).
        """
        return self._bilinear_interpolate('gravity_earth', lat, lon)

    def get_surface_gravity(self, lat: float, lon: float, elevation: float) -> float:
        """
        Compute the actual gravity at the Earth's surface at a given
        topographic elevation.

        This uses the gravity disturbance (δg) definition:
            g_surface = γ_surface + δg
        where γ_surface is normal gravity evaluated at the topographic
        height (using the free‑air gradient), and δg is the gravity
        disturbance obtained from the grid.

        Parameters
        ----------
        lat, lon : float
            Geographic coordinates in decimal degrees.
        elevation : float
            Orthometric or ellipsoidal height in metres (typically DEM
            height plus geoid undulation if ellipsoidal height is desired).

        Returns
        -------
        float
            Surface gravity in milligal (mGal).
        """
        # Normal gravity at ellipsoid
        gamma0 = self.get_normal_gravity(lat, lon)
        # Free‑air correction to bring normal gravity to the given elevation
        # Using constant gradient -0.3086 mGal/m (valid for WGS84)
        gamma_surface = gamma0 - 0.3086 * elevation
        delta_g = self.get_gravity_disturbance(lat, lon)
        return gamma_surface + delta_g

    def get_gravitation(self, lat: float, lon: float) -> float:
        """
        Get the gravitational acceleration on the ellipsoid.

        Returns
        -------
        float
            Gravitation in milligal (mGal).
        """
        return self._bilinear_interpolate('gravitation_ell', lat, lon)

    def get_potential(self, lat: float, lon: float) -> float:
        """
        Get the normal potential (U₀) on the ellipsoid.

        Returns
        -------
        float
            Normal potential in m²/s².
        """
        return self._bilinear_interpolate('potential_ell', lat, lon)

    def get_second_r_derivative(self, lat: float, lon: float) -> float:
        """
        Get the second radial derivative of the potential (vertical gradient).

        Returns
        -------
        float
            Vertical gravity gradient in Eötvös (1 E = 10⁻⁹ s⁻²).
        """
        return self._bilinear_interpolate('second_r_derivative', lat, lon)

    def get_water_column(self, lat: float, lon: float) -> float:
        """
        Get the equivalent water column height.

        Returns
        -------
        float
            Water column height in metres.
        """
        return self._bilinear_interpolate('water_column', lat, lon)

    def get_gravity_disturbance(self, lat: float, lon: float) -> float:
        """
        Get the precise gravity disturbance (δg) with elevation correction.

        The gravity disturbance is defined as the difference between the
        magnitude of actual gravity and normal gravity at the same point in space:
            δg(h,λ,φ) = |∇W(h,λ,φ)| - |∇U(h,φ)|

        This grid includes the `h_over_geoid` column, which provides the
        topographic height used to evaluate normal gravity at the observation
        point.  This is the most rigorous gravity functional for comparing
        observed gravity with a global model, as it avoids the need for
        downward continuation.

        Returns
        -------
        float
            Gravity disturbance in milligal (mGal).
        """
        return self._bilinear_interpolate('gravity_disturbance', lat, lon)

    def get_gravity_disturbance_sa(self, lat: float, lon: float) -> float:
        """
        Get the gravity disturbance in spherical approximation.

        This is an approximation of the gravity disturbance computed using
        spherical coordinates and the radial derivative of the disturbing
        potential:
            δg_sa(0,λ,φ) = -∂T^c/∂r

        It is computationally simpler but less accurate in mountainous regions
        (errors can exceed 300 mGal) because it neglects the ellipsoidal shape
        of the Earth and the lateral variations of normal gravity.

        Returns
        -------
        float
            Gravity disturbance (spherical approximation) in milligal (mGal).
        """
        return self._bilinear_interpolate('gravity_disturbance_sa', lat, lon)

    def get_gravity_anomaly_cl(self, lat: float, lon: float) -> float:
        """
        Get the classical (free‑air) gravity anomaly (Δg_cl).

        The classical gravity anomaly is defined as the difference between the
        harmonic downward continuation of gravity to the geoid and normal
        gravity on the ellipsoid:
            Δg_cl(λ,φ) = |∇W^c(N)| - |∇U(0)|

        This is the traditional quantity used in Stokes's integral and in
        terrestrial gravimetry.  It corresponds to the free‑air anomaly without
        terrain corrections.

        Returns
        -------
        float
            Classical gravity anomaly in milligal (mGal).
        """
        return self._bilinear_interpolate('gravity_anomaly_cl', lat, lon)

    def get_gravity_anomaly_csb(self, lat: float, lon: float) -> float:
        """
        Get the complete spherical Bouguer anomaly (Δg_CSB) in mGal.

        **Definition (Uz & Ince, 2025):**
            The complete spherical Bouguer anomaly is obtained by subtracting
            the gravitational effect of all topographic masses (including
            terrain corrections) from the free‑air anomaly, and replacing water
            masses below the geoid with standard crustal density.  The reduction
            is performed in spherical geometry, accounting for the curvature of
            the Earth and the full topography.

            Mathematically:
                Δg_CSB = Δg_FA - δg_TGM
            where δg_TGM is the gravity disturbance computed from a topographic
            gravity model (TGM) that represents the attraction of all surface
            masses (rock, water, ice) with realistic densities.

        **Scientific significance:**
            - Represents the gravity field after removing the predictable
              effect of topography, thus revealing lateral density variations
              in the subsurface.
            - Suitable for deep crustal studies, mineral exploration, and
              geothermal assessments.
            - Compared to simple Bouguer, CSB yields significantly smoother
              anomalies over rugged terrain and is physically more correct.

        **Reference:**
            Uz, M. & Ince, E.S. (2025). Retrieving complete spherical Bouguer
            and isostatic gravity anomalies using global gravity forward models.
            Geophysical Journal International, 244(2), ggaf473.

        Returns
        -------
        float
            Complete spherical Bouguer anomaly (mGal).
        """
        return self._bilinear_interpolate('gravity_anomaly_csb', lat, lon)

    def get_gravity_anomaly_bg(self, lat: float, lon: float) -> float:
        """
        Get the simple Bouguer gravity anomaly (Δg_BG) in mGal.

        **Definition:**
            The simple Bouguer anomaly is the free‑air anomaly corrected for
            the gravitational attraction of an infinite horizontal slab
            (Bouguer plate) of constant crustal density (ρ_c = 2670 kg/m³)
            and, over oceans, replacement of water with crustal material.
            Terrain corrections and spherical curvature are ignored.

            Mathematically:
                Δg_BG = Δg_FA - 2πG ρ_c H + (water correction)
            where H is the topographic height (positive on land, negative at sea).

        **Scientific significance:**
            - Quick and easy to compute; useful for regional mapping where
              terrain effects are moderate.
            - Less accurate in mountainous areas, where terrain corrections
              can exceed 100 mGal, but adequate for many reconnaissance studies.

        **Reference:**
            Hofmann-Wellenhof, B. & Moritz, H. (2006). Physical Geodesy.
            Springer.

        Returns
        -------
        float
            Simple Bouguer anomaly (mGal).
        """
        return self._bilinear_interpolate('gravity_anomaly_bg', lat, lon)

    def get_gravity_anomaly_iso(self, lat: float, lon: float) -> float:
        """
        Get the isostatic gravity anomaly (Δg_ISO) in mGal.

        **Definition (Airy‑Heiskanen model):**
            The isostatic anomaly is the Bouguer anomaly further corrected for
            the gravitational effect of isostatic compensation.  In the
            Airy‑Heiskanen model, topographic masses are compensated by
            crustal roots (under continents) or anti‑roots (under oceans) at a
            depth of compensation T (typically 30 km).  The anomaly is computed
            as:
                Δg_ISO = Δg_FA - δg_TGM_ISO
            where δg_TGM_ISO is the gravity disturbance from a combined
            topographic‑isostatic model (e.g., RWI.TOIS_2012).

        **Scientific significance:**
            - Indicates departures from local isostatic equilibrium.
            - Positive anomalies suggest under‑compensation (mass excess),
              negative anomalies suggest over‑compensation (mass deficit).
            - Used to infer lithospheric flexure, tectonic uplift, and mantle
              dynamics.

        **Reference:**
            Uz, M. & Ince, E.S. (2025), ibid.
            Grombein, T. et al. (2014). A Wavelet‑Based Assessment of
            Topographic‑Isostatic Reductions for GOCE Gravity Gradients.
            Surveys in Geophysics, 35(4), 959–982.

        Returns
        -------
        float
            Isostatic gravity anomaly (mGal).
        """
        return self._bilinear_interpolate('gravity_anomaly_iso', lat, lon)

    def get_gravity_anomaly_sa(self, lat: float, lon: float) -> float:
        """
        Get the spherical approximation gravity anomaly (Δg_SA) in mGal.

        **Definition:**
            The spherical approximation anomaly is computed directly from the
            spherical harmonic coefficients of the disturbing potential using
            only the radial derivative (spherical approximation), without
            applying any topographic or isostatic corrections.  It assumes
            a spherical Earth and ignores the ellipsoidal shape.

            It is essentially the gravity anomaly evaluated on the ellipsoid
            (height_over_ell = 0) using the formula:
                Δg_SA(λ,φ) = (GM/r²) Σ (n+1) (R/r)^n ΔC_nm Y_nm(λ,φ)

        **Scientific significance:**
            - Convenient for quick global comparisons and for validating
              more advanced reductions.
            - Less accurate than the classical anomaly (Δg_cl) in regions of
              significant topography, because it does not account for the
              vertical gradient of normal gravity or the ellipsoidal shape.

        **Reference:**
            Barthelmes, F. (2013). Definition of Functionals of the Geopotential.
            GFZ Scientific Technical Report STR09/02.

        Returns
        -------
        float
            Spherical approximation gravity anomaly (mGal).
        """
        return self._bilinear_interpolate('gravity_anomaly_sa', lat, lon)

    def get_normal_gravity_at_45(self) -> float:
        """
        Get normal gravity at 45° latitude (for dynamic height calculation).
        Returns in mGal.
        """
        return self.get_normal_gravity(45.0, 0.0)

    def get_horizontal_gradient(self, lat: float, lon: float,
                                grid_name: str = 'gravity_anomaly') -> Tuple[float, float]:
        """
        Compute horizontal gradient (∂g/∂x, ∂g/∂y) at a point using finite differences.

        Parameters:
            lat, lon : float (decimal degrees)
            grid_name : str (default 'gravity_anomaly')

        Returns:
            (gx, gy) : float (mGal/m), or (nan, nan) if outside grid
        """
        g = self.grids.get(grid_name)
        if g is None:
            return (np.nan, np.nan)

        lons = g['lons']
        lats = g['lats']
        grid = g['grid']

        idx_lat = np.searchsorted(lats, lat) - 1
        idx_lon = np.searchsorted(lons, lon) - 1

        if idx_lat < 1 or idx_lat >= len(lats) - 2 or idx_lon < 1 or idx_lon >= len(lons) - 2:
            return (np.nan, np.nan)

        # Cell dimensions in metres
        lat_r = math.radians(lat)
        dx = (lons[idx_lon+1] - lons[idx_lon]) * (np.pi/180.0) * 6371000.0 * math.cos(lat_r)
        dy = (lats[idx_lat+1] - lats[idx_lat]) * (np.pi/180.0) * 6371000.0

        # Central difference
        gx = (grid[idx_lat, idx_lon+1] - grid[idx_lat, idx_lon-1]) / (2.0 * dx)
        gy = (grid[idx_lat+1, idx_lon] - grid[idx_lat-1, idx_lon]) / (2.0 * dy)

        return (gx, gy)

# ============================================================================
# CLASS: Pawitra Stratigraphy (Mt. Penanggungan)
# ============================================================================

class PawitraStratigraphy:
    """
    High-resolution 2D lithological model of Mt. Penanggungan (Pawitra)
    based on Paripurno et al. (2018) IOP Conf. Ser. Earth Environ. Sci. 212:012045.

    Implements anisotropic distance-weighted interpolation for 19 volcanic units
    (lavas, pyroclastic flows, lahars) with directional ellipses, providing
    local density estimates and material descriptions for geodetic-gravimetric inversion.

    Usage:
        strat = PawitraStratigraphy()
        density, unit_code, description = strat.query(lat, lon)
        density_only = strat.density_at(lat, lon)
    """

    # Density constants [kg/m³]
    DENSITY_LAVA = 2650.0
    DENSITY_PYRO = 2250.0
    DENSITY_LAHAR = 2050.0
    DENSITY_BACKGROUND = 2450.0  # Regional basement

    # Reference peak coordinates
    PEAK_LAT = -7.6156
    PEAK_LON = 112.6200

    def __init__(self, peak_lat: float = PEAK_LAT, peak_lon: float = PEAK_LON):
        self.peak_lat = peak_lat
        self.peak_lon = peak_lon
        self._units: List[Dict[str, Any]] = []
        self._build_units()

    def _build_units(self):
        """Populate the unit database with geometry, density, and description."""
        # Lava units (Plv)
        lavas = [
            ("Plv1_Jambe", 2650.0, -7.6156, 112.6050, 270.0, 1.5, 1.0,
             "Jambe cone: blackish-grey pyroxene andesite lava, massive, hypocrystalline, aphanitic-phaneritic, euhedral-subhedral, plagioclase, pyroxene, hornblende"),
            ("Plv2_Gajahmungkur", 2650.0, -7.6036, 112.6300, 45.0, 1.8, 1.2,
             "Gajahmungkur cone: brown andesite lava, hypocrystalline, subhedral-anhedral, fractured, good aquifer"),
            ("Plv3_Bekel", 2650.0, -7.6236, 112.6080, 225.0, 1.5, 1.0,
             "Bekel cone: blackish-grey andesite–basalt lava, hypocrystalline, phaneritic-aphantic, anhedral-subhedral, plagioclase, pyroxene, hornblende, olivine"),
            ("Plv4_Bendo", 2650.0, -7.6306, 112.6200, 180.0, 1.5, 1.0,
             "Bendo cone: blackish-brownish grey andesite lava, massive, hypocrystalline, soft phaneritic, anhedral-subhedral"),
            ("Plv5_Genting", 2650.0, -7.6006, 112.6200, 0.0, 1.8, 1.2,
             "Genting cone: brown andesite lava, hypocrystalline, subhedral-anhedral, plagioclase, hornblende, pyroxene"),
            ("Plv6_Wangi", 2650.0, -7.6156, 112.6350, 90.0, 1.5, 1.0,
             "Wangi cone: brown andesite lava, hypocrystalline, subhedral-anhedral, plagioclase, opaque minerals, glass mass"),
            ("Plv7_Kemuncup", 2650.0, -7.6256, 112.6350, 135.0, 1.5, 1.0,
             "Kemuncup cone: grey andesite lava, hypocrystalline, subhedral-anhedral, plagioclase, pyroxene"),
            ("Plv8_Watesnegoro", 2650.0, -7.6156, 112.6200, 20.0, 2.5, 1.8,
             "Watesnegoro unit: grey andesite lava, brecciated, hypocrystalline, subhedral-anhedral, plagioclase, pyroxene, widespread"),
            ("Plv9_Kedungudi", 2650.0, -7.6156, 112.6200, 0.0, 0.8, 0.8,
             "Kedungudi unit: hornblende andesite lava, grey, hypocrystalline, subhedral-anhedral, plagioclase, pyroxene, hornblende, summit area"),
        ]
        for code, dens, lat, lon, az, maj, mn, desc in lavas:
            self._units.append({
                "code": code, "type": "lava", "density": dens, "lat": lat, "lon": lon,
                "azimuth": az, "sigma_major_km": maj, "sigma_minor_km": mn,
                "description": desc
            })

        # Pyroclastic flow units (Pap)
        pyros = [
            ("Pap1_Bekel", 2250.0, -7.6236, 112.6080, 0.0, 1.2, 1.0,
             "Bekel pyroclastic flow: grey, poorly sorted, close fabric, subangular-angular, andesite fragments 2-12 cm, sand matrix"),
            ("Pap2_Bendo", 2250.0, -7.6306, 112.6200, 0.0, 1.0, 1.0,
             "Bendo pyroclastic flow: brown, close fabric, subangular-angular, andesite fragments 4-11 cm, sand matrix"),
            ("Pap3_Wangi", 2250.0, -7.6156, 112.6350, 120.0, 2.0, 1.5,
             "Wangi pyroclastic flow: brown, poorly sorted, open fabric, angular-subangular, andesite fragments 3-5 cm, silica cement, glass mass, SE direction"),
            ("Pap4_Kemuncup", 2250.0, -7.6256, 112.6350, 100.0, 1.8, 1.2,
             "Kemuncup pyroclastic flow: brownish-grey, poorly sorted, close fabric, subangular-subrounded, andesite fragments 0.2-20 cm, coarse sand matrix"),
            ("Pap5_Masjedong", 2250.0, -7.6156, 112.6200, 0.0, 3.0, 2.5,
             "Masjedong pyroclastic flow: youngest, brown, poorly sorted, close fabric, subangular-subrounded, andesite fragments 2-15 cm, sand matrix, spreads N, NE, W"),
            ("Pap6", 2250.0, -7.6156, 112.6200, 270.0, 0.6, 0.6,
             "Summit pyroclastic flow: youngest near peak, andesite and pumice fragments 2 mm-20 cm, sand matrix, western summit area"),
        ]
        for code, dens, lat, lon, az, maj, mn, desc in pyros:
            self._units.append({
                "code": code, "type": "pyroclastic", "density": dens, "lat": lat, "lon": lon,
                "azimuth": az, "sigma_major_km": maj, "sigma_minor_km": mn,
                "description": desc
            })

        # Lahar units (Plh, Alh)
        lahars = [
            ("Alh1_Janjing", 2050.0, -7.6300, 112.6300, 150.0, 2.0, 1.5,
             "Arjuna-Welirang lahar: sand layer + pyroclastic flow, andesite fragments 8-95 cm, sand matrix, SE direction (external source)"),
            ("Plh1_Bekel", 2050.0, -7.6236, 112.6080, 0.0, 1.2, 1.0,
             "Bekel lahar: sand and minor mud, andesite fragments 6-36 cm"),
            ("Plh2_Kemucup", 2050.0, -7.6256, 112.6350, 270.0, 1.5, 1.2,
             "Kemuncup lahar: brownish-grey, poorly sorted, open fabric, subangular-subrounded, andesite fragments 0.2-40 cm, sand matrix, westwards"),
            ("Plh3_Masjedong", 2050.0, -7.6156, 112.6200, 30.0, 2.0, 1.5,
             "Masjedong lahar: grey, poorly sorted, open fabric, subangular-subrounded, andesite fragments 0.2-35 cm, sand matrix, NE direction"),
        ]
        for code, dens, lat, lon, az, maj, mn, desc in lahars:
            self._units.append({
                "code": code, "type": "lahar", "density": dens, "lat": lat, "lon": lon,
                "azimuth": az, "sigma_major_km": maj, "sigma_minor_km": mn,
                "description": desc
            })

    def _anisotropic_distance(self, lat: float, lon: float,
                              unit_lat: float, unit_lon: float,
                              azimuth_deg: float, sigma_major_km: float,
                              sigma_minor_km: float) -> float:
        """
        Compute anisotropic (elliptical) distance from a point to a unit's source.

        Args:
            lat, lon: Target coordinates (deg)
            unit_lat, unit_lon: Source coordinates (deg)
            azimuth_deg: Direction of major axis (deg clockwise from north)
            sigma_major_km: Half‑length of major axis (km)
            sigma_minor_km: Half‑length of minor axis (km)

        Returns:
            Normalised distance (dimensionless)
        """
        phi = math.radians(azimuth_deg)
        # Convert to km: 1° ≈ 111 km
        dy = (lat - unit_lat) * 111.0
        dx = (lon - unit_lon) * 111.0 * math.cos(math.radians((lat + unit_lat) * 0.5))
        # Rotate to ellipse coordinates
        x_rot = dx * math.cos(phi) + dy * math.sin(phi)
        y_rot = -dx * math.sin(phi) + dy * math.cos(phi)
        # Normalised distance in ellipse space
        dist_major = x_rot / sigma_major_km
        dist_minor = y_rot / sigma_minor_km
        return math.hypot(dist_major, dist_minor)

    def query(self, lat: float, lon: float) -> Tuple[float, str, str]:
        """
        Retrieve the most representative lithological unit at a given coordinate.

        Args:
            lat: Latitude (deg)
            lon: Longitude (deg)

        Returns:
            Tuple (density_kg_m3, unit_code, material_description)
        """
        best_density = self.DENSITY_BACKGROUND
        best_code = "Regional"
        best_desc = "Regional basement: breccia, tuff, older lahar deposits (undifferentiated)"
        best_weight = 0.0

        for u in self._units:
            dist = self._anisotropic_distance(lat, lon, u["lat"], u["lon"],
                                              u["azimuth"], u["sigma_major_km"], u["sigma_minor_km"])
            if dist > 3.0:   # influence limited to 3 sigma
                continue
            weight = 1.0 / (dist ** 2.0 + 1e-6)   # inverse square distance
            if weight > best_weight:
                best_weight = weight
                best_density = u["density"]
                best_code = u["code"]
                best_desc = u["description"]

        # Enforce known stratigraphy at Jolotundo observatory (W-NW flank)
        # Coordinates: -7.609444°, 112.595556°
        # This area is dominated by Pap5_Masjedong pyroclastic flow.
        if (-7.615 < lat < -7.605) and (112.590 < lon < 112.605):
            best_density = self.DENSITY_PYRO
            best_code = "Pap5_Masjedong"
            best_desc = "Masjedong pyroclastic flow: youngest unit, brown, poorly sorted, andesite fragments 2-15 cm, sand matrix, spreads westwards to the flank."

        return best_density, best_code, best_desc

    def density_at(self, lat: float, lon: float) -> float:
        """Return only density (kg/m³) at a point."""
        return self.query(lat, lon)[0]

    def generate_density_matrix(self, lats_grid: np.ndarray, lons_grid: np.ndarray) -> np.ndarray:
        """
        Generate a 2D density matrix over the DEM grid.

        Args:
            lats_grid, lons_grid: 2D arrays of latitudes and longitudes (same shape)

        Returns:
            2D array of densities (kg/m³)
        """
        rows, cols = lats_grid.shape
        density = np.full_like(lats_grid, self.DENSITY_BACKGROUND, dtype=np.float64)
        for i in range(rows):
            for j in range(cols):
                density[i, j] = self.density_at(lats_grid[i, j], lons_grid[i, j])
        return density

# ============================================================================
# CLASS: Isostatic Calculator (Airy-Heiskanen)
# ============================================================================

class IsostaticCalculator:
    """
    Computes Airy‑Heiskanen isostatic compensation parameters.

    **Scientific Background (Heiskanen & Moritz, 1967):**
        The Airy‑Heiskanen model assumes that topographic masses are compensated
        by crustal roots (under continents) or anti‑roots (under oceans) relative
        to a normal crustal thickness T₀. The root depth is computed as:
            On land:  t = (ρ_c / (ρ_m - ρ_c)) * H
            On ocean: t = ((ρ_c - ρ_w) / (ρ_m - ρ_c)) * |H|
        where H is the elevation (positive on land, negative at sea).

    **References:**
        Heiskanen, W.A. & Moritz, H. (1967). Physical Geodesy.
        Uz, M. & Ince, E.S. (2025). Retrieving complete spherical Bouguer... GJI.
    """

    def __init__(self, rho_crust: float = RHO_CRUST,
                 rho_mantle: float = RHO_MANTLE,
                 rho_water: float = 1025.0,
                 T0: float = 30000.0):
        """
        Parameters:
            rho_crust : float (kg/m³) default 2670
            rho_mantle : float (kg/m³) default 3270
            rho_water : float (kg/m³) default 1025
            T0 : float (metres) normal crustal thickness, default 30000 (30 km)
        """
        self.rho_c = rho_crust
        self.rho_m = rho_mantle
        self.rho_w = rho_water
        self.T0 = T0

    def compute_root_depth(self, elevation: float) -> float:
        """
        Compute the depth of the isostatic root (or anti-root) for a given
        elevation (positive on land, negative over oceans).

        Returns:
            float : Root depth in metres.
        """
        if elevation >= 0:
            # Continental root
            return (self.rho_c / (self.rho_m - self.rho_c)) * elevation
        else:
            # Oceanic anti-root (negative elevation)
            return (self.rho_c - self.rho_w) / (self.rho_m - self.rho_c) * abs(elevation)

    def compute_moho_depth(self, elevation: float) -> float:
        """
        Compute the depth of the Mohorovičić discontinuity (Moho) below sea level.

        Returns:
            float : Moho depth in metres.
        """
        return self.T0 + self.compute_root_depth(elevation)


# ============================================================================
# CLASS: Spectral Analyzer (FFT 2D)
# ============================================================================

class SpectralAnalyzer:
    """
    Performs 2D spectral analysis (FFT) on gridded data to extract dominant
    wavelengths and power spectra.

    **Algorithm (Cooley & Tukey, 1965):**
        The 2D Fast Fourier Transform is applied to the input grid. The power
        spectrum is computed as |FFT|². A radial average is computed from the
        centre of the spectrum to obtain a 1D power profile. The dominant
        wavelength corresponds to the frequency with maximum power.

    **References:**
        Cooley, J.W. & Tukey, J.W. (1965). An algorithm for the machine
        calculation of complex Fourier series. Math. Comput., 19, 297–301.
    """

    @staticmethod
    def power_spectrum_2d(grid: np.ndarray, cellsize_deg: float) -> Dict[str, Any]:
        """
        Compute the 2D power spectrum and return dominant wavelength and
        radial power profile.

        Parameters:
            grid : np.ndarray (2D)
            cellsize_deg : float (degrees)

        Returns:
            dict: Contains:
                - wavelength_dominant_m: Dominant wavelength (metres)
                - total_energy: Total spectral energy
                - radial_frequency: Array of radial frequencies (1/m)
                - radial_power: Array of radial power values
        """
        # 2D FFT
        f = np.fft.fft2(grid)
        fshift = np.fft.fftshift(f)
        power = np.abs(fshift) ** 2

        # Spatial frequencies (cycles per metre)
        ny, nx = grid.shape
        dx = cellsize_deg * 111320.0  # metres per pixel

        fx = np.fft.fftfreq(nx, d=dx)
        fy = np.fft.fftfreq(ny, d=dx)

        fx_shift = np.fft.fftshift(fx)
        fy_shift = np.fft.fftshift(fy)

        fx_grid, fy_grid = np.meshgrid(fx_shift, fy_shift)
        fr = np.sqrt(fx_grid**2 + fy_grid**2)

        # Radial averaging
        r_max = np.max(fr)
        bins = 100
        r_edges = np.linspace(0, r_max, bins + 1)
        r_centers = (r_edges[:-1] + r_edges[1:]) / 2

        radial_power = np.zeros(bins)
        for i in range(bins):
            mask = (fr >= r_edges[i]) & (fr < r_edges[i+1])
            if np.any(mask):
                radial_power[i] = np.mean(power[mask])

        # Dominant wavelength
        idx_max = np.argmax(radial_power)
        f_dom = r_centers[idx_max]
        wavelength_dom = 1.0 / f_dom if f_dom > 0 else np.inf

        return {
            'wavelength_dominant_m': wavelength_dom,
            'total_energy': np.sum(power),
            'radial_frequency': r_centers,
            'radial_power': radial_power,
        }

    @staticmethod
    def analyze_dem(dem_reader: ASCDEMReader) -> Dict[str, Any]:
        """Convenience method to analyse a DEM grid."""
        return SpectralAnalyzer.power_spectrum_2d(
            dem_reader.elevations,
            dem_reader.cellsize
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def geodetic_to_cartesian(
    lat_deg: float,
    lon_deg: float,
    h_m: float,
    a: float = WGS84_A,
    f: float = WGS84_F
) -> Tuple[float, float, float]:
    """
    Convert geodetic coordinates (latitude, longitude, ellipsoidal height)
    to Earth‑centred, Earth‑fixed Cartesian coordinates (X, Y, Z) in metres.

    Parameters
    ----------
    lat_deg, lon_deg : float
        Geographic coordinates in decimal degrees.
    h_m : float
        Ellipsoidal height in metres.
    a, f : float
        Semi‑major axis and flattening of the reference ellipsoid.
        Default: WGS84.

    Returns
    -------
    x, y, z : float
        Cartesian coordinates in metres.

    Notes
    -----
    Uses the standard closed‑form conversion:
        X = (N + h) cos φ cos λ
        Y = (N + h) cos φ sin λ
        Z = (N(1 − e²) + h) sin φ
    where N is the radius of curvature of the prime vertical.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    e2 = 2.0 * f - f * f
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * math.cos(lon)
    y = (N + h_m) * cos_lat * math.sin(lon)
    z = (N * (1.0 - e2) + h_m) * sin_lat
    return x, y, z


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    radius: float = 6371000.0
) -> float:
    """
    Compute the great‑circle distance between two geographic points using the
    haversine formula.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        Geographic coordinates in decimal degrees.
    radius : float, optional
        Mean Earth radius in metres. Default: 6,371,000 m.

    Returns
    -------
    float
        Distance in metres.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius * c


def normal_gravity_somigliana(lat_deg: float) -> float:
    """
    Compute normal gravity on the WGS84 ellipsoid at a given latitude.
    Returns value in mGal.
    Uses Somigliana's formula with WGS84 constants.
    """
    # WGS84 constants
    a = 6378137.0          # semi-major axis [m]
    f = 1.0 / 298.257223563
    gm = 3.986004418e14    # geocentric gravitational constant [m³/s²]
    omega = 7.292115e-5    # angular velocity [rad/s]

    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    sin2 = sin_lat * sin_lat

    e2 = 2*f - f*f
    b = a * (1 - f)
    m_param = (omega**2 * a**2 * b) / gm

    # q0 and q0_prime for Somigliana
    e_prime2 = e2 / (1 - e2)
    e_prime = math.sqrt(e_prime2)
    atan_e = math.atan(e_prime)
    q0 = 0.5 * ((1 + 3/e_prime2) * atan_e - 3/e_prime)
    q0_prime = 3 * (1 + 1/e_prime2) * (1 - atan_e/e_prime) - 1

    # gamma_a (equatorial) and gamma_b (polar)
    gamma_a = (gm / (a * b)) * (1 - m_param - (m_param/6) * e_prime * q0_prime / q0)
    gamma_b = (gm / (a * b)) * (1 + (m_param/3) * e_prime * q0_prime / q0 - m_param)

    # Somigliana formula
    k = (gamma_b / gamma_a) - 1
    gamma = gamma_a * (1 + k * sin2) / math.sqrt(1 - e2 * sin2)

    return gamma * 1e5   # convert m/s² → mGal


# ============================================================================
# SELF‑TEST & ELEGANT DISPLAY FORMATTER
# ============================================================================

if __name__ == "__main__":
    """
    ========================================================================
    BAGIAN UTAMA: DEMONSTRASI KOMPUTASI GEODESI-GEOFISIKA TERINTEGRASI
    ========================================================================

    Bagian ini mengintegrasikan:
        - Model Elevasi Digital (Copernicus DSM 30m)
        - Model medan gravitasi global XGM2019e-2159 (Zingerle et al., 2020)
        - Stratigrafi lokal Gunung Penanggungan (Paripurno et al., 2018)
        - Koreksi medan tesseroid (Heck & Seitz, 2007)
        - Reduksi Bouguer dan isostasi Airy-Heiskanen (Heiskanen & Moritz, 1967)
        - Analisis spektral FFT (Cooley & Tukey, 1965)

    Semua perhitungan mengacu pada sistem referensi WGS84 dan EGM2008,
    konsisten dengan model XGM2019e (tide‑free).
    ========================================================================
    """

    import textwrap
    import shutil
    import contextlib
    import io
    import math
    import numpy as np

    # =========================================================================
    #  TERMINAL WIDTH DETECTION & PRINTING UTILITY
    # =========================================================================
    term_width = shutil.get_terminal_size(fallback=(58, 24)).columns
    BOX_WIDTH = min(term_width - 2, 72)
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Titik observasi: Jolotundo, lereng barat‑laut Gunung Penanggungan
    LAT, LON = -7.609444, 112.595556

    def print_scientific_box(title: str, data: dict, width: int = BOX_WIDTH) -> None:
        """Cetak kotak hasil dengan tata letak profesional dan pembungkusan teks."""
        border = BOLD + "=" * width + RESET
        sep = BOLD + "-" * width + RESET

        print(border)
        print(BOLD + title.center(width) + RESET)
        print(sep)

        if not data:
            print(border)
            return

        max_key_len = max(len(str(k)) for k in data.keys())
        indent_size = max_key_len + 3
        val_max_width = width - indent_size
        if val_max_width < 15:
            val_max_width = width

        for key, value in data.items():
            key_str = str(key)
            if key_str.strip().startswith('─'):
                print('─' * width)
                continue
            val_str = str(value)
            formatted_key = BOLD + key_str.ljust(max_key_len) + RESET + " : "
            if '\n' in val_str:
                lines = val_str.split('\n')
                print(formatted_key + lines[0])
                for line in lines[1:]:
                    print(" " * indent_size + line)
            else:
                wrapped_val = textwrap.fill(val_str, width=val_max_width)
                lines = wrapped_val.split('\n')
                print(formatted_key + lines[0])
                for line in lines[1:]:
                    print(" " * indent_size + line)

        print(border)
        print()

    # =========================================================================
    #  KONSTANTA FISIKA – Acuan dari IERS Conventions (2010) & WGS84
    # =========================================================================
    GRAVITATIONAL_CONSTANT = 6.67430e-11      # m³ kg⁻¹ s⁻² (CODATA 2018)
    RHO_CRUST = 2670.0                        # Densitas kerak standar (kg/m³)
    RHO_MANTLE = 3270.0                       # Densitas mantel untuk isostasi Airy
    FREE_AIR_GRADIENT = -0.3086               # Gradien udara bebas (mGal/m)
    PI = math.pi
    T0 = 30000.0                              # Ketebalan kerak normal (30 km) – Airy

    # =========================================================================
    #  1. MEMUAT DATA – DEM & GRID GRAVITASI GLOBAL
    # =========================================================================
    dem_loaded = False
    xgm_loaded = False

    print("⚙️ [DIAGNOSTIC] Initialising local grids and DEM...")

    # Alihkan output sementara agar tidak mengganggu tampilan
    with contextlib.redirect_stdout(io.StringIO()):
        # ---- 1a. DEM Copernicus DSM 30m (resolusi ~30 m) ----
        try:
            dem = ASCDEMReader("jolotundo_cop30.asc")
            meta = dem.get_metadata()
            elev = dem.get_elevation(LAT, LON)   # Orthometric height (m)
            dem_loaded = True
        except FileNotFoundError:
            pass

        # ---- 1b. Model gravitasi XGM2019e-2159 (Zingerle et al., 2020) ----
        try:
            xgm = XGMGridReader(data_dir=".", grid_prefix="XGM2019e_2159")
            xgm_loaded = True
        except FileNotFoundError:
            pass

    if not dem_loaded or not xgm_loaded:
        print("❌ Data not found. Please ensure jolotundo_cop30.asc and XGM grids exist.")
        exit(1)

    # ---- 1c. Stratigrafi lokal (Paripurno et al., 2018) ----
    print("⚙️ [DIAGNOSTIC] Initialising Pawitra Stratigraphy...")
    strat = PawitraStratigraphy()
    density_local, unit_code, unit_desc = strat.query(LAT, LON)
    print(f"✅ Stratigraphy loaded: density={density_local:.1f} kg/m³, unit={unit_code}")

    # =========================================================================
    #  2. ANALISIS DEM – TURUNAN MORFOMETRI & KOREKSI MEDAN
    # =========================================================================
    # --- Koreksi medan tesseroid (Heck & Seitz, 2007; Uieda et al., 2016) ---
    # Koreksi medan positif menghitung tarikan massa topografi di sekitar titik,
    # menggunakan integrasi Gauss‑Legendre 3×3×3 dan subdivisi adaptif.
    tc_tess_global = dem.get_terrain_correction_tesseroid(LAT, LON,
                                                          radius_deg=0.05,
                                                          rho=RHO_CRUST)

    tc_tess_local = dem.get_terrain_correction_tesseroid(LAT, LON,
                                                         radius_deg=0.05,
                                                         rho=density_local)

    # --- Turunan morfometrik (Horn, 1981) ---
    slope = dem.get_slope(LAT, LON)
    aspect = dem.get_aspect(LAT, LON)
    prof_curv = dem.get_profile_curvature(LAT, LON)  # kelengkungan searah lereng
    plan_curv = dem.get_plan_curvature(LAT, LON)     # kelengkungan tegak lereng

    # Siapkan data untuk output
    bounds_str = f"Lon: {meta['lon_min']:.4f}° to {meta['lon_max']:.4f}°\nLat: {meta['lat_min']:.4f}° to {meta['lat_max']:.4f}°"
    elev_stats_str = f"Min: {meta['elevation_min']:.2f} m, Max: {meta['elevation_max']:.2f} m, Mean: {meta['elevation_mean']:.2f} m"

    dem_data = {
        "Data Source": "Copernicus Global DSM 30m (COP30)",
        "File Path": meta['filepath'],
        "Grid Dimensions": f"{meta['ncols']} columns × {meta['nrows']} rows",
        "Spatial Bounds": bounds_str,
        "Elevation Stats": elev_stats_str,
        "Jolotundo Elevation": f"{elev:.3f} m (Lat -7.609444°, Lon 112.595556°)",
        "Slope": f"{slope:.2f}°" if not math.isnan(slope) else "N/A",
        "Aspect": f"{aspect:.2f}° (from North)" if not math.isnan(aspect) else "N/A",
        "Profile Curvature": f"{prof_curv:.6f} 1/m" if not math.isnan(prof_curv) else "N/A",
        "Plan Curvature": f"{plan_curv:.6f} 1/m" if not math.isnan(plan_curv) else "N/A",
        "TC (global ρ=2670)": f"{tc_tess_global:.4f} mGal (Tesseroid)",
        "TC (local ρ=2250)": f"{tc_tess_local:.4f} mGal (Tesseroid)",
    }

    # ---- Stratigrafi lokal ----
    strat_data = {
        "Reference": "Paripurno et al. (2018) IOP Conf. Ser. 212:012045",
        "Global Density (WGS84/GRS80)": f"{RHO_CRUST:.0f} kg/m³",
        "Local Density (Pawitra)": f"{density_local:.1f} kg/m³",
        "Unit Code": unit_code,
        "Material Description": unit_desc,
    }

    # =========================================================================
    #  3. MEDAN GRAVITASI DASAR – XGM2019e-2159
    # =========================================================================
    """
    Menurut Barthelmes (2013), fungsional medan gravitasi dapat dihitung dari
    koefisien harmonik bola. Di sini kita ambil beberapa besaran dasar:
        - γ₀ : gravitasi normal di ellipsoid (Somigliana)
        - ζ  : height anomaly (Molodensky) – mengikuti garis unting
        - N  : undulasi geoid – sepanjang normal ellipsoid
        - Δg : anomali gravitasi modern (Molodensky)
        - δg : gravity disturbance – selisih g(P) – γ(P)
        - Δg_cl : anomali klasik (Stokes) – g di geoid – γ₀
        - g_geoid : gravitasi di geoid = γ₀ + Δg_cl (downward continued)
    """
    lat, lon = LAT, LON
    gamma0 = xgm.get_normal_gravity(lat, lon)
    zeta = xgm.get_undulation(lat, lon, use_geoid=False)
    N = xgm.get_undulation(lat, lon, use_geoid=True)

    # Fungsi aman untuk grid opsional (mengembalikan NaN jika tidak ada)
    def safe_get(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (KeyError, ValueError, TypeError):
            return float('nan')

    dg = safe_get(xgm.get_gravity_anomaly, lat, lon)
    g_geoid = safe_get(xgm.get_gravity_earth, lat, lon)   # ≡ γ₀ + Δg_cl
    grad = safe_get(xgm.get_second_r_derivative, lat, lon)
    dg_dist = safe_get(xgm.get_gravity_disturbance, lat, lon)
    dg_dist_sa = safe_get(xgm.get_gravity_disturbance_sa, lat, lon)
    dg_cl = safe_get(xgm.get_gravity_anomaly_cl, lat, lon)
    csb = safe_get(xgm.get_gravity_anomaly_csb, lat, lon)
    bg = safe_get(xgm.get_gravity_anomaly_bg, lat, lon)
    iso = safe_get(xgm.get_gravity_anomaly_iso, lat, lon)
    dg_sa_anom = safe_get(xgm.get_gravity_anomaly_sa, lat, lon)

    # ---- Gravitasi permukaan awal (sementara) ----
    # Menggunakan koreksi udara bebas dan gravity disturbance
    h_ell = elev + N                     # tinggi ellipsoidal (m)
    gamma_surface_est = gamma0 - 0.3086 * h_ell
    g_surface_est = gamma_surface_est + dg_dist if not math.isnan(dg_dist) else float('nan')

    xgm_data = {
        "Data Source": "XGM2019e-2159 (GFZ-Potsdam/ICGEM, d/o 2159, tide-free)",
        "Coordinates": f"Lat {lat:.6f}°, Lon {lon:.6f}°",
        "Undulation (ζ)": f"{zeta:.8f} m (Height Anomaly, plumb line)",
        "Undulation (N)": f"{N:.8f} m (Geoid, ellipsoid normal)",
        "Normal Gravity (γ₀)": f"{gamma0:.4f} mGal",
        "Gravity Anomaly (Δg)": f"{dg:.4f} mGal" if not math.isnan(dg) else "N/A",
        "Gravity at Geoid (g_geoid)": f"{g_geoid:.4f} mGal" if not math.isnan(g_geoid) else "N/A",
        "Surface Gravity (g)": f"{g_surface_est:.4f} mGal (using δg)" if not math.isnan(g_surface_est) else "N/A",
        "Vertical Gradient": f"{grad:.2f} Eötvös" if not math.isnan(grad) else "N/A",
        "Gravity Disturbance (δg)": f"{dg_dist:.4f} mGal" if not math.isnan(dg_dist) else "N/A",
        "Gravity Disturbance (sa)": f"{dg_dist_sa:.4f} mGal" if not math.isnan(dg_dist_sa) else "N/A",
        "Classical Anomaly (Δg_cl)": f"{dg_cl:.4f} mGal" if not math.isnan(dg_cl) else "N/A",
        "Complete Bouguer (CSB)": f"{csb:.4f} mGal" if not math.isnan(csb) else "N/A",
        "Simple Bouguer (BG)": f"{bg:.4f} mGal" if not math.isnan(bg) else "N/A",
        "Isostatic (ISO)": f"{iso:.4f} mGal" if not math.isnan(iso) else "N/A",
        "Spherical Approx (SA)": f"{dg_sa_anom:.4f} mGal" if not math.isnan(dg_sa_anom) else "N/A",
    }

    # =========================================================================
    #  MODEL ACCURACY & UNCERTAINTY (berdasarkan dokumentasi resmi)
    # =========================================================================
    """
    AKURASI DAN KETIDAKPASTIAN MODEL
    ---------------------------------
    Informasi ini mengacu pada:
        - Copernicus DEM Product Handbook (Airbus, 2022)
        - Zingerle et al. (2020), Journal of Geodesy, XGM2019e
        - Barthelmes (2013), STR09/02
    """
    accuracy_data = {
        "───────────────────────────── ": None,
        "COPERNICUS DEM GLO-30 (30m)": "",
        "Vertical Datum": "EGM2008 (EPSG 3855)",
        "Horizontal Datum": "WGS84-G1150 (EPSG 4326)",
        "Absolute Vertical (LE90, spec)": "< 4 m",
        "Absolute Vertical (LE90, global)": "1.92 m (mean, excl. Antarctica/Greenland)",
        "Relative Vertical (LE90)": "< 2 m (slope ≤20°) / < 4 m (slope >20°)",
        "Horizontal (CE90)": "< 6 m",
        "Est. local uncertainty (LE90)": f"±1.5–3.0 m (at Jolotundo, slope ~14°)",
        "───────────────────────────── ": None,
        "XGM2019e-2159 GRAVITY FIELD": "",
        "Max degree/order": "2159 (~9.3 km resolution)",
        "Tide system": "tide‑free",
        "Reference system": "WGS84 (consistent)",
        "Geoid accuracy (global RMS)": "~0.205 m (from GNSS/levelling studies)",
        "Geoid accuracy (local, Indonesia)": "±0.10–0.20 m (estimated, 95% CI)",
        "Gravity anomaly (1σ)": "±5–10 mGal (mountainous terrain)",
        "Gravity disturbance (1σ)": "±5–10 mGal + 0.3 mGal/m × δh (elevation error)",
        "───────────────────────────── ": None,
        "REFERENCES": "",
        "Copernicus DEM": "Airbus (2022). Copernicus DEM Product Handbook.",
        "XGM2019e": "Zingerle et al. (2020). J. Geod., 94(7).",
        "Functionals": "Barthelmes (2013). GFZ STR09/02.",
    }

    # =========================================================================
    #  4. DERIVASI GEOFISIKA LANJUTAN (Barthelmes, 2013; Uz & Ince, 2025)
    # =========================================================================
    """
    Bagian ini menggabungkan efek topografi dan isostasi untuk menghasilkan
    anomali Bouguer lengkap dan anomali isostatik, serta besaran geodesi seperti
    potensial gangguan (T), bilangan geopotensial (C), dan tinggi dinamik.
    """
    if not any(math.isnan(v) for v in [elev, gamma0, zeta, N, csb, bg,
                                       dg_dist, dg_dist_sa, dg_cl, dg_sa_anom]):

        # ---- 4a. Koreksi medan global dari selisih CSB – BG ----
        # CSB adalah anomali Bouguer spherical lengkap; BG adalah simple Bouguer.
        # Selisihnya mencerminkan efek topografi 3D yang hilang pada reduksi pelat.
        TC_global = csb - bg
        # Menurut Uz & Ince (2025), TC_global = pengaruh topografi penuh.

        # ---- 4b. Koreksi spherical shell (Barthelmes, 2013, pers. 69–71) ----
        # Koreksi ini memperbaiki undulasi geoid akibat massa topografi di atas geoid.
        gamma_ms2 = gamma0 * 1e-5          # mGal → m/s²
        coeff_shell = 2.0 * PI * GRAVITATIONAL_CONSTANT * RHO_CRUST / gamma_ms2
        dN_top = -coeff_shell * (elev ** 2)
        # Nilai negatif berarti geoid turun di bawah gunung (efek massa).

        # ---- 4c. Gradien udara bebas dan gravitasi normal di permukaan ----
        # Tinggi ellipsoidal (h = H + N) digunakan untuk koreksi yang lebih akurat.
        h_ell = elev + N
        FAC = FREE_AIR_GRADIENT * h_ell          # koreksi udara bebas (mGal)
        gamma_surface = gamma0 + FAC             # gravitasi normal di permukaan

        # ---- 4d. GRAVITASI PERMUKAAN YANG BENAR (menggunakan δg) ----
        # Berdasarkan definisi gravity disturbance (Barthelmes, 2013):
        #   δg(P) = g(P) – γ(P)  →  g(P) = γ(P) + δg(P)
        # Dengan γ(P) = γ₀ + FAC (pada tinggi ellipsoidal), dan δg dari grid.
        g_surface = xgm.get_surface_gravity(lat, lon, h_ell)

        # ---- 4e. Potensial gangguan, bilangan geopotensial, tinggi dinamik ----
        # T = γ₀ · ζ   (Bruns' formula, aproksimasi orde pertama)
        T_disturb = gamma0 * zeta * 1e-5          # m²/s²

        # C = g_surface · H   (geopotential number, Heiskanen & Moritz, 1967)
        C = gamma_surface * elev * 1e-5           # m²/s²

        # Tinggi dinamik = C / γ₄₅   (γ₄₅ = gravitasi normal di 45°)
        gamma45 = normal_gravity_somigliana(45.0)
        gamma45_ms2 = gamma45 * 1e-5
        dyn_height = C / gamma45_ms2              # metre

        # ---- 4f. Koreksi pelat Bouguer (global & lokal) ----
        # 0.04193 = 2πGρ (dengan ρ dalam g/cm³) dalam mGal/m
        bc_slab_global = 0.04193 * (RHO_CRUST / 1000.0) * elev
        bc_slab_local  = 0.04193 * (density_local / 1000.0) * elev
        diff_bc = bc_slab_local - bc_slab_global

        # ---- 4g. Anomali Bouguer lengkap dengan densitas lokal ----
        # CBA_local = Δg – Bouguer_slab_local + TC_local
        cba_local = dg - bc_slab_local + tc_tess_local

        # ---- 4h. Model isostasi Airy-Heiskanen (Heiskanen & Moritz, 1967) ----
        # Akar kerak: t = (ρ_c / (ρ_m – ρ_c)) · H  (di darat)
        root_std = (RHO_CRUST / (RHO_MANTLE - RHO_CRUST)) * elev
        moho_depth_std = T0 + root_std
        root_local = (density_local * elev) / (RHO_MANTLE - density_local)
        moho_depth_local = T0 + root_local
        diff_root = root_local - root_std

        # ---- 4i. Perbandingan antar definisi anomali ----
        diff_cl_mod = dg_cl - dg          # klasik vs modern
        diff_sa_mod = dg_sa_anom - dg     # spherical aprox vs modern
        diff_dist = dg_dist_sa - dg_dist  # spherical disturbance vs exact

        # ---- 4j. Perbedaan geoid vs height anomaly ----
        N_zeta_diff = N - zeta

        # Susun data untuk output
        adv_data = {
            "────────── ": None,
            "GLOBAL PARAMETERS (ρ=2670)": "",
            "TC (global, CSB - BG)": f"{TC_global:.4f} mGal",
            "CBA (global, XGM2019e)": f"{csb:.4f} mGal",
            "Bouguer Slab (global)": f"{bc_slab_global:.4f} mGal",

            "──────────  ": None,
            "LOCAL PARAMETERS (ρ=2250)": "",
            f"TC (local, tesseroid)": f"{tc_tess_local:.4f} mGal",
            f"CBA (local)": f"{cba_local:.4f} mGal",
            f"Bouguer Slab (local)": f"{bc_slab_local:.4f} mGal",

            "──────────   ": None,
            "BC diff (local - global)": f"{diff_bc:+.4f} mGal",

            "──────────    ": None,
            "Spherical Shell Corr (ΔN_top)": f"{dN_top:.6f} m (Barthelmes eq.69)",
            "Disturbing Potential (T)": f"{T_disturb:.3f} m²/s²",
            "Geopotential Number (C)": f"{C:.3f} m²/s²",
            "Dynamic Height": f"{dyn_height:.4f} m (C / γ₄₅)" if not math.isnan(dyn_height) else "N/A",
            "Free-Air Correction (FAC)": f"{FAC:.3f} mGal",
            "Normal Gravity at Surface": f"{gamma_surface:.4f} mGal",
            "Surface Gravity (g)": f"{g_surface:.4f} mGal (using δg)",
            "N - ζ difference": f"{N_zeta_diff:.8f} m (geoid - height anomaly)",

            "Δg_cl - Δg_modern": f"{diff_cl_mod:.4f} mGal (classical - modern)",
            "Δg_sa - Δg_modern": f"{diff_sa_mod:.4f} mGal (spherical approx)",
            "δg_sa - δg": f"{diff_dist:.4f} mGal (spherical disturbance)",
        }
    else:
        adv_data = {"Status": "Some gravity grids missing. Cannot compute derivatives."}

    # =========================================================================
    #  5. MODEL ISOSTASI AIRY-HEISKANEN (jika tidak dihitung di atas)
    # =========================================================================
    if 'root_std' in locals() and 'root_local' in locals():
        iso_data = {
            "Topographic Density (local)": f"{density_local:.0f} kg/m³",
            "Crust Reference Density (global)": f"{RHO_CRUST:.0f} kg/m³",
            "Mantle Density": f"{RHO_MANTLE:.0f} kg/m³",
            "Normal Crust T₀": f"{T0/1000:.1f} km",
            "Root (global ρ=2670)": f"{root_std/1000:.3f} km",
            "Moho (global)": f"{moho_depth_std/1000:.3f} km",
            f"Root (local ρ={density_local:.0f})": f"{root_local/1000:.3f} km",
            "Moho (local)": f"{moho_depth_local/1000:.3f} km",
            "Root difference (local - global)": f"{diff_root/1000:+.3f} km",
        }
    else:
        iso_calc = IsostaticCalculator()
        root = iso_calc.compute_root_depth(elev)
        moho = iso_calc.compute_moho_depth(elev)
        iso_data = {
            "Topographic Density (local)": f"{density_local:.0f} kg/m³",
            "Crust Reference Density (global)": f"{RHO_CRUST:.0f} kg/m³",
            "Mantle Density": f"{RHO_MANTLE:.0f} kg/m³",
            "Normal Crust T₀": f"{T0/1000:.1f} km",
            "Root (standard)": f"{root/1000:.3f} km",
            "Moho (standard)": f"{moho/1000:.3f} km",
        }

    # =========================================================================
    #  6. ANALISIS SPEKTRAL (FFT 2D) – Cooley & Tukey (1965)
    # =========================================================================
    spec = SpectralAnalyzer.analyze_dem(dem)
    spec_data = {
        "DEM Dominant Wavelength": f"{spec['wavelength_dominant_m']/1000:.2f} km",
        "DEM Total Energy": f"{spec['total_energy']:.2e} (arb. units)",
    }

    # =========================================================================
    #  7. KOMPUTASI GEODETIK PENDUKUNG
    # =========================================================================
    # Konversi koordinat geodetik ke kartesian (WGS84)
    x, y, z_cart = geodetic_to_cartesian(LAT, LON, elev)
    # Jarak great‑circle ke puncak Pawitra (Haversine)
    dist = haversine_distance(LAT, LON, -7.6156, 112.6200)
    # Gradien horizontal anomali gravitasi (mGal/m)
    gx, gy = xgm.get_horizontal_gradient(lat, lon)

    helper_data = {
        "Data Source": "WGS84 Reference Ellipsoid (a=6378137 m, f=1/298.257223563)",
        "Jolotundo Cartesian (X, Y, Z)": f"X = {x:.3f} m\nY = {y:.3f} m\nZ = {z_cart:.3f} m",
        "Great-Circle Distance to Pawitra Peak": f"{dist:.3f} metres",
        "Horizontal Gradient (∂g/∂x, ∂g/∂y)": f"{gx:.4f}, {gy:.4f} mGal/m" if not math.isnan(gx) else "N/A",
    }

    # =========================================================================
    #  8. CETAK HASIL – LAPORAN ILMIAH TERSTRUKTUR
    # =========================================================================
    print_scientific_box("JOLOTUNDO OBSV – DIGITAL ELEVATION MODEL (COP30 DSM)", dem_data)
    print_scientific_box("LOCAL STRATIGRAPHY (Mt. Penanggungan)", strat_data)
    print_scientific_box("GRAVITY FIELD MODEL (XGM2019e-2159)", xgm_data)
    print_scientific_box("GEOPHYSICAL DERIVATIONS", adv_data)
    print_scientific_box("AIRY-HEISKANEN ISOSTATIC MODEL", iso_data)
    print_scientific_box("SPECTRAL ANALYSIS (FFT 2D)", spec_data)
    print_scientific_box("GEODETIC COMPUTATIONS", helper_data)
    print_scientific_box("MODEL ACCURACY & UNCERTAINTY", accuracy_data)    

    print(f"\n{BOLD}{'=' * term_width}{RESET}")
    print(f"{BOLD}{' ANALYSIS COMPLETE '.center(term_width, '=')}{RESET}")
    print(f"{BOLD}{'=' * term_width}{RESET}")