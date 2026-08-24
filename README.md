ASTERID

https://img.shields.io/badge/License-Non--Commercial-red.svg
https://img.shields.io/badge/Research-Only-blue.svg
https://img.shields.io/badge/Education-Use-green.svg
https://img.shields.io/badge/python-3.8+-blue.svg
https://img.shields.io/badge/IERS-2010-orange.svg
https://img.shields.io/badge/IAU-2006-2000A-purple.svg
https://img.shields.io/badge/IMCCE-VSOP2013-ELP-purple.svg

ASTERID — A high‑precision astrometric, geodetic, and geophysical reduction toolkit implementing IERS Conventions (2010) and IAU 2006/2000A standards.

---

📖 Overview

ASTERID is a Python‑based suite for rigorous astrometric and geodetic calculations, designed for research applications requiring sub‑arcsecond and sub‑millimetre accuracy. It implements the full reduction chain from IERS Conventions (2010) and IAU 2006/2000A resolutions, including:

· Earth orientation — CIP‑CIO and equinox‑based transformations (ITRS ↔ GCRS) with full IERS 2010 EOP, nutation, precession, frame bias, and polar motion.
· High‑precision ephemerides — VSOP2013 (Sun, Earth‑Moon barycentre) and ELP/MPP02 (Moon) with light‑time, aberration, gravitational deflection, and topocentric corrections.
· Geophysical site corrections — Solid Earth tides, ocean tide loading (FES2014), pole tide, atmospheric loading, and ITRF2020‑PMM plate kinematics.
· Atmospheric modelling — VMF3 and GPT3 tropospheric delay with real‑time ECMWF IFS HRES ingestion.
· Local gravity & geoid — XGM2019e‑2159 grids, tesseroid‑based terrain correction, vertical deflection, and complete Bouguer anomalies.
· Lunar libration — LLIB04 physical libration model.

The system is calibrated for the Jolotundo Observatory (Mount Penanggungan, East Java) but is fully generalisable to any location.

---

🚀 Core Modules

Time & Earth Orientation

Module Description
Timescales.py UTC, UT1, TAI, TT, TCG, TCB, TDB conversions with ΔT from IERS HMNAO tables and eclipse‑anchored historical corrections.
EarthRotation.py CIP‑CIO and equinox‑based ITRS ↔ GCRS transformations; nutation (IAU 2000A), precession (IAU 2006), frame bias; quaternion implementation (Bizouard & Cheng 2023).
EOPDelta.py Earth Orientation Parameters (EOP) from IERS C04 and Bulletin‑A with cubic spline interpolation and sub‑diurnal libration.

Ephemerides (Sun, Moon)

Module Description
VSOP2013.py Full VSOP2013 analytical theory for the Sun (Earth‑Moon barycentre), validated against IMCCE reference.
ELP_MPP02_full.py ELP/MPP02 lunar ephemerides with DE405/LLR corrections.
SM_VSOP2013.py Orchestrator combining VSOP2013 (Sun) and ELP/MPP02 (Moon) with all IERS 2010 corrections.
DT18_VSOP2013_Realtime.py Real‑time ephemeris and geodetic report generator.

Geophysical & Atmospheric Modelling

Module Description
StationDispl.py Station displacement due to solid Earth tides, ocean tide loading (FES2014), pole tide, and atmospheric loading.
Site_Geophysic.py Local gravity field (XGM2019e‑2159), tesseroid terrain correction, vertical deflection (DoV), plate kinematics (ITRF2020‑PMM), and stratigraphy.
Atmospheric_refraction.py Refraction models (Bennett, VMF3, GPT3) with real‑time ECMWF IFS HRES assimilation.
gpt3.py / vmf3.py GPT3 grid loader and VMF3 mapping functions.
ecmwf_realtime.py Smart cached retrieval of ECMWF IFS HRES 9 km atmospheric fields.

Coordinate & Transformation Utilities

Module Description
Coord_Transform.py Spherical ↔ Cartesian, equatorial ↔ ecliptic, GCRS ↔ ITRS, horizontal (az/alt) with refraction and diurnal corrections.
llib04.py LLIB04 lunar physical libration.
solar_lunar_events.py Solar/lunar event calculator (equinoxes, solstices, eclipses, standstills, phases).

---

📂 Repository Structure

```
ASTERID/
├── Timescales.py                # Time scales (UTC, UT1, TAI, TT, TDB, TCG, TCB)
├── EarthRotation.py             # ITRS ↔ GCRS, nutation, precession, quaternions
├── EOPDelta.py                  # IERS EOP (C04 + Bulletin‑A)
├── Coord_Transform.py           # Spherical, equatorial, ecliptic, horizontal, GCRS‑ITRS
├── VSOP2013.py                  # VSOP2013 Sun/EMB ephemeris
├── ELP_MPP02_full.py            # ELP/MPP02 lunar ephemeris
├── SM_VSOP2013.py               # Orchestrator: Sun + Moon with IERS corrections
├── DT18_VSOP2013_Realtime.py    # Real‑time ephemeris & geodetic report
├── StationDispl.py              # IERS 2010 station displacement
├── Site_Geophysic.py            # Local gravity, geoid, terrain correction, plate kinematics
├── Atmospheric_refraction.py    # Refraction models (Bennett, VMF3, GPT3, ECMWF)
├── gpt3.py                      # GPT3 grid loader
├── vmf3.py                      # VMF3 mapping functions
├── ecmwf_realtime.py            # ECMWF IFS HRES retrieval with smart caching
├── llib04.py                    # LLIB04 lunar libration
├── solar_lunar_events.py        # Solar/lunar events calculator
│
├── Data files
│   ├── EOP_20u24_C04_one_file_1962-now.txt   # IERS EOP C04
│   ├── gpt3_1.npz                            # GPT3 grid (1°)
│   ├── VSOP2013p3.dat                        # VSOP2013 full series
│   ├── VSOP2013.ctl                          # VSOP2013 reference control file
│   ├── VSOP87A_ear.txt                       # VSOP87A Earth series
│   ├── ELP_MAIN_S1.txt – ELP_PERT.S3         # ELP/MPP02 series
│   ├── LLIB04_DAT.txt                        # LLIB04 libration series
│   ├── Ray_Ponte_2003_mbar.txt               # Atmospheric loading coefficients
│   ├── opoleloadcoefcmcor.npz                # Ocean pole tide grid (Desai 2002)
│   ├── height_anomaly_ell_XGM2019e_2159.txt  # XGM2019e‑2159 geoid grids
│   └── output_hh.asc                         # Local DEM (Copernicus 30 m)
│
└── README.md
```

⚠️ All data files must be present in the same directory as the corresponding Python modules.

---

🔧 Installation

Requirements

· Python 3.8+
· NumPy (≥1.20), SciPy (≥1.8)
· Pandas (optional, for ECMWF caching)
· No other external dependencies

```bash
git clone https://github.com/yourusername/ASTERID.git
cd ASTERID
```

All data files are included — no additional setup required.

---

💻 Usage

1. Real‑Time Ephemeris & Geodetic Report

```python
from DT18_VSOP2013_Realtime import generate_report, print_report

report = generate_report()
print_report(report)
```

2. Earth Orientation (ITRS ↔ GCRS)

```python
from EarthRotation import EarthOrientation
import numpy as np

eo = EarthOrientation()
tt_jd = 2451545.0
vec = np.array([1.0, 0.0, 0.0])
vec_itrs = eo.gcrs_to_itrs(vec, tt_jd, paradigm='cip', use_eop=True)
```

3. Sun & Moon Apparent Topocentric Positions

```python
from SM_VSOP2013 import AstronomicalEphemeris
import math

ephem = AstronomicalEphemeris()
tt_jd = 2451545.0
lat = math.radians(-7.609444)
lon = math.radians(112.595556)
h = 554.509

sun = ephem.sun_apparent_topocentric(tt_jd, lat, lon, h, apply_refraction=True)
moon = ephem.moon_apparent_topocentric(tt_jd, lat, lon, h, apply_refraction=True)

print(f"Sun  Az/Alt: {sun['az_deg']:.6f}° / {sun['alt_app_deg']:.6f}°")
print(f"Moon Az/Alt: {moon['az_deg']:.6f}° / {moon['alt_app_deg']:.6f}°")
```

4. Local Gravity & Geoid

```python
from Site_Geophysic import LocalEGM2008Grids, PawitraGeophysics

grids = LocalEGM2008Grids('.')
geophys = PawitraGeophysics(grids)
anomalies = geophys.dynamic_gravity_anomalies(-7.609444, 112.595556, 554.509)
print(f"Complete Bouguer anomaly: {anomalies['complete_bouguer_anomaly_mgal']:.3f} mGal")
```

---

✅ Validation

Module Validation Method Reference
VSOP2013.py Compare with VSOP2013.ctl 11 JD epochs
ELP82B.py Compare with Table H values 5 JD epochs
EarthRotation.py Round‑trip GCRS↔ITRS, CIP/CIO at J2000 IERS 2010
EOPDelta.py Interpolation against IERS C04 Various epochs
SM_VSOP2013.py JPL Horizons comparison (Sun & Moon) 2024‑12‑25

Expected residuals:

· Sun position (VSOP2013 vs DE441): ~45 km
· Moon position (ELP/MPP02 vs DE441): ~5 m (radial)
· EOP interpolation: ≤ 1 µas
· Gravity anomalies: ≤ 1 mGal

---

📚 References

· IERS Conventions (2010), IERS TN36.
· IAU 2006/2000A Resolutions.
· Bizouard, C. & Cheng, Y. (2023). Quaternions for Earth rotation. J. Geod. 97, 53.
· Fienga, A., et al. (2013). VSOP2013. IMCCE.
· Chapront, J. & Francou, G. (2003). ELP revisited. A&A 404, 735.
· Altamimi, Z., et al. (2023). ITRF2020‑PMM. GRL 50, e2023GL106373.
· Zingerle, P., et al. (2020). XGM2019e. J. Geod. 94(7).
· Böhm, J., et al. (2016). VMF3. J. Geod. 90, 449–460.

---

📝 License & Terms

```
NON‑COMMERCIAL USE ONLY.
Provided for educational and research purposes.
Commercial use strictly prohibited without permission.
All data and algorithms provided "as‑is".
```

Third‑Party Attribution

· VSOP2013, ELP, LLIB04: IMCCE (Paris Observatory)
· XGM2019e‑2159: ICGEM (GFZ Potsdam)
· IERS EOP: IERS (Paris Observatory)
· FES2014: IERS / LEGOS
· ECMWF IFS: Open‑Meteo (ECMWF license)

---

🏛️ About the Jolotundo Research Consortium

The Jolotundo Research Consortium (JRC) is an independent research collective for archaeoastronomy, epigraphy, and geodetic heritage. It is not a registered legal entity; it serves as a scholarly identity for the developers.

---

👥 Author

Rakawi:
Jolotundo Research Consortium

---

🤝 Contributing

Contributions improving accuracy, fixing bugs, adding documentation, or extending validation are welcome. Full planetary implementations are out of scope.

---

⚠️ Disclaimer

```
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
```

---

🌟 Acknowledgements

· IMCCE / Paris Observatory — VSOP2013, ELP, LLIB04
· IERS — EOP data and Conventions
· ICGEM / GFZ — XGM2019e‑2159
· Open‑Meteo — ECMWF IFS HRES data

---

Last updated: 2026‑08‑24
ASTERID — 
Jolotundo Research Consortium
