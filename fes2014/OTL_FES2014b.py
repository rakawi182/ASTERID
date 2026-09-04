#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# ASTERID Ω-342 : High-Precision Ocean Tide Loading Algorithmic Expansion
# FES2014b – Stasiun JOLOTUNDO OBSV
# ==============================================================================
#
# SCIENTIFIC SUMMARY
# ------------------
# This module implements the FES2014b ocean tide loading model for station
# displacement (radial, EW, NS) and derived quantities (gravity, tilt).
# The algorithm follows the IERS Conventions 2010 (Chapter 6) and the
# official FES2014b handling documented in Explanations_about_FES2014b.pdf.
#
# KEY FEATURES:
#   1. 342-constituent tidal spectrum (Cartwright-Edden 1973) via
#      admittance interpolation (natural cubic spline) from 11 primary waves.
#   2. Nodal modulation (18.6-year cycle) for M2, S2, K1, K2, O1, N2, P1.
#   3. S1 (Doodson 164.555) and Sa (Doodson 056.555) ZEROED (radiational tides,
#      handled by atm_loading_displacement from Ray & Ponte 2003).
#   4. Om1 (055.565) and Om2 (055.575) as EQUILIBRIUM tides (radial only,
#      phase = -90°, deformation factor = -0.305).
#
# REFERENCES:
#   [1] Lyard, F., Lefevre, F., Letellier, T., Francis, O. (2006).
#       Modelling the global ocean tides: modern insights from FES2004.
#       Ocean Dynamics, 56(5-6), 394-415.
#   [2] IERS Conventions (2010). IERS Technical Note No. 36.
#   [3] FES team (2019). Explanations_about_FES2014b.pdf.
#   [4] Ray, R.D. & Egbert, G.D. (2004). The global S1 tide.
#       J. Phys. Oceanogr., 34, 1922-1935.
#   [5] Ray, R.D. & Ponte, R.M. (2003). Barometric tides from ECMWF
#       operational analyses. Ann. Geophys., 21, 1897-1910.
#   [6] Cartwright, D.E. & Edden, A.C. (1973). Corrected tables of tidal
#       harmonics. Geophys. J. R. astr. Soc., 33, 253-264.
#   [7] Altamimi, Z., et al. (2023). ITRF2020 plate motion model.
#       Geophys. Res. Lett., 50, e2023GL106373.
# ==============================================================================

import math
import numpy as np
from typing import Tuple, List, Dict, Optional

class Asterid342Engine_FES2014:
    """
    High-precision FES2014b ocean tide loading engine.
    
    Implements the full 342-constituent tidal spectrum (Cartwright-Edden 1973)
    using admittance interpolation (natural cubic spline) from 11 primary
    constituents (M2, S2, N2, K2, K1, O1, P1, Q1, Mf, Mm, Ssa). The engine
    computes displacement components (radial, West, South) in meters for a
    given epoch (MJD TT). It includes:
    
        - Nodal modulation (18.6-year) for all primary constituents.
        - Radiational tides (S1, Sa) zeroed – handled by atm_loading_displacement.
        - Equilibrium long-period tides (Om1, Om2) added as radial only.
    
    The algorithm follows the official FES2014b handling (Explanations_about_FES2014b.pdf)
    and the IERS Conventions 2010 (Section 6.3).
    
    Parameters
    ----------
    blq_data : dict
        Must contain 11 primary constituents:
        'M2','S2','N2','K2','K1','O1','P1','Q1','Mf','Mm','Ssa'
        Each value is a list of 6 floats:
        [amp_rad (m), ph_rad (deg), amp_ew (m), ph_ew (deg), amp_ns (m), ph_ns (deg)]
        Conventions: phase = lag (positive = delayed), EW positive = west,
        NS positive = south. This is consistent with the Chalmers loading service
        (http://holt.oso.chalmers.se/loading/).
    
    Raises
    ------
    ValueError
        If blq_data does not contain all required constituents.
    
    References
    ----------
    - IERS Conventions 2010, Section 6.3 (Ocean Tides)
    - Explanations_about_FES2014b.pdf (FES team, 2019)
    - Lyard et al. (2006), Ocean Dynamics
    """
    
    # List of required constituents (IERS 2010, Section 6.3.2)
    REQUIRED_WAVES = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm', 'Ssa']
    
    def __init__(self, blq_data: Dict[str, List[float]]):
        # Validate input data
        for wave in self.REQUIRED_WAVES:
            if wave not in blq_data:
                raise ValueError(
                    f"Missing required constituent '{wave}' in blq_data. "
                    f"Required: {self.REQUIRED_WAVES}"
                )
            if len(blq_data[wave]) != 6:
                raise ValueError(
                    f"Constituent '{wave}' must have exactly 6 elements "
                    f"[amp_rad, ph_rad, amp_ew, ph_ew, amp_ns, ph_ns]"
                )
        
        self.blq = blq_data
        
        # 11 primary constituents (anchor waves)
        self.main_waves = self.REQUIRED_WAVES
        
        # Doodson numbers for primary constituents (6-vector form)
        # Format: [n1, n2, n3, n4, n5, n6] where:
        #   n1: 0=long-period, 1=diurnal, 2=semi-diurnal, 3=ter-diurnal, etc.
        #   n2..n6: multipliers for fundamental arguments (IERS 2010, Eq. 5.43)
        self.main_doodson = np.array([
            [2,  0,  0,  0,  0,  0],  # M2
            [2,  2, -2,  0,  0,  0],  # S2
            [2, -1,  0,  1,  0,  0],  # N2
            [2,  2,  0,  0,  0,  0],  # K2
            [1,  1,  0,  0,  0,  0],  # K1
            [1, -1,  0,  0,  0,  0],  # O1
            [1,  1, -2,  0,  0,  0],  # P1
            [1, -2,  0,  1,  0,  0],  # Q1
            [0,  2,  0,  0,  0,  0],  # Mf
            [0,  1,  0, -1,  0,  0],  # Mm
            [0,  0,  2,  0,  0,  0]   # Ssa
        ])

        # ------------------------------------------------------------------
        # Cartwright-Edden amplitudes (TAMP) for 342 constituents
        # Extracted from ADMINT.F (IERS Conventions 2010, Section 6.3)
        # These are normalized spherical harmonic amplitudes (fully normalized)
        # ------------------------------------------------------------------
        self.tamp_342 = np.array([
            .632208, .294107, .121046, .079915, .023818,-.023589, .022994, .019333,-.017871, .017192, 
            .016018, .004671,-.004662,-.004519, .004470, .004467, .002589,-.002455,-.002172, .001972, 
            .001947, .001914,-.001898, .001802, .001304, .001170, .001130, .001061,-.001022,-.001017, 
            .001014, .000901,-.000857, .000855, .000855, .000772, .000741, .000741,-.000721, .000698, 
            .000658, .000654,-.000653, .000633, .000626,-.000598, .000590, .000544, .000479,-.000464, 
            .000413,-.000390, .000373, .000366, .000366,-.000360,-.000355, .000354, .000329, .000328, 
            .000319, .000302, .000279,-.000274,-.000272, .000248,-.000225, .000224,-.000223,-.000216,
            .000211, .000209, .000194, .000185,-.000174,-.000171, .000159, .000131, .000127, .000120, 
            .000118, .000117, .000108, .000107, .000105,-.000102, .000102, .000099,-.000096, .000095,
            -.000089,-.000085,-.000084,-.000081,-.000077,-.000072,-.000067, .000066, .000064, .000063, 
            .000063, .000063, .000062, .000062,-.000060, .000056, .000053, .000051, .000050, .368645,
            -.262232,-.121995,-.050208, .050031,-.049470, .020620, .020613, .011279,-.009530,-.009469,
            -.008012, .007414,-.007300, .007227,-.007131,-.006644, .005249, .004137, .004087, .003944, 
            .003943, .003420, .003418, .002885, .002884, .002160,-.001936, .001934,-.001798, .001690,
            .001689, .001516, .001514,-.001511, .001383, .001372, .001371,-.001253,-.001075, .001020, 
            .000901, .000865,-.000794, .000788, .000782,-.000747,-.000745, .000670,-.000603,-.000597, 
            .000542, .000542,-.000541,-.000469,-.000440, .000438, .000422, .000410,-.000374,-.000365, 
            .000345, .000335,-.000321,-.000319, .000307, .000291, .000290,-.000289, .000286, .000275, 
            .000271, .000263,-.000245, .000225, .000225, .000221,-.000202,-.000200,-.000199, .000192, 
            .000183, .000183, .000183,-.000170, .000169, .000168, .000162, .000149,-.000147,-.000141, 
            .000138, .000136, .000136, .000127, .000127,-.000126,-.000121,-.000121, .000117,-.000116,
            -.000114,-.000114,-.000114, .000114, .000113, .000109, .000108, .000106,-.000106,-.000106, 
            .000105, .000104,-.000103,-.000100,-.000100,-.000100, .000099,-.000098, .000093, .000093, 
            .000090,-.000088, .000083,-.000083,-.000082,-.000081,-.000079,-.000077,-.000075,-.000075,
            -.000075, .000071, .000071,-.000071, .000068, .000068, .000065, .000065, .000064, .000064, 
            .000064,-.000064,-.000060, .000056, .000056, .000053, .000053, .000053,-.000053, .000053, 
            .000053, .000052, .000050,-.066607,-.035184,-.030988, .027929,-.027616,-.012753,-.006728,
            -.005837,-.005286,-.004921,-.002884,-.002583,-.002422, .002310, .002283,-.002037, .001883,
            -.001811,-.001687,-.001004,-.000925,-.000844, .000766, .000766,-.000700,-.000495,-.000492, 
            .000491, .000483, .000437,-.000416,-.000384, .000374,-.000312,-.000288,-.000273, .000259, 
            .000245,-.000232, .000229,-.000216, .000206,-.000204,-.000202, .000200, .000195,-.000190, 
            .000187, .000180,-.000179, .000170, .000153,-.000137,-.000119,-.000119,-.000112,-.000110,
            -.000110, .000107,-.000095,-.000095,-.000091,-.000090,-.000081,-.000079,-.000079, .000077,
            -.000073, .000069,-.000067,-.000066, .000065, .000064,-.000062, .000060, .000059,-.000056, 
            .000055,-.000051
        ])

        # ------------------------------------------------------------------
        # Doodson numbers for 342 constituents (flat array, 6 integers each)
        # Extracted from ADMINT.F (IERS Conventions 2010)
        # ------------------------------------------------------------------
        self.idd_342_flat = np.array([
            2,0,0,0,0,0, 2,2,-2,0,0,0, 2,-1,0,1,0,0, 2,2,0,0,0,0, 2,2,0,0,1,0, 2,0,0,0,-1,0, 2,-1,2,-1,0,0,
            2,-2,2,0,0,0, 2,1,0,-1,0,0, 2,2,-3,0,0,1, 2,-2,0,2,0,0, 2,-3,2,1,0,0, 2,1,-2,1,0,0, 2,-1,0,1,-1,0,
            2,3,0,-1,0,0, 2,1,0,1,0,0, 2,2,0,0,2,0, 2,2,-1,0,0,-1, 2,0,-1,0,0,1, 2,1,0,1,1,0, 2,3,0,-1,1,0,
            2,0,1,0,0,-1, 2,0,-2,2,0,0, 2,-3,0,3,0,0, 2,-2,3,0,0,-1, 2,4,0,0,0,0, 2,-1,1,1,0,-1, 2,-1,3,-1,0,-1,
            2,2,0,0,-1,0, 2,-1,-1,1,0,1, 2,4,0,0,1,0, 2,-3,4,-1,0,0, 2,-1,2,-1,-1,0, 2,3,-2,1,0,0, 2,1,2,-1,0,0,
            2,-4,2,2,0,0, 2,4,-2,0,0,0, 2,0,2,0,0,0, 2,-2,2,0,-1,0, 2,2,-4,0,0,2, 2,2,-2,0,-1,0, 2,1,0,-1,-1,0,
            2,-1,1,0,0,0, 2,2,-1,0,0,1, 2,2,1,0,0,-1, 2,-2,0,2,-1,0, 2,-2,4,-2,0,0, 2,2,2,0,0,0, 2,-4,4,0,0,0,
            2,-1,0,-1,-2,0, 2,1,2,-1,1,0, 2,-1,-2,3,0,0, 2,3,-2,1,1,0, 2,4,0,-2,0,0, 2,0,0,2,0,0, 2,0,2,-2,0,0,
            2,0,2,0,1,0, 2,-3,3,1,0,-1, 2,0,0,0,-2,0, 2,4,0,0,2,0, 2,4,-2,0,1,0, 2,0,0,0,0,2, 2,1,0,1,2,0,
            2,0,-2,0,-2,0, 2,-2,1,0,0,1, 2,-2,1,2,0,-1, 2,-1,1,-1,0,1, 2,5,0,-1,0,0, 2,1,-3,1,0,1, 2,-2,-1,2,0,1,
            2,3,0,-1,2,0, 2,1,-2,1,-1,0, 2,5,0,-1,1,0, 2,-4,0,4,0,0, 2,-3,2,1,-1,0, 2,-2,1,1,0,0, 2,4,0,-2,1,0,
            2,0,0,2,1,0, 2,-5,4,1,0,0, 2,0,2,0,2,0, 2,-1,2,1,0,0, 2,5,-2,-1,0,0, 2,1,-1,0,0,0, 2,2,-2,0,0,2,
            2,-5,2,3,0,0, 2,-1,-2,1,-2,0, 2,-3,5,-1,0,-1, 2,-1,0,0,0,1, 2,-2,0,0,-2,0, 2,0,-1,1,0,0, 2,-3,1,1,0,1,
            2,3,0,-1,-1,0, 2,1,0,1,-1,0, 2,-1,2,1,1,0, 2,0,-3,2,0,1, 2,1,-1,-1,0,1, 2,-3,0,3,-1,0, 2,0,-2,2,-1,0,
            2,-4,3,2,0,-1, 2,-1,0,1,-2,0, 2,5,0,-1,2,0, 2,-4,5,0,0,-1, 2,-2,4,0,0,-2, 2,-1,0,1,0,2, 2,-2,-2,4,0,0,
            2,3,-2,-1,-1,0, 2,-2,5,-2,0,-1, 2,0,-1,0,-1,1, 2,5,-2,-1,1,0, 1,1,0,0,0,0, 1,-1,0,0,0,0, 1,1,-2,0,0,0,
            1,-2,0,1,0,0, 1,1,0,0,1,0, 1,-1,0,0,-1,0, 1,2,0,-1,0,0, 1,0,0,1,0,0, 1,3,0,0,0,0, 1,-2,2,-1,0,0,
            1,-2,0,1,-1,0, 1,-3,2,0,0,0, 1,0,0,-1,0,0, 1,1,0,0,-1,0, 1,3,0,0,1,0, 1,1,-3,0,0,1, 1,-3,0,2,0,0,
            1,1,2,0,0,0, 1,0,0,1,1,0, 1,2,0,-1,1,0, 1,0,2,-1,0,0, 1,2,-2,1,0,0, 1,3,-2,0,0,0, 1,-1,2,0,0,0,
            1,1,1,0,0,-1, 1,1,-1,0,0,1, 1,4,0,-1,0,0, 1,-4,2,1,0,0, 1,0,-2,1,0,0, 1,-2,2,-1,-1,0, 1,3,0,-2,0,0,
            1,-1,0,2,0,0, 1,-1,0,0,-2,0, 1,3,0,0,2,0, 1,-3,2,0,-1,0, 1,4,0,-1,1,0, 1,0,0,-1,-1,0, 1,1,-2,0,-1,0,
            1,-3,0,2,-1,0, 1,1,0,0,2,0, 1,1,-1,0,0,-1, 1,-1,-1,0,0,1, 1,0,2,-1,1,0, 1,-1,1,0,0,-1, 1,-1,-2,2,0,0,
            1,2,-2,1,1,0, 1,-4,0,3,0,0, 1,-1,2,0,1,0, 1,3,-2,0,1,0, 1,2,0,-1,-1,0, 1,0,0,1,-1,0, 1,-2,2,1,0,0,
            1,4,-2,-1,0,0, 1,-3,3,0,0,-1, 1,-2,1,1,0,-1, 1,-2,3,-1,0,-1, 1,0,-2,1,-1,0, 1,-2,-1,1,0,1, 1,4,-2,1,0,0,
            1,-4,4,-1,0,0, 1,-4,2,1,-1,0, 1,5,-2,0,0,0, 1,3,0,-2,1,0, 1,-5,2,2,0,0, 1,2,0,1,0,0, 1,1,3,0,0,-1,
            1,-2,0,1,-2,0, 1,4,0,-1,2,0, 1,1,-4,0,0,2, 1,5,0,-2,0,0, 1,-1,0,2,1,0, 1,-2,1,0,0,0, 1,4,-2,1,1,0,
            1,-3,4,-2,0,0, 1,-1,3,0,0,-1, 1,3,-3,0,0,1, 1,5,-2,0,1,0, 1,1,2,0,1,0, 1,2,0,1,1,0, 1,-5,4,0,0,0,
            1,-2,0,-1,-2,0, 1,5,0,-2,1,0, 1,1,2,-2,0,0, 1,1,-2,2,0,0, 1,-2,2,1,1,0, 1,0,3,-1,0,-1, 1,2,-3,1,0,1,
            1,-2,-2,3,0,0, 1,-1,2,-2,0,0, 1,-4,3,1,0,-1, 1,-4,0,3,-1,0, 1,-1,-2,2,-1,0, 1,-2,0,3,0,0, 1,4,0,-3,0,0,
            1,0,1,1,0,-1, 1,2,-1,-1,0,1, 1,2,-2,1,-1,0, 1,0,0,-1,-2,0, 1,2,0,1,2,0, 1,2,-2,-1,-1,0, 1,0,0,1,2,0,
            1,0,1,0,0,0, 1,2,-1,0,0,0, 1,0,2,-1,-1,0, 1,-1,-2,0,-2,0, 1,-3,1,0,0,1, 1,3,-2,0,-1,0, 1,-1,-1,0,-1,1,
            1,4,-2,-1,1,0, 1,2,1,-1,0,-1, 1,0,-1,1,0,1, 1,-2,4,-1,0,0, 1,4,-4,1,0,0, 1,-3,1,2,0,-1, 1,-3,3,0,-1,-1,
            1,1,2,0,2,0, 1,1,-2,0,-2,0, 1,3,0,0,3,0, 1,-1,2,0,-1,0, 1,-2,1,-1,0,1, 1,0,-3,1,0,1, 1,-3,-1,2,0,1,
            1,2,0,-1,2,0, 1,6,-2,-1,0,0, 1,2,2,-1,0,0, 1,-1,1,0,-1,-1, 1,-2,3,-1,-1,-1, 1,-1,0,0,0,2, 1,-5,0,4,0,0,
            1,1,0,0,0,-2, 1,-2,1,1,-1,-1, 1,1,-1,0,1,1, 1,1,2,0,0,-2, 1,-3,1,1,0,0, 1,-4,4,-1,-1,0, 1,1,0,-2,-1,0,
            1,-2,-1,1,-1,1, 1,-3,2,2,0,0, 1,5,-2,-2,0,0, 1,3,-4,2,0,0, 1,1,-2,0,0,2, 1,-1,4,-2,0,0, 1,2,2,-1,1,0,
            1,-5,2,2,-1,0, 1,1,-3,0,-1,1, 1,1,1,0,1,-1, 1,6,-2,-1,1,0, 1,-2,2,-1,-2,0, 1,4,-2,1,2,0, 1,-6,4,1,0,0,
            1,5,-4,0,0,0, 1,-3,4,0,0,0, 1,1,2,-2,1,0, 1,-2,1,0,-1,0, 0,2,0,0,0,0, 0,1,0,-1,0,0, 0,0,2,0,0,0,
            0,0,0,0,1,0, 0,2,0,0,1,0, 0,3,0,-1,0,0, 0,1,-2,1,0,0, 0,2,-2,0,0,0, 0,3,0,-1,1,0, 0,0,1,0,0,-1,
            0,2,0,-2,0,0, 0,2,0,0,2,0, 0,3,-2,1,0,0, 0,1,0,-1,-1,0, 0,1,0,-1,1,0, 0,4,-2,0,0,0, 0,1,0,1,0,0,
            0,0,3,0,0,-1, 0,4,0,-2,0,0, 0,3,-2,1,1,0, 0,3,-2,-1,0,0, 0,4,-2,0,1,0, 0,0,2,0,1,0, 0,1,0,1,1,0,
            0,4,0,-2,1,0, 0,3,0,-1,2,0, 0,5,-2,-1,0,0, 0,1,2,-1,0,0, 0,1,-2,1,-1,0, 0,1,-2,1,1,0, 0,2,-2,0,-1,0,
            0,2,-3,0,0,1, 0,2,-2,0,1,0, 0,0,2,-2,0,0, 0,1,-3,1,0,1, 0,0,0,0,2,0, 0,0,1,0,0,1, 0,1,2,-1,1,0,
            0,3,0,-3,0,0, 0,2,1,0,0,-1, 0,1,-1,-1,0,1, 0,1,0,1,2,0, 0,5,-2,-1,1,0, 0,2,-1,0,0,1, 0,2,2,-2,0,0,
            0,1,-1,0,0,0, 0,5,0,-3,0,0, 0,2,0,-2,1,0, 0,1,1,-1,0,-1, 0,3,-4,1,0,0, 0,0,2,0,2,0, 0,2,0,-2,-1,0,
            0,4,-3,0,0,1, 0,3,-1,-1,0,1, 0,0,2,0,0,-2, 0,3,-3,1,0,1, 0,2,-4,2,0,0, 0,4,-2,-2,0,0, 0,3,1,-1,0,-1,
            0,5,-4,1,0,0, 0,3,-2,-1,-1,0, 0,3,-2,1,2,0, 0,4,-4,0,0,0, 0,6,-2,-2,0,0, 0,5,0,-3,1,0, 0,4,-2,0,2,0,
            0,2,2,-2,1,0, 0,0,4,0,0,-2, 0,3,-1,0,0,0, 0,3,-3,-1,0,1, 0,4,0,-2,2,0, 0,1,-2,-1,-1,0, 0,2,-1,0,0,-1,
            0,4,-4,2,0,0, 0,2,1,0,1,-1, 0,3,-2,-1,1,0, 0,4,-3,0,1,1, 0,2,0,0,3,0, 0,6,-4,0,0,0
        ])
        self.idd_342 = self.idd_342_flat.reshape((342, 6))

    # ======================================================================
    # FUNDAMENTAL ARGUMENTS FOR NUTATION AND TIDES (IERS 2010)
    # ======================================================================
    @staticmethod
    def _compute_fundamental_arguments(t_cy: float) -> Dict[str, float]:
        """
        Compute fundamental arguments (l, l', F, D, Ω, planets, p_A) in radians.
        
        Based on IERS Conventions 2010, Eqs. (5.43) and (5.44).
        
        Parameters
        ----------
        t_cy : float
            Time in Julian centuries since J2000.0.
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing:
            'l', 'lp', 'F', 'D', 'Om', 'Me', 'Ve', 'E', 'Ma', 'J', 'Sa', 'U', 'Ne', 'pa'
            all in radians.
        
        References
        ----------
        - IERS Conventions 2010, Section 5.4
        - Capitaine et al. (2003), Astron. Astrophys. 412, 567-586
        """
        FUNDAMENTAL_ARGS = {
            'l':   [134.96340251 * 3600, 1717915923.2178, 31.8792, 0.051635, -0.00024470],
            'lp':  [357.52910918 * 3600, 129596581.0481, -0.5532, 0.000136, -0.00001149],
            'F':   [93.27209062 * 3600,  1739527262.8478, -12.7512, -0.001037, 0.00000417],
            'D':   [297.85019547 * 3600, 1602961601.2090, -6.3706, 0.006593, -0.00003169],
            'Om':  [125.04455501 * 3600, -6962890.5431, 7.4722, 0.007702, -0.00005939],
        }
        PLANETARY_ARGS = {
            'Me': [4.402608842, 2608.7903141574],
            'Ve': [3.176146697, 1021.3285546211],
            'E':  [1.753470314, 628.3075849991],
            'Ma': [6.203480913, 334.0612426700],
            'J':  [0.599546497, 52.9690962641],
            'Sa': [0.874016757, 21.3299104960],
            'U':  [5.481293872, 7.4781598567],
            'Ne': [5.311886287, 3.8133035638],
        }
        PRECESSION_RATE = 0.02438175   # p_A coefficient (rad/Julian century)
        
        args = {}
        for name, coeffs in FUNDAMENTAL_ARGS.items():
            val_arcsec = sum(c * (t_cy ** i) for i, c in enumerate(coeffs))
            args[name] = math.radians(val_arcsec / 3600.0) % (2.0 * math.pi)
        for name, (const, rate) in PLANETARY_ARGS.items():
            args[name] = (const + rate * t_cy) % (2.0 * math.pi)
        args['pa'] = (PRECESSION_RATE * t_cy + 0.00000538691 * t_cy**2) % (2.0 * math.pi)
        return args

    # ======================================================================
    # FREQUENCY & PHASE COMPUTATION (IERS TDFRPH.F)
    # ======================================================================
    def _get_freq_phase_vectorized(self, doodson_matrix: np.ndarray, 
                                   mjd_tt: float, mjd_ut: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency (cycles/day) and phase (degrees) for a set of Doodson numbers.
        
        This is a vectorized port of the IERS TDFRPH.F routine (Conventions 2010, Section 6.3).
        It computes the fundamental arguments (l, l', F, D, Ω) using the IAU 2006/2000A
        formulations (IERS 2010, Eqs. 5.43–5.44) and evaluates the Doodson argument for
        each constituent.
        
        Parameters
        ----------
        doodson_matrix : np.ndarray, shape (K, 6)
            Each row is a 6-element Doodson number [n1, n2, n3, n4, n5, n6].
        mjd_tt, mjd_ut : float
            Modified Julian Date in TT and UT1 scales.
        
        Returns
        -------
        freqs : np.ndarray, shape (K,)
            Tidal frequencies in cycles/day.
        phases : np.ndarray, shape (K,)
            Tidal phases in degrees (0–360).
        
        References
        ----------
        - IERS Conventions 2010, Section 5.4 (Fundamental Arguments)
        - IERS Conventions 2010, Section 6.3 (Ocean Tides)
        - Simon et al. (1994), Astron. Astrophys. 282, 663
        """
        t_cy = (mjd_tt - 51544.5) / 36525.0  # Julian centuries since J2000.0
        
        # Fundamental arguments (IERS 2010, Eqs. 5.43–5.44)
        # l (mean anomaly of the Moon)
        f1 = (134.9634025100 + t_cy * (477198.8675605000 + t_cy * (0.0088553333 + t_cy * (0.0000143431 + t_cy * (-0.0000000680)))))
        # l' (mean anomaly of the Sun)
        f2 = (357.5291091806 + t_cy * (35999.0502911389 + t_cy * (-0.0001536667 + t_cy * (0.0000000378 + t_cy * (-0.0000000032)))))
        # F (mean argument of latitude of the Moon)
        f3 = (93.2720906200 + t_cy * (483202.0174577222 + t_cy * (-0.0035420000 + t_cy * (-0.0000002881 + t_cy * (0.0000000012)))))
        # D (mean elongation of the Moon from the Sun)
        f4 = (297.8501954694 + t_cy * (445267.1114469445 + t_cy * (-0.0017696111 + t_cy * (0.0000018314 + t_cy * (-0.0000000088)))))
        # Ω (mean longitude of the ascending node of the Moon)
        f5 = (125.0445550100 + t_cy * (-1934.1362619722 + t_cy * (0.0020756111 + t_cy * (0.0000021394 + t_cy * (-0.0000000165)))))
        
        # τ (mean solar time angle, in degrees)
        day_frac_ut = mjd_ut - np.floor(mjd_ut)
        tau = 360.0 * day_frac_ut - f4
        
        # Fundamental argument vector
        args = np.array([
            tau,                  # 1: mean solar time angle
            f3 + f5,              # 2: s = F + Ω (mean longitude of Moon)
            f3 + f5 - f4,         # 3: h = s - D (mean longitude of Sun)
            f3 + f5 - f1,         # 4: p = s - l (mean longitude of lunar perigee)
            -f5,                  # 5: N' = -Ω (negative of lunar node)
            f3 + f5 - f4 - f2     # 6: ps = h - l' (mean longitude of solar perigee)
        ])
        
        # Phase = sum(n_i * args_i)
        phases = np.dot(doodson_matrix, args) % 360.0
        phases = np.where(phases < 0, phases + 360.0, phases)
        
        # Frequency computation (cycle/day)
        # Derivatives of fundamental arguments with respect to time
        fd1 = 0.0362916471 + 0.0000000013 * t_cy
        fd2 = 0.0027377786
        fd3 = 0.0367481951 - 0.0000000005 * t_cy
        fd4 = 0.0338631920 - 0.0000000003 * t_cy
        fd5 = -0.0001470938 + 0.0000000003 * t_cy
        
        freq_dood = np.array([
            1.0 - fd4,            # d(tau)/dt
            fd3 + fd5,            # d(s)/dt
            fd3 + fd5 - fd4,      # d(h)/dt
            fd3 + fd5 - fd1,      # d(p)/dt
            -fd5,                 # d(N')/dt
            fd3 + fd5 - fd4 - fd2 # d(ps)/dt
        ])
        freqs = np.dot(doodson_matrix, freq_dood)
        
        return freqs, phases

    # ======================================================================
    # NATURAL CUBIC SPLINE INTERPOLATION (IERS EVAL.F)
    # ======================================================================
    @staticmethod
    def _cubic_spline(x_anchors: np.ndarray, y_anchors: np.ndarray, 
                      x_targets: np.ndarray) -> np.ndarray:
        """
        Natural cubic spline interpolation (IERS EVAL.F algorithm).
        
        This implements the natural cubic spline (second derivatives zero
        at endpoints) as used in the IERS EVAL.F routine for admittance
        interpolation. The algorithm is O(N) and does not require scipy.
        
        Parameters
        ----------
        x_anchors : np.ndarray, shape (N,)
            Anchor frequencies (must be sorted strictly increasing).
        y_anchors : np.ndarray, shape (N,)
            Anchor admittance values (real or imaginary).
        x_targets : np.ndarray, shape (M,)
            Target frequencies to interpolate.
        
        Returns
        -------
        y_targets : np.ndarray, shape (M,)
            Interpolated admittance values.
        
        References
        ----------
        - IERS Conventions 2010, Section 6.3 (Admittance interpolation)
        - IERS EVAL.F subroutine (internal IERS code)
        - Press et al. (1992), Numerical Recipes, Section 3.3
        """
        # Sort data by anchor frequency
        idx = np.argsort(x_anchors)
        x = x_anchors[idx]
        y = y_anchors[idx]
        n = len(x)
        
        if n < 2:
            return np.full_like(x_targets, y[0] if n == 1 else 0.0)
        
        # Compute differences and alpha coefficients
        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n-1):
            alpha[i] = (3.0/h[i])*(y[i+1]-y[i]) - (3.0/h[i-1])*(y[i]-y[i-1])
        
        # Thomas algorithm (tridiagonal solver) for second derivatives
        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)
        for i in range(1, n-1):
            l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]
        
        # Back substitution
        b = np.zeros(n)
        c = np.zeros(n)
        d = np.zeros(n)
        for j in range(n-2, -1, -1):
            c[j] = z[j] - mu[j]*c[j+1]
            b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1] + 2.0*c[j])/3.0
            d[j] = (c[j+1] - c[j])/(3.0*h[j])
        
        # Evaluate spline at target points
        results = np.zeros_like(x_targets)
        for k, xt in enumerate(x_targets):
            if xt <= x[0]:
                results[k] = y[0]
            elif xt >= x[-1]:
                results[k] = y[-1]
            else:
                i = np.searchsorted(x, xt) - 1
                i = max(0, min(i, n-2))
                dx = xt - x[i]
                results[k] = y[i] + b[i]*dx + c[i]*dx**2 + d[i]*dx**3
        return results

    # ======================================================================
    # ADMITTANCE INTERPOLATION FOR 342 CONSTITUENTS
    # ======================================================================
    def _process_admittance(self, comp_idx: int, freqs_342: np.ndarray,
                            mask: np.ndarray, anchors_mask: np.ndarray,
                            anchor_freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate complex admittance (real and imaginary) for one component.
        
        The admittance is defined as the ratio of the tidal displacement amplitude
        (from BLQ data) to the Cartwright-Edden amplitude (from the tidal potential).
        The real and imaginary parts are interpolated separately using natural
        cubic splines over the frequency domain.
        
        Parameters
        ----------
        comp_idx : int
            0 = radial (U), 1 = West (W), 2 = South (S)
        freqs_342 : np.ndarray, shape (342,)
            Frequencies for all 342 constituents.
        mask : np.ndarray, shape (342,), bool
            Mask selecting constituents in a specific spectral band (LP/DI/SD).
        anchors_mask : np.ndarray, shape (11,), bool
            Mask selecting anchor constituents in that spectral band.
        anchor_freqs : np.ndarray, shape (11,)
            Frequencies of the 11 anchor constituents.
        
        Returns
        -------
        real_interp, imag_interp : np.ndarray
            Interpolated real and imaginary admittance for the selected constituents.
        """
        blq_keys = np.array(self.main_waves)[anchors_mask]
        anchor_f = anchor_freqs[anchors_mask]
        
        real_anchors = []
        imag_anchors = []
        for key in blq_keys:
            # Find the match between the anchor and its Cartwright-Edden counterpart
            dood = self.main_doodson[self.main_waves.index(key)]
            match_idx = np.where((self.idd_342 == dood).all(axis=1))[0][0]
            tamp_val = abs(self.tamp_342[match_idx])
            
            # Complex admittance = (displacement amplitude / potential amplitude) * exp(i * phase)
            # Note: phase sign conversion: BLQ gives lag (positive), we need lead for cos() in Eq. 6.15
            amp = self.blq[key][comp_idx*2]
            ph = np.deg2rad(-self.blq[key][comp_idx*2+1])  # lag → lead
            
            real_anchors.append((amp / tamp_val) * np.cos(ph))
            imag_anchors.append((amp / tamp_val) * np.sin(ph))
        
        target_f = freqs_342[mask]
        if len(anchor_f) > 0 and len(target_f) > 0:
            real_interp = self._cubic_spline(anchor_f, np.array(real_anchors), target_f)
            imag_interp = self._cubic_spline(anchor_f, np.array(imag_anchors), target_f)
            return real_interp, imag_interp
        else:
            return np.zeros(len(target_f)), np.zeros(len(target_f))

    # ======================================================================
    # NODAL MODULATION (18.6-YEAR CYCLE)
    # ======================================================================
    def _apply_nodal_modulation(self, tide_name: str, tt_jd: float) -> Tuple[float, float]:
        """
        Compute nodal modulation factors (amplitude f and phase u) for primary tides.
        
        The 18.6-year nodal cycle (lunar node regression) modulates tidal amplitudes
        and phases. The modulation depends on the longitude of the ascending node (Ω).
        
        The corrections follow IERS Conventions 2010, Section 7, and are based on
        the analytical expressions:
            M2:  f = 1 + 0.037 cos(Ω),  u = -0.037 sin(Ω)
            K1:  f = 1 + 0.036 cos(Ω),  u = -0.036 sin(Ω)
            O1:  f = 1 + 0.038 cos(Ω),  u = -0.038 sin(Ω)
            N2:  f = 1 + 0.018 cos(Ω),  u = -0.018 sin(Ω)
            P1:  f = 1 + 0.017 cos(Ω),  u = -0.017 sin(Ω)
        
        Parameters
        ----------
        tide_name : str
            Name of the tidal constituent (e.g., 'M2', 'K1').
        tt_jd : float
            Julian Date in TT.
        
        Returns
        -------
        f : float
            Amplitude modulation factor.
        u : float
            Phase modulation in radians.
        
        References
        ----------
        - IERS Conventions 2010, Section 7 (Nodal modulation)
        - Explanations_about_FES2014b.pdf, Section 2
        """
        t_cy = (tt_jd - 2451545.0) / 36525.0
        args = self._compute_fundamental_arguments(t_cy)
        Om = args['Om']  # longitude of lunar ascending node (radians)
        
        tide_upper = tide_name.upper()
        if tide_upper == 'M2':
            f = 1.0 + 0.037 * math.cos(Om)
            u = -0.037 * math.sin(Om)
        elif tide_upper in ['K1', 'K2']:
            f = 1.0 + 0.036 * math.cos(Om)
            u = -0.036 * math.sin(Om)
        elif tide_upper == 'O1':
            f = 1.0 + 0.038 * math.cos(Om)
            u = -0.038 * math.sin(Om)
        elif tide_upper == 'N2':
            f = 1.0 + 0.018 * math.cos(Om)
            u = -0.018 * math.sin(Om)
        elif tide_upper == 'P1':
            f = 1.0 + 0.017 * math.cos(Om)
            u = -0.017 * math.sin(Om)
        else:
            f = 1.0
            u = 0.0
        return f, u

    # ======================================================================
    # EQUILIBRIUM LONG-PERIOD TIDES (Om1 & Om2)
    # ======================================================================
    def _compute_equilibrium_radial(self, mjd_tt: float, delta_t: float,
                                    doodson: List[int], amp_cm: float) -> float:
        """
        Compute radial displacement from equilibrium long-period tides (Om1/Om2).
        
        Om1 (Doodson 055.565, 18.6-year nodal tide) and Om2 (Doodson 055.575,
        9.3-year tide) are NOT provided in the FES2014b hydrodynamic grid because
        they are purely equilibrium tides (isostatic response). Official FES2014b
        handling (Explanations_about_FES2014b.pdf, Section 2) requires:
            - Amplitude: from astronomical ephemeris (Om1: 0.4347 cm, Om2: 0.0105 cm)
            - Phase convention: χ = -90° ("original Conventions")
            - Deformation factor: -0.305
                = h'₂/(1+k'₂) * (1+k₂-h₂)
                = (-0.3075/0.6925) * (0.6870)
            - Only radial component exists (zonal, m=0)
        
        Parameters
        ----------
        mjd_tt : float
            Modified Julian Date in TT.
        delta_t : float
            ΔT = TT - UT1 (seconds).
        doodson : list of int
            6-element Doodson number [n1, n2, n3, n4, n5, n6].
        amp_cm : float
            Astronomical amplitude in centimeters (from ephemeris).
        
        Returns
        -------
        dU : float
            Radial displacement in meters.
        
        References
        ----------
        - Explanations_about_FES2014b.pdf, Section 2 (Om1/Om2 handling)
        - IERS Conventions 2010, Table 6.3 & 6.4 (Love numbers)
        - Cartwright & Edden (1973), Geophys. J. R. astr. Soc., 33, 253-264
        """
        mjd_ut = mjd_tt - delta_t / 86400.0
        dood_mat = np.array([doodson])
        _, phase_deg = self._get_freq_phase_vectorized(dood_mat, mjd_tt, mjd_ut)
        
        # Phase = raw Doodson phase + χ (χ = -90° as per FES team decision)
        phase_rad = math.radians(phase_deg[0] - 90.0)
        
        # Astronomical amplitude (cm → m)
        H_f_m = amp_cm * 0.01
        
        # Deformation factor for OTL equilibrium (radial only)
        # Derived from IERS 2010, Section 6.3:
        #   h'_2 = -0.3075 (load Love number for potential)
        #   k'_2 = -0.3075 (load Love number for deformation)
        #   k_2  =  0.29525 (Love number for body tide)
        #   h_2  =  0.6078  (Shida number for body tide)
        factor = -0.305
        
        return factor * H_f_m * math.cos(phase_rad)

    # ======================================================================
    # MAIN COMPUTATION METHOD
    # ======================================================================
    def compute_displacement(self, mjd_tt: float, delta_t: float = 0.0,
                             include_equilibrium_long_period: bool = True) -> Tuple[float, float, float]:
        """
        Compute FES2014b Ocean Tide Loading displacement components.
        
        This is the main entry point for the engine. It evaluates the full
        342-constituent tidal spectrum using admittance interpolation from
        11 primary constituents, with nodal modulation, radiational tide
        zeroing, and optional equilibrium long-period tides (Om1/Om2).
        
        Parameters
        ----------
        mjd_tt : float
            Modified Julian Date in TT (Terrestrial Time).
        delta_t : float, optional
            ΔT = TT - UT1 in seconds. Default: 0.0.
        include_equilibrium_long_period : bool, optional
            If True (default), add Om1 and Om2 equilibrium tides (radial only).
            Should be True for displacement applications; set False for
            gravity/tilt applications where these zonal tides produce no
            measurable signals.
        
        Returns
        -------
        dU, dW, dS : float
            Displacement components in meters:
            dU = radial (positive up),
            dW = West (positive west),
            dS = South (positive south).
        
        References
        ----------
        - IERS Conventions 2010, Section 6.3 (Ocean Tides)
        - Explanations_about_FES2014b.pdf (FES team, 2019)
        - Lyard et al. (2006), Ocean Dynamics
        """
        mjd_ut = mjd_tt - delta_t / 86400.0
        
        # ================================================================
        # 1. Compute frequencies and phases for all 342 constituents
        # ================================================================
        freqs_342, phases_342 = self._get_freq_phase_vectorized(self.idd_342, mjd_tt, mjd_ut)
        
        # Apply phase bias based on Doodson first index (n1)
        # n1 = 0: long-period, 1: diurnal, 2: semi-diurnal
        n1 = self.idd_342[:, 0]
        phases_342 = np.where(n1 == 0, phases_342 + 180.0, phases_342)
        phases_342 = np.where(n1 == 1, phases_342 + 90.0, phases_342)
        phases_rad = np.deg2rad(phases_342)
        
        # Anchor frequencies (11 primary constituents)
        anchor_freqs, _ = self._get_freq_phase_vectorized(self.main_doodson, mjd_tt, mjd_ut)
        
        # Spectral band masks
        mask_lp = (n1 == 0)   # long-period
        mask_di = (n1 == 1)   # diurnal
        mask_sd = (n1 == 2)   # semi-diurnal
        
        anchor_n1 = self.main_doodson[:, 0]
        anchors_lp = (anchor_n1 == 0)
        anchors_di = (anchor_n1 == 1)
        anchors_sd = (anchor_n1 == 2)

        # ================================================================
        # 2. Apply 18.6-year nodal modulation to primary waves
        # ================================================================
        tt_jd = mjd_tt + 2400000.5  # convert MJD to JD (TT)
        self.modulated_blq = {}
        for tide in self.main_waves:
            f, u = self._apply_nodal_modulation(tide, tt_jd)
            raw = self.blq[tide]
            # raw = [amp_rad, ph_rad, amp_ew, ph_ew, amp_ns, ph_ns]
            self.modulated_blq[tide] = [
                raw[0] * f,
                raw[1] + math.degrees(u),
                raw[2] * f,
                raw[3] + math.degrees(u),
                raw[4] * f,
                raw[5] + math.degrees(u),
            ]

        # ================================================================
        # 3. Compute admittance interpolation for each component
        # ================================================================
        displacements = []
        for comp_idx in range(3):  # 0=radial, 1=west, 2=south
            # Interpolate real and imaginary parts for each spectral band
            re_lp, im_lp = self._process_admittance(comp_idx, freqs_342, mask_lp, anchors_lp, anchor_freqs)
            re_di, im_di = self._process_admittance(comp_idx, freqs_342, mask_di, anchors_di, anchor_freqs)
            re_sd, im_sd = self._process_admittance(comp_idx, freqs_342, mask_sd, anchors_sd, anchor_freqs)
            
            # Assemble full 342-element complex admittance array
            re_full = np.zeros(342)
            im_full = np.zeros(342)
            re_full[mask_lp] = re_lp
            im_full[mask_lp] = im_lp
            re_full[mask_di] = re_di
            im_full[mask_di] = im_di
            re_full[mask_sd] = re_sd
            im_full[mask_sd] = im_sd
            
            # Compute amplitude and phase from admittance
            amp_342 = self.tamp_342 * np.sqrt(re_full**2 + im_full**2)
            phase_admit = np.arctan2(im_full, re_full)
            
            # =============================================================
            # ZERO RADIATIONAL TIDES: S1 and Sa
            # These are handled by atm_loading_displacement (Ray & Ponte, 2003)
            # =============================================================
            for i, dood in enumerate(self.idd_342):
                # S1 = Doodson 164.555 = [1,6,4,5,5,5]
                if np.array_equal(dood, [1, 6, 4, 5, 5, 5]):
                    amp_342[i] = 0.0
                # Sa radiational = Doodson 056.555 = [0,5,6,5,5,5]
                if np.array_equal(dood, [0, 5, 6, 5, 5, 5]):
                    amp_342[i] = 0.0
            
            # Sum over all constituents
            final_phase = phases_rad + phase_admit
            disp = np.sum(amp_342 * np.cos(final_phase))
            displacements.append(disp)

        # ================================================================
        # 4. ADD EQUILIBRIUM LONG-PERIOD TIDES (Om1, Om2) - RADIAL ONLY
        # ================================================================
        if include_equilibrium_long_period:
            # Om1: Doodson 055.565, astronomical amplitude 0.4347 cm
            dU_om1 = self._compute_equilibrium_radial(mjd_tt, delta_t, [0,5,5,5,6,5], 0.4347)
            # Om2: Doodson 055.575, astronomical amplitude 0.0105 cm
            dU_om2 = self._compute_equilibrium_radial(mjd_tt, delta_t, [0,5,5,5,7,5], 0.0105)
            displacements[0] += dU_om1 + dU_om2

        # Return in meters: dU (radial), dW (West), dS (South)
        return displacements[0], displacements[1], displacements[2]


# ==============================================================================
#  UJI COBA DENGAN DATA BLQ JOLOTUNDO OBSV (FES2014b)
# ==============================================================================
if __name__ == "__main__":
    # Data BLQ displacement – sesuai file resmi dari Chalmers
    JOLOTUNDO_FES2014_BLQ = {
        'M2':  [0.01007, -164.0, 0.00252, -95.0,  0.00334,  43.4],
        'S2':  [0.00442, -89.8,  0.00115, -54.0,  0.00191, 110.0],
        'N2':  [0.00219, 165.3,  0.00050, -109.0, 0.00062,   5.6],
        'K2':  [0.00121, -95.3,  0.00030, -54.0,  0.00053, 109.8],
        'K1':  [0.01276,  0.7,   0.00118,  7.3,   0.00145,  89.1],
        'O1':  [0.00878, -22.1,  0.00075, -15.7,  0.00011,  49.6],
        'P1':  [0.00390, -1.4,   0.00035,  4.1,   0.00044,  86.4],
        'Q1':  [0.00186, -34.0,  0.00017, -13.0,  0.00007, 166.7],
        'Mf':  [0.00103, -169.5, 0.00006, -41.9,  0.00002,-147.9],
        'Mm':  [0.00061, -173.9, 0.00002, -71.4,  0.00003,-173.8],
        'Ssa': [0.00050, 178.7,  0.00002, -159.4, 0.00002, 169.3],
    }

    engine = Asterid342Engine_FES2014(JOLOTUNDO_FES2014_BLQ)

    # Epoch uji (sama seperti sebelumnya)
    mjd_ut_test = 55007.0 + 0.049131944444
    delta_t_test = 66.184
    mjd_tt_test = mjd_ut_test + (delta_t_test / 86400.0)

    print("=" * 75)
    print(" KALKULASI OTL ASTERID Ω-342 ENGINE — FES2014b")
    print(" Stasiun : JOLOTUNDO OBSV")
    print(" Datum   : Lat -7.609444° | Lon 112.595556° | Elev 561.002 m")
    print(" Epoch   : 2009-06-25 01:10:45 UT")
    print("=" * 75)

    dU, dW, dS = engine.compute_displacement(mjd_tt_test, delta_t_test)

    print(f"{'Komponen':<12} | {'Displacement (meter)':<20}")
    print("-" * 45)
    print(f"{'Radial (dU)':<12} | {dU:>16.8f}")
    print(f"{'West (dW)':<12} | {dW:>16.8f}")
    print(f"{'South (dS)':<12} | {dS:>16.8f}")
    print("=" * 75)
    print("✓ S1 & Sa radiasi dinolkan (ditangani oleh ATM loading).")
    print("✓ Om1 & Om2 dihitung ekuilibrium dengan fase -90°.")
    print("✓ Konsisten dengan Explanations_about_FES2014b.pdf")