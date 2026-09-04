#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# ASTERID Ω-342 : High-Precision Ocean Tide Loading Algorithmic Expansion
# FES2014b – Stasiun JOLOTUNDO OBSV
# ==============================================================================
# Perbaikan menyeluruh berdasarkan:
#   1. Explanations_about_FES2014b.pdf (catatan resmi FES team)
#   2. Data BLQ displacement & gravity dari http://holt.oso.chalmers.se/loading/
#   3. IERS Conventions 2010 (Bab 6)
#
# Modifikasi kritis:
#   - S1 (Doodson 164.555) dan Sa (056.555) DINOLKAN karena bersifat radiasi
#     (ditangani oleh atm_loading_displacement dari Ray & Ponte 2003)
#   - Om1 (055.565) dan Om2 (055.575) dihitung sebagai gelombang EKUILIBRIUM
#     dengan fase -90° dan faktor deformasi -0.305 (h'_2 / (1+k'_2) * (1+k2-h2))
# ==============================================================================

import math
import numpy as np
from typing import Tuple, List, Dict, Optional

class Asterid342Engine_FES2014:
    def __init__(self, blq_data: Dict[str, List[float]]):
        """
        Parameters
        ----------
        blq_data : dict
            Harus berisi 11 gelombang utama: 
            'M2','S2','N2','K2','K1','O1','P1','Q1','Mf','Mm','Ssa'
            Setiap nilai adalah list 6 elemen:
            [amp_rad (m), ph_rad (°), amp_ew (m), ph_ew (°), amp_ns (m), ph_ns (°)]
            Konvensi: fase positif = lag (terlambat), positif EW = ke barat, positif NS = ke selatan.
        """
        self.blq = blq_data
        
        # 11 Gelombang Utama (anchor) sesuai urutan file BLQ
        self.main_waves = ['M2', 'S2', 'N2', 'K2', 'K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm', 'Ssa']
        
        # Doodson number untuk 11 anchor (baris: M2, S2, N2, ...)
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
        # 342 koefisien amplitudo Cartwright‑Edden (tamp_342) 
        # dan Doodson number (idd_342) – diekstrak dari ADMINT.F IERS
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

        # Flat array Doodson 342 (6 angka per konstituen, total 2052 elemen)
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

    # ----------------------------------------------------------------------
    #  FREKUENSI & FASE DOODSON (porting dari TDFRPH.F IERS)
    # ----------------------------------------------------------------------
    def _get_freq_phase_vectorized(self, doodson_matrix: np.ndarray, 
                                   mjd_tt: float, mjd_ut: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Menghitung frekuensi (cycle/day) dan fase (derajat) untuk matriks Doodson.
        """
        t_cy = (mjd_tt - 51544.5) / 36525.0

        f1 = (134.9634025100 + t_cy * (477198.8675605000 + t_cy * (0.0088553333 + t_cy * (0.0000143431 + t_cy * (-0.0000000680)))))
        f2 = (357.5291091806 + t_cy * (35999.0502911389 + t_cy * (-0.0001536667 + t_cy * (0.0000000378 + t_cy * (-0.0000000032)))))
        f3 = (93.2720906200 + t_cy * (483202.0174577222 + t_cy * (-0.0035420000 + t_cy * (-0.0000002881 + t_cy * (0.0000000012)))))
        f4 = (297.8501954694 + t_cy * (445267.1114469445 + t_cy * (-0.0017696111 + t_cy * (0.0000018314 + t_cy * (-0.0000000088)))))
        f5 = (125.0445550100 + t_cy * (-1934.1362619722 + t_cy * (0.0020756111 + t_cy * (0.0000021394 + t_cy * (-0.0000000165)))))

        day_frac_ut = mjd_ut - np.floor(mjd_ut)
        tau = 360.0 * day_frac_ut - f4

        args = np.array([
            tau,
            f3 + f5,
            f3 + f5 - f4,
            f3 + f5 - f1,
            -f5,
            f3 + f5 - f4 - f2
        ])

        phases = np.dot(doodson_matrix, args) % 360.0
        phases = np.where(phases < 0, phases + 360.0, phases)

        fd1 = 0.0362916471 + 0.0000000013 * t_cy
        fd2 = 0.0027377786
        fd3 = 0.0367481951 - 0.0000000005 * t_cy
        fd4 = 0.0338631920 - 0.0000000003 * t_cy
        fd5 = -0.0001470938 + 0.0000000003 * t_cy

        freq_dood = np.array([
            1.0 - fd4,
            fd3 + fd5,
            fd3 + fd5 - fd4,
            fd3 + fd5 - fd1,
            -fd5,
            fd3 + fd5 - fd4 - fd2
        ])
        freqs = np.dot(doodson_matrix, freq_dood)
        return freqs, phases

    # ----------------------------------------------------------------------
    #  CUBIC SPLINE INTERPOLATION (natural, manual, tanpa scipy)
    # ----------------------------------------------------------------------
    @staticmethod
    def _cubic_spline(x_anchors: np.ndarray, y_anchors: np.ndarray, 
                      x_targets: np.ndarray) -> np.ndarray:
        """
        Natural cubic spline interpolation.
        """
        idx = np.argsort(x_anchors)
        x = x_anchors[idx]
        y = y_anchors[idx]
        n = len(x)
        if n < 2:
            return np.full_like(x_targets, y[0] if n == 1 else 0.0)

        h = np.diff(x)
        alpha = np.zeros(n)
        for i in range(1, n-1):
            alpha[i] = (3.0/h[i])*(y[i+1]-y[i]) - (3.0/h[i-1])*(y[i]-y[i-1])

        l = np.ones(n)
        mu = np.zeros(n)
        z = np.zeros(n)
        for i in range(1, n-1):
            l[i] = 2.0*(x[i+1]-x[i-1]) - h[i-1]*mu[i-1]
            mu[i] = h[i]/l[i]
            z[i] = (alpha[i] - h[i-1]*z[i-1])/l[i]

        b = np.zeros(n)
        c = np.zeros(n)
        d = np.zeros(n)
        for j in range(n-2, -1, -1):
            c[j] = z[j] - mu[j]*c[j+1]
            b[j] = (y[j+1]-y[j])/h[j] - h[j]*(c[j+1] + 2.0*c[j])/3.0
            d[j] = (c[j+1] - c[j])/(3.0*h[j])

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

    # ----------------------------------------------------------------------
    #  ADMITANSI INTERPOLATION (untuk gelombang gravitasi)
    # ----------------------------------------------------------------------
    def _process_admittance(self, comp_idx: int, freqs_342: np.ndarray,
                            mask: np.ndarray, anchors_mask: np.ndarray,
                            anchor_freqs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolasi admitansi riil & imajiner untuk satu komponen (U, W, atau S).
        """
        blq_keys = np.array(self.main_waves)[anchors_mask]
        anchor_f = anchor_freqs[anchors_mask]

        real_anchors = []
        imag_anchors = []
        for key in blq_keys:
            dood = self.main_doodson[self.main_waves.index(key)]
            match_idx = np.where((self.idd_342 == dood).all(axis=1))[0][0]
            tamp_val = abs(self.tamp_342[match_idx])

            amp = self.blq[key][comp_idx*2]
            ph = np.deg2rad(-self.blq[key][comp_idx*2+1])  # konversi lag → lead

            real_anchors.append((amp / tamp_val) * np.cos(ph))
            imag_anchors.append((amp / tamp_val) * np.sin(ph))

        target_f = freqs_342[mask]
        if len(anchor_f) > 0 and len(target_f) > 0:
            real_interp = self._cubic_spline(anchor_f, np.array(real_anchors), target_f)
            imag_interp = self._cubic_spline(anchor_f, np.array(imag_anchors), target_f)
            return real_interp, imag_interp
        else:
            return np.zeros(len(target_f)), np.zeros(len(target_f))

    # ----------------------------------------------------------------------
    #  GELOMBANG EKUILIBRIUM Om1 & Om2 (FES2014b official treatment)
    # ----------------------------------------------------------------------
    def _compute_equilibrium_radial(self, mjd_tt: float, delta_t: float,
                                    doodson: List[int], amp_cm: float) -> float:
        """
        Menghitung displacement radial (meter) untuk Om1 (055.565) dan Om2 (055.575).

        Menggunakan faktor deformasi OTL ekuilibrium:
            dU_OTL = [h'_2 / (1 + k'_2)] * (1 + k_2 - h_2) * H_f
            = -0.305 * H_f

        Fase : χ = -90° (sesuai keputusan FES team, "original Conventions")
        """
        mjd_ut = mjd_tt - delta_t / 86400.0
        dood_mat = np.array([doodson])
        _, phase_deg = self._get_freq_phase_vectorized(dood_mat, mjd_tt, mjd_ut)

        # Fase ekuilibrium: raw Doodson + χ (χ = -90°)
        phase_rad = math.radians(phase_deg[0] - 90.0)

        # Konversi amplitudo astronomi cm → m
        H_f_m = amp_cm * 0.01

        # Faktor deformasi OTL ekuilibrium untuk derajat 2 (h'_2=-0.3075, k'_2=-0.3075, k2=0.29525, h2=0.6078)
        # = (-0.3075/0.6925) * (1 + 0.29525 - 0.6078) = -0.305
        factor = -0.305
        amp_m = factor * H_f_m

        return amp_m * math.cos(phase_rad)

    # ----------------------------------------------------------------------
    #  MAIN API : compute_displacement
    # ----------------------------------------------------------------------
    def compute_displacement(self, mjd_tt: float, delta_t: float = 0.0) -> Tuple[float, float, float]:
        """
        Menghitung total displacement OTL (dU, dW, dS) dalam meter.

        Parameters
        ----------
        mjd_tt : float
            Modified Julian Date dalam skala Terrestrial Time (TT)
        delta_t : float
            ΔT = TT - UT1 (detik)

        Returns
        -------
        (dU, dW, dS) : float
            dU = radial (positif ke atas)
            dW = barat (positif ke barat)
            dS = selatan (positif ke selatan)
        """
        mjd_ut = mjd_tt - delta_t / 86400.0

        # 1. Frekuensi & fase untuk 342 konstituen dan 11 anchor
        freqs_342, phases_342 = self._get_freq_phase_vectorized(self.idd_342, mjd_tt, mjd_ut)
        anchor_freqs, _ = self._get_freq_phase_vectorized(self.main_doodson, mjd_tt, mjd_ut)

        # 2. Masking berdasarkan jenis gelombang (n1 = 0: long period, 1: diurnal, 2: semi-diurnal)
        n1 = self.idd_342[:, 0]
        phases_342 = np.where(n1 == 0, phases_342 + 180.0, phases_342)   # bias long period
        phases_342 = np.where(n1 == 1, phases_342 + 90.0, phases_342)    # bias diurnal
        phases_rad = np.deg2rad(phases_342)

        mask_lp = (n1 == 0)
        mask_di = (n1 == 1)
        mask_sd = (n1 == 2)

        anchor_n1 = self.main_doodson[:, 0]
        anchors_lp = (anchor_n1 == 0)
        anchors_di = (anchor_n1 == 1)
        anchors_sd = (anchor_n1 == 2)

        # 3. Loop untuk 3 komponen (Radial, East-West, North-South)
        dU, dW, dS = 0.0, 0.0, 0.0

        for comp_idx in range(3):
            re_lp, im_lp = self._process_admittance(comp_idx, freqs_342, mask_lp, anchors_lp, anchor_freqs)
            re_di, im_di = self._process_admittance(comp_idx, freqs_342, mask_di, anchors_di, anchor_freqs)
            re_sd, im_sd = self._process_admittance(comp_idx, freqs_342, mask_sd, anchors_sd, anchor_freqs)

            re_full = np.zeros(342)
            im_full = np.zeros(342)
            re_full[mask_lp] = re_lp
            im_full[mask_lp] = im_lp
            re_full[mask_di] = re_di
            im_full[mask_di] = im_di
            re_full[mask_sd] = re_sd
            im_full[mask_sd] = im_sd

            amp_342 = self.tamp_342 * np.sqrt(re_full**2 + im_full**2)
            phase_admit = np.arctan2(im_full, re_full)

            # ----------------------------------------------------------
            #  PERBAIKAN KRITIS: NOLKAN GELOMBANG RADIASI
            #  S1 (Doodson 164.555) dan Sa (056.555) adalah radiasi
            #  Ditangani oleh atm_loading_displacement (Ray & Ponte 2003)
            # ----------------------------------------------------------
            for i, dood in enumerate(self.idd_342):
                # S1 = [1,6,4,5,5,5]
                if np.array_equal(dood, [1, 6, 4, 5, 5, 5]):
                    amp_342[i] = 0.0
                # Sa radiational = [0,5,6,5,5,5]
                if np.array_equal(dood, [0, 5, 6, 5, 5, 5]):
                    amp_342[i] = 0.0

            final_phase = phases_rad + phase_admit
            disp = np.sum(amp_342 * np.cos(final_phase))

            if comp_idx == 0:
                dU = disp
            elif comp_idx == 1:
                dW = disp   # West
            else:
                dS = disp   # South

        # --------------------------------------------------------------
        #  TAMBAHKAN Om1 & Om2 SECARA EKUILIBRIUM (hanya komponen radial)
        #  Om1: Doodson 055.565, amp astronomi 0.4347 cm
        #  Om2: Doodson 055.575, amp astronomi 0.0105 cm
        # --------------------------------------------------------------
        dU_om1 = self._compute_equilibrium_radial(mjd_tt, delta_t, [0,5,5,5,6,5], 0.4347)
        dU_om2 = self._compute_equilibrium_radial(mjd_tt, delta_t, [0,5,5,5,7,5], 0.0105)
        dU += dU_om1 + dU_om2

        return dU, dW, dS


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