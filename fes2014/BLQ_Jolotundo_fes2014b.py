blq_data = {
    "metadata": {
        "model": "FES2014b",
        "station": "JOLOTUNDO OBSV",
        "longitude": 112.595556,   # derajat, positif ke timur
        "latitude": -7.609444,     # derajat, negatif berarti selatan
        "height": 561.002,         # meter, tinggi ellipsoid (tide-free)
        "description": {
            "amplitudes_unit": "meter",
            "phases_unit": "derajat",
            "phase_convention": "fase lag relatif terhadap Greenwich, positif berarti lag (terlambat)",
            "radial_positive": "ke atas (upwards)",
            "tangential_EW_positive": "ke barat (west)",
            "tangential_NS_positive": "ke selatan (south)",
            "constituents_order": ["M2", "S2", "N2", "K2", "K1", "O1", "P1", "Q1", "MF", "MM", "SSA"],
            "source": "http://holt.oso.chalmers.se/loading/",
            "green_function": "PREM",
            "corrections": "subtracted uniform tidal layer to conserve water mass, seawater density = 1030 kg/m^3"
        }
    },
    "radial": {
        "amplitudes": [0.01007, 0.00442, 0.00219, 0.00121, 0.01276, 0.00878, 0.00390, 0.00186, 0.00103, 0.00061, 0.00050],
        "phases": [-164.0, -89.8, 165.3, -95.3, 0.7, -22.1, -1.4, -34.0, -169.5, -173.9, 178.7]
    },
    "tangential_EW": {
        "amplitudes": [0.00252, 0.00115, 0.00050, 0.00030, 0.00118, 0.00075, 0.00035, 0.00017, 0.00006, 0.00002, 0.00002],
        "phases": [-95.0, -54.0, -109.0, -54.0, 7.3, -15.7, 4.1, -13.0, -41.9, -71.4, -159.4]
    },
    "tangential_NS": {
        "amplitudes": [0.00334, 0.00191, 0.00062, 0.00053, 0.00145, 0.00011, 0.00044, 0.00007, 0.00002, 0.00003, 0.00002],
        "phases": [43.4, 110.0, 5.6, 109.8, 89.1, 49.6, 86.4, 166.7, -147.9, -173.8, 169.3]
    }
}