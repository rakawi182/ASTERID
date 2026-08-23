import urllib.request
import urllib.parse
import json

def get_elevation(lat: float, lon: float, dataset: str = "COP30", api_key: str = None) -> float:
    """
    Ambil elevasi satu titik dari OpenTopography API (endpoint /v1/elevation).

    Parameters
    ----------
    lat, lon : float
        Koordinat geodetik (derajat desimal).
    dataset : str
        Nama dataset, misal 'COP30', 'SRTMGL1', 'NASADEM', dll.
    api_key : str
        API key dari OpenTopography. Wajib diisi.

    Returns
    -------
    float
        Elevasi dalam meter.
    """
    if api_key is None:
        raise ValueError("API key harus diberikan.")

    url = "https://portal.opentopography.org/API/v1/elevation"
    params = {
        "latitude": lat,
        "longitude": lon,
        "dataset": dataset,
        "API_Key": api_key
    }
    full_url = url + "?" + urllib.parse.urlencode(params)

    with urllib.request.urlopen(full_url) as response:
        if response.getcode() != 200:
            raise Exception(f"Gagal: {response.getcode()}")
        data = json.loads(response.read().decode())

    # Perbaikan: gunakan "Elevation" (huruf kapital E)
    elev = data.get("Elevation")
    if elev is None:
        raise KeyError(f"Kunci 'Elevation' tidak ditemukan. Keys: {list(data.keys())}")

    return float(elev)


if __name__ == "__main__":
    API_KEY = "c653fbe7ef769b5643eb8d057c73a811"

    # Contoh pertama: Jolotundo
    try:
        elev = get_elevation(
            lat=-7.609444,
            lon=112.595556,
            dataset="COP30",
            api_key=API_KEY
        )
        print(f"✅ Elevasi Jolotundo: {elev:.2f} m")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Contoh kedua: koordinat baru (112.566089, -7.521951)
    try:
        elev2 = get_elevation(
            lat=-7.521951,   # latitude
            lon=112.566089,  # longitude
            dataset="COP30",
            api_key=API_KEY
        )
        print(f"✅ Elevasi Home (112.566089, -7.521951): {elev2:.2f} m")
    except Exception as e:
        print(f"❌ Error pada titik kedua: {e}")