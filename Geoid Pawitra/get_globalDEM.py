import urllib.request
import urllib.parse
import json

def download_dem(
    demtype: str,
    south: float,
    north: float,
    west: float,
    east: float,
    api_key: str,
    output_format: str = "AAIGrid",
    save_path: str = "dem.asc"
) -> str:
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": output_format,
        "API_Key": api_key
    }
    full_url = url + "?" + urllib.parse.urlencode(params)

    try:
        with urllib.request.urlopen(full_url) as response:
            if response.getcode() != 200:
                raise Exception(f"Gagal mengunduh: {response.getcode()}")
            with open(save_path, "wb") as f:
                f.write(response.read())
        print(f"✅ DEM berhasil diunduh ke: {save_path}")
        return save_path
    except Exception as e:
        raise Exception(f"Error: {e}")

def get_elevation(lat: float, lon: float, dataset: str = "COP30", api_key: str = None) -> float:
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
    try:
        with urllib.request.urlopen(full_url) as response:
            if response.getcode() != 200:
                raise Exception(f"Gagal: {response.getcode()}")
            data = json.loads(response.read().decode())
            return data.get("elevation", None)
    except Exception as e:
        raise Exception(f"Error: {e}")

if __name__ == "__main__":
    API_KEY = "c653fbe7ef769b5643eb8d057c73a811"

    # Contoh download DEM
    try:
        download_dem(
            demtype="COP30",
            south=-7.665416655556,
            north=-7.565416655556,
            west=112.565416655556,
            east=112.675416655556,
            api_key=API_KEY,
            output_format="AAIGrid",
            save_path="jolotundo_cop30.asc"
        )
    except Exception as e:
        print(f"❌ Error download DEM: {e}")

