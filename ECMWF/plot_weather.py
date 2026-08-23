import sys
sys.dont_write_bytecode = True

import matplotlib.pyplot as plt
from datetime import datetime, timezone
import home_ecmwf as em

def plot_advanced_weather_data():
    # 1. Set coordinates 
    lat = -7.521951
    lon = 112.566089
    
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Fetching advanced weather data for {date_str}...")
    
    # 2. Call the internal function to retrieve raw data
    raw_data = em._fetch_from_openmeteo(lat, lon, date_str, date_str)
    
    if raw_data is None or "hourly" not in raw_data:
        print("Failed to retrieve data from the API.")
        return

    # 3. Extract required data
    hourly = raw_data['hourly']
    times = [t.split("T")[1] for t in hourly['time']]
    temps = hourly['temperature_2m']
    rh = hourly['relative_humidity_2m']
    precip = hourly['precipitation']
    solar_rad = hourly['shortwave_radiation']

    # 4. Create Figure with adjusted height and proportions
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [3, 2, 1.5]})
    
    # FIX 1: Unified Title to save space and avoid collisions. Placed explicitly at the top.
    fig.suptitle(f'ECMWF IFS HRES Weather & Radiation Forecast (9 km)\nAtmospheric Conditions at {lat}, {lon}', 
                 fontsize=15, fontweight='bold', y=0.97)

    # --- Top Plot: Temperature and Humidity ---
    color_temp = 'tab:red'
    ax1.set_ylabel('Temperature (°C)', color=color_temp, fontweight='bold')
    line1, = ax1.plot(times, temps, color=color_temp, marker='o', label='Temperature (°C)')
    ax1.tick_params(axis='y', labelcolor=color_temp)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xticklabels([]) 

    ax1_twin = ax1.twinx()
    color_rh = 'tab:blue'
    ax1_twin.set_ylabel('Relative Humidity (%)', color=color_rh, fontweight='bold')
    line2, = ax1_twin.plot(times, rh, color=color_rh, marker='s', linestyle='--', label='Relative Humidity (%)')
    ax1_twin.tick_params(axis='y', labelcolor=color_rh)
    ax1_twin.set_ylim(0, 105)

    # FIX 2: Legend placed neatly just above the top axis border, without a frame to look cleaner.
    ax1.legend(handles=[line1, line2], loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    # (ax1.set_title is removed because it is now combined in fig.suptitle)

    # --- Middle Plot: Solar Radiation ---
    color_solar = 'tab:orange'
    ax2.set_xlabel('Time (UTC)')
    ax2.set_ylabel('Solar Rad (W/m²)', color=color_solar, fontweight='bold')
    ax2.plot(times, solar_rad, color=color_solar, marker='^', label='Shortwave Radiation')
    ax2.fill_between(times, solar_rad, color=color_solar, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color_solar)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_title('Shortwave Solar Radiation', pad=10)

    # --- Bottom Plot / Table: Summary Data ---
    ax3.axis('tight')
    ax3.axis('off')
    
    table_data = []
    col_labels = ["Time", "Temp (°C)", "RH (%)", "Precip (mm)", "Solar (W/m²)"]
    
    for i in range(0, len(times), 3):
        table_data.append([times[i], f"{temps[i]:.1f}", f"{rh[i]:.0f}", f"{precip[i]:.1f}", f"{solar_rad[i]:.0f}"])
        
    table = ax3.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center', bbox=[0, 0, 1, 0.9])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    ax3.set_title('Summary Table (3-Hour Samples)', fontweight='bold', pad=10)

    # FIX 3: Strict layout boundaries. 
    # rect=[left, bottom, right, top] ensures the plots don't overwrite the suptitle.
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    # Explicit vertical space between subplots
    plt.subplots_adjust(hspace=0.35) 
    
    plt.show()

if __name__ == "__main__":
    plot_advanced_weather_data()
