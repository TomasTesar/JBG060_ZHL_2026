import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from processing_data.loading_impact_data import load_admin_boundaries
from processing_data.loading import load_flood_masks, load_rainfall_runoff

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# ---------------------------------------------------------------
# 1. Pilot region bbox (same as before)
# ---------------------------------------------------------------
admin1, admin2 = load_admin_boundaries()
pilot_name = "Aweil East"
pilot_county = admin2[admin2.adm2_name == pilot_name]
bounds = pilot_county.geometry.total_bounds

bbox = {
    "lon_min": bounds[0],
    "lat_min": bounds[1],
    "lon_max": bounds[2],
    "lat_max": bounds[3],
}

# ---------------------------------------------------------------
# 2. Flood events per month (2023-2024) — reuse from before
# ---------------------------------------------------------------
flood_events = load_flood_masks(np.array([2023, 2024]), bbox=bbox)
flood_events["year_month"] = pd.to_datetime(flood_events["date"]).dt.to_period("M")
flood_monthly = flood_events.groupby("year_month").size()

# ---------------------------------------------------------------
# 3. Rainfall + runoff for the same years, clipped to bbox
# ---------------------------------------------------------------
print("Loading rainfall/runoff for 2023-2024 (this may take a moment)...")
rainfall_runoff = load_rainfall_runoff(np.array([2023, 2024]))

# Clip to bbox
rr_clipped = rainfall_runoff.sel(
    latitude=slice(bbox["lat_max"], bbox["lat_min"]),  # note: often stored north->south
    longitude=slice(bbox["lon_min"], bbox["lon_max"]),
)

# Spatial mean over the region, daily
tp_daily = rr_clipped["tp"].mean(dim=["latitude", "longitude"]).to_series()
ro_daily = rr_clipped["ro"].mean(dim=["latitude", "longitude"]).to_series()

# Resample to monthly totals for comparison with flood counts
tp_monthly = tp_daily.resample("MS").sum()
ro_monthly = ro_daily.resample("MS").sum()

print("\n=== Monthly rainfall (mm, region average) ===")
print(tp_monthly)
print("\n=== Monthly runoff (mm, region average) ===")
print(ro_monthly)

# ---------------------------------------------------------------
# 4. Combined plot: flood pixel-days vs rainfall vs runoff
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

flood_monthly.plot(ax=axes[0], kind="bar", color="steelblue")
axes[0].set_title(f"Flood pixel-days per month — {pilot_name}")
axes[0].set_ylabel("Pixel-days")

tp_monthly.plot(ax=axes[1], color="green")
axes[1].set_title("Monthly rainfall (region average, mm)")
axes[1].set_ylabel("mm")

ro_monthly.plot(ax=axes[2], color="darkorange")
axes[2].set_title("Monthly runoff (region average, mm)")
axes[2].set_ylabel("mm")

plt.tight_layout()
plt.savefig("aweil_east_flood_rainfall_runoff.png", dpi=150)
print("\nSaved plot to aweil_east_flood_rainfall_runoff.png")
plt.show()