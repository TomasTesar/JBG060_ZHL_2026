import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from processing_data.loading_impact_data import load_admin_boundaries, load_farmland_mask
from processing_data.loading import load_flood_masks

# ---------------------------------------------------------------
# 1. National boundary / bbox
# ---------------------------------------------------------------
admin1, admin2 = load_admin_boundaries()
bounds = admin1.total_bounds  # [lon_min, lat_min, lon_max, lat_max]
national_bbox = {
    "lon_min": bounds[0], "lat_min": bounds[1],
    "lon_max": bounds[2], "lat_max": bounds[3],
}
print(f"National bbox: {national_bbox}")

# ---------------------------------------------------------------
# 2. Load cropland and rangeland masks nationally
# ---------------------------------------------------------------
print("\nLoading cropland and rangeland masks...")
crop_da = load_farmland_mask(national_bbox, mask_type="crop")
range_da = load_farmland_mask(national_bbox, mask_type="rangeland")

print(f"Cropland grid shape: {crop_da.shape}, dims: {crop_da.dims}")
print(f"Rangeland grid shape: {range_da.shape}, dims: {range_da.dims}")

lat_dim = "latitude" if "latitude" in crop_da.dims else "lat"
lon_dim = "longitude" if "longitude" in crop_da.dims else "lon"

lat_vals = crop_da[lat_dim].values
lon_vals = crop_da[lon_dim].values

# imshow needs to know whether latitude runs ascending or descending
# so the image isn't flipped upside down
origin = "upper" if lat_vals[0] > lat_vals[-1] else "lower"
extent = [lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max()]

# ---------------------------------------------------------------
# 3. Load Nov 2024 flood pixels nationally (for the overlay)
# ---------------------------------------------------------------
print("\nLoading national flood mask data for Nov 2024...")
flood_events = load_flood_masks(np.array([2024]), bbox=None)
flood_events["date"] = pd.to_datetime(flood_events["date"])
nov_2024 = flood_events[
    (flood_events["date"] >= "2024-11-01") & (flood_events["date"] < "2024-12-01")
]
flooded_pixels = nov_2024[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
print(f"Unique flooded pixel locations nationally, Nov 2024: {len(flooded_pixels):,}")

# ---------------------------------------------------------------
# 4. Plot: cropland map, rangeland map, and a combined map with flood overlay
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# --- Cropland ---
im0 = axes[0].imshow(
    crop_da.values, extent=extent, origin=origin, cmap="YlGn", vmin=0, vmax=100
)
admin1.boundary.plot(ax=axes[0], color="black", linewidth=0.5)
axes[0].set_title("Cropland cover (%)")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

# --- Rangeland ---
im1 = axes[1].imshow(
    range_da.values, extent=extent, origin=origin, cmap="YlOrBr", vmin=0, vmax=100
)
admin1.boundary.plot(ax=axes[1], color="black", linewidth=0.5)
axes[1].set_title("Rangeland cover (%)")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

# --- Cropland + flood overlay ---
im2 = axes[2].imshow(
    crop_da.values, extent=extent, origin=origin, cmap="Greens", vmin=0, vmax=100, alpha=0.8
)
axes[2].scatter(
    flooded_pixels["lon"], flooded_pixels["lat"],
    s=0.5, color="crimson", alpha=0.25, label="Flooded pixels (Nov 2024)",
)
admin1.boundary.plot(ax=axes[2], color="black", linewidth=0.5)
axes[2].set_title("Cropland + Nov 2024 flood extent")
axes[2].set_xlabel("Longitude")
axes[2].set_ylabel("Latitude")
axes[2].legend(markerscale=15, loc="lower left")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("south_sudan_agriculture_map.png", dpi=150)
print("\nSaved plot to south_sudan_agriculture_map.png")
plt.show()

# ---------------------------------------------------------------
# 5. Quantify: how much of the flooded area was cropland/rangeland?
# ---------------------------------------------------------------
print("\n=== National summary: flooded pixels' land cover ===")


def nearest_value(da_values, lat_arr, lon_arr, lats, lons):
    lat_idx = np.abs(lat_arr[:, None] - np.array(lats)[None, :]).argmin(axis=0)
    lon_idx = np.abs(lon_arr[:, None] - np.array(lons)[None, :]).argmin(axis=0)
    return da_values[lat_idx, lon_idx]


crop_at_flood = nearest_value(
    crop_da.values, lat_vals, lon_vals, flooded_pixels["lat"].values, flooded_pixels["lon"].values
)
range_at_flood = nearest_value(
    range_da.values, lat_vals, lon_vals, flooded_pixels["lat"].values, flooded_pixels["lon"].values
)

print(f"Average cropland cover in flooded pixels: {crop_at_flood.mean():.1f}%")
print(f"Average rangeland cover in flooded pixels: {range_at_flood.mean():.1f}%")
print(f"Flooded pixels with >10% cropland: {(crop_at_flood > 10).sum():,} "
      f"({(crop_at_flood > 10).mean() * 100:.1f}% of all flooded pixels)")
print(f"Flooded pixels with >10% rangeland: {(range_at_flood > 10).sum():,} "
      f"({(range_at_flood > 10).mean() * 100:.1f}% of all flooded pixels)")