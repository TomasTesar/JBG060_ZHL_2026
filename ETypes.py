import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path

from processing_data.loading_impact_data import load_admin_boundaries, load_farmland_mask
from processing_data.loading import load_flood_masks

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def get_dim_names(da):
    """Figure out which dimension names this DataArray actually uses."""
    lat_candidates = ["lat", "latitude", "y"]
    lon_candidates = ["lon", "longitude", "x"]
    lat_dim = next((d for d in lat_candidates if d in da.dims), None)
    lon_dim = next((d for d in lon_candidates if d in da.dims), None)
    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not determine lat/lon dimension names from {da.dims}")
    return lat_dim, lon_dim


def sample_dataarray_nearest(da, lat_dim, lon_dim, lats, lons):
    """
    Manual nearest-neighbor lookup into a 2D xarray DataArray, bypassing
    xarray's own vectorized .sel() (which can choke on some coordinate
    array shapes/dtypes). Grids here are small (a few hundred cells per
    axis), so a simple argmin per point is plenty fast.
    """
    lat_vals = da[lat_dim].values
    lon_vals = da[lon_dim].values
    arr = da.values
    dims = da.dims
    lat_axis = dims.index(lat_dim)
    lon_axis = dims.index(lon_dim)

    results = np.empty(len(lats))
    for i, (lat, lon) in enumerate(zip(lats, lons)):
        lat_idx = np.abs(lat_vals - lat).argmin()
        lon_idx = np.abs(lon_vals - lon).argmin()
        idx = [slice(None)] * arr.ndim
        idx[lat_axis] = lat_idx
        idx[lon_axis] = lon_idx
        results[i] = arr[tuple(idx)]
    return results


def sample_raster_deduped(raster_path, coords):
    """
    Point-sample a raster at (lon, lat) coords, but collapse coords that
    fall into the SAME raster cell before summing. This matters whenever
    the raster's resolution is coarser than the spacing of your sample
    points (e.g. a 9.3km cattle grid sampled at 250m flood-pixel locations)
    — otherwise you count that one cell's value once per overlapping point,
    massively inflating any total.

    Returns:
        unique_cell_values -> use this for SUMMING (totals)
        per_point_values   -> use this for per-pixel inspection / averaging
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        rows_cols = [src.index(lon, lat) for lon, lat in coords]

    per_point_values = np.clip(
        np.array([arr[r, c] for r, c in rows_cols], dtype=float), 0, None
    )

    unique_cells = list(set(rows_cols))
    unique_cell_values = np.clip(
        np.array([arr[r, c] for r, c in unique_cells], dtype=float), 0, None
    )

    return unique_cell_values, per_point_values


# ---------------------------------------------------------------
# 0. Setup: pilot region bbox
# ---------------------------------------------------------------
admin1, admin2 = load_admin_boundaries()
pilot_name = "Aweil East"
pilot_county = admin2[admin2.adm2_name == pilot_name]
bounds = pilot_county.geometry.total_bounds
bbox = {
    "lon_min": bounds[0], "lat_min": bounds[1],
    "lon_max": bounds[2], "lat_max": bounds[3],
}

pop_2024_path = next(Path("raw_data").rglob("ssd_pop_2024_*.tif"), None)
cattle_path = next(Path("raw_data").rglob("geonode__cattle_gha.tif"), None)

if pop_2024_path is None or cattle_path is None:
    raise FileNotFoundError(
        "Could not find one of the raster files under raw_data/. "
        "Run: list(Path('raw_data').rglob('*.tif')) to check paths."
    )

print(f"Using population raster: {pop_2024_path}")
print(f"Using cattle raster: {cattle_path}")

# ---------------------------------------------------------------
# PART A: Exposure overlay for the Nov 2024 flood peak
# ---------------------------------------------------------------
print("\n### PART A: Exposure overlay, Nov 2024 flood peak ###\n")

flood_events = load_flood_masks(np.array([2024]), bbox=bbox)
flood_events["date"] = pd.to_datetime(flood_events["date"])
nov_2024 = flood_events[
    (flood_events["date"] >= "2024-11-01") & (flood_events["date"] < "2024-12-01")
]

flooded_pixels = nov_2024[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
print(f"Unique flooded pixel locations in Nov 2024: {len(flooded_pixels)}")

coords = list(zip(flooded_pixels["lon"], flooded_pixels["lat"]))

# --- Population: dedupe by raster cell before summing ---
pop_unique_cells, pop_per_point = sample_raster_deduped(pop_2024_path, coords)
flooded_pixels["population"] = pop_per_point
total_flooded_population = pop_unique_cells.sum()
print(f"Estimated population in flooded pixels (Nov 2024, deduped): {total_flooded_population:,.0f}")

# --- Cattle: dedupe is CRITICAL here — cattle grid (~9.3km) is much
#     coarser than flood pixels (~250m), so naive summing wildly inflates ---
cattle_unique_cells, cattle_per_point = sample_raster_deduped(cattle_path, coords)
flooded_pixels["cattle"] = cattle_per_point
total_flooded_cattle = cattle_unique_cells.sum()
print(f"Estimated cattle in flooded pixels (Nov 2024, deduped): {total_flooded_cattle:,.0f}")
print(
    f"(For comparison, naively summing per flood-pixel — WRONG — would give "
    f"{cattle_per_point.sum():,.0f}, since it counts each coarse cattle cell "
    f"once per overlapping 250m flood pixel instead of once per cell)"
)

# --- Cropland / rangeland: fine for averaging as-is, no dedupe needed ---
crop_da = load_farmland_mask(bbox, mask_type="crop")
range_da = load_farmland_mask(bbox, mask_type="rangeland")

lat_dim, lon_dim = get_dim_names(crop_da)
print(f"Cropland DataArray dims: {crop_da.dims} -> using lat_dim='{lat_dim}', lon_dim='{lon_dim}'")

crop_vals = sample_dataarray_nearest(
    crop_da, lat_dim, lon_dim, flooded_pixels["lat"].values, flooded_pixels["lon"].values
)
range_vals = sample_dataarray_nearest(
    range_da, lat_dim, lon_dim, flooded_pixels["lat"].values, flooded_pixels["lon"].values
)

flooded_pixels["crop_pct"] = crop_vals
flooded_pixels["rangeland_pct"] = range_vals

print(f"Average cropland cover in flooded pixels: {flooded_pixels['crop_pct'].mean():.1f}%")
print(f"Average rangeland cover in flooded pixels: {flooded_pixels['rangeland_pct'].mean():.1f}%")
print("\nSample rows:")
print(flooded_pixels.head(10))

# ---------------------------------------------------------------
# PART B: Spatial pattern — recurring vs unusual flood pixels
# ---------------------------------------------------------------
print("\n### PART B: Recurring vs unusual flood locations, 2023-2024 ###\n")

all_flood = load_flood_masks(np.array([2023, 2024]), bbox=bbox)

pixel_type_counts = (
    all_flood.groupby(["lat", "lon", "flood_type"]).size().reset_index(name="days")
)
pixel_dominant = (
    pixel_type_counts.sort_values("days", ascending=False)
    .drop_duplicates(subset=["lat", "lon"], keep="first")
    .reset_index(drop=True)
)

print(f"Unique flooded pixels overall (2023-2024): {len(pixel_dominant)}")
print(pixel_dominant["flood_type"].value_counts().rename({0: "recurring", 1: "unusual"}))

coords_all = list(zip(pixel_dominant["lon"], pixel_dominant["lat"]))
_, pop_per_point_all = sample_raster_deduped(pop_2024_path, coords_all)
pixel_dominant["population"] = pop_per_point_all

pixel_dominant["crop_pct"] = sample_dataarray_nearest(
    crop_da, lat_dim, lon_dim, pixel_dominant["lat"].values, pixel_dominant["lon"].values
)

summary = pixel_dominant.groupby("flood_type").agg(
    n_pixels=("lat", "size"),
    avg_population=("population", "mean"),
    avg_crop_pct=("crop_pct", "mean"),
)
summary.index = summary.index.map({0: "recurring", 1: "unusual"})
print("\nComparison: recurring vs unusual flood pixels")
print(summary)

# ---------------------------------------------------------------
# Plot: spatial map of recurring vs unusual pixels
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))
colors = pixel_dominant["flood_type"].map({0: "steelblue", 1: "crimson"})
ax.scatter(pixel_dominant["lon"], pixel_dominant["lat"], c=colors, s=4, alpha=0.6)
ax.set_title(f"Flood pixel locations by type — {pilot_name}\n(blue=recurring, red=unusual)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.tight_layout()
plt.savefig("aweil_east_flood_type_map.png", dpi=150)
print("\nSaved map to aweil_east_flood_type_map.png")
plt.show()