import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box
import time

from processing_data.loading_impact_data import load_admin_boundaries
from processing_data.loading import load_flood_masks

# ---------------------------------------------------------------
# 1. National boundary (South Sudan) — used to bound the road download
# ---------------------------------------------------------------
admin1, admin2 = load_admin_boundaries()
national_bounds = admin1.total_bounds  # [lon_min, lat_min, lon_max, lat_max]
print(f"South Sudan bounding box: {national_bounds}")

# ---------------------------------------------------------------
# 2. Download the national road network
#    Using network_type="drive" (major roads/tracks) rather than "all"
#    keeps this from being enormous — OSM coverage in South Sudan is
#    sparse anyway, so "all" vs "drive" likely won't differ hugely here,
#    but "drive" is meaningfully faster to download and process.
#    Using graph_from_place (country polygon) rather than a bbox avoids
#    pulling in roads from neighboring countries that fall inside the
#    bounding rectangle but outside South Sudan itself.
# ---------------------------------------------------------------
print("\nDownloading South Sudan's road network (needs internet, may take a few minutes)...")
start = time.time()
try:
    G = ox.graph_from_place("South Sudan", network_type="drive")
except Exception as e:
    print(f"graph_from_place failed ({e}), falling back to bbox-based download...")
    G = ox.graph_from_bbox(
        bbox=(national_bounds[0], national_bounds[1], national_bounds[2], national_bounds[3]),
        network_type="drive",
    )

edges = ox.graph_to_gdfs(G, nodes=False)
print(f"Loaded {len(edges)} road segments in {time.time() - start:.1f}s")
print("NOTE: OpenStreetMap coverage in rural South Sudan can be sparse — "
      "a low road count reflects mapping gaps, not necessarily an actual "
      "absence of roads/tracks on the ground.")

# ---------------------------------------------------------------
# 3. Load Nov 2024 flood pixels, nationally
# ---------------------------------------------------------------
print("\nLoading national flood mask data for 2024 (this can take a while)...")
start = time.time()
flood_events = load_flood_masks(np.array([2024]), bbox=None)
flood_events["date"] = pd.to_datetime(flood_events["date"])
print(f"Loaded in {time.time() - start:.1f}s")

nov_2024 = flood_events[
    (flood_events["date"] >= "2024-11-01") & (flood_events["date"] < "2024-12-01")
]
flooded_pixels = nov_2024[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
print(f"Unique flooded pixel locations nationally, Nov 2024: {len(flooded_pixels):,}")

# ---------------------------------------------------------------
# 4. Build flood-extent squares (one per unique flooded pixel) and
#    use a spatial join (not buffer+union) to find intersecting roads —
#    far more scalable at national scale with potentially millions of pixels
# ---------------------------------------------------------------
pixel_size_deg = 0.00225  # ~250m at this latitude, matching MODIS/VIIRS grid
half = pixel_size_deg / 2

print("\nBuilding flood-pixel squares...")
flood_squares = gpd.GeoDataFrame(
    geometry=[
        box(lon - half, lat - half, lon + half, lat + half)
        for lat, lon in zip(flooded_pixels["lat"], flooded_pixels["lon"])
    ],
    crs="EPSG:4326",
)

print("Running spatial join between roads and flood squares "
      "(this is the main cost — can take a few minutes at national scale)...")
start = time.time()
joined = gpd.sjoin(edges, flood_squares, how="inner", predicate="intersects")
flooded_edge_idx = joined.index.unique()
print(f"Spatial join finished in {time.time() - start:.1f}s")

edges["flooded"] = edges.index.isin(flooded_edge_idx)
n_flooded_segments = edges["flooded"].sum()
flooded_length_km = edges.loc[edges["flooded"], "length"].sum() / 1000
total_length_km = edges["length"].sum() / 1000

print(f"\nRoad segments intersecting Nov 2024 flood extent: {n_flooded_segments} / {len(edges)}")
if total_length_km > 0:
    print(
        f"Flooded road length: {flooded_length_km:.1f} km out of "
        f"{total_length_km:.1f} km total ({flooded_length_km / total_length_km * 100:.1f}%)"
    )

# ---------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

# Plot flood pixels as small, semi-transparent points so density is visible
ax.scatter(
    flooded_pixels["lon"], flooded_pixels["lat"],
    s=0.5, color="steelblue", alpha=0.15, label="Flooded pixels (Nov 2024)",
)

if len(edges) > 0:
    edges[~edges["flooded"]].plot(ax=ax, color="gray", linewidth=0.4, label="Roads (not flooded)")
    if n_flooded_segments > 0:
        edges[edges["flooded"]].plot(ax=ax, color="red", linewidth=1.0, label="Roads crossing flood extent")

# Overlay admin1 boundaries for geographic context
admin1.boundary.plot(ax=ax, color="black", linewidth=0.5, alpha=0.5)

ax.set_title("South Sudan: national road network vs. flood extent, Nov 2024")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(markerscale=10)
plt.tight_layout()
plt.savefig("south_sudan_road_flooding.png", dpi=150)
print("\nSaved plot to south_sudan_road_flooding.png")
plt.show()