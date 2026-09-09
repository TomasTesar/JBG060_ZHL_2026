import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
from shapely.geometry import box
from scipy.spatial import cKDTree

from processing_data.loading_impact_data import (
    load_admin_boundaries,
    load_health_facilities,
)
from processing_data.loading import load_flood_masks

# ---------------------------------------------------------------
# 0. Pick your pilot county here
# ---------------------------------------------------------------
pilot_name = "Aweil East"

# ---------------------------------------------------------------
# 1. Pilot region bbox
# ---------------------------------------------------------------
admin1, admin2 = load_admin_boundaries()
pilot_county = admin2[admin2.adm2_name == pilot_name]
if len(pilot_county) == 0:
    raise ValueError(
        f"'{pilot_name}' not found in admin2 — check exact spelling against "
        f"admin2['adm2_name'].unique(). Names are case-sensitive."
    )

bounds = pilot_county.geometry.total_bounds  # [lon_min, lat_min, lon_max, lat_max]
bbox = {
    "lon_min": bounds[0], "lat_min": bounds[1],
    "lon_max": bounds[2], "lat_max": bounds[3],
}
print(f"{pilot_name} bbox: {bbox}")

# ---------------------------------------------------------------
# 2. Flood pixels, Nov 2024, for this bbox
# ---------------------------------------------------------------
print("Loading flood mask data for Nov 2024...")
flood_events = load_flood_masks(np.array([2024]), bbox=bbox)
flood_events["date"] = pd.to_datetime(flood_events["date"])
nov_2024 = flood_events[
    (flood_events["date"] >= "2024-11-01") & (flood_events["date"] < "2024-12-01")
]
flooded_pixels = nov_2024[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
print(f"Unique flooded pixel locations in {pilot_name}, Nov 2024: {len(flooded_pixels):,}")

# ---------------------------------------------------------------
# 3. Roads for this bbox
# ---------------------------------------------------------------
print("\nDownloading road network for this county (needs internet)...")
try:
    G = ox.graph_from_bbox(
        bbox=(bbox["lon_min"], bbox["lat_min"], bbox["lon_max"], bbox["lat_max"]),
        network_type="all",
    )
except TypeError:
    G = ox.graph_from_bbox(
        bbox["lat_max"], bbox["lat_min"], bbox["lon_max"], bbox["lon_min"],
        network_type="all",
    )

edges = ox.graph_to_gdfs(G, nodes=False)
print(f"Loaded {len(edges)} road segments")
print("NOTE: OSM coverage in rural South Sudan is often sparse — a low count "
      "reflects mapping gaps, not necessarily an absence of real roads/tracks.")

if len(edges) > 0 and len(flooded_pixels) > 0:
    pixel_size_deg = 0.00225  # ~250m, matching MODIS/VIIRS grid
    half = pixel_size_deg / 2
    flood_squares = gpd.GeoDataFrame(
        geometry=[
            box(lon - half, lat - half, lon + half, lat + half)
            for lat, lon in zip(flooded_pixels["lat"], flooded_pixels["lon"])
        ],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(edges, flood_squares, how="inner", predicate="intersects")
    flooded_edge_idx = joined.index.unique()
    edges["flooded"] = edges.index.isin(flooded_edge_idx)
else:
    edges["flooded"] = False

n_flooded_segments = edges["flooded"].sum() if len(edges) > 0 else 0
if len(edges) > 0:
    flooded_km = edges.loc[edges["flooded"], "length"].sum() / 1000
    total_km = edges["length"].sum() / 1000
    print(f"Road segments crossing flood extent: {n_flooded_segments} / {len(edges)}")
    if total_km > 0:
        print(f"Flooded road length: {flooded_km:.1f} km / {total_km:.1f} km total "
              f"({flooded_km / total_km * 100:.1f}%)")

# ---------------------------------------------------------------
# 4. Health facilities for this bbox
# ---------------------------------------------------------------
print("\nLoading health facilities...")
facilities = load_health_facilities()
facilities = facilities.dropna(subset=["Lat", "Long"])
facilities = facilities[
    (facilities["Lat"] >= bbox["lat_min"]) & (facilities["Lat"] <= bbox["lat_max"]) &
    (facilities["Long"] >= bbox["lon_min"]) & (facilities["Long"] <= bbox["lon_max"])
].reset_index(drop=True)
print(f"Health facilities within {pilot_name}: {len(facilities)}")

DEG_PER_KM = 1 / 111.0
DIRECT_THRESHOLD_KM = 0.15
NEARBY_THRESHOLD_KM = 5.0

facilities["exposure"] = "unaffected"
if len(facilities) > 0 and len(flooded_pixels) > 0:
    tree = cKDTree(flooded_pixels[["lat", "lon"]].values)
    distances_deg, _ = tree.query(facilities[["Lat", "Long"]].values, k=1)
    facilities["dist_to_flood_km"] = distances_deg / DEG_PER_KM
    facilities.loc[facilities["dist_to_flood_km"] <= NEARBY_THRESHOLD_KM, "exposure"] = "nearby (access risk)"
    facilities.loc[facilities["dist_to_flood_km"] <= DIRECT_THRESHOLD_KM, "exposure"] = "directly flooded"
    print(f"\nFacility exposure summary:\n{facilities['exposure'].value_counts()}")
elif len(facilities) == 0:
    print(f"No health facilities found within {pilot_name} — that panel will be empty, "
          f"which can genuinely happen for smaller/rural counties.")

# ---------------------------------------------------------------
# DIAGNOSTIC: confirm all layers share the same coordinate extent
# ---------------------------------------------------------------
print("\n=== DIAGNOSTIC: bounding box of each layer (should all roughly match) ===")
print(f"County bbox (from admin2):        lon [{bbox['lon_min']:.4f}, {bbox['lon_max']:.4f}], "
      f"lat [{bbox['lat_min']:.4f}, {bbox['lat_max']:.4f}]")

if len(flooded_pixels) > 0:
    print(f"Flood pixels actual extent:       lon [{flooded_pixels['lon'].min():.4f}, {flooded_pixels['lon'].max():.4f}], "
          f"lat [{flooded_pixels['lat'].min():.4f}, {flooded_pixels['lat'].max():.4f}]")
else:
    print("Flood pixels: NONE FOUND in this bbox")

if len(edges) > 0:
    road_bounds = edges.total_bounds  # [minx, miny, maxx, maxy] = [lon_min, lat_min, lon_max, lat_max]
    print(f"Road network actual extent:       lon [{road_bounds[0]:.4f}, {road_bounds[2]:.4f}], "
          f"lat [{road_bounds[1]:.4f}, {road_bounds[3]:.4f}]")
    print(f"Road network CRS: {edges.crs}")
else:
    print("Roads: NONE FOUND in this bbox")

if len(facilities) > 0:
    print(f"Facilities actual extent:         lon [{facilities['Long'].min():.4f}, {facilities['Long'].max():.4f}], "
          f"lat [{facilities['Lat'].min():.4f}, {facilities['Lat'].max():.4f}]")
else:
    print("Facilities: NONE FOUND in this bbox")

print("\nIf any of the above extents are wildly different from the others "
      "(e.g. off by a factor of 10, or lat/lon appear swapped), that's the "
      "misalignment. A common cause: facilities' Lat/Long columns swapped, "
      "or the road graph's CRS not actually being EPSG:4326 degrees.")

# ---------------------------------------------------------------
# 5. Plot: ONE combined map — roads, flood, and healthcare together
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 12))

# Force all three layers onto the SAME axis limits (the county bbox),
# rather than letting each layer's .plot() call auto-scale independently
ax.set_xlim(bbox["lon_min"], bbox["lon_max"])
ax.set_ylim(bbox["lat_min"], bbox["lat_max"])
ax.set_aspect("equal")

# --- Flood extent (background layer) ---
ax.scatter(
    flooded_pixels["lon"], flooded_pixels["lat"],
    s=4, color="steelblue", alpha=0.25, label=f"Flooded pixels (n={len(flooded_pixels):,})",
)

# --- Roads (flooded vs not) ---
if len(edges) > 0:
    edges[~edges["flooded"]].plot(ax=ax, color="gray", linewidth=0.8, label="Roads (not flooded)")
    if n_flooded_segments > 0:
        edges[edges["flooded"]].plot(ax=ax, color="darkred", linewidth=2.5,
                                      label=f"Roads crossing flood (n={n_flooded_segments})")

# --- Health facilities (exposure-coded) ---
colors = {"unaffected": "gray", "nearby (access risk)": "orange", "directly flooded": "red"}
sizes = {"unaffected": 40, "nearby (access risk)": 90, "directly flooded": 140}
markers = {"unaffected": "o", "nearby (access risk)": "^", "directly flooded": "X"}
for status, color in colors.items():
    subset = facilities[facilities["exposure"] == status]
    if len(subset) == 0:
        continue
    ax.scatter(
        subset["Long"], subset["Lat"],
        s=sizes[status], color=color, marker=markers[status],
        label=f"Facility: {status} (n={len(subset)})",
        edgecolors="black", linewidths=0.8, zorder=5,
    )

ax.set_title(f"{pilot_name}: roads, flood extent & health facility exposure — Nov 2024")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

plt.tight_layout()
outname = f"{pilot_name.replace(' ', '_').lower()}_combined_impact_map.png"
plt.savefig(outname, dpi=150)
print(f"\nSaved plot to {outname}")
plt.show()