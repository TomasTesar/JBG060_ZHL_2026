"""
Full data-quality / exploration audit across every dataset in the project,
plus (at the end) the full hard-coded demo pipelines from loading.py and
loading_impact_data.py.

WARNING on the demo pipelines: loading.py processes ~9,500 daily
evapotranspiration files into annual CSVs across 2000-2025, and
loading_impact_data.py downloads a live OpenStreetMap road network and
opens several interactive plot windows you'll need to close manually
for the script to continue. This can take a long time and needs internet.
"""

import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

from processing_data.loading_impact_data import (
    load_admin_boundaries,
    load_worldpop_area,
    load_ipc_data,
    load_GDP,
    load_health_facilities,
    load_farmland_mask,
)
from processing_data.loading import (
    load_flood_masks,
    load_dartmouth_data,
    load_lake_stations,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------
# 1. Administrative boundaries
# ---------------------------------------------------------------
def check_admin_boundaries():
    section("1. Administrative boundaries")
    admin1, admin2 = load_admin_boundaries()

    print(f"Admin1 (states): {len(admin1)} rows")
    print(f"Admin2 (counties): {len(admin2)} rows  (expected 79 incl. Abyei)")

    for name, gdf in [("admin1", admin1), ("admin2", admin2)]:
        invalid = (~gdf.geometry.is_valid).sum()
        empty = gdf.geometry.is_empty.sum()
        print(f"{name}: {invalid} invalid geometries, {empty} empty geometries")
        print(f"{name}: CRS = {gdf.crs}")
        dupes = gdf.geometry.duplicated().sum()
        print(f"{name}: {dupes} duplicate geometries")

    return admin1, admin2


# ---------------------------------------------------------------
# 2. Flood masks — auto-detects every year actually present on disk
# ---------------------------------------------------------------
import re


def detect_available_flood_years():
    """Scan raw_data for flood_events_{tile}_{year}.parquet files and return
    the actual set of years present, instead of assuming a fixed range."""
    files = list(Path("raw_data").rglob("flood_events_*.parquet"))
    years = set()
    for f in files:
        m = re.search(r"_(\d{4})\.parquet$", f.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def check_flood_masks(years=None):
    if years is None:
        years = detect_available_flood_years()
        if not years:
            print("Could not auto-detect flood mask years from raw_data/ — "
                  "check that the parquet files are present and named as expected.")
            return None
        print(f"Auto-detected flood mask years available on disk: {years[0]}-{years[-1]} "
              f"({len(years)} years total)")

    section(f"2. Flood masks, national scale, years {min(years)}-{max(years)}")
    import time
    start = time.time()

    years_arr = np.array(list(years))
    print(f"Loading {len(years_arr)} years of national flood mask data — this may take a while...")
    flood = load_flood_masks(years_arr, bbox=None)

    print(f"Loaded in {time.time() - start:.1f} seconds")
    print(f"Total rows: {len(flood):,}")
    print(f"Columns: {flood.columns.tolist()}")
    print(f"Date range: {flood['date'].min()} to {flood['date'].max()}")
    print(f"Missing values per column:\n{flood.isna().sum()}")

    print(f"\nFlood type breakdown:\n{flood['flood_type'].value_counts()}")
    print(f"\nTiles present: {flood['tile'].unique()}")

    dupes = flood.duplicated(subset=["date", "lat", "lon"]).sum()
    print(f"\nDuplicate (date, lat, lon) rows: {dupes}")

    print(f"\nLat range: {flood['lat'].min():.3f} to {flood['lat'].max():.3f}  (South Sudan ~ 3.5 to 12.5)")
    print(f"Lon range: {flood['lon'].min():.3f} to {flood['lon'].max():.3f}  (South Sudan ~ 24 to 36)")

    flood["year"] = pd.to_datetime(flood["date"]).dt.year
    print(f"\nRows per year:\n{flood.groupby('year').size()}")

    return flood


# ---------------------------------------------------------------
# 3. Population rasters
# ---------------------------------------------------------------
def check_population():
    section("3. Population rasters")

    pop_dir = Path("raw_data/worldpop") if Path("raw_data/worldpop").exists() else Path("raw_data")
    pop_files = sorted(pop_dir.rglob("ssd_pop_*_CN_100m_R2025A_v1.tif"))
    print(f"Found {len(pop_files)} population raster files")

    for f in pop_files:
        with rasterio.open(f) as src:
            arr = src.read(1)
            nodata = src.nodata
            valid = arr[arr != nodata] if nodata is not None else arr
            total = valid[valid > 0].sum()
            n_nodata = (arr == nodata).sum() if nodata is not None else 0
            print(
                f"  {f.name}: shape={arr.shape}, nodata_value={nodata}, "
                f"n_nodata_pixels={n_nodata:,}, total_population≈{total:,.0f}"
            )
    print("\n(Sanity check: South Sudan's total population is commonly cited around 11-12 million.")
    print(" If totals above look off by an order of magnitude, double check the nodata handling.)")


# ---------------------------------------------------------------
# 4. Cattle raster
# ---------------------------------------------------------------
def check_cattle():
    section("4. Cattle raster (Greater Horn of Africa — South Sudan is a subset)")
    cattle_path = next(Path("raw_data").rglob("geonode__cattle_gha.tif"), None)
    if cattle_path is None:
        print("Cattle raster not found under raw_data/ — check the path.")
        return

    with rasterio.open(cattle_path) as src:
        arr = src.read(1)
        nodata = src.nodata
        print(f"Shape: {arr.shape}, nodata value: {nodata}")
        valid = arr[arr != nodata] if nodata is not None else arr
        print(f"Min: {valid.min():,.1f}, Max: {valid.max():,.1f}, Mean: {valid.mean():,.2f}")
        print(f"Negative values (should be none after masking nodata): {(valid < 0).sum()}")
        print("NOTE: this raster covers the whole Greater Horn of Africa, not just South Sudan —")
        print("      always clip to your region of interest before summing totals.")


# ---------------------------------------------------------------
# 5. Cropland / rangeland masks
# ---------------------------------------------------------------
def check_farmland(national_bbox):
    section("5. Cropland / rangeland masks")
    for mask_type in ["crop", "rangeland"]:
        da = load_farmland_mask(national_bbox, mask_type=mask_type)
        arr = da.values
        print(f"\n{mask_type.upper()} mask:")
        print(f"  Shape: {arr.shape}, dims: {da.dims}")
        print(f"  Min: {np.nanmin(arr):.1f}, Max: {np.nanmax(arr):.1f}  (expected 0-100%)")
        print(f"  NaN count: {np.isnan(arr).sum():,} / {arr.size:,}")
        out_of_range = ((arr < 0) | (arr > 100)) & ~np.isnan(arr)
        print(f"  Values outside [0, 100]: {out_of_range.sum()}")


# ---------------------------------------------------------------
# 6. IPC food insecurity data
# ---------------------------------------------------------------
def check_ipc():
    section("6. IPC food insecurity data")
    ipc = load_ipc_data()
    print(f"Shape: {ipc.shape}")
    print(f"Reporting periods: {len(ipc)}")
    print(ipc[["Start Date", "End Date"]])

    county_cols = [c for c in ipc.columns if c not in ("Start Date", "End Date")]
    print(f"\nNumber of county columns: {len(county_cols)}  (expected ~79-83)")

    missing_per_period = ipc[county_cols].isna().sum(axis=1)
    print(f"\nMissing county values per reporting period:\n{missing_per_period}")

    print("\nNOTE: this table gives Phase 3+ population only, per the loader's docstring —")
    print("      it is NOT the full breakdown by phase (1 through 5). If you need the full")
    print("      phase breakdown, you'll need to parse the raw IPC Excel files directly.")


# ---------------------------------------------------------------
# 7. GDP
# ---------------------------------------------------------------
def check_gdp():
    section("7. GDP (World Bank)")
    gdp = load_GDP()
    print(gdp)
    print(
        "\nNOTE: the raw World Bank CSV covers 1960-2025 (per Data_overview.xlsx), "
        "but the loader function's docstring says it returns 2008-2015 only — "
        "double check load_GDP()'s implementation if you need more recent years, "
        "since South Sudan only became independent in 2011 and WB data before "
        "that may not exist or may refer to a different entity."
    )


# ---------------------------------------------------------------
# 8. Discharge (Dartmouth Flood Observatory)
# ---------------------------------------------------------------
def check_discharge():
    section("8. Discharge stations (Dartmouth Flood Observatory)")
    discharge = load_dartmouth_data()
    print(f"Number of stations: {len(discharge)}")

    for area_id, df in discharge.items():
        n_missing = df["Discharge (m3/s)"].isna().sum()
        n_negative = (df["Discharge (m3/s)"] < 0).sum()
        date_range = f"{df.index.min()} to {df.index.max()}"
        expected_days = (df.index.max() - df.index.min()).days + 1
        print(
            f"  Station {area_id}: {len(df)} rows, dates {date_range}, "
            f"{n_missing} missing, {n_negative} negative values, "
            f"expected ~{expected_days} days if daily -> "
            f"{'looks daily' if len(df) > expected_days * 0.8 else 'looks sparser than daily (monthly?)'}"
        )


# ---------------------------------------------------------------
# 9. Lake levels
# ---------------------------------------------------------------
def check_lake_levels():
    section("9. Lake level stations")
    lakes = load_lake_stations()
    for name, df in lakes.items():
        print(f"\n{name}: {len(df)} rows")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {df.columns.tolist()}")
        n_dupes = df.index.duplicated().sum()
        print(f"  Duplicate timestamps: {n_dupes}")
        if "error" in df.columns:
            print(f"  Error stats: min={df['error'].min()}, max={df['error'].max()}, mean={df['error'].mean():.4f}")
        if "ice_flag" in df.columns:
            print(f"  Rows flagged as frozen surface: {(df['ice_flag'] != 0).sum()}")


# ---------------------------------------------------------------
# 10. Health facilities
# ---------------------------------------------------------------
def check_health_facilities():
    section("10. Health facilities")
    facilities = load_health_facilities()
    print(f"Total South Sudan facilities: {len(facilities)}  (expected 1,747 per Data_overview.xlsx)")
    print(f"\nFacility types:\n{facilities['Facility_t'].value_counts()}")
    print(f"\nOwnership breakdown:\n{facilities['Ownership'].value_counts()}")

    missing_coords = facilities[["Lat", "Long"]].isna().any(axis=1).sum()
    print(f"\nRows with missing coordinates: {missing_coords}")

    out_of_range = (
        (facilities["Lat"] < 3.0) | (facilities["Lat"] > 13.0) |
        (facilities["Long"] < 23.0) | (facilities["Long"] > 37.0)
    )
    print(f"Rows with coordinates outside plausible South Sudan range: {out_of_range.sum()}")


# ---------------------------------------------------------------
# Run everything
# ---------------------------------------------------------------
if __name__ == "__main__":
    admin1, admin2 = check_admin_boundaries()

    national_bbox = {
        "lon_min": admin1.total_bounds[0],
        "lat_min": admin1.total_bounds[1],
        "lon_max": admin1.total_bounds[2],
        "lat_max": admin1.total_bounds[3],
    }

    check_flood_masks()
    check_population()
    check_cattle()
    check_farmland(national_bbox)
    check_ipc()
    check_gdp()
    check_discharge()
    check_lake_levels()
    check_health_facilities()

    print("\n" + "=" * 70)
    print("Audit complete. Review flagged issues above before choosing your")
    print("final research direction / pilot region.")
    print("=" * 70)

    # -----------------------------------------------------------
    # Full demo pipelines (loading.py / loading_impact_data.py)
    # -----------------------------------------------------------
    section("11. FULL DEMO PIPELINES (loading.py + loading_impact_data.py)")
    print("Running the full hard-coded demonstrations from both modules.")
    print("This will be slow and will open interactive plot windows —")
    print("close each plot window to let the script continue.\n")

    import subprocess
    import sys

    print(">>> Running processing_data/loading.py ...")
    subprocess.run([sys.executable, "processing_data/loading.py"], check=False)

    print("\n>>> Running processing_data/loading_impact_data.py ...")
    subprocess.run([sys.executable, "processing_data/loading_impact_data.py"], check=False)

    print("\nFull demo pipelines finished.")