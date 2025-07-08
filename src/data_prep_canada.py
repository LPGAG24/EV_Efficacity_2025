#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified data-prep script for:
  • CKAN (open.canada.ca) via REST
  • StatCan Web Data Service via stats-can

Usage (examples in __main__):
  python data_prep.py ckan        # runs the CKAN download example
  python data_prep.py statcan     # runs the StatCan fleet pivot example
"""

from __future__ import annotations
import json, time, re, sys
from pathlib import Path
from typing import Literal
import pandas as pd
import requests

# ─── Paths for raw/processed storage ────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[0]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

CKAN_ELE_ID = "026e45b4-eb63-451f-b34f-d9308ea3a3d9"
CKAN_HYBR_ID = "8812228b-a6aa-4303-b3d0-66489225120d"

# ─── CKAN settings ─────────────────────────────────────────────────────────
CKAN_BASE = "https://open.canada.ca/data/en/api/3/action"

def km_to_kwh(distance_km: float, efficiency_kwh_per_100km: float = 18.0) -> float:
    """Convert travelled kilometres to required kWh.

    Parameters
    ----------
    distance_km : float
        Distance in kilometres.
    efficiency_kwh_per_100km : float, default 18.0
        Energy consumed per 100 km.

    Returns
    -------
    float
        Energy in kilowatt‑hours required to travel ``distance_km``.
    """
    return distance_km * efficiency_kwh_per_100km / 100.0

def _slugify(d: dict[str, str] | None) -> str:
    """Turn {'Make':'Tesla'} → '__Make-Tesla' (safe for filenames)."""
    if not d:
        return ""
    items = sorted(d.items())
    parts = [f"{k}-{v}" for k, v in items]
    raw   = "__".join(parts)
    safe  = re.sub(r"[^A-Za-z0-9_\-\.]", "_", raw)
    return f"__{safe}"

def _ckan_get(endpoint: str, **params) -> dict:
    r = requests.get(f"{CKAN_BASE}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    j = r.json()
    if not j.get("success"):
        raise RuntimeError(j)
    return j["result"]

def fetch_all_records(
    resource_id: str = CKAN_ELE_ID,
    batch: int = 1000,
    filters: dict[str, str] | None = None,
    sleep: float = 0.2,
) -> pd.DataFrame:
    """Paginate through a CKAN datastore_search, return full DataFrame."""
    offset = 0
    frames = []
    filt_param = json.dumps(filters) if filters else None

    while True:
        res = _ckan_get(
            "datastore_search",
            resource_id=resource_id,
            limit=batch,
            offset=offset,
            filters=filt_param,
        )
        recs = res["records"]
        if not recs:
            break
        frames.append(pd.DataFrame(recs))
        offset += batch
        time.sleep(sleep)

    return pd.concat(frames, ignore_index=True)

def download_ckan_resource(
    resource_id: str,
    filename: str | None = None,
    filters: dict[str, str] | None = None,
    usecols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    Fetch a full CKAN resource, optionally filter columns, 
    save the raw JSON in data/raw/, and return DataFrame.
    """
    df = fetch_all_records(resource_id, filters=filters)
    if usecols is not None:
        df = df[[c for c in usecols if c in df.columns]]

    slug = _slugify(filters)
    tag  = filename or f"{resource_id}{slug}.json"
    (RAW / tag).write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")
    print(f"[CKAN] saved raw to {RAW/tag!s}")
    return df

# ─── StatCan settings ───────────────────────────────────────────────────────
from stats_can import StatsCan

def fetch_statcan_fleet(table_id: str = "23-10-0308-01") -> pd.DataFrame:
    """
    Use stats-can to pull the full vehicle-registration table,
    then pivot to one row per province with total stock for latest year.
    """
    sc = StatsCan()
    tbl = sc.table_to_df(table_id)

    latest_year = tbl["REF_DATE"].max()
    fleet = tbl[
        (tbl["REF_DATE"] == latest_year) 
    ][
        ["GEO", "VALUE", "Vehicle Type", "Fuel Type"]
    ].rename(columns={"GEO": "Province",
                      "VALUE": "Vehicles nb",
                      "Vehicle Type": "Vehicle Type",
                      "Fuel Type": "Fuel Type"})

    prov_fleet = (fleet.set_index("Province")
                       .sort_index())
    print(f"[StatCan] {table_id} | REF_DATE={latest_year}")
    return fleet

def fetch_statcan(link:str)->pd.DataFrame:
    """
    Fetch a StatCan table by link, return DataFrame.
    """
    sc = StatsCan()
    tbl = sc.table_to_df(link)
    if tbl.empty:
        raise ValueError(f"No data found for {link}")
    return tbl



# ─── Main entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    for mode in ["ckan", "statcan"]:
        if mode == "ckan":
            # Example: download all Tesla and Nissan entries, keep key cols
            RID = "026e45b4-eb63-451f-b34f-d9308ea3a3d9"
            df = download_ckan_resource(
                resource_id=RID,
                usecols=("Make", "Model", "Vehicle class", "Combined (kWh/100 km)")
            )
            print(df.info())
            print(df.head())

        elif mode == "statcan":
            # Example: pivot active light-duty fleet by province for latest year
            pivot = fetch_statcan_fleet("23-10-0308-01")
            print(pivot)
            for prov in pivot["Province"].unique():
                prov_df = pivot[pivot["Province"] == prov]
                #addition each vehicle type from each provinces
                print(f"{prov}: {prov_df['Vehicles nb'].sum()} vehicles")
            
            
        
