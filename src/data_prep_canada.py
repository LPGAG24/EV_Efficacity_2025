# ──────────────────────────────────────────────────────────
# src/data_prep.py
# Téléchargement & préparation des datasets CKAN (open.canada.ca)
# ──────────────────────────────────────────────────────────
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal
import re

import pandas as pd
import requests

# Racines des dossiers
ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

CKAN_BASE = "https://open.canada.ca/data/en/api/3/action"


# ----------------------------------------------------------------------
def _ckan_get(endpoint: str, **params) -> dict:
    """Appelle l'API CKAN, gère les erreurs et retourne le JSON décodé."""
    r = requests.get(f"{CKAN_BASE}/{endpoint}", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError(data)
    return data["result"]


# ----------------------------------------------------------------------
def fetch_all_records(resource_id: str, batch: int = 1000, filters: dict[str, str] | None = None, sleep: float = 0.2) -> pd.DataFrame:
    """Récupère *tous* les enregistrements d'une ressource CKAN (pagination)."""
    offset, frames = 0, []
    filt_param = json.dumps(filters) if filters else None
    while True:
        res = _ckan_get("datastore_search", resource_id=resource_id, limit=batch, filters=filt_param, offset=offset)
        records = res["records"]
        if not records:
            break
        frames.append(pd.DataFrame(records))
        offset += batch
        time.sleep(sleep)          # politesse / anti‑ratelimit
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------
# 
# Téléchargement de la ressource CKAN, sauvegarde en JSON dans data/raw/,
#
def download_ckan_resource(
    resource_id: str,
    filename: str | None = None,
    filters_params: dict[str, str] | None = None,
    usecols: tuple[str, ...] | None = None, 
) -> pd.DataFrame:
    """
    Télécharge la ressource CKAN, la sauvegarde en JSON dans data/raw/,
    renvoie un DataFrame. `mode="sample"` → limit & q sont appliqués.
    """
    df = fetch_all_records(resource_id, filters=filters_params)
    tag = "full"

    if usecols is not None:
        df = df[[c for c in usecols if c in df.columns]]

    # écriture brute pour traçabilité
    if filters_params:
        fname = (filename or f"{resource_id}_{_slugify(filters_params)}.json")
    elif filters_params is None:
        fname = (filename or f"{resource_id}_{tag}.json")
    (RAW / fname).write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")

    return df


def _slugify(d: dict[str, str] | None) -> str:
    """
    Transforme {'Make':'Tesla','Model year':'2022'}
    -> '__Make-Tesla__Model_year-2022'
    """
    if not d:
        return ""
    # 1) ordonner pour une sortie stable
    items = sorted(d.items())
    # 2) concaténer et nettoyer les caractères non alphanum
    parts = [f"{k}-{v}" for k, v in items]
    raw   = "__".join(parts)
    safe  = re.sub(r"[^A-Za-z0-9_\-\.]", "_", raw)
    return f"__{safe}"

# ----------------------------------------------------------------------
if __name__ == "__main__":
    #Info electric vehicles (Le/Km)
    # list of possible filters
    """
    filters = {
        "_id":25471,"Model year":"2012","Make":"Mitsubishi","Model":"i-MiEV","Vehicle class":"Subcompact",
        "Motor (kW)":"49","Transmission":"A1","Fuel type":"B",
        "City (kWh/100 km)":"16.9","Highway (kWh/100 km)":"21.4","Combined (kWh/100 km)":"18.7",
        "City (Le/100 km)":"1.9","Highway (Le/100 km)":"2.4","Combined (Le/100 km)":"2.1",
        "Range (km)":"100","CO2 emissions (g/km)":"0","CO2 rating":"n/a","Smog rating":"n/a",
        "Recharge time (h)":"7"
    }
    """
    full = download_ckan_resource("026e45b4-eb63-451f-b34f-d9308ea3a3d9", usecols=("Make", "Model", "Combined (Le/100 km)"))

    print(full.info())
    print(full.head())