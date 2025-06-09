# src/utils/normalizer.py
from __future__ import annotations
from collections.abc import Mapping

import re
import pandas as pd
from typing import Any

# --- dictionnaires maîtres --------------------------------------------------
PROVINCES: dict[str, list[str]] = {
    "Alberta":          ["ab", "alberta", "albert"],
    "British Columbia": ["bc", "b.-c.", "bct" ,"b c", "british columbia"],
    "Manitoba":         ["mb", "manitoba"],
    "New Brunswick":    ["nb", "new brunswick"],
    "Newfoundland and Labrador": ["nl", "n.-l.", "newfoundland", "labrador"],
    "Nova Scotia":      ["ns", "nova scotia"],
    "Ontario":          ["on", "ontario"],
    "Prince Edward Island": ["pe", "pei", "prince edward", "p.e.i."],
    "Quebec":           ["qc", "québec", "queb", "pq"],
    "Saskatchewan":     ["sk", "saskatchewan"],
    "Yukon":            ["yt", "yukon"],
    "Northwest Territories": ["nt", "nwt", "northwest territories"],
    "Nunavut":          ["nu", "nunavut"],
    "Canada":          ["canada", "ca", "can"],
    "Unknown":         ["unknown", "inconnu", "non spécifié", "non spécifié(e)", "non spécifié(e)s"],
}

VEHICLE_CLASSES: dict[str, list[str]] = {
    "Subcompact": ["subcompact", "mini car"],
    "Compact SUV": ["suv compact", "suv-c", "compact suv"],
    "SUV": ["suv", "sport utility vehicle", "vus"],
    "Pickup": ["pickup", "truck", "pickup truck"],
    "Minivan": ["minivan", "multi-purpose van"],
    "Sedan": ["sedan", "berline", "compact"],
}


COLUMN_NAMES: dict[str, list[str]] = {
    "Province":       ["prov", "province_name", "prv", "location"],
    "Vehicle class":  ["vehicle_class", "veh class", "class"],
    "Year":           ["yr", "année", "fiscal_year"],
    "Count":          ["n", "nb", "number", "total"],
    "Type":           ["type", "vehicle_type", "veh type", "class"],
    "Fuel type":      ["fuel_type", "fuel", "energy_type", "energy"],
}


def _simplify(text: str) -> str:
    """Retourne une version uniforme : minuscules + alphanum uniquement."""
    return re.sub(r"[^a-z0-9]", "", text.casefold())

def canonicalize(value: str, mapping: Mapping[str, list[str]]) -> str | None:
    """
    Transforme *value* en forme canonique selon *mapping*.
    - Retourne la clé canonique si trouvée
    - Sinon renvoie None
    """
    key = _simplify(value)
    for canonical, aliases in mapping.items():
        if key == _simplify(canonical):
            return canonical          # déjà canonique
        if key in {_simplify(a) for a in aliases}:
            return canonical
    return None


def canonical_province(name: str) -> str:
    result = canonicalize(name, PROVINCES)
    if result is None:
        raise ValueError(f"Province inconnue : {name!r}")
    return result

def canonical_vehicle_class(name: str) -> str:
    result = canonicalize(name, VEHICLE_CLASSES)
    if result is None:
        raise ValueError(f"Classe de véhicule inconnue : {name!r}")
    return result


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise nom de province et classe véhicule dans *df*."""
    df = df.copy()
    if "Province" in df.columns:
        df["Province"] = df["Province"].map(canonical_province)

    if "Vehicle class" in df.columns:
        df["Vehicle class"] = df["Vehicle class"].map(canonical_vehicle_class)

    return df

def normalize_columns(df: pd.DataFrame,
    mapping: Mapping[str, list[str]] = COLUMN_NAMES,
) -> pd.DataFrame:
    """
    Renomme les colonnes du DataFrame selon le dictionnaire *mapping*.
    - Conservation des colonnes déjà canoniques.
    - Les colonnes non reconnues restent inchangées.
    """
    rename_dict = {}
    for col in df.columns:
        key = _simplify(col)
        for canonical, aliases in mapping.items():
            if key == _simplify(canonical) or key in {_simplify(a) for a in aliases}:
                rename_dict[col] = canonical
                break
    return df.rename(columns=rename_dict)