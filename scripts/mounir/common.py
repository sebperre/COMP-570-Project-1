"""Shared constants and helpers for the 311 pipeline."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "mounir" / "raw"
OUTPUT_DIR = ROOT / "data" / "mounir" / "processed"
SUMMARY_DIR = OUTPUT_DIR / "summary_2014_present"
CENSUS_PATH = ROOT / "data" / "adil" / "processed" / "borough_population_2021.csv"

WINDOW_START = "2023-01-01"
WINDOW_END = "2025-12-31"
WINDOW_YEARS = [2023, 2024, 2025]

# Each creation year is read from exactly one extract. The archives overlap
# (2016 is in both 2014-2016 and 2016-2018; 2017-2018 is a strict subset of
# 2016-2018), and Information rows carry no ID_UNIQUE, so ID-based
# deduplication alone cannot remove their duplicates. The 2016-2018 archive is
# the most recently republished copy of 2016-2018 (portal, retrieved
# 2026-10-07), so it supersedes the older 2017-2018 file, which is not read.
SOURCE_YEARS = {
    "requetes311.csv": range(2022, 2100),
    "requetes311_2019-2021.csv": range(2019, 2022),
    "requetes311_2016-2018.csv": range(2016, 2019),
    "requetes311_2014-2016.csv": range(2014, 2016),
}
SUPERSEDED_FILES = ["requetes311_2017-2018.csv"]

CANONICAL_BOROUGHS = [
    "Ahuntsic-Cartierville",
    "Anjou",
    "Côte-des-Neiges-Notre-Dame-de-Grâce",
    "Lachine",
    "LaSalle",
    "Le Plateau-Mont-Royal",
    "Le Sud-Ouest",
    "L'Île-Bizard-Sainte-Geneviève",
    "Mercier-Hochelaga-Maisonneuve",
    "Montréal-Nord",
    "Outremont",
    "Pierrefonds-Roxboro",
    "Rivière-des-Prairies-Pointe-aux-Trembles",
    "Rosemont-La Petite-Patrie",
    "Saint-Laurent",
    "Saint-Léonard",
    "Verdun",
    "Ville-Marie",
    "Villeray-Saint-Michel-Parc-Extension",
]

# The 15 municipalities of the agglomeration that are not part of the City of
# Montreal. They appear in ARRONDISSEMENT_GEO but are outside the 19 boroughs.
LINKED_MUNICIPALITIES = [
    "Baie-d'Urfé", "Beaconsfield", "Côte-Saint-Luc", "Dollard-des-Ormeaux",
    "Dorval", "Hampstead", "Kirkland", "L'Île-Dorval", "Montréal-Est",
    "Montréal-Ouest", "Mont-Royal", "Pointe-Claire", "Sainte-Anne-de-Bellevue",
    "Senneville", "Westmount",
]


def normalize_label(value: object) -> str:
    """Lowercase, strip accents and punctuation: a stable lookup key."""
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", "", text.lower())


BOROUGH_LOOKUP = {normalize_label(name): name for name in CANONICAL_BOROUGHS}
LINKED_LOOKUP = {normalize_label(name) for name in LINKED_MUNICIPALITIES}


def reconcile_borough(geo: pd.Series, handling: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (borough, borough_rule) for each row.

    Rule: use the location field ARRONDISSEMENT_GEO when it names one of the 19
    boroughs; fall back to the handling unit ARRONDISSEMENT only when the
    location field is empty. A non-empty location outside the 19 boroughs is
    never overridden by the handling unit.
    """
    geo_key = geo.map(normalize_label)
    handling_key = handling.map(normalize_label)
    geo_borough = geo_key.map(BOROUGH_LOOKUP)
    fallback_borough = handling_key.map(BOROUGH_LOOKUP)

    geo_empty = geo_key.eq("")
    borough = geo_borough.where(~geo_empty, fallback_borough)

    rule = pd.Series("dropped_other_value", index=geo.index, dtype="object")
    rule[geo_borough.notna()] = "location_field"
    rule[geo_empty & fallback_borough.notna()] = "fallback_handling_field"
    rule[geo_empty & fallback_borough.isna()] = "dropped_no_location"
    rule[geo_key.isin(LINKED_LOOKUP)] = "dropped_linked_municipality"
    return borough, rule
