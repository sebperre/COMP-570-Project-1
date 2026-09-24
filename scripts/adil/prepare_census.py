"""Clean Adil's 2021 census extract into one row per borough.

Input:  data/adil/raw/POPULATION TOTALE EN 2016 ET EN 2021.csv (cp1252)
Output: data/adil/processed/borough_population_2021.csv (utf-8) with
        borough, population (2021), population_2016, growth_rate_pct,
        density_per_km2

The 'borough' and 'population' columns are the contract read by
scripts/mounir/audit_311.py (CENSUS_PATH) for its per-capita rates.

Usage:
  .venv/bin/python scripts/adil/prepare_census.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "adil" / "raw" / "POPULATION TOTALE EN 2016 ET EN 2021.csv"
OUT = ROOT / "data" / "adil" / "processed" / "borough_population_2021.csv"

# Statistics Canada 2021 census population of the City of Montreal (19 boroughs)
CITY_POPULATION_2021 = 1_762_949


def main():
    df = pd.read_csv(RAW, encoding="cp1252").rename(columns={
        "Borough": "borough",
        "Population 2021": "population",
        "Population 2016": "population_2016",
        "Growth Rate (%)": "growth_rate_pct",
        "Density (per km²)": "density_per_km2",
    })
    df["borough"] = df["borough"].str.strip()

    assert len(df) == 19 and df["borough"].is_unique, "expected 19 distinct boroughs"
    assert df["population"].sum() == CITY_POPULATION_2021, df["population"].sum()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(df)} boroughs, population {df['population'].sum():,}")


if __name__ == "__main__":
    main()
