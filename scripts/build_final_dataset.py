"""Join 311 (Mounir), census (Adil) and crime (Shirley) into one row per borough.

Inputs (run the three pipelines first, see README):
  data/mounir/processed/borough_311_2023_2025.csv   311 counts per borough
  data/mounir/processed/requests_311_2023_2025.csv  cleaned 311 rows, for the incident columns
  data/mounir/processed/acti_nom_mapping.csv        ACTI_NOM -> group, theme, confidence
  data/mounir/processed/audit_2023_2025.csv         311 funnel, for the integrity checks
  data/adil/processed/borough_population_2021.csv   2021 Census population
  data/shirley/processed/crimes_by_borough_2023_2025.csv
  data/shirley/processed/crimes_join_report.txt
  data/sebastien/raw/bourough-geo-coords.json       official names and MAMH codes
  data/sebastien/raw/isq_borough_population_2023.csv  ISQ estimate, sensitivity only

The three sources spell some borough names differently (hyphen or en dash,
accents), so every name is normalized and mapped to the borough's MAMH code
(e.g. REM17) before joining. The script stops if any of the 19 boroughs is
missing or duplicated on any side.

311 columns follow Mounir's rules: Requete + Plainte rows located in one of
the 19 boroughs, excluding rows logged at a borough office (LOC_ERREUR_GDT = 1).
The outcome (requests_condition) counts requests that report a problem in the
neighbourhood; service and administrative requests are kept for comparison.
The incident columns (cleanliness, rats, potholes) count specific 311
categories (INCIDENTS below) on the same rows.

Rates are per 1,000 residents per year: the 2023-2025 count divided by 3 years
and by the 2021 Census population. The ISQ 1 July 2023 estimate is carried
along as a sensitivity check on the denominator.

Outputs:
  data/final_dataset.csv          19 rows, one per borough
  data/final_dataset_report.json  join and integrity-check results

Usage:
  .venv/bin/python scripts/build_final_dataset.py
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MOUNIR = DATA / "mounir" / "processed"
ADIL = DATA / "adil" / "processed"
SHIRLEY = DATA / "shirley" / "processed"

BOROUGH_311 = MOUNIR / "borough_311_2023_2025.csv"
REQUESTS_311 = MOUNIR / "requests_311_2023_2025.csv"
MAPPING_311 = MOUNIR / "acti_nom_mapping.csv"
AUDIT_311 = MOUNIR / "audit_2023_2025.csv"
CENSUS = ADIL / "borough_population_2021.csv"
CRIME = SHIRLEY / "crimes_by_borough_2023_2025.csv"
CRIME_REPORT = SHIRLEY / "crimes_join_report.txt"
BOUNDARIES = DATA / "sebastien" / "raw" / "bourough-geo-coords.json"
ISQ = DATA / "sebastien" / "raw" / "isq_borough_population_2023.csv"

OUT = DATA / "final_dataset.csv"
REPORT = DATA / "final_dataset_report.json"

YEARS = 3
CITY_POPULATION_2021 = 1_762_949
REQUEST_NATURES = ["Requete", "Plainte"]
THEMES = ["road_sidewalk", "waste_cleanliness", "graffiti_vandalism", "noise_nuisance", "trees_greenspace"]

# 311 categories (ACTI_NOM) counted for each incident column; the four
# cleanliness_* columns are summed into cleanliness_total.
INCIDENTS = {
    "cleanliness_illegal_dumping": [
        "Dépôt illégal - Déchets", "Dépôt illégal déchet - ruelle", "Dépôt illégal-Ramassage",
        "Dépôt sauvage - Domaine public", "Dépôt sauvage - Terrain privé",
    ],
    "cleanliness_public_cleaning": ["Nettoyage du domaine public", "Parc - Propreté"],
    "cleanliness_graffiti": [
        "Graffitis - Domaine public", "Graffitis - Domaine privé", "Graffitis haineux",
        "*Graffitis - Domaine privé - Inspection", "Graffitis sur signalisation",
        "Graffitis sur bâtiment public",
    ],
    "cleanliness_debris": [
        "Débris sur la voie publique", "Débris sur chantier et rues",
        "Environnement - débris voie publique",
    ],
    "rats": ["Extermination à l'extérieur - Rats"],
    "potholes": ["Nid-de-poule"],
}

# Mounir's borough table column -> final dataset column
COLUMNS_311 = {
    "rows_requete_plainte": "requests_311",
    "office_flag_rows": "requests_office_flagged",
    "condition_rows": "requests_condition",
    "condition_rows_clear_only": "requests_condition_clear_only",
    "service_rows": "requests_service",
    "administrative_rows": "requests_administrative",
    **{f"theme_{t}": f"condition_{t}" for t in THEMES},
}


def borough_key(name):
    """Lowercase, no accents, no punctuation: 'Rosemont–La Petite-Patrie' -> 'rosemontlapetitepatrie'."""
    name = unicodedata.normalize("NFKD", str(name))
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "", name.casefold())


def borough_codes():
    features = json.loads(BOUNDARIES.read_text(encoding="utf-8"))["features"]
    rows = [f["properties"] for f in features if f["properties"]["TYPE"] == "Arrondissement"]
    return pd.DataFrame({"code_mamh": [r["CODEMAMH"] for r in rows], "borough": [r["NOM"] for r in rows]})


def attach_code(df, codes, source):
    """Replace the 'borough' name column by the MAMH code; stop unless all 19 boroughs appear once."""
    lookup = {borough_key(b): c for b, c in zip(codes["borough"], codes["code_mamh"])}
    df = df.assign(code_mamh=df["borough"].map(lambda b: lookup.get(borough_key(b))))
    unmatched = df.loc[df["code_mamh"].isna(), "borough"].tolist()
    if unmatched:
        raise SystemExit(f"{source}: borough names not in the boundary file: {unmatched}")
    if len(df) != 19 or df["code_mamh"].nunique() != 19:
        raise SystemExit(f"{source}: expected each of the 19 boroughs exactly once")
    return df.drop(columns="borough")


def slug(text):
    return re.sub(r"\W+", "_", text.casefold()).strip("_")


def load_311(codes):
    t = pd.read_csv(BOROUGH_311, encoding="utf-8")
    t = t[["borough", *COLUMNS_311]].rename(columns=COLUMNS_311)
    return attach_code(t, codes, "311 (Mounir)")


def load_incidents(codes):
    """Incident counts on Mounir's outcome base: Requete + Plainte, office-flagged rows excluded."""
    df = pd.read_csv(REQUESTS_311, usecols=["NATURE", "ACTI_NOM", "LOC_ERREUR_GDT", "borough"],
                     dtype=str, encoding="utf-8")
    base = df[df["NATURE"].isin(REQUEST_NATURES) & df["LOC_ERREUR_GDT"].ne("1")]

    labels = [a for group in INCIDENTS.values() for a in group]
    missing = sorted(set(labels) - set(base["ACTI_NOM"]))
    if missing:
        raise SystemExit(f"incident categories not found in the 311 data: {missing}")
    mapping = pd.read_csv(MAPPING_311, dtype=str, encoding="utf-8").set_index("ACTI_NOM")["group"]
    not_condition = {a: mapping.get(a) for a in labels if mapping.get(a) != "condition"}

    label_to_metric = {a: m for m, group in INCIDENTS.items() for a in group}
    picked = base.assign(metric=base["ACTI_NOM"].map(label_to_metric)).dropna(subset=["metric"])
    counts = (picked.pivot_table(index="borough", columns="metric", values="NATURE",
                                 aggfunc="size", fill_value=0)
              .reindex(columns=list(INCIDENTS), fill_value=0))
    cleanliness = [m for m in INCIDENTS if m.startswith("cleanliness_")]
    counts.insert(len(cleanliness), "cleanliness_total", counts[cleanliness].sum(axis=1))
    counts.columns.name = None
    return attach_code(counts.reset_index(), codes, "311 incidents"), not_condition


def load_census(codes):
    census = pd.read_csv(CENSUS, encoding="utf-8").rename(columns={
        "population": "population_2021",
        "growth_rate_pct": "growth_2016_2021_pct",
        "density_per_km2": "density_2021_per_km2",
    })
    return attach_code(census, codes, "census (Adil)")


def load_crime(codes):
    crime = pd.read_csv(CRIME, encoding="utf-8-sig")
    categories = [c for c in crime.columns if c not in ("borough", "Total")]
    crime = crime.rename(columns={"Total": "crimes_total", **{c: f"crimes_{slug(c)}" for c in categories}})
    crime = crime[["borough", *sorted(c for c in crime.columns if c.startswith("crimes_") and c != "crimes_total"),
                   "crimes_total"]]
    return attach_code(crime, codes, "crime (Shirley)")


def crime_kept_in_report():
    text = CRIME_REPORT.read_text(encoding="utf-8")
    return int(re.search(r"KEPT in the 19 boroughs: ([\d,]+)", text).group(1).replace(",", ""))


def main():
    inputs = [BOROUGH_311, REQUESTS_311, MAPPING_311, AUDIT_311, CENSUS, CRIME, CRIME_REPORT, BOUNDARIES, ISQ]
    for path in inputs:
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)}; run the earlier pipeline steps first (see README)")

    codes = borough_codes()
    census = load_census(codes)
    isq = pd.read_csv(ISQ, encoding="utf-8").drop(columns="borough")
    s311 = load_311(codes)
    incidents, not_condition = load_incidents(codes)
    crime = load_crime(codes)

    df = (codes.merge(census, on="code_mamh", validate="1:1")
          .merge(isq, on="code_mamh", validate="1:1")
          .merge(s311, on="code_mamh", validate="1:1")
          .merge(incidents, on="code_mamh", validate="1:1")
          .merge(crime, on="code_mamh", validate="1:1"))
    if len(df) != 19:
        raise SystemExit(f"join produced {len(df)} rows, expected 19")

    df["population_change_2021_2023_pct"] = ((df["population_2023_07_01"] / df["population_2021"] - 1) * 100).round(1)
    after_office = df["requests_311"] - df["requests_office_flagged"]
    df["office_flagged_share_pct"] = (df["requests_office_flagged"] / df["requests_311"] * 100).round(1)
    df["administrative_share_pct"] = (df["requests_administrative"] / after_office * 100).round(1)

    per_1000 = 1000 / YEARS / df["population_2021"]
    counts_311 = [*COLUMNS_311.values(), *[c for c in incidents.columns if c != "code_mamh"]]
    crime_counts = [c for c in crime.columns if c.startswith("crimes_")]
    for c in counts_311:
        df[f"rate_{c}"] = (df[c] * per_1000).round(2)
    for c in crime_counts:
        df[f"rate_{c}"] = (df[c] * per_1000).round(2)
    per_1000_isq = 1000 / YEARS / df["population_2023_07_01"]
    df["rate_requests_condition_isq2023"] = (df["requests_condition"] * per_1000_isq).round(2)
    df["rate_crimes_total_isq2023"] = (df["crimes_total"] * per_1000_isq).round(2)

    ids = ["code_mamh", "borough", "population_2021", "population_2016", "growth_2016_2021_pct",
           "density_2021_per_km2", "population_2023_07_01", "population_change_2021_2023_pct"]
    df = df[[*ids, *counts_311, "office_flagged_share_pct", "administrative_share_pct",
             *[f"rate_{c}" for c in counts_311], "rate_requests_condition_isq2023",
             *crime_counts, *[f"rate_{c}" for c in crime_counts], "rate_crimes_total_isq2023"]]
    df = df.sort_values("borough").reset_index(drop=True)
    df.to_csv(OUT, index=False, encoding="utf-8")

    audit = pd.read_csv(AUDIT_311, encoding="utf-8").set_index("metric")["value"]
    mounir_rate = attach_code(pd.read_csv(BOROUGH_311, encoding="utf-8")[["borough", "outcome_per_1000_per_year"]],
                              codes, "311 rates (Mounir)").set_index("code_mamh")["outcome_per_1000_per_year"]
    rate_gap = (df.set_index("code_mamh")["rate_requests_condition"] - mounir_rate).abs().max()
    checks = {
        "population_2021_matches_city_total": int(df["population_2021"].sum()) == CITY_POPULATION_2021,
        "requests_311_match_mounir_audit": int(df["requests_311"].sum()) == int(audit["requete_plainte_rows"]),
        "condition_matches_mounir_outcome": int(df["requests_condition"].sum()) == int(audit["outcome_rows"]),
        "condition_rate_matches_mounir": bool(rate_gap <= 0.005 + 1e-9),
        "crimes_match_shirley_report": int(df["crimes_total"].sum()) == crime_kept_in_report(),
        "crime_categories_sum_to_total": bool(
            (df[[c for c in crime_counts if c != "crimes_total"]].sum(axis=1) == df["crimes_total"]).all()),
        "incident_categories_all_in_condition_group": not not_condition,
        "no_missing_values": not df.isna().any().any(),
    }
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise SystemExit(f"integrity checks failed: {failed} (incident labels outside 'condition': {not_condition})")

    def spearman(a, b):
        return round(df[a].rank().corr(df[b].rank()), 2)

    admin = df.set_index("borough")["administrative_share_pct"]
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "checks": checks,
        "population_2021": int(df["population_2021"].sum()),
        "population_2023_07_01": int(df["population_2023_07_01"].sum()),
        "totals_2023_2025": {c: int(df[c].sum()) for c in [*counts_311, "crimes_total"]},
        "city_rate_per_1000_per_year": {
            c: round(df[c].sum() * 1000 / YEARS / df["population_2021"].sum(), 2)
            for c in ["requests_condition", "cleanliness_total", "rats", "potholes", "crimes_total"]
        },
        "administrative_share_pct": {
            "city": round(df["requests_administrative"].sum()
                          / (df["requests_311"] - df["requests_office_flagged"]).sum() * 100, 1),
            "min": {"borough": admin.idxmin(), "pct": float(admin.min())},
            "max": {"borough": admin.idxmax(), "pct": float(admin.max())},
        },
        "spearman_with_crime_rate": {
            c: spearman(f"rate_{c}", "rate_crimes_total")
            for c in ["requests_condition", "cleanliness_total", "rats", "potholes"]
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    pd.set_option("display.width", 200)
    print(df[["borough", "population_2021", "rate_requests_condition", "rate_cleanliness_total",
              "rate_rats", "rate_potholes", "administrative_share_pct", "rate_crimes_total"]].to_string(index=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(df)} rows x {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
