"""Audit tables for the 311 source: every 311 number quoted in the report.

Run after build_dataset.py and acti_nom_mapping.py. Writes to data/mounir/processed/:

- audit_2023_2025.csv                metric, value, definition (the funnel)
- borough_311_2023_2025.csv          one row per borough: counts, shares, outcome
- missingness_2023_2025.csv          missing values per field (raw and cleaned)
- missingness_by_borough_2023_2025.csv
- biases_2023_2025.csv               quantified biases
- summary_2014_present/yearly_summary.csv and borough_year_summary.csv

If data/adil/processed/borough_population_2021.csv exists (columns: borough,
population) the borough table also gets rates per 1,000 residents.
"""

from __future__ import annotations

import pandas as pd

from common import (
    CANONICAL_BOROUGHS, CENSUS_PATH, OUTPUT_DIR, RAW_DIR, SUMMARY_DIR, WINDOW_YEARS, normalize_label,
)


REQUEST_NATURES = ["Requete", "Plainte"]
DIGITAL_CHANNELS = ["Mobile", "Internet", "Médias sociaux"]
THEMES = ["road_sidewalk", "waste_cleanliness", "graffiti_vandalism", "noise_nuisance", "trees_greenspace"]


class Metrics:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, metric: str, value, definition: str) -> None:
        if isinstance(value, float):
            value = round(value, 4)
        self.rows.append({"metric": metric, "value": value, "definition": definition})

    def save(self, path) -> None:
        pd.DataFrame(self.rows).to_csv(path, index=False, encoding="utf-8")


def load_window() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "requests_311_2023_2025.csv", dtype=str, encoding="utf-8")
    mapping = pd.read_csv(OUTPUT_DIR / "acti_nom_mapping.csv", dtype=str, encoding="utf-8").fillna("")
    df = df.merge(mapping[["ACTI_NOM", "group", "theme", "subtheme", "confidence"]], on="ACTI_NOM", how="left")
    df["office_flag"] = df["LOC_ERREUR_GDT"].eq("1")
    df["handling_borough"] = df["ARRONDISSEMENT"].map(normalize_label).map(
        {normalize_label(b): b for b in CANONICAL_BOROUGHS})
    return df


def raw_window_missingness() -> pd.DataFrame:
    """Missing values per raw field in the 2023-2025 rows of the current extract."""
    total = info = 0
    missing_all: dict[str, int] = {}
    missing_noninfo: dict[str, int] = {}
    for chunk in pd.read_csv(RAW_DIR / "requetes311.csv", dtype=str, chunksize=250_000):
        chunk = chunk[chunk["DDS_DATE_CREATION"].str[:4].isin([str(y) for y in WINDOW_YEARS])]
        total += len(chunk)
        is_info = chunk["NATURE"].eq("Information")
        info += int(is_info.sum())
        for column, count in chunk.isna().sum().items():
            missing_all[column] = missing_all.get(column, 0) + int(count)
        for column, count in chunk.loc[~is_info].isna().sum().items():
            missing_noninfo[column] = missing_noninfo.get(column, 0) + int(count)
    return pd.DataFrame({
        "field": list(missing_all),
        "stage": "raw_2023_2025",
        "missing_all_rows": list(missing_all.values()),
        "fraction_all_rows": [v / total for v in missing_all.values()],
        "missing_excluding_information": [missing_noninfo[c] for c in missing_all],
        "fraction_excluding_information": [missing_noninfo[c] / (total - info) for c in missing_all],
    })


def borough_table(df: pd.DataFrame, rp: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    by = lambda frame: frame.groupby("borough")  # noqa: E731
    t = pd.DataFrame(index=CANONICAL_BOROUGHS)
    t.index.name = "borough"
    t["rows_all_natures"] = by(df).size()
    t["rows_requete_plainte"] = by(rp).size()
    t["office_flag_rows"] = by(rp[rp["office_flag"]]).size()
    t["office_flag_share"] = t["office_flag_rows"] / t["rows_requete_plainte"]
    t["rows_after_office_exclusion"] = by(base).size()
    for group in ["condition", "service", "administrative"]:
        t[f"{group}_rows"] = by(base[base["group"].eq(group)]).size()
        t[f"{group}_share"] = t[f"{group}_rows"] / t["rows_after_office_exclusion"]
    cond = base[base["group"].eq("condition")]
    t["condition_rows_clear_only"] = by(cond[cond["confidence"].eq("clear")]).size()
    for theme in THEMES:
        t[f"theme_{theme}"] = by(cond[cond["theme"].eq(theme)]).size()
    t["outcome_share_of_requete_plainte"] = t["condition_rows"] / t["rows_requete_plainte"]
    t["raw_to_outcome_inflation"] = t["rows_requete_plainte"] / t["condition_rows"] - 1
    flags = pd.DataFrame({
        "borough": rp["borough"],
        "channel_phone_share": rp["PROVENANCE_ORIGINALE"].eq("Téléphone"),
        "channel_digital_share": rp["PROVENANCE_ORIGINALE"].isin(DIGITAL_CHANNELS),
        "channel_in_person_share": rp["PROVENANCE_ORIGINALE"].eq("Personne"),
        "share_retired_labels": rp["ACTI_NOM"].str.startswith("*"),
        "handling_unit_differs_rows": rp["handling_borough"].notna() & rp["handling_borough"].ne(rp["borough"]),
    })
    means = flags.groupby("borough").mean()
    for column in ["channel_phone_share", "channel_digital_share", "channel_in_person_share", "share_retired_labels"]:
        t[column] = means[column]
    t["distinct_acti_labels"] = by(rp)["ACTI_NOM"].nunique()
    labels_per_borough = rp.groupby("ACTI_NOM")["borough"].nunique()
    rare = labels_per_borough[labels_per_borough <= 3].index
    t["rows_in_labels_used_by_le3_boroughs"] = by(rp[rp["ACTI_NOM"].isin(rare)]).size()
    t["share_in_labels_used_by_le3_boroughs"] = t["rows_in_labels_used_by_le3_boroughs"] / t["rows_requete_plainte"]
    t["handling_unit_differs_rows"] = flags.groupby("borough")["handling_unit_differs_rows"].sum()
    t["fallback_rule_rows"] = by(rp[rp["borough_rule"].eq("fallback_handling_field")]).size()
    t = t.fillna(0)
    t["rank_raw_requete_plainte"] = t["rows_requete_plainte"].rank(ascending=False).astype(int)
    t["rank_outcome"] = t["condition_rows"].rank(ascending=False).astype(int)

    if CENSUS_PATH.exists():
        census = pd.read_csv(CENSUS_PATH, encoding="utf-8")
        census["key"] = census["borough"].map(normalize_label)
        population = census.set_index("key")["population"]
        t["population_2021"] = [population.get(normalize_label(b)) for b in t.index]
        per_1000 = 1000 / t["population_2021"] / len(WINDOW_YEARS)
        t["requete_plainte_per_1000_per_year"] = t["rows_requete_plainte"] * per_1000
        t["outcome_per_1000_per_year"] = t["condition_rows"] * per_1000
        t["outcome_clear_only_per_1000_per_year"] = t["condition_rows_clear_only"] * per_1000
        t["rank_raw_rate"] = t["requete_plainte_per_1000_per_year"].rank(ascending=False)
        t["rank_outcome_rate"] = t["outcome_per_1000_per_year"].rank(ascending=False)
    return t


def main() -> None:
    m = Metrics()
    yearly = pd.read_csv(OUTPUT_DIR / "yearly_counts.csv")
    yw = yearly[yearly["year"].isin(WINDOW_YEARS)]
    raw_total = int(yw["rows"].sum())
    nature = yw.groupby("NATURE")["rows"].sum()
    rule = yw[yw["NATURE"].ne("Information")].groupby("borough_rule")["rows"].sum()
    non_info = int(raw_total - nature.get("Information", 0))

    m.add("window", "2023-01-01 to 2025-12-31", "Inclusive; by request creation date (DDS_DATE_CREATION)")
    m.add("raw_rows", raw_total, "All raw rows created in the window (single source: requetes311.csv)")
    for key in ["Requete", "Plainte", "Commentaire", "Information"]:
        m.add(f"raw_rows_{key}", int(nature.get(key, 0)), f"Raw rows with NATURE = {key}")
    m.add("information_share_of_raw", nature.get("Information", 0) / raw_total,
          "Information calls: no ID_UNIQUE, no location, excluded")
    m.add("non_information_rows", non_info, "Rows with a request/complaint/comment that carry an ID")
    for name, definition in [
        ("location_field", "Borough taken from ARRONDISSEMENT_GEO (location of the work)"),
        ("fallback_handling_field", "ARRONDISSEMENT_GEO empty; borough taken from handling unit ARRONDISSEMENT"),
        ("dropped_linked_municipality", "Location in one of the 15 linked municipalities (not one of the 19 boroughs)"),
        ("dropped_no_location", "Both borough fields empty or handling unit not a borough"),
        ("dropped_other_value", "Location value not recognised"),
    ]:
        m.add(f"borough_rule_{name}", int(rule.get(name, 0)), definition)
    dropped = int(rule.filter(like="dropped").sum())
    m.add("non_information_rows_dropped_by_borough_rule", dropped, "Sum of dropped_* rules")
    m.add("non_information_drop_fraction", dropped / non_info, "Dropped / non-Information rows")

    raw_values = pd.read_csv(OUTPUT_DIR / "raw_borough_values_2023_2025.csv", encoding="utf-8")
    for field in ["ARRONDISSEMENT", "ARRONDISSEMENT_GEO"]:
        values = raw_values[raw_values["field"].eq(field) & raw_values["raw_value"].ne("(missing)")]
        m.add(f"distinct_raw_values_{field}", len(values), f"Distinct non-missing {field} strings, non-Information rows")

    df = load_window()
    m.add("cleaned_rows", len(df), "Rows in requests_311_2023_2025.csv (19 boroughs, non-Information)")
    m.add("cleaned_distinct_acti_labels", df["ACTI_NOM"].nunique(), "Distinct ACTI_NOM in the cleaned window extract")
    rp = df[df["NATURE"].isin(REQUEST_NATURES)]
    m.add("cleaned_rows_Commentaire", int(df["NATURE"].eq("Commentaire").sum()), "Comments, excluded from the outcome")
    m.add("requete_plainte_rows", len(rp), "Requete + Plainte rows in the 19 boroughs")
    m.add("requete_plainte_distinct_labels", rp["ACTI_NOM"].nunique(), "Distinct ACTI_NOM among Requete + Plainte rows")
    handling_differs = rp["handling_borough"].notna() & rp["handling_borough"].ne(rp["borough"])
    m.add("handling_unit_differs_from_location", int(handling_differs.sum()),
          "Requete+Plainte rows assigned to their location borough although another borough handled them")
    m.add("handling_unit_not_a_borough", int(rp["handling_borough"].isna().sum()),
          "Requete+Plainte rows whose handling unit is not a borough (e.g. 'Ville de Montréal')")
    m.add("unmapped_label_rows", int(rp["group"].isna().sum() + rp["group"].eq("unmapped").sum()),
          "Requete+Plainte rows whose label has no mapping (should be 0)")

    office = rp["office_flag"]
    m.add("office_flag_rows", int(office.sum()),
          "LOC_ERREUR_GDT = 1: request logged at a borough office, location imprecise (City documentation)")
    m.add("office_flag_share", office.mean(), "Office-flagged / Requete+Plainte")
    m.add("office_flag_missing", int(rp["LOC_ERREUR_GDT"].isna().sum()), "LOC_ERREUR_GDT empty")
    base = rp[~office]
    m.add("rows_after_office_exclusion", len(base), "Requete+Plainte rows not office-flagged")
    for group in ["condition", "service", "administrative"]:
        sub = base[base["group"].eq(group)]
        m.add(f"{group}_rows", len(sub), f"Rows mapped to group '{group}' after office exclusion")
        m.add(f"{group}_share", len(sub) / len(base), f"{group} rows / rows after office exclusion")
        m.add(f"{group}_distinct_labels", sub["ACTI_NOM"].nunique(), f"Distinct labels in group '{group}'")
    cond = base[base["group"].eq("condition")]
    m.add("outcome_rows", len(cond), "PRIMARY OUTCOME: condition rows (clear + ambiguous labels), 2023-2025")
    m.add("outcome_rows_clear_only", int(cond["confidence"].eq("clear").sum()),
          "Sensitivity: condition rows using only unambiguous labels")
    m.add("outcome_share_of_raw_311", len(cond) / raw_total, "Outcome rows / all raw 311 rows in the window")
    m.add("outcome_share_of_requete_plainte", len(cond) / len(rp), "Outcome rows / in-city Requete+Plainte rows")
    office_admin = rp["office_flag"] | rp["group"].eq("administrative")
    m.add("office_or_administrative_rows", int(office_admin.sum()), "Office-flagged or administrative-label rows")
    m.add("office_or_administrative_share", office_admin.mean(), "... / in-city Requete+Plainte rows")
    for theme in THEMES:
        sub = cond[cond["theme"].eq(theme)]
        m.add(f"theme_{theme}", len(sub), f"Outcome rows in theme {theme}")
    for year in WINDOW_YEARS:
        m.add(f"outcome_rows_{year}", int(cond["year"].eq(str(year)).sum()), f"Outcome rows created in {year}")

    m.add("missing_coordinates_requete_plainte", int(rp["LOC_LAT"].isna().sum()), "Requete+Plainte rows without LOC_LAT")
    m.add("missing_last_status", int(rp["DERNIER_STATUT"].isna().sum()), "Requete+Plainte rows without DERNIER_STATUT")
    m.add("missing_provenance", int(rp["PROVENANCE_ORIGINALE"].isna().sum()), "Requete+Plainte rows without channel")
    m.save(OUTPUT_DIR / "audit_2023_2025.csv")

    boroughs = borough_table(df, rp, base)
    boroughs.to_csv(OUTPUT_DIR / "borough_311_2023_2025.csv", encoding="utf-8")

    # ---- missingness ----
    raw_missing = raw_window_missingness()
    clean_missing = pd.DataFrame({
        "field": df.columns[:20],
        "stage": "cleaned_2023_2025",
        "missing_all_rows": [int(df[c].isna().sum()) for c in df.columns[:20]],
        "fraction_all_rows": [df[c].isna().mean() for c in df.columns[:20]],
    })
    pd.concat([raw_missing, clean_missing]).to_csv(OUTPUT_DIR / "missingness_2023_2025.csv", index=False, encoding="utf-8")
    key_fields = ["ACTI_NOM", "TYPE_LIEU_INTERV", "LOC_LAT", "LOC_ERREUR_GDT", "PROVENANCE_ORIGINALE",
                  "DERNIER_STATUT", "last_status_at"]
    by_borough = rp.groupby("borough")[key_fields].apply(lambda g: g.isna().mean())
    by_borough["complete_record_share"] = rp.groupby("borough")[key_fields].apply(lambda g: g.notna().all(axis=1).mean())
    by_borough["rows"] = rp.groupby("borough").size()
    by_borough.to_csv(OUTPUT_DIR / "missingness_by_borough_2023_2025.csv", encoding="utf-8")

    # ---- biases ----
    b = boroughs
    median_office = b["office_flag_share"].median()
    median_admin = b["administrative_share"].median()
    rank_shift = (b["rank_raw_requete_plainte"] - b["rank_outcome"]).abs()
    biases = [
        ("office_flag_share_min", b["office_flag_share"].min(), b["office_flag_share"].idxmin()),
        ("office_flag_share_median", median_office, ""),
        ("office_flag_share_max", b["office_flag_share"].max(), b["office_flag_share"].idxmax()),
        ("administrative_share_min", b["administrative_share"].min(), b["administrative_share"].idxmin()),
        ("administrative_share_median", median_admin, ""),
        ("administrative_share_max", b["administrative_share"].max(), b["administrative_share"].idxmax()),
        ("raw_to_outcome_inflation_min", b["raw_to_outcome_inflation"].min(), b["raw_to_outcome_inflation"].idxmin()),
        ("raw_to_outcome_inflation_median", b["raw_to_outcome_inflation"].median(), ""),
        ("raw_to_outcome_inflation_max", b["raw_to_outcome_inflation"].max(), b["raw_to_outcome_inflation"].idxmax()),
        ("boroughs_changing_rank_raw_vs_outcome", int((rank_shift > 0).sum()), "count-rank, not per-capita"),
        ("max_rank_shift_raw_vs_outcome", int(rank_shift.max()), rank_shift.idxmax()),
        ("distinct_labels_min", b["distinct_acti_labels"].min(), b["distinct_acti_labels"].idxmin()),
        ("distinct_labels_max", b["distinct_acti_labels"].max(), b["distinct_acti_labels"].idxmax()),
        ("labels_used_by_le3_boroughs", int((rp.groupby("ACTI_NOM")["borough"].nunique() <= 3).sum()),
         f"of {rp['ACTI_NOM'].nunique()} labels"),
        ("share_in_labels_used_by_le3_boroughs_max", b["share_in_labels_used_by_le3_boroughs"].max(),
         b["share_in_labels_used_by_le3_boroughs"].idxmax()),
        ("share_in_labels_used_by_le3_boroughs_median", b["share_in_labels_used_by_le3_boroughs"].median(), ""),
        ("digital_channel_share_min", b["channel_digital_share"].min(), b["channel_digital_share"].idxmin()),
        ("digital_channel_share_max", b["channel_digital_share"].max(), b["channel_digital_share"].idxmax()),
        ("in_person_channel_share_min", b["channel_in_person_share"].min(), b["channel_in_person_share"].idxmin()),
        ("in_person_channel_share_max", b["channel_in_person_share"].max(), b["channel_in_person_share"].idxmax()),
        ("ambiguous_share_of_outcome_min", (1 - b["condition_rows_clear_only"] / b["condition_rows"]).min(),
         (1 - b["condition_rows_clear_only"] / b["condition_rows"]).idxmin()),
        ("ambiguous_share_of_outcome_max", (1 - b["condition_rows_clear_only"] / b["condition_rows"]).max(),
         (1 - b["condition_rows_clear_only"] / b["condition_rows"]).idxmax()),
    ]
    pd.DataFrame(biases, columns=["metric", "value", "borough_or_note"]).round(4).to_csv(
        OUTPUT_DIR / "biases_2023_2025.csv", index=False, encoding="utf-8")

    summary_2014_present(yearly)
    print(pd.read_csv(OUTPUT_DIR / "audit_2023_2025.csv").to_string())


def summary_2014_present(yearly: pd.DataFrame) -> None:
    """Separate, non-report summary of every year, for later project phases."""
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    mapping = pd.read_csv(OUTPUT_DIR / "acti_nom_mapping.csv", dtype=str, encoding="utf-8").fillna("")
    group_of = mapping.set_index("ACTI_NOM")["group"]
    parts = []
    columns = ["year", "NATURE", "ACTI_NOM", "borough", "LOC_ERREUR_GDT", "DERNIER_STATUT", "LOC_LAT"]
    for chunk in pd.read_csv(OUTPUT_DIR / "requests_311_2014_present.csv.gz", usecols=columns, dtype=str,
                             chunksize=500_000):
        chunk = chunk[chunk["NATURE"].isin(REQUEST_NATURES)].copy()
        chunk["year"] = chunk["year"].astype(int)
        chunk["group"] = chunk["ACTI_NOM"].map(group_of).fillna("unmapped")
        chunk["office"] = chunk["LOC_ERREUR_GDT"].eq("1")
        chunk["outcome"] = chunk["group"].eq("condition") & ~chunk["office"]
        chunk["status_missing"] = chunk["DERNIER_STATUT"].isna()
        chunk["coords_missing"] = chunk["LOC_LAT"].isna()
        parts.append(chunk.groupby(["year", "borough"]).agg(
            requete_plainte=("NATURE", "size"), office_flag=("office", "sum"), outcome=("outcome", "sum"),
            status_missing=("status_missing", "sum"), coords_missing=("coords_missing", "sum"),
            administrative=("group", lambda s: s.eq("administrative").sum()),
            unmapped=("group", lambda s: s.eq("unmapped").sum()),
            distinct_labels=("ACTI_NOM", "nunique")))
    by = pd.concat(parts).groupby(level=[0, 1]).agg(
        {"requete_plainte": "sum", "office_flag": "sum", "outcome": "sum", "status_missing": "sum",
         "coords_missing": "sum", "administrative": "sum",
         "unmapped": "sum", "distinct_labels": "max"})
    by.to_csv(SUMMARY_DIR / "borough_year_summary.csv", encoding="utf-8")

    labels = pd.read_csv(OUTPUT_DIR / "label_year_counts.csv", encoding="utf-8")
    labels_rp = labels[labels["NATURE"].isin(REQUEST_NATURES)]
    years = by.groupby(level=0).sum(numeric_only=True).drop(columns="distinct_labels")
    years["distinct_labels_requete_plainte"] = labels_rp.groupby("year")["ACTI_NOM"].nunique()
    raw = yearly.groupby("year")["rows"].sum()
    info = yearly[yearly["NATURE"].eq("Information")].groupby("year")["rows"].sum()
    in_city = yearly[yearly["borough"].notna() & yearly["borough"].ne("")].groupby("year")["rows"].sum()
    years.insert(0, "raw_rows", raw)
    years.insert(1, "information_rows", info)
    years.insert(2, "in_city_non_information_rows", in_city)
    years["office_flag_share"] = years["office_flag"] / years["requete_plainte"]
    years["outcome_share_of_requete_plainte"] = years["outcome"] / years["requete_plainte"]
    years["complete_year"] = years.index < 2026
    years.round(4).to_csv(SUMMARY_DIR / "yearly_summary.csv", encoding="utf-8")


if __name__ == "__main__":
    main()
