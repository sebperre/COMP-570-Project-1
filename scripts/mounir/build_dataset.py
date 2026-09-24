"""Build the cleaned, borough-level 311 request dataset and its build audit.

One chunked pass over the raw extracts in data/mounir/raw/ produces
(in data/mounir/processed/):

- requests_311_2014_present.csv.gz : every non-Information request located in
  one of the 19 boroughs, all years (kept for Project 2 / future phases)
- requests_311_2023_2025.csv       : the same records for the study window
- build_audit.csv                  : row accounting for every rule applied
- yearly_counts.csv                : year x NATURE x borough rule x borough counts
- label_year_counts.csv            : year x NATURE x ACTI_NOM counts (in-city rows)
- raw_borough_values_2023_2025.csv : distinct raw borough strings in the window
"""

from __future__ import annotations

import gzip

import pandas as pd

from common import (
    OUTPUT_DIR, RAW_DIR, SOURCE_YEARS, SUPERSEDED_FILES, WINDOW_YEARS, reconcile_borough,
)


CHUNK_SIZE = 200_000

OUTPUT_COLUMNS = [
    "ID_UNIQUE", "year", "created_at", "NATURE", "ACTI_NOM", "TYPE_LIEU_INTERV",
    "borough", "borough_rule", "ARRONDISSEMENT", "ARRONDISSEMENT_GEO", "LOC_ERREUR_GDT",
    "PROVENANCE_ORIGINALE", "UNITE_RESP_PARENT", "DERNIER_STATUT", "last_status_at",
    "LOC_LAT", "LOC_LONG", "LOC_X", "LOC_Y", "source_file",
]
# RUE, RUE_INTERSECTION1/2 and LIN_CODE_POSTAL are not retained: they are not
# needed at borough level and dropping them minimises location detail.


def parse_datetime(values: pd.Series) -> pd.Series:
    """Parse both raw formats: ISO '2023-07-10T20:15:39' and '2018/01/21 20:04:02'."""
    text = values.str.replace("/", "-", regex=False).str.replace("T", " ", regex=False)
    return pd.to_datetime(text, format="%Y-%m-%d %H:%M:%S", errors="coerce")


def add(counter: dict, key, value: int) -> None:
    counter[key] = counter.get(key, 0) + int(value)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_path = OUTPUT_DIR / "requests_311_2014_present.csv.gz"
    window_path = OUTPUT_DIR / "requests_311_2023_2025.csv"

    audit: dict[str, int] = {}
    yearly: dict[tuple, int] = {}
    labels: dict[tuple, int] = {}
    raw_values: dict[tuple, int] = {}
    seen_ids: set[str] = set()
    first_full = first_window = True

    with gzip.open(full_path, "wt", encoding="utf-8", newline="") as full_fh, \
            open(window_path, "w", encoding="utf-8", newline="") as window_fh:
        for file_name, years in SOURCE_YEARS.items():
            for chunk in pd.read_csv(RAW_DIR / file_name, dtype=str, chunksize=CHUNK_SIZE):
                add(audit, f"rows_read:{file_name}", len(chunk))
                chunk["created_at"] = parse_datetime(chunk["DDS_DATE_CREATION"])
                add(audit, "unparseable_creation_date", chunk["created_at"].isna().sum())
                chunk["year"] = chunk["created_at"].dt.year.astype("Int64")
                in_partition = chunk["year"].isin(list(years))
                add(audit, f"rows_outside_year_partition:{file_name}", (~in_partition).sum())
                chunk = chunk.loc[in_partition].copy()
                add(audit, f"rows_used:{file_name}", len(chunk))
                chunk["source_file"] = file_name

                # Information calls have no ID_UNIQUE and no location by design.
                information = chunk["NATURE"].eq("Information")
                add(audit, "information_rows", information.sum())
                add(audit, "information_rows_with_id", (information & chunk["ID_UNIQUE"].notna()).sum())
                for (year, nature), count in chunk.loc[information].groupby(["year", "NATURE"]).size().items():
                    add(yearly, (year, nature, "dropped_information_no_location", ""), count)
                chunk = chunk.loc[~information].copy()

                missing_id = chunk["ID_UNIQUE"].isna()
                add(audit, "non_information_rows_missing_id", missing_id.sum())
                duplicate = chunk["ID_UNIQUE"].notna() & (
                    chunk["ID_UNIQUE"].isin(seen_ids) | chunk["ID_UNIQUE"].duplicated()
                )
                add(audit, "duplicate_id_rows_removed", duplicate.sum())
                chunk = chunk.loc[~duplicate].copy()
                seen_ids.update(chunk["ID_UNIQUE"].dropna())

                window = chunk["year"].isin(WINDOW_YEARS)
                for column in ["ARRONDISSEMENT", "ARRONDISSEMENT_GEO"]:
                    counts = chunk.loc[window, column].fillna("(missing)").value_counts()
                    for value, count in counts.items():
                        add(raw_values, (column, value), count)

                chunk["borough"], chunk["borough_rule"] = reconcile_borough(
                    chunk["ARRONDISSEMENT_GEO"], chunk["ARRONDISSEMENT"]
                )
                group = chunk.groupby(["year", "NATURE", "borough_rule", chunk["borough"].fillna("")]).size()
                for key, count in group.items():
                    add(yearly, key, count)

                chunk = chunk.loc[chunk["borough"].notna()].copy()
                for key, count in chunk.groupby(["year", "NATURE", "ACTI_NOM"]).size().items():
                    add(labels, key, count)

                chunk["last_status_at"] = parse_datetime(chunk["DATE_DERNIER_STATUT"].fillna(""))
                for column in ["LOC_LAT", "LOC_LONG", "LOC_X", "LOC_Y"]:
                    chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                out = chunk[OUTPUT_COLUMNS]
                out.to_csv(full_fh, index=False, header=first_full)
                first_full = False
                in_window = out["year"].isin(WINDOW_YEARS)
                if in_window.any():
                    out.loc[in_window].to_csv(window_fh, index=False, header=first_window)
                    first_window = False
                add(audit, "rows_written_2014_present", len(out))
                add(audit, "rows_written_2023_2025", in_window.sum())
            print(f"processed {file_name}")

    for name in SUPERSEDED_FILES:
        audit[f"superseded_file_not_read:{name}"] = 1

    pd.DataFrame([{"metric": k, "value": v} for k, v in audit.items()]).to_csv(
        OUTPUT_DIR / "build_audit.csv", index=False
    )
    pd.DataFrame(
        [(*k, v) for k, v in yearly.items()],
        columns=["year", "NATURE", "borough_rule", "borough", "rows"],
    ).sort_values(["year", "NATURE", "borough_rule", "borough"]).to_csv(OUTPUT_DIR / "yearly_counts.csv", index=False)
    pd.DataFrame(
        [(*k, v) for k, v in labels.items()], columns=["year", "NATURE", "ACTI_NOM", "rows"]
    ).sort_values(["year", "NATURE", "rows"], ascending=[True, True, False]).to_csv(
        OUTPUT_DIR / "label_year_counts.csv", index=False
    )
    pd.DataFrame(
        [(*k, v) for k, v in raw_values.items()], columns=["field", "raw_value", "rows"]
    ).sort_values(["field", "rows"], ascending=[True, False]).to_csv(
        OUTPUT_DIR / "raw_borough_values_2023_2025.csv", index=False
    )
    print(f"wrote {audit['rows_written_2014_present']:,} rows (2014-present), "
          f"{audit['rows_written_2023_2025']:,} rows (2023-2025)")


if __name__ == "__main__":
    main()
