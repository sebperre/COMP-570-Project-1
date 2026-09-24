"""Make the report's figures and tables from the pipeline outputs.

Inputs (run the pipeline first, see README):
  data/final_dataset.csv                          one row per borough
  data/mounir/processed/requests_311_2023_2025.csv, acti_nom_mapping.csv,
    audit_2023_2025.csv, missingness_2023_2025.csv, build_audit.csv
  data/shirley/processed/crimes_join_report.txt
  data/sebastien/raw/bourough-geo-coords.json     boundaries, for the maps
  data/sources_manifest.json                      retrieval dates

Outputs:
  report/figures/requests_over_time.pdf   monthly 311 volume by request type and incident
  report/figures/borough_maps.pdf         rates per 1,000 residents per year, four maps
  report/figures/admin_share.pdf          office-flagged and administrative share of 311 per borough
  report/figures/cleanliness_vs_crime.pdf cleanliness rate vs crime rate per borough
  report/tables/sources.tex, join_loss.tex, filtering.tex, missingness.tex, incidents.tex, population.tex

Boroughs are shown in alphabetical order or on maps, never sorted by value:
the report describes the data and does not rank neighbourhoods.

Usage:
  .venv/bin/python scripts/make_report_figures.py
"""

import json
import re
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_final_dataset import INCIDENTS, REQUEST_NATURES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MOUNIR = DATA / "mounir" / "processed"
SHIRLEY = DATA / "shirley" / "processed"
FINAL = DATA / "final_dataset.csv"
FIG_DIR = ROOT / "report" / "figures"
TAB_DIR = ROOT / "report" / "tables"


SHORT = {
    "Ahuntsic-Cartierville": "AC", "Anjou": "AN", "Côte-des-Neiges-Notre-Dame-de-Grâce": "CDN-NDG",
    "L'Île-Bizard-Sainte-Geneviève": "IBSG", "LaSalle": "LS", "Lachine": "LC",
    "Le Plateau-Mont-Royal": "PMR", "Le Sud-Ouest": "SO", "Mercier-Hochelaga-Maisonneuve": "MHM",
    "Montréal-Nord": "MN", "Outremont": "OUT", "Pierrefonds-Roxboro": "PR",
    "Rivière-des-Prairies-Pointe-aux-Trembles": "RDP-PAT", "Rosemont-La Petite-Patrie": "RPP",
    "Saint-Laurent": "SLA", "Saint-Léonard": "SLE", "Verdun": "VER", "Ville-Marie": "VM",
    "Villeray-Saint-Michel-Parc-Extension": "VSP",
}

# label offsets (points) for the crowded lower-left of the scatter plot
LABEL_OFFSETS = {
    "PR": (-4, 0, "right"), "IBSG": (4, -3, "left"), "SLA": (-4, 2, "right"),
    "SLE": (-4, -1, "right"), "LS": (-4, -2, "right"), "AN": (0, 5, "center"),
    "RDP-PAT": (4, -5, "left"), "AC": (-1, -6, "center"), "CDN-NDG": (4, -3, "left"),
    "LC": (4, 1, "left"), "MN": (0, 5, "center"), "VSP": (3, 3, "left"),
    "SO": (-4, 3, "right"), "MHM": (4, -4, "left"), "PMR": (-4, 0, "right"), "VM": (-4, 0, "right"),
}

COL_W, PAGE_W = 3.3, 7.0
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def fmt(n):
    return f"{n:,}"


def tex_escape(s):
    return (s.replace("\\", r"\textbackslash{}").replace("&", r"\&").replace("%", r"\%")
            .replace("_", r"\_").replace("#", r"\#"))


def write_table(name, body):
    (TAB_DIR / name).write_text(body, encoding="utf-8")


# ---------------------------------------------------------------- figures

def fig_requests_over_time(requests):
    month = requests["created_at"].str[:7]
    by_type = requests.groupby([month, "group"]).size().unstack(fill_value=0)
    label_to_metric = {a: m for m, labels in INCIDENTS.items() for a in labels}
    metric = requests["ACTI_NOM"].map(label_to_metric)
    metric = metric.where(~metric.fillna("").str.startswith("cleanliness_"), "cleanliness")
    by_incident = requests.groupby([month, metric]).size().unstack(fill_value=0)
    idx = pd.to_datetime(by_type.index)

    fig, (a, b) = plt.subplots(2, 1, figsize=(COL_W, 3.6), sharex=True)
    for col, style in [("condition", "-"), ("service", "--"), ("administrative", ":")]:
        a.plot(idx, by_type[col] / 1000, style, color="black", lw=1, label=col.capitalize())
    a.set_ylabel("Requests per month (000s)")
    a.set_title("(a) Requests by type", loc="left")
    a.legend(frameon=False, ncol=3, loc="upper left")
    a.set_ylim(0, by_type.to_numpy().max() / 1000 * 1.35)
    for col, color in [("cleanliness", "#1b7837"), ("potholes", "#762a83"), ("rats", "#b35806")]:
        b.plot(by_incident.index.map(pd.Timestamp), by_incident[col], color=color, lw=1, label=col.capitalize())
    b.set_yscale("log")
    b.set_ylabel("Requests per month (log)")
    b.set_title("(b) Incident columns", loc="left")
    b.set_ylim(10, by_incident[["cleanliness", "potholes"]].to_numpy().max() * 8)
    b.legend(frameon=False, ncol=3, loc="upper left")
    b.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=[1, 7]))
    b.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "requests_over_time.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_borough_maps(table):
    shapes = gpd.read_file(DATA / "sebastien" / "raw" / "bourough-geo-coords.json")
    shapes["geometry"] = shapes.simplify(0.0003)
    boroughs = shapes[shapes["TYPE"] == "Arrondissement"].merge(
        table, left_on="CODEMAMH", right_on="code_mamh")
    linked = shapes[shapes["TYPE"] != "Arrondissement"]
    panels = [("rate_cleanliness_total", "Cleanliness requests", "Greens"),
              ("rate_potholes", "Pothole requests", "Purples"),
              ("rate_rats", "Rat requests", "Oranges"),
              ("rate_crimes_total", "Crimes", "Greys")]
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, 2.3))
    for ax, (col, title, cmap) in zip(axes, panels):
        linked.plot(ax=ax, color="#eeeeee", edgecolor="white", linewidth=0.3)
        boroughs.plot(column=col, ax=ax, cmap=cmap, edgecolor="black", linewidth=0.3, legend=True,
                      legend_kwds={"orientation": "horizontal", "shrink": 0.8, "pad": 0.02})
        ax.set_title(f"{title}\nper 1,000 residents per year")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "borough_maps.pdf", metadata={"CreationDate": None})
    plt.close(fig)


def fig_admin_share(table):
    """Office-flagged and administrative requests as a share of each borough's 311 requests."""
    t = table.sort_values("borough", ascending=False)
    office = t["requests_office_flagged"] / t["requests_311"] * 100
    admin = t["requests_administrative"] / t["requests_311"] * 100
    city = (t["requests_office_flagged"].sum() + t["requests_administrative"].sum()) / t["requests_311"].sum() * 100
    top = (office + admin).idxmax()
    fig, ax = plt.subplots(figsize=(COL_W, 3.2))
    ax.barh(t["borough"], office, color=["#b2182b" if i == top else "#555555" for i in t.index],
            label="Office-flagged")
    ax.barh(t["borough"], admin, left=office, color=["#f4a582" if i == top else "#bbbbbb" for i in t.index],
            label="Administrative")
    ax.axvline(city, color="black", lw=0.8, ls="--")
    ax.text(city + 0.3, len(t) - 0.6, f"city {city:.1f}%", fontsize=7)
    ax.set_xlabel("Share of 311 requests (%)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2, loc="upper center")
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG_DIR / "admin_share.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    return city


def fig_cleanliness_vs_crime(inc):
    fig, ax = plt.subplots(figsize=(COL_W, 2.8))
    ax.scatter(inc["rate_cleanliness_total"], inc["rate_crimes_total"], s=14, color="black")
    for _, r in inc.iterrows():
        label = SHORT[r["borough"]]
        dx, dy, ha = LABEL_OFFSETS.get(label, (4, -2, "left"))
        ax.annotate(label, (r["rate_cleanliness_total"], r["rate_crimes_total"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=6, ha=ha, va="center")
    ax.set_xlim(0, None)
    rho = inc["rate_cleanliness_total"].rank().corr(inc["rate_crimes_total"].rank())
    ax.set_xlabel("Cleanliness requests per 1,000 residents per year")
    ax.set_ylabel("Crimes per 1,000 residents per year")
    ax.text(0.03, 0.95, f"Spearman $\\rho$ = {rho:.2f}, n = 19", transform=ax.transAxes, va="top")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cleanliness_vs_crime.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    return rho


# ---------------------------------------------------------------- tables

def crime_report():
    return (SHIRLEY / "crimes_join_report.txt").read_text(encoding="utf-8")


def audit_311():
    return pd.read_csv(MOUNIR / "audit_2023_2025.csv", encoding="utf-8").set_index("metric")["value"]


def table_sources():
    manifest = {Path(s["file"]).name: s for s in
                json.loads((DATA / "sources_manifest.json").read_text(encoding="utf-8"))["sources"]}
    crime_rows = re.search(r"Rows in source file \(all years\): ([\d,]+)", crime_report()).group(1)
    build = pd.read_csv(MOUNIR / "build_audit.csv", encoding="utf-8").set_index("metric")["value"]
    archives = ["requetes311_2019-2021.csv", "requetes311_2016-2018.csv", "requetes311_2014-2016.csv"]
    archive_dates = sorted({manifest[f]["retrieved"] for f in archives})
    rows = [
        ("311 requests, 2022--26", "City",
         manifest["requetes311.csv"]["retrieved"], fmt(int(build["rows_read:requetes311.csv"]))),
        ("311 archives, 2014--21$^\\ddagger$", "City",
         " / ".join(archive_dates), fmt(sum(int(build[f"rows_read:{f}"]) for f in archives))),
        ("Crime, 2015--26", "SPVM",
         manifest["actes-criminels.csv"]["retrieved"], crime_rows),
        ("Boundaries", "City",
         manifest["bourough-geo-coords.json"]["retrieved"], "34"),
        ("Census, 2021", "StatCan", "2026-10-06", "19"),
        ("Population, 2023$^\\dagger$", "ISQ", "2026-10-06", "19"),
    ]
    body = "\n".join(f"    {a} & {b} & {c} & {d} \\\\" for a, b, c, d in rows)
    write_table("sources.tex", f"""\\begin{{tabular}}{{@{{}}llrr@{{}}}}
    \\toprule
    Source & Publisher & Retrieved & Records \\\\
    \\midrule
{body}
    \\bottomrule
\\end{{tabular}}
""")


def table_join_loss():
    a = audit_311().astype(str)

    def n(metric):
        return int(float(a[metric]))

    crime = crime_report()

    def grab(pattern):
        return int(re.search(pattern, crime).group(1).replace(",", ""))

    crime_in = grab(r"Rows in 2023-2025: ([\d,]+)")
    crime_noloc = grab(r"No usable location[^:]*: ([\d,]+)")
    crime_linked = grab(r"in a linked city[^:]*: ([\d,]+)")
    crime_kept = grab(r"KEPT in the 19 boroughs: ([\d,]+)")
    assert crime_in - crime_noloc - crime_linked == crime_kept

    raw, info = n("raw_rows"), n("raw_rows_Information")
    non_info, dropped = n("non_information_rows"), n("non_information_rows_dropped_by_borough_rule")
    cleaned, comments = n("cleaned_rows"), n("cleaned_rows_Commentaire")
    rp, office = n("requete_plainte_rows"), n("office_flag_rows")
    assert raw - info == non_info and non_info - dropped == cleaned and cleaned - comments == rp
    assert rp - office == n("rows_after_office_exclusion")
    rows = [
        ("311 Information", raw, info, "calls: no ID, no location"),
        ("311 borough rule", non_info, dropped,
         f"linked city {fmt(n('borough_rule_dropped_linked_municipality'))}; "
         f"no location {fmt(n('borough_rule_dropped_no_location'))}"),
        ("311 comments", cleaned, comments, "not a request"),
        ("311 office flag", rp, office, "logged at a borough office"),
        ("Crime", crime_in, crime_in - crime_kept,
         f"no usable location {fmt(crime_noloc)}; linked city {fmt(crime_linked)}"),
        ("Census, ISQ", 19, 0, "joined on borough code"),
    ]
    body = "\n".join(f"    {a} & {fmt(b)} & {fmt(c)} & {d} \\\\" for a, b, c, d in rows)
    write_table("join_loss.tex", f"""\\begin{{tabular}}{{@{{}}lrr>{{\\raggedright\\arraybackslash}}p{{2.6cm}}@{{}}}}
    \\toprule
    Step & In & Lost & Reason \\\\
    \\midrule
{body}
    \\bottomrule
\\end{{tabular}}
""")


def table_filtering():
    a = audit_311().astype(str)

    def n(metric):
        return int(float(a[metric]))

    crime = crime_report()

    def grab(pattern):
        return int(re.search(pattern, crime).group(1).replace(",", ""))

    build = pd.read_csv(MOUNIR / "build_audit.csv", encoding="utf-8").set_index("metric")["value"]
    file_rows = int(build["rows_read:requetes311.csv"])
    raw, info = n("raw_rows"), n("raw_rows_Information")
    non_info, dropped = n("non_information_rows"), n("non_information_rows_dropped_by_borough_rule")
    cleaned, comments = n("cleaned_rows"), n("cleaned_rows_Commentaire")
    rp, office = n("requete_plainte_rows"), n("office_flag_rows")
    kept_311 = n("rows_after_office_exclusion")
    assert raw - info == non_info and non_info - dropped == cleaned and cleaned - comments == rp
    assert rp - office == kept_311

    crime_all = grab(r"Rows in source file \(all years\): ([\d,]+)")
    crime_in = grab(r"Rows in 2023-2025: ([\d,]+)")
    crime_noloc = grab(r"No usable location[^:]*: ([\d,]+)")
    crime_linked = grab(r"in a linked city[^:]*: ([\d,]+)")
    crime_kept = grab(r"KEPT in the 19 boroughs: ([\d,]+)")
    assert crime_in - crime_noloc - crime_linked == crime_kept

    rows_311 = [
        ("Created outside 2023--25", file_rows - raw, raw),
        ("\\textit{Information} calls", info, non_info),
        ("Located outside the 19 boroughs", dropped, cleaned),
        ("Comments", comments, rp),
        ("Logged at a borough office", office, kept_311),
    ]
    rows_crime = [
        ("Reported outside 2023--25", crime_all - crime_in, crime_in),
        ("No usable coordinates", crime_noloc, crime_in - crime_noloc),
        ("Located in a linked city", crime_linked, crime_kept),
    ]

    def lines(rows):
        return "\n".join(f"    \\quad {a} & {fmt(b)} & {fmt(c)} \\\\" for a, b, c in rows)

    write_table("filtering.tex", f"""\\begin{{tabular}}{{@{{}}lrr@{{}}}}
    \\toprule
    Step & Dropped & Kept \\\\
    \\midrule
    \\textbf{{311 requests}}, current file & & {fmt(file_rows)} \\\\
{lines(rows_311)}
    \\midrule
    \\textbf{{Crime incidents}}, source file & & {fmt(crime_all)} \\\\
{lines(rows_crime)}
    \\bottomrule
\\end{{tabular}}
""")


def table_missingness():
    fields = {
        "ID_UNIQUE": "Request ID", "ACTI_NOM": "Category", "LOC_LAT": "Coordinates",
        "ARRONDISSEMENT": "Borough responsible", "ARRONDISSEMENT_GEO": "Borough of location",
        "TYPE_LIEU_INTERV": "Location type", "LOC_ERREUR_GDT": "Office flag",
        "PROVENANCE_ORIGINALE": "Channel", "DERNIER_STATUT": "Last status",
        "UNITE_RESP_PARENT": "Unit responsible",
    }
    miss = pd.read_csv(MOUNIR / "missingness_2023_2025.csv", encoding="utf-8")
    miss = miss[miss["stage"] == "raw_2023_2025"].set_index("field")
    crime_pct = float(re.search(r"No usable location.*?: [\d,]+ \(([\d.]+)%\)", crime_report()).group(1))
    rows = [f"    {label} & {miss.at[f, 'fraction_all_rows'] * 100:.1f} & "
            f"{miss.at[f, 'fraction_excluding_information'] * 100:.1f} \\\\"
            for f, label in fields.items()]
    rows.append(f"    Crime: coordinates & \\multicolumn{{2}}{{r}}{{{crime_pct:.1f}}} \\\\")
    write_table("missingness.tex", f"""\\begin{{tabular}}{{@{{}}lrr@{{}}}}
    \\toprule
    Field & All rows & Excl.\\ Information \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def table_population(table):
    pop = table.sort_values("borough")
    rows = [f"    {tex_escape(r.borough)} & {fmt(int(r.population_2021))} & "
            f"{fmt(int(r.population_2023_07_01))} & {r.population_change_2021_2023_pct:+.1f} \\\\"
            for r in pop.itertuples()]
    p21, p23 = int(pop["population_2021"].sum()), int(pop["population_2023_07_01"].sum())
    write_table("population.tex", f"""\\begin{{tabular}}{{@{{}}>{{\\raggedright\\arraybackslash}}p{{3.8cm}}rrr@{{}}}}
    \\toprule
    Borough & 2021 & 2023 & \\% \\\\
    \\midrule
{chr(10).join(rows)}
    \\midrule
    Total & {fmt(p21)} & {fmt(p23)} & {(p23 / p21 - 1) * 100:+.1f} \\\\
    \\bottomrule
\\end{{tabular}}
""")


def table_incidents(inc):
    names = {
        "cleanliness_illegal_dumping": "Illegal dumping",
        "cleanliness_public_cleaning": "Public cleaning",
        "cleanliness_graffiti": "Graffiti",
        "cleanliness_debris": "Debris on roads",
        "rats": "Rats",
        "potholes": "Potholes",
    }
    rows = []
    for metric, label in names.items():
        cats = INCIDENTS[metric]
        shown = ", ".join(f"\\textit{{{tex_escape(c)}}}" for c in cats[:2])
        if len(cats) > 2:
            shown += f" + {len(cats) - 2} more"
        rows.append(f"    {label} & {shown} & {fmt(int(inc[metric].sum()))} \\\\")
        if metric == "cleanliness_debris":
            rows.append(f"    \\textbf{{Cleanliness}} & sum of the four above & "
                        f"\\textbf{{{fmt(int(inc['cleanliness_total'].sum()))}}} \\\\")
            rows.append("    \\midrule")
    write_table("incidents.tex", f"""\\begin{{tabular}}{{@{{}}l>{{\\raggedright\\arraybackslash}}p{{4.3cm}}r@{{}}}}
    \\toprule
    Column & 311 categories (\\texttt{{ACTI\\_NOM}}) & Requests \\\\
    \\midrule
{chr(10).join(rows)}
    \\bottomrule
\\end{{tabular}}
""")


def load_requests():
    """Mounir's outcome base: Requete + Plainte rows in the 19 boroughs, office-flagged rows excluded."""
    requests = pd.read_csv(MOUNIR / "requests_311_2023_2025.csv",
                           usecols=["NATURE", "ACTI_NOM", "LOC_ERREUR_GDT", "created_at"], dtype=str)
    requests = requests[requests["NATURE"].isin(REQUEST_NATURES) & requests["LOC_ERREUR_GDT"].ne("1")]
    mapping = pd.read_csv(MOUNIR / "acti_nom_mapping.csv", usecols=["ACTI_NOM", "group"], dtype=str)
    return requests.merge(mapping, on="ACTI_NOM", how="left", validate="m:1")


def main():
    for path in (FINAL, MOUNIR / "requests_311_2023_2025.csv", MOUNIR / "audit_2023_2025.csv"):
        if not path.exists():
            raise SystemExit(f"missing {path.relative_to(ROOT)}; run the pipeline first (see README)")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(FINAL)
    requests = load_requests()

    fig_requests_over_time(requests)
    fig_borough_maps(table)
    city_admin = fig_admin_share(table)
    rho = fig_cleanliness_vs_crime(table)
    table_sources()
    table_join_loss()
    table_filtering()
    table_missingness()
    table_incidents(table)
    table_population(table)

    print(f"figures -> {FIG_DIR.relative_to(ROOT)}, tables -> {TAB_DIR.relative_to(ROOT)}")
    print(f"city office-flagged + administrative share {city_admin:.1f}%, "
          f"cleanliness vs crime rho {rho:.2f}, outcome-base rows {len(requests):,}")


if __name__ == "__main__":
    main()
