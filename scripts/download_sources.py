"""Download the raw City of Montreal sources used by the project.

  311 service requests (Mounir)     data/mounir/raw/      current file + 2014-2021 archives
  Crime incidents, SPVM (Shirley)   data/shirley/raw/
  Borough boundaries (Sebastien)    data/sebastien/raw/

The census (Adil, data/adil/raw/) and the ISQ population estimate
(data/sebastien/raw/) are committed, not downloaded.

Files that already exist are skipped (the 311 files total ~2 GB); pass
--force to re-download. Every run rewrites data/sources_manifest.json with the
URL, retrieval date, licence, and size of each file, since the City replaces
its files without notice.

The 2017-2018 311 archive is not downloaded: the 2016-2018 archive covers the
same years and is the copy Mounir's pipeline reads (scripts/mounir/common.py).

Usage:
  .venv/bin/python scripts/download_sources.py
  .venv/bin/python scripts/download_sources.py --force
"""

import argparse
import json
import shutil
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MANIFEST = DATA / "sources_manifest.json"

# donnees.montreal.ca rejects requests without a browser user agent.
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) COMP570-project"

CKAN = "https://donnees.montreal.ca/dataset"
CC_BY = "CC BY 4.0"
REQ_311 = f"{CKAN}/5866f832-676d-4b07-be6a-e99c21eb17e4/resource"


def source_311(file_name, resource, years):
    return {
        "path": f"mounir/raw/{file_name}",
        "source": f"311 service requests (Requetes 3-1-1, {years})",
        "publisher": "Ville de Montreal",
        "page": f"{CKAN}/requete-311",
        "url": f"{REQ_311}/{resource}/download/{file_name}",
        "licence": CC_BY,
    }


SOURCES = [
    source_311("requetes311.csv", "2cfa0e06-9be4-49a6-b7f1-ee9f2363a872", "2022 to present"),
    source_311("requetes311_2019-2021.csv", "dbfc05f8-b939-4639-ae52-2e77f738e43f", "archive 2019-2021"),
    source_311("requetes311_2016-2018.csv", "dbc02208-907c-46ff-a052-e4c42042d327", "archive 2016-2018"),
    source_311("requetes311_2014-2016.csv", "f62595b0-b26a-4e7b-ba67-05b84521ab64", "archive 2014-2016"),
    {
        "path": "shirley/raw/actes-criminels.csv",
        "source": "Criminal acts reported to the SPVM (Actes criminels)",
        "publisher": "Ville de Montreal / SPVM",
        "page": f"{CKAN}/actes-criminels",
        "url": f"{CKAN}/5829b5b0-ea6f-476f-be94-bc2b8797769a/resource/"
        "c6f482bf-bf0f-4960-8b2f-9982c211addd/download/actes-criminels.csv",
        "licence": CC_BY,
    },
    {
        "path": "sebastien/raw/bourough-geo-coords.json",
        "source": "Borough and linked-city boundaries, WGS 84 "
        "(Limites administratives de l'agglomeration)",
        "publisher": "Ville de Montreal",
        "page": f"{CKAN}/limites-administratives-agglomeration",
        "url": f"{CKAN}/9797a946-9da8-41ec-8815-f6b276dec7e9/resource/"
        "e18bfd07-edc8-4ce8-8a5a-3b617662a794/download/"
        "limites-administratives-agglomeration.geojson",
        "licence": CC_BY,
    },
]


def fetch(url, dest, timeout=3600, retries=3):
    """Stream url to dest through a temporary file, so a failed download leaves no partial file."""
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            tmp.replace(dest)
            return
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            print(f"  retry {attempt} after error: {exc}")
            time.sleep(5 * attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    # keyed by file name so dates survive the files moving between folders
    retrieved = {Path(e["file"]).name: e.get("retrieved") for e in old.get("sources", [])}
    today = date.today().isoformat()

    entries = []
    for meta in SOURCES:
        path = DATA / meta["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not args.force:
            print(f"skip {meta['path']} (exists)")
            mtime = date.fromtimestamp(path.stat().st_mtime).isoformat()
            when = retrieved.get(path.name) or mtime
        else:
            print(f"download {meta['path']}")
            fetch(meta["url"], path)
            when = today
        entries.append({
            "file": f"data/{meta['path']}",
            "source": meta["source"],
            "publisher": meta["publisher"],
            "url": meta["url"],
            "page": meta["page"],
            "licence": meta["licence"],
            "bytes": path.stat().st_size,
            "retrieved": when,
        })

    MANIFEST.write_text(json.dumps({"sources": entries}, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
