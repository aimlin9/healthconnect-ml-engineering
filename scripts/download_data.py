"""Download the 27 HealthConnect Rwanda Week 6 source datasets listed in
data/dataset_manifest.json into data/raw/.

Drive files are pulled with gdown (handles the large-file virus-scan
interstitial). The two multi-tab Google Sheets workbooks (Health Facilities,
Clinical Transactions) are exported as .xlsx and later split per-sheet during
cleaning, since each tab is a distinct entity/table.
"""
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

import gdown
import requests

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "dataset_manifest.json"
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def slugify(name: str) -> str:
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def download_drive_file(file_id: str, dest_stub: Path) -> Path:
    out = gdown.download(id=file_id, output=str(dest_stub) + ".download", quiet=False)
    if out is None:
        raise RuntimeError(f"gdown failed for file_id={file_id}")
    out_path = Path(out)
    # Sniff real type: CSV text vs. Excel zip signature
    with open(out_path, "rb") as f:
        head = f.read(8)
    if head[:2] == b"PK":
        final = dest_stub.with_suffix(".xlsx")
    else:
        final = dest_stub.with_suffix(".csv")
    out_path.rename(final)
    return final


def download_sheet(file_id: str, dest_stub: Path) -> Path:
    url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=xlsx"
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    final = dest_stub.with_suffix(".xlsx")
    final.write_bytes(r.content)
    return final


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    results = []
    for entry in manifest:
        name = entry["dataset"]
        slug = slugify(name)
        stub = RAW_DIR / slug
        existing = list(RAW_DIR.glob(slug + ".*"))
        if existing:
            print(f"[skip] {name} already downloaded -> {existing[0].name}")
            results.append({**entry, "local_path": str(existing[0].relative_to(ROOT))})
            continue
        print(f"[get ] {name} ({entry['kind']}, id={entry['file_id']})")
        try:
            if entry["kind"] == "sheet":
                path = download_sheet(entry["file_id"], stub)
            else:
                path = download_drive_file(entry["file_id"], stub)
            print(f"       -> {path.relative_to(ROOT)} ({path.stat().st_size/1024:.1f} KB)")
            results.append({**entry, "local_path": str(path.relative_to(ROOT))})
        except Exception as e:
            print(f"       !! FAILED: {e}", file=sys.stderr)
            results.append({**entry, "local_path": None, "error": str(e)})
        time.sleep(1)

    (RAW_DIR / "_download_results.json").write_text(json.dumps(results, indent=2))
    failed = [r for r in results if not r.get("local_path")]
    print(f"\nDone. {len(results)-len(failed)}/{len(results)} succeeded.")
    if failed:
        print("Failed:", [r["dataset"] for r in failed])


if __name__ == "__main__":
    main()
