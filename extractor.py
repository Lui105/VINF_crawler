
import re
import sys
import json
import html
import argparse
from pathlib import Path
import csv

import unicodedata

RE_S = re.S | re.I

def read_html(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def strip_tags(s: str) -> str:
    s = re.sub(r"<script\b.*?</script>", "", s, flags=RE_S)
    s = re.sub(r"<style\b.*?</style>", "", s, flags=RE_S)
    s = re.sub(r"<!--.*?-->", "", s, flags=RE_S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    return s.strip()

def extract_name(doc: str):
    m = re.search(r'"@type"\s*:\s*"Person"[\s\S]*?"name"\s*:\s*"([^"]+)"', doc, RE_S)
    if m: return html.unescape(m.group(1)).strip()
    m = re.search(r"<h1[^>]*>(.*?)</h1>", doc, RE_S)
    if m: return strip_tags(m.group(1)).split("|")[0].strip()
    m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', doc, RE_S)
    if m: return m.group(1).split("Stats")[0].strip(" |-")
    return None


def extract_teams_played_from_totals(tables: list[dict]) -> list[str]:
    target = None
    for t in tables:
        h = (t.get("headline") or "").strip().lower()
        if h.startswith("totals table") and "playoffs" not in h:
            target = t
            break
    if not target:
        return []
    header = target.get("header") or []
    body = target.get("body") or []
    team_col = None
    for i, col in enumerate(header):
        if (col or "").strip().lower() in {"team", "tm"}:
            team_col = i
            break
    if team_col is None:
        return []
    seen = set()
    ordered = []
    for row in body:
        if team_col >= len(row):
            continue
        raw = (row[team_col] or "").strip()
        if not raw:
            continue
        m = re.match(r"^([A-Za-z]{2,4})(?:\b|[^A-Za-z])", raw)
        if not m:
            continue
        code = m.group(1).upper()
        if code in {"TOT", "2TM"}:
            continue
        if code not in seen:
            seen.add(code)
            ordered.append(code)

    return ordered






def clean(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = normalize_spaces(s)
    return re.sub(r"\s+", " ", s).strip()

def balanced_div_inner(html_doc: str, open_pos: int) -> str:
    m_open_tag = re.match(r"<div\b[^>]*>", html_doc[open_pos:], flags=RE_S | re.I)
    if not m_open_tag:
        return None
    i = open_pos + m_open_tag.end()
    depth = 1
    tag_re = re.compile(r"</?div\b[^>]*>", re.I)
    while True:
        m = tag_re.search(html_doc, i)
        if not m:
            return None
        if html_doc[m.start():m.end()].lower().startswith("</div"):
            depth -= 1
            if depth == 0:
                return html_doc[open_pos + m_open_tag.end(): m.start()]
        else:
            depth += 1
        i = m.end()

def extract_summary_table(doc: str) -> dict:

    m_start = re.search(r'<div[^>]*class="[^"]*\bstats_pullout\b[^"]*"[^>]*>', doc, RE_S)
    if not m_start:
        return None

    inner = balanced_div_inner(doc, m_start.start())
    if inner is None:
        return None

    rows = []

    for div_html in re.findall(r"<div[^>]*>(.*?)</div>", inner, RE_S):
        m_pair = re.search(
            r"<strong[^>]*>(.*?)</strong>.*?<p[^>]*>(?:\s*|&nbsp;|&#160;)*</p>\s*<p[^>]*>(.*?)</p>",
            div_html, RE_S
        )
        if not m_pair:
            m_pair = re.search(
                r"<strong[^>]*>(.*?)</strong>.*?<p[^>]*>(.*?)</p>",
                div_html, RE_S
            )
        if not m_pair:
            continue

        label = clean(m_pair.group(1))
        value = clean(m_pair.group(2))
        if not label or not value or label.upper() == "SUMMARY" or value.upper() == "CAREER":
            continue

        rows.append([label, value])

    if not rows:
        return None

    return {
        "id": "summary",
        "headline": "Summary",
        "header": ["Stat", "Value"],
        "body": rows
    }



def strip_tags(s: str) -> str:
    s = re.sub(r"<script\b.*?</script>", "", s, flags=RE_S)
    s = re.sub(r"<style\b.*?</style>", "", s, flags=RE_S)
    s = re.sub(r"<!--.*?-->", "", s, flags=RE_S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = normalize_spaces(s)
    return s.strip()

NBSP_CHARS = {
    "\u00A0",
    "\u202F",
    "\u2007",
    "\u2009",
    "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
}

def normalize_spaces(s: str) -> str:
    if not s:
        return s
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2007", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def find_balanced_block(html_doc: str, open_pos: int) -> tuple[int, int]:

    m_open = re.match(r"<div\b[^>]*>", html_doc[open_pos:], flags=RE_S)
    if not m_open:
        return None
    start = open_pos
    i = open_pos + m_open.end()
    depth = 1
    tag_re = re.compile(r"</?div\b[^>]*>", re.I)
    while True:
        m = tag_re.search(html_doc, i)
        if not m:
            return None
        if html_doc[m.start():m.end()].lower().startswith("</div"):
            depth -= 1
            if depth == 0:
                end = m.end()
                return (start, end)
        else:
            depth += 1
        i = m.end()

def remove_media_item_blocks(meta_inner: str) -> str:

    out = meta_inner
    while True:
        m = re.search(r'<div[^>]*class="[^"]*\bmedia-item\b[^"]*"[^>]*>', out, RE_S)
        if not m:
            break
        bounds = find_balanced_block(out, m.start())
        if not bounds:
            break
        s, e = bounds
        out = out[:s] + out[e:]
    return out

def first_child_div_inner(html_fragment: str) -> str:
    m = re.search(r"<div\b[^>]*>", html_fragment, RE_S)
    if not m:
        return None
    bounds = find_balanced_block(html_fragment, m.start())
    if not bounds:
        return None
    s, e = bounds
    m_open = re.match(r"<div\b[^>]*>", html_fragment[s:], flags=RE_S)
    if not m_open:
        return None
    inner_start = s + m_open.end()
    inner = html_fragment[inner_start:e-6]
    return inner

def extract_personal_info_from_meta_inner(meta_inner: str) -> str:
    cleaned = remove_media_item_blocks(meta_inner)
    content = first_child_div_inner(cleaned)
    if content is None:
        content = cleaned
    return strip_tags(content)

def find_meta_inner(doc: str) -> str:
    m = re.search(r'<div[^>]*id="meta"[^>]*>', doc, RE_S)
    if m:
        bounds = find_balanced_block(doc, m.start())
        if bounds:
            s, e = bounds
            m_open = re.match(r"<div\b[^>]*>", doc[s:], flags=RE_S)
            if m_open:
                return doc[s + m_open.end(): e-6]

    for com in re.findall(r"<!--([\s\S]*?)-->", doc, RE_S):
        m2 = re.search(r'<div[^>]*id="meta"[^>]*>', com, RE_S)
        if m2:
            bounds2 = find_balanced_block(com, m2.start())
            if bounds2:
                s2, e2 = bounds2
                m_open2 = re.match(r"<div\b[^>]*>", com[s2:], flags=RE_S)
                if m_open2:
                    return com[s2 + m_open2.end(): e2-6]
    return None

def extract_meta_block_text(doc: str) -> str:
    meta_inner = find_meta_inner(doc)
    if not meta_inner:
        return None
    return extract_personal_info_from_meta_inner(meta_inner)




def extract_position(meta_text: str) -> str:
    if not meta_text:
        return None
    s = normalize_spaces(meta_text)

    sep = r"(?:[•▪\|/])"

    m = re.search(
        r"\bPosition\s*:\s*(.+?)\s*(?=(?:Shoots\s*:|Born\s*:|Team\s*:|College\s*:|Draft\s*:|NBA Debut\s*:|Experience\s*:|"
        + sep + r"|$))",
        s, re.I
    )
    if not m:
        return None

    pos = m.group(1).strip(" -/|")
    pos = re.split(r"\s*/\s*|\s*,\s*|\s*\|\s*", pos)[0].strip()
    return pos or None

def extract_jersey_numbers(doc: str):
    nums = set()
    for mm in re.finditer(r'<svg[^>]*class="[^"]*jersey[^"]*"[^>]*>(.*?)</svg>', doc, RE_S):
        inner = mm.group(1)
        t = re.search(r"<text[^>]*>(.*?)</text>", inner, RE_S)
        if t:
            val = strip_tags(t.group(1))
            for d in re.findall(r"\b\d{1,3}\b", val):
                nums.add(str(int(d)))
    if not nums:
        for m in re.finditer(r"\b(?:No\.?|Number)\s*[:#]\s*(\d{1,3})\b", doc, RE_S):
            nums.add(m.group(1))
    return sorted(nums, key=lambda x: int(x)) if nums else []

def unhide_commented_tables(doc: str) -> str:
    def repl(m):
        block = m.group(0)
        if re.search(r"<table\b", block, RE_S):
            return re.sub(r"^<!--\s*|\s*-->$", "", block, flags=RE_S)
        return block
    return re.sub(r"<!--[\s\S]*?-->", repl, doc)

def extract_tables(doc: str):
    d2 = unhide_commented_tables(doc)
    out = []
    for tm in re.finditer(r'(<table\b[^>]*id="([^"]+)"[^>]*>[\s\S]*?</table>)', d2, RE_S):
        full_html = tm.group(1)
        table_id = tm.group(2)

        headline = None
        cap = re.search(r"<caption[^>]*>([\s\S]*?)</caption>", full_html, RE_S)
        if cap:
            headline = strip_tags(cap.group(1)).strip()
        if not headline:
            start = tm.start()
            back = d2[max(0, start-2500):start]
            sec = re.search(r'<div[^>]*class="[^"]*section_heading[^"]*"[^>]*>[\s\S]*?<span[^>]*>(.*?)</span>', back, RE_S)
            if sec:
                headline = strip_tags(sec.group(1)).strip()
        if not headline:
            headline = table_id.replace("_", " ").title()

        rows = []
        for rm in re.finditer(r"<tr[^>]*>([\s\S]*?)</tr>", full_html, RE_S):
            row_html = rm.group(1)
            cells = re.findall(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", row_html, RE_S)
            if cells:
                rows.append([strip_tags(c).replace("\n", " ").strip() for c in cells])

        header = None
        body = rows
        thead = re.search(r"<thead[^>]*>([\s\S]*?)</thead>", full_html, RE_S)
        if thead:
            hcells = re.findall(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", thead.group(1), RE_S)
            if hcells:
                header = [strip_tags(c).replace("\n", " ").strip() for c in hcells]
                if rows and len(rows[0]) == len(header):
                    body = rows[1:]
        if header is None and rows:
            if all(not re.search(r"\d", c or "") for c in rows[0]):
                header = rows[0]
                body = rows[1:]
        out.append({"id": table_id, "headline": headline, "header": header, "body": body})
    return out

def first5_parts_from_stem(stem: str):

    parts = [p.strip().lower() for p in stem.split("-")]
    parts = [p for p in parts if p != ""]
    return parts[:5]

def disambiguate_headlines(tables: list) -> list:
    seen = {}
    for t in tables:
        h = t["headline"]
        count = seen.get(h, 0)
        if count == 1:
            t["headline"] = f"{h} Playoffs"
        seen[h] = count + 1
    return tables


def parse_top_metadata_block(path: Path) -> dict:
    meta = {}
    began = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("\ufeff#"):
                s = s[1:]

            if s.upper() == "# METADATA BEGIN":
                began = True
                continue
            if s.upper() == "# METADATA END":
                break

            if not began:
                if not s.startswith("#"):
                    break

            if s.startswith("#"):
                body = s.lstrip("#").strip()
                if ":" in body:
                    key, _, value = body.partition(":")
                    key = key.strip().lower()
                    value = value.strip()
                    meta[key] = value
            else:
                break
    return meta

def load_side_metadata(meta_dir: Path, html_stem: str) -> dict:
    if not meta_dir or not meta_dir.exists():
        return {}

    target_prefix = first5_parts_from_stem(html_stem)
    if not target_prefix:
        return {}

    candidates = []
    for p in meta_dir.glob("*.tsv"):
        if first5_parts_from_stem(p.stem) == target_prefix:
            candidates.append(p)

    if not candidates:
        return {}

    candidates.sort(key=lambda p: (len(p.name), p.name.lower()))
    path = candidates[0]

    data = parse_top_metadata_block(path)
    return {
        "source_url": data.get("source_url", ""),
        "crawl_depth": data.get("crawl_depth", ""),
        "timestamp":  data.get("timestamp", ""),
        "_meta_file": str(path),
    }


def write_single_tsv(tsv_path: Path, meta: dict, tables: list):
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["# name", meta.get("name","")])
        w.writerow(["# position", meta.get("position","")])
        w.writerow(["# jersey_numbers", ",".join(meta.get("jersey_numbers", []))])
        w.writerow(["# personal_info_text", (meta.get("personal_info_text","").replace("\t"," ").replace("\n"," / "))])
        w.writerow(["# teams_played", ",".join(meta.get("teams_played", []))])
        w.writerow(["# source_url", meta.get("source_url","")])
        w.writerow(["# crawl_depth", meta.get("crawl_depth","")])
        w.writerow(["# timestamp", meta.get("timestamp","")])
        w.writerow([])

        for t in tables:
            w.writerow(["# TABLE", t["id"], t["headline"]])
            width = 0
            if t["header"]:
                width = max(width, len(t["header"]))
            for r in t["body"]:
                width = max(width, len(r))
            def norm(row):
                return (row or []) + [""] * (width - len(row or []))
            if t["header"]:
                w.writerow(norm(t["header"]))
            for r in t["body"]:
                w.writerow(norm(r))
            w.writerow([])

def process_file(fp: Path, dest_dir: Path, meta_dir: Path):
    doc = read_html(fp)
    name = extract_name(doc)
    meta_text = extract_meta_block_text(doc)
    position = extract_position(meta_text or "")
    jerseys = extract_jersey_numbers(doc)
    tables = extract_tables(doc)
    tables = disambiguate_headlines(tables)

    teams_played = extract_teams_played_from_totals(tables)

    summary_table = extract_summary_table(doc)
    if summary_table:
        tables = [summary_table] + tables

    side = load_side_metadata(meta_dir, fp.stem) if meta_dir else {}

    meta = {
        "name": name,
        "position": position,
        "jersey_numbers": jerseys,
        "personal_info_text": meta_text or "",
        "teams_played": teams_played,
        "source_url": side.get("source_url",""),
        "crawl_depth": side.get("crawl_depth",""),
        "timestamp": side.get("timestamp",""),
    }

    tsv_out = dest_dir / (fp.stem + ".tsv")
    write_single_tsv(tsv_out, meta, tables)



def main():
    ap = argparse.ArgumentParser(description="Extract BBR player data (regex parsing) into one TSV per page; metadata matched by first 5 hyphen-split parts of filename.")
    ap.add_argument("--src", required=True, help="Source folder with .html files")
    ap.add_argument("--dest", required=True, help="Destination folder for TSV files")
    ap.add_argument("--meta", required=True, help="Directory with sidecar TSVs")
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.dest)
    meta_dir = Path(args.meta)

    if not src.exists() or not src.is_dir():
        print(f"ERROR: src folder does not exist or is not a directory: {src}", file=sys.stderr)
        sys.exit(2)
    if not meta_dir.exists() or not meta_dir.is_dir():
        print(f"ERROR: meta folder does not exist or is not a directory: {meta_dir}", file=sys.stderr)
        sys.exit(2)

    pattern = "*.html"
    html_files = sorted(src.glob(pattern))
    if not html_files:
        print("No .html files found.", file=sys.stderr)
        sys.exit(0)
    count = 0
    for fp in html_files:
        try:
            process_file(fp, dest, meta_dir)
            count += 1
            if count % 100 == 0:
                print(f"Processed file number {count}")
        except Exception as e:
            print(json.dumps({"file": str(fp), "error": str(e)}), file=sys.stderr)

if __name__ == "__main__":
    main()
