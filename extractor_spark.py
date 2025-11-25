import os
import sys
import json
import html
import argparse
from pathlib import Path
import csv
import unicodedata
import re

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StringType,
    ArrayType,
    MapType,
    StructType,
    StructField,
)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable



RE_S = re.S | re.I

NBSP_CHARS = {
    "\u00A0", "\u202F", "\u2007", "\u2009",
    "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
}


def normalize_spaces_py(s: str) -> str:
    if not s:
        return s
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    for ch in NBSP_CHARS:
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def strip_tags_py(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<script\b.*?</script>", "", s, flags=RE_S)
    s = re.sub(r"<style\b.*?</style>", "", s, flags=RE_S)
    s = re.sub(r"<!--.*?-->", "", s, flags=RE_S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", "\n", s)
    s = normalize_spaces_py(s)
    return s.strip()




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


def find_balanced_block(html_doc: str, open_pos: int):
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
    inner = html_fragment[inner_start:e - 6]
    return inner


def extract_personal_info_from_meta_inner(meta_inner: str) -> str:
    cleaned = remove_media_item_blocks(meta_inner)
    content = first_child_div_inner(cleaned)
    if content is None:
        content = cleaned
    return strip_tags_py(content)


def find_meta_inner(doc: str) -> str:
    m = re.search(r'<div[^>]*id="meta"[^>]*>', doc, RE_S)
    if m:
        bounds = find_balanced_block(doc, m.start())
        if bounds:
            s, e = bounds
            m_open = re.match(r"<div\b[^>]*>", doc[s:], flags=RE_S)
            if m_open:
                return doc[s + m_open.end(): e - 6]

    for com in re.findall(r"<!--([\s\S]*?)-->", doc, RE_S):
        m2 = re.search(r'<div[^>]*id="meta"[^>]*>', com, RE_S)
        if m2:
            bounds2 = find_balanced_block(com, m2.start())
            if bounds2:
                s2, e2 = bounds2
                m_open2 = re.match(r"<div\b[^>]*>", com[s2:], flags=RE_S)
                if m_open2:
                    return com[s2 + m_open2.end(): e2 - 6]
    return None


def extract_meta_block_text_py(doc: str) -> str:
    meta_inner = find_meta_inner(doc)
    if not meta_inner:
        return None
    return extract_personal_info_from_meta_inner(meta_inner)


def extract_summary_table_py(doc: str):
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

        label = strip_tags_py(m_pair.group(1))
        value = strip_tags_py(m_pair.group(2))
        if not label or not value or label.upper() == "SUMMARY" or value.upper() == "CAREER":
            continue
        rows.append([label, value])

    if not rows:
        return None

    return {
        "id": "summary",
        "headline": "Summary",
        "header": ["Stat", "Value"],
        "body": rows,
    }


def unhide_commented_tables_py(doc: str) -> str:
    def repl(m):
        block = m.group(0)
        if re.search(r"<table\b", block, RE_S):
            return re.sub(r"^<!--\s*|\s*-->$", "", block, flags=RE_S)
        return block

    return re.sub(r"<!--[\s\S]*?-->", repl, doc)


def extract_tables_py(doc: str):
    d2 = unhide_commented_tables_py(doc)
    out = []
    for tm in re.finditer(r'(<table\b[^>]*id="([^"]+)"[^>]*>[\s\S]*?</table>)', d2, RE_S):
        full_html = tm.group(1)
        table_id = tm.group(2)

        headline = None
        cap = re.search(r"<caption[^>]*>([\s\S]*?)</caption>", full_html, RE_S)
        if cap:
            headline = strip_tags_py(cap.group(1)).strip()
        if not headline:
            start = tm.start()
            back = d2[max(0, start - 2500):start]
            sec = re.search(
                r'<div[^>]*class="[^"]*section_heading[^"]*"[^>]*>[\s\S]*?<span[^>]*>(.*?)</span>',
                back, RE_S
            )
            if sec:
                headline = strip_tags_py(sec.group(1)).strip()
        if not headline:
            headline = table_id.replace("_", " ").title()

        rows = []
        for rm in re.finditer(r"<tr[^>]*>([\s\S]*?)</tr>", full_html, RE_S):
            row_html = rm.group(1)
            cells = re.findall(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", row_html, RE_S)
            if cells:
                rows.append([strip_tags_py(c).replace("\n", " ").strip() for c in cells])

        header = None
        body = rows
        thead = re.search(r"<thead[^>]*>([\s\S]*?)</thead>", full_html, RE_S)
        if thead:
            hcells = re.findall(r"<t[hd][^>]*>([\s\S]*?)</t[hd]>", thead.group(1), RE_S)
            if hcells:
                header = [strip_tags_py(c).replace("\n", " ").strip() for c in hcells]
                if rows and len(rows[0]) == len(header):
                    body = rows[1:]

        if header is None and rows:
            if all(not re.search(r"\d", c or "") for c in rows[0]):
                header = rows[0]
                body = rows[1:]

        out.append({"id": table_id, "headline": headline, "header": header, "body": body})
    return out


def disambiguate_headlines_py(tables: list) -> list:
    seen = {}
    for t in tables:
        h = t["headline"]
        count = seen.get(h, 0)
        if count == 1:
            t["headline"] = f"{h} Playoffs"
        seen[h] = count + 1
    return tables

def extract_jersey_numbers_py(doc: str):
    nums = set()
    for mm in re.finditer(r'<svg[^>]*class="[^"]*jersey[^"]*"[^>]*>(.*?)</svg>', doc, RE_S):
        inner = mm.group(1)
        t = re.search(r"<text[^>]*>(.*?)</text>", inner, RE_S)
        if t:
            val = strip_tags_py(t.group(1))
            for d in re.findall(r"\b\d{1,3}\b", val):
                nums.add(str(int(d)))
    if not nums:
        for m in re.finditer(r"\b(?:No\.?|Number)\s*[:#]\s*(\d{1,3})\b", doc, RE_S):
            nums.add(m.group(1))
    return sorted(nums, key=lambda x: int(x)) if nums else []


def extract_teams_played_from_totals_py(tables: list) -> list[str]:
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


def first5_parts_from_stem(stem: str):
    parts = [p.strip().lower() for p in stem.split("-")]
    parts = [p for p in parts if p != ""]
    return parts[:5]


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
        "timestamp": data.get("timestamp", ""),
        "_meta_file": str(path),
    }




def compute_name_columns(df):
    json_name_pattern = r'"@type"\s*:\s*"Person"[\s\S]*?"name"\s*:\s*"([^"]+)"'
    df = df.withColumn("name_json", F.regexp_extract("html", json_name_pattern, 1))

    h1_pattern = r"<h1[^>]*>(.*?)</h1>"
    df = df.withColumn("h1_html", F.regexp_extract("html", h1_pattern, 1))

    df = df.withColumn(
        "h1_text",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace("h1_html", r"<script\b.*?</script>", ""),
                    r"<[^>]+>", " "
                ),
                r"\s+", " "
            )
        )
    )

    og_pattern = r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"'
    df = df.withColumn("og_title", F.regexp_extract("html", og_pattern, 1))

    df = df.withColumn(
        "name",
        F.when(F.length(F.col("name_json")) > 0, F.trim(F.col("name_json")))
        .when(F.length(F.col("h1_text")) > 0,
              F.trim(F.element_at(F.split(F.col("h1_text"), r"\|"), 1)))
        .when(F.length(F.col("og_title")) > 0,
              F.trim(
                  F.regexp_replace(
                      F.regexp_replace(F.col("og_title"), r"Stats.*$", ""),
                      r"[-\|]+$", ""
                  )
              ))
        .otherwise(F.lit(None))
    )
    return df



def extract_position_spark(df):

    df = df.withColumn("personal_info_text", udf_extract_meta_block_text("html"))

    position_pattern = r"Position\s*:\s*([^•▪\|/]+?)(?=\s*(?:Shoots|Born|Team|College|Draft|NBA Debut|Experience|•|▪|\||/|$))"

    df = df.withColumn(
        "position_raw",
        F.regexp_extract("personal_info_text", position_pattern, 1)
    )

    df = df.withColumn(
        "position",
        F.when(
            F.length("position_raw") > 0,
            F.trim(
                F.element_at(
                    F.split(
                        F.regexp_replace("position_raw", r"[-/\|]+$", ""),
                        r"\s*/\s*|\s*,\s*|\s*\|\s*"
                    ),
                    1
                )
            )
        ).otherwise(F.lit(None))
    )

    return df


@F.udf(StringType())
def udf_extract_meta_block_text(doc: str):
    return extract_meta_block_text_py(doc) or ""


@F.udf(StringType())
def udf_tables_full(doc: str):
    tables = extract_tables_py(doc)
    tables = disambiguate_headlines_py(tables)
    summary = extract_summary_table_py(doc)
    if summary:
        tables = [summary] + tables
    return json.dumps(tables, ensure_ascii=False)


@F.udf(ArrayType(StringType()))
def udf_teams_played(tables_json: str):
    try:
        tables = json.loads(tables_json)
    except Exception:
        return []
    return extract_teams_played_from_totals_py(tables)

@F.udf(ArrayType(StringType()))
def udf_extract_jersey_numbers(doc: str):
    return extract_jersey_numbers_py(doc)




def write_single_tsv(tsv_path: Path, meta: dict, tables: list):
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["# name", meta.get("name", "")])
        w.writerow(["# position", meta.get("position", "")])
        w.writerow(["# jersey_numbers", ",".join(meta.get("jersey_numbers", []))])
        w.writerow([
            "# personal_info_text",
            (meta.get("personal_info_text", "").replace("\t", " ").replace("\n", " / "))
        ])
        w.writerow(["# teams_played", ",".join(meta.get("teams_played", []))])
        w.writerow(["# source_url", meta.get("source_url", "")])
        w.writerow(["# crawl_depth", meta.get("crawl_depth", "")])
        w.writerow(["# timestamp", meta.get("timestamp", "")])
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




def main():
    ap = argparse.ArgumentParser(
        description="Extract BBR player data with Spark; maximized native regex usage."
    )
    ap.add_argument("--src", required=True, help="Source folder with .html files")
    ap.add_argument("--out", required=True, help="Destination folder for TSV files")
    ap.add_argument("--meta", required=True, help="Directory with sidecar TSVs")
    ap.add_argument("--batch-size", type=int, default=50, help="Files per batch")
    args = ap.parse_args()

    src = Path(args.src)
    dest = Path(args.out)
    meta_dir = Path(args.meta)

    if not src.exists() or not src.is_dir():
        print(f"ERROR: src folder does not exist: {src}", file=sys.stderr)
        sys.exit(2)
    if not meta_dir.exists() or not meta_dir.is_dir():
        print(f"ERROR: meta folder does not exist: {meta_dir}", file=sys.stderr)
        sys.exit(2)

    html_files = sorted(src.glob("*.html"))
    if not html_files:
        print("No .html files found.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(html_files)} HTML files to process")

    spark = (
        SparkSession.builder
        .appName("BBRPlayerExtractorSparkOptimized")
        .master("local[2]")
        .config("spark.python.worker.reuse", "true")
        .config("spark.network.timeout", "1200s")
        .config("spark.executor.heartbeatInterval", "120s")
        .config("spark.rpc.askTimeout", "1200s")
        .config("spark.rpc.lookupTimeout", "1200s")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    sc = spark.sparkContext
    b_meta_dir = sc.broadcast(str(meta_dir))

    dest.mkdir(parents=True, exist_ok=True)

    total_count = 0
    batch_size = args.batch_size

    for batch_num in range(0, len(html_files), batch_size):
        batch = html_files[batch_num:batch_num + batch_size]
        print(f"\nBatch {batch_num // batch_size + 1}/{(len(html_files) - 1) // batch_size + 1}: {len(batch)} files")

        paths_rdd = sc.parallelize([str(p) for p in batch])

        def load_file(path_str):
            p = Path(path_str)
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception as e:
                sys.stderr.write(json.dumps({"file": path_str, "error": str(e)}) + "\n")
                return None
            return (path_str, text)

        pair_rdd = paths_rdd.map(load_file).filter(lambda x: x is not None)
        df = pair_rdd.toDF(["path", "html"])

        df = compute_name_columns(df)
        df = df.withColumn("jersey_numbers", udf_extract_jersey_numbers("html"))
        df = extract_position_spark(df)

        df = df.withColumn("tables_json", udf_tables_full("html"))
        df = df.withColumn("teams_played", udf_teams_played("tables_json"))

        try:
            rows = df.select(
                "path", "name", "position", "jersey_numbers",
                "personal_info_text", "teams_played", "tables_json"
            ).collect()
        except Exception as e:
            print(f"ERROR in batch {batch_num // batch_size + 1}: {e}", file=sys.stderr)
            continue

        for row in rows:
            path = Path(row["path"])
            stem = path.stem
            side = load_side_metadata(Path(b_meta_dir.value), stem)

            try:
                tables = json.loads(row["tables_json"] or "[]")
            except Exception:
                tables = []

            meta = {
                "name": row["name"],
                "position": row["position"],
                "jersey_numbers": row["jersey_numbers"] or [],
                "personal_info_text": row["personal_info_text"] or "",
                "teams_played": row["teams_played"] or [],
                "source_url": side.get("source_url", ""),
                "crawl_depth": side.get("crawl_depth", ""),
                "timestamp": side.get("timestamp", ""),
            }

            fixed_tables = []
            for t in tables:
                fixed_tables.append({
                    "id": t.get("id", ""),
                    "headline": t.get("headline", ""),
                    "header": t.get("header") or [],
                    "body": t.get("body") or [],
                })

            out_path = dest / f"{stem}.tsv"
            write_single_tsv(out_path, meta, fixed_tables)
            total_count += 1

        print(f"Batch complete: {len(rows)} files written")

    print(f"\nTotal: {total_count} files -> {dest}")
    spark.stop()


if __name__ == "__main__":
    main()