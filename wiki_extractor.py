import os
import sys
import re
import html
import json
import argparse
import unicodedata
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import udf
from pyspark.sql.types import (
    StringType,
    StructType,
    StructField,
)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable



def read_player_tsv_metadata(tsv_path: Path):

    name = None
    with tsv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break
            line = line.strip()
            if not line:
                continue
            if not line.startswith("#"):
                continue
            body = line[1:].strip()
            if not body:
                continue
            if "\t" in body:
                key, val = body.split("\t", 1)
            elif ":" in body:
                key, val = body.split(":", 1)
            else:
                continue
            key = key.strip().lower()
            val = val.strip()
            if key == "name" and val:
                name = val
    if not name:
        return None
    return {"name": name, "file": str(tsv_path)}


def load_players(tsv_src: Path):
    players = []
    for i, p in enumerate(sorted(tsv_src.glob("*.tsv"))):
        meta = read_player_tsv_metadata(p)
        if meta:
            players.append(meta)
    return players


def _oneline(s: str) -> str:
    return " ".join((s or "").split())


RE_S = re.S | re.I

NBSP_CHARS = {
    "\u00A0", "\u202F", "\u2007", "\u2009",
    "\u2002", "\u2003", "\u2004", "\u2005", "\u2006",
}


def normalize_ws(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    for ch in NBSP_CHARS:
        s = s.replace(ch, " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r" ?\n ?", "\n", s)
    return s.strip()


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


def strip_ref_tags(text: str) -> str:
    text = re.sub(r"<ref[^>/]*/>", "", text, flags=re.I)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text, flags=re.I | re.S)
    return text


def strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text)


def strip_tables(text: str) -> str:
    out = []
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith("{|", i):
            depth += 1
            i += 2
        elif depth > 0 and text.startswith("|}", i):
            depth -= 1
            i += 2
        elif depth > 0:
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def strip_templates(text: str) -> str:
    out = []
    i = 0
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith("{{", i):
            depth += 1
            i += 2
        elif depth > 0 and text.startswith("}}", i):
            depth -= 1
            i += 2
        elif depth > 0:
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def replace_links(text: str) -> str:
    if not text:
        return ""
    def _file_repl(m):
        inner = m.group(1)
        parts = inner.split("|")
        if len(parts) >= 2:
            for part in reversed(parts):
                part = part.strip()
                if part and part.lower() not in {"thumb", "thumbnail", "right", "left", "center"}:
                    return part
        return ""

    text = re.sub(
        r"\[\[(?:File|Image):([^\]]+)\]\]",
        _file_repl,
        text,
        flags=re.I
    )

    text = re.sub(r"\[\[([^|\]]+)\|([^\]]+)\]\]", r"\2", text)

    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)

    text = re.sub(r"\[https?:[^\s\]]+\s+([^\]]+)\]", r"\1", text)

    text = re.sub(r"\[https?:[^\]]+\]", "", text)

    return text


def strip_categories(text: str) -> str:
    return re.sub(r"\[\[Category:[^\]]+\]\]", "", text, flags=re.I)


def clean_wikitext_for_text_sections(text: str) -> str:
    if not text:
        return ""
    text = strip_ref_tags(text)
    text = strip_tables(text)
    text = strip_templates(text)
    text = strip_categories(text)
    text = replace_links(text)
    text = strip_html_tags(text)
    text = text.replace("\r", "")

    cleaned_lines = []
    for line in text.splitlines():
        ls = line.lstrip()
        if not ls:
            cleaned_lines.append("")
            continue
        if ls.startswith(("{|", "|-", "|}", "||", "!", "|")):
            continue
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize_ws(text)


def split_sections_on_raw(text: str):
    if not text:
        return "", {}


    raw = strip_comments(text)
    raw = strip_ref_tags(raw)


    headings = []
    pattern = re.compile(
        r"^(?P<eq>={2,})\s*(?P<title>.+?)\s*(?P=eq)\s*$",
        flags=re.M
    )

    for m in pattern.finditer(raw):
        eq = m.group("eq")
        level = len(eq)
        title = m.group("title").strip()
        headings.append({
            "start": m.start(),
            "end": m.end(),
            "title": title,
            "level": level,
        })

    sections = {}
    if not headings:
        lead_clean = clean_wikitext_for_text_sections(raw)
        return lead_clean, sections


    first = headings[0]
    lead_raw = raw[:first["start"]]
    lead_clean = clean_wikitext_for_text_sections(lead_raw)

    n = len(headings)
    text_len = len(raw)

    for i, h in enumerate(headings):
        level = h["level"]
        start_body = h["end"]

        end_body = text_len
        for j in range(i + 1, n):
            if headings[j]["level"] <= level:
                end_body = headings[j]["start"]
                break

        body_raw = raw[start_body:end_body]
        body_clean = clean_wikitext_for_text_sections(body_raw).strip()

        if body_clean:
            sections[h["title"]] = body_clean

    return lead_clean, sections



def choose_section(sections: dict, names_like, allow_substring=False, exclude_sub=None):

    if not sections:
        return ""

    if exclude_sub is None:
        exclude_sub = []

    def norm(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    wanted_norm = [norm(n) for n in names_like]

    for h, body in sections.items():
        hnorm = norm(h)
        if hnorm in wanted_norm and body:
            return body.strip()

    if allow_substring:
        for h, body in sections.items():
            hlow = h.lower()
            if any(w in hlow for w in wanted_norm) and not any(bad in hlow for bad in exclude_sub):
                if body:
                    return body.strip()

    return ""


def looks_like_basketball_player(title: str, text: str) -> bool:
    if not text:
        return False
    tlow = text.lower()

    if "{{disambiguation" in tlow:
        return False
    if "#redirect" in tlow[:120]:
        return False

    patterns = [
        r"is an? [a-z\s]*basketball player",
        r"professional basketball player",
        r"college basketball player",
        r"nba player",
    ]
    for pat in patterns:
        if re.search(pat, tlow):
            return True

    if "infobox basketball" in tlow or "infobox nba" in tlow:
        return True

    head = tlow[:800]
    if "basketball" in head:
        return True

    return False


def extract_sections_from_wikitext(text: str):
    if not text:
        return ("", "", "", "")

    bio, sections = split_sections_on_raw(text)

    hs = choose_section(
        sections,
        names_like=[
            "High school",
            "High school career",
        ],
        allow_substring=True,
        exclude_sub=["statistics", "table"],
    )

    if not hs:
        for h, body in sections.items():
            if "high school" in h.lower() and body.strip():
                hs = body.strip()
                break


    college = choose_section(
        sections,
        names_like=[
            "College career",
            "College",
            "College years",
            "Amateur career",
        ],
        allow_substring=True,
        exclude_sub=["statistics", "table"],
    )

    if not college:
        for h, body in sections.items():
            hl = h.lower()
            if ("college" in hl or "ncaa" in hl) and "statistic" not in hl and body.strip():
                college = body.strip()
                break

    pl = choose_section(
        sections,
        names_like=["Personal life"],
        allow_substring=True,
        exclude_sub=["statistics", "table"],
    )

    return (bio.strip(), (hs or "").strip(), (college or "").strip(), (pl or "").strip())


sections_schema = StructType([
    StructField("bio", StringType(), True),
    StructField("high_school", StringType(), True),
    StructField("college", StringType(), True),
    StructField("personal_life", StringType(), True),
])



extract_sections_udf = udf(
    lambda txt: dict(
        zip(
            ["bio", "high_school", "college", "personal_life"],
            extract_sections_from_wikitext(txt)
        )
    ),
    sections_schema,
)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv-src", required=True)
    ap.add_argument("--wiki-xml", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tsv_src = Path(args.tsv_src)
    wiki_xml = Path(args.wiki_xml)
    out_dir = Path(args.out)

    if not tsv_src.is_dir():
        print(f"ERROR: TSV source dir not found: {tsv_src}", file=sys.stderr)
        sys.exit(2)
    if not wiki_xml.is_file():
        print(f"ERROR: Wiki XML file not found: {wiki_xml}", file=sys.stderr)
        sys.exit(2)

    out_dir.mkdir(parents=True, exist_ok=True)

    players = load_players(tsv_src)
    if not players:
        print("No players with # name found in TSVs.", file=sys.stderr)
        sys.exit(0)

    print(f"Collected {len(players)} candidate players from TSVs")

    names = sorted({p["name"] for p in players})
    title_candidates = set()
    for name in names:
        title_candidates.add(name)
        title_candidates.add(f"{name} (basketball)")
        title_candidates.add(f"{name} (basketball player)")

    spark = (
        SparkSession.builder
        .appName("WikiPlayerBiosJoin")
        .master("local[*]")
        .config("spark.jars.packages", "com.databricks:spark-xml_2.12:0.16.0")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.hadoop.io.native.lib.available", "false")
        .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem")
        .config("spark.hadoop.fs.AbstractFileSystem.file.impl", "org.apache.hadoop.fs.local.LocalFs")
        .config("spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version", "1")
        .getOrCreate()
    )



    sc = spark.sparkContext
    b_titles = sc.broadcast(list(title_candidates))
    b_names = sc.broadcast(set(names))

    wiki_xml_uri = Path(args.wiki_xml).resolve().as_uri()
    out_dir_uri = Path(args.out).resolve().as_uri()

    wiki_raw = (
        spark.read.format("xml")
        .options(rowTag="page", mode="PERMISSIVE")
        .load(str(wiki_xml_uri))
    )

    text_col = F.col("revision.text._VALUE")
    title_col = F.col("title")

    title_plain = F.regexp_replace(
        title_col,
        r" \((basketball.*?|basketball player.*?)\)$",
        ""
    )

    wiki = (
        wiki_raw
        .select(
            title_col.alias("title"),
            title_plain.alias("title_plain"),
            text_col.alias("text"),
        )
        .where(
            (F.col("title_plain").isin(list(b_names.value))) |
            (F.col("title").isin(list(b_titles.value)))
        )
    )

    wiki = wiki.withColumn("text_lower", F.lower(F.col("text")))
    wiki = wiki.filter(
        (F.col("text").isNotNull()) &
        (~F.col("text_lower").startswith("#redirect")) &
        (~F.col("text_lower").contains("{{disambiguation")) &
        (~F.col("text_lower").contains("{{hndis")) &
        (F.col("text_lower").contains("basketball"))
    )

    wiki = wiki.withColumn("secs", extract_sections_udf(F.col("text")))

    wiki = wiki.select(
        "title",
        "title_plain",
        F.col("secs.bio").alias("wiki_bio"),
        F.col("secs.high_school").alias("wiki_high_school"),
        F.col("secs.college").alias("wiki_college"),
        F.col("secs.personal_life").alias("wiki_personal_life"),
    )

    wiki_rows = wiki.collect()


    wiki_by_plain = {}

    for r in wiki_rows:
        t = r["title"]
        tp = r["title_plain"]

        text_blob = " ".join(
            x for x in [
                r["wiki_bio"] or "",
                r["wiki_high_school"] or "",
                r["wiki_college"] or "",
                r["wiki_personal_life"] or "",
            ]
            if x
        )

        if not looks_like_basketball_player(t, text_blob):
            continue

        key = tp if tp in b_names.value else t
        prev = wiki_by_plain.get(key)
        if prev is None:
            wiki_by_plain[key] = r
            print(f"[match] {key} <- {t}")
        else:
            prev_bio = prev["wiki_bio"] or ""
            if len((r["wiki_bio"] or "")) > len(prev_bio):
                wiki_by_plain[key] = r
                print(f"[match-update] {key} <- {t} (longer bio)")

    print(f"Matched {len(wiki_by_plain)} wiki pages to player names")

    players_df = spark.createDataFrame(players)

    wiki_records = []
    for key, r in wiki_by_plain.items():
        wiki_records.append({
            "name": key,
            "wiki_bio": (r["wiki_bio"] or "").strip(),
            "wiki_high_school": (r["wiki_high_school"] or "").strip(),
            "wiki_college": (r["wiki_college"] or "").strip(),
            "wiki_personal_life": (r["wiki_personal_life"] or "").strip(),
        })
    wiki_df = spark.createDataFrame(wiki_records)

    joined = (
        players_df
        .join(wiki_df, on="name", how="left")
    )

    (out_dir / "joined").mkdir(parents=True, exist_ok=True)
    joined.coalesce(1).write.mode("overwrite") \
        .option("header", True) \
        .option("delimiter", "\t") \
        .csv(str(out_dir_uri + "/joined"))

    print(f"Wrote Spark-joined TSV to {out_dir / 'joined'}")

    spark.stop()


if __name__ == "__main__":
    main()
