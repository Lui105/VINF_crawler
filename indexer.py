
import argparse, json, re, html, unicodedata, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional
import hashlib
import gzip
import orjson


def sha1_first_byte(s: str) -> int:
    return hashlib.sha1(s.encode('utf-8')).digest()[0]  # 0..255

def write_sharded_inverted_index(out_dir: Path, scope_name: str, inv: Dict[str, List[int]]):

    tokdir = out_dir / f"{scope_name}_tokens"
    tokdir.mkdir(parents=True, exist_ok=True)

    shards: List[Dict[str, List[int]]] = [dict() for _ in range(256)]
    for tok, postings in inv.items():
        b = sha1_first_byte(tok)
        shards[b][tok] = postings

    for i, shard in enumerate(shards):
        if not shard:
            continue
        with gzip.open(tokdir / f"bucket_{i:02x}.json.gz", "wb") as gz:
            gz.write(orjson.dumps(shard))







def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def make_table_uid(path: Path, doc_id: int, ord_idx: int) -> str:
    return f"tbl_{short_hash(str(path))}_{doc_id}_{ord_idx}"


NBSPS = {"\u00A0","\u202F","\u2007","\u2009","\u2002","\u2003","\u2004","\u2005","\u2006"}
def normalize_spaces(s: str) -> str:
    if not s: return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    for ch in NBSPS: s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

TOKEN_RE = re.compile(r"[A-Za-z0-9%]+")
def tokenize(text: str) -> List[str]:
    text = normalize_spaces(text).lower()
    return TOKEN_RE.findall(text)

def uniq_preserve(seq: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out



@dataclass
class Doc:
    id: int
    path: str
    name: str
    position: str
    jersey_numbers: List[str]
    summary: str
    personal_info_text: str
    source_url: str
    crawl_depth: str
    timestamp: str
    source_file: str
    source_file_sha256: str
    teams_played: List[str]
    page_title: str
    canonical_url: str
    og_url: str
    html_lang: str
    player_slug: str
    table_count: int
    table_ids: List[str]
    has_summary_table: bool
    text_name: str
    text_meta: str
    text_tables: str

def parse_tsv(path: Path) -> Tuple[Dict[str, str], List[Dict]]:
    meta: Dict[str, str] = {}
    tables: List[Dict] = []
    cur_table = None
    header_read = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                header_read = False; cur_table = None; continue
            if line.startswith("# "):
                parts = line[2:].split("\t")
                if parts and parts[0] == "TABLE":
                    table_id = parts[1] if len(parts)>1 else ""
                    headline = parts[2] if len(parts)>2 else ""
                    cur_table = {"id": table_id, "headline": headline, "header": None, "body": []}
                    tables.append(cur_table); header_read = False
                else:
                    key = parts[0].strip(); val = parts[1].strip() if len(parts)>1 else ""
                    meta[key] = val
                continue
            if cur_table is not None:
                row = line.split("\t")
                if not header_read:
                    cur_table["header"] = row; header_read = True
                else:
                    cur_table["body"].append(row)
    return meta, tables

def get_meta(meta: Dict[str,str], key: str, default: str = "") -> str:
    return meta.get(key, default)

def parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    return [x.strip() for x in s.split(",")] if s else []


@dataclass
class LightDoc:
    id: int
    path: str
    name: str
    position: str
    teams_played: List[str]
    jersey_numbers: List[str]
    source_url: str
    timestamp: str
    player_slug: str
    table_ids: List[str]
    table_count: int

def doc_to_light(d: Doc) -> LightDoc:
    return LightDoc(
        id=d.id, path=d.path, name=d.name, position=d.position,
        teams_played=d.teams_played, jersey_numbers=d.jersey_numbers,
        source_url=d.source_url, timestamp=d.timestamp, player_slug=d.player_slug,
        table_ids=d.table_ids, table_count=d.table_count
    )


def doc_from_tsv(path: Path, doc_id: int) -> Tuple[Doc, Dict, List[Dict]]:
    meta, tables = parse_tsv(path)

    name = get_meta(meta, "name"); position = get_meta(meta, "position")
    jersey_numbers = parse_csv_list(get_meta(meta, "jersey_numbers"))
    summary = get_meta(meta, "summary"); personal = get_meta(meta, "personal_info_text")
    source_url = get_meta(meta, "source_url"); crawl_depth = get_meta(meta, "crawl_depth")
    timestamp = get_meta(meta, "timestamp"); source_file = get_meta(meta, "source_file")
    source_sha = get_meta(meta, "source_file_sha256")
    teams_played = parse_csv_list(get_meta(meta, "teams_played"))
    page_title = get_meta(meta, "page_title"); canonical_url = get_meta(meta, "canonical_url")
    og_url = get_meta(meta, "og_url"); html_lang = get_meta(meta, "html_lang")
    player_slug = get_meta(meta, "player_slug")


    unique_ids = []
    for k, t in enumerate(tables):
        uid = make_table_uid(path, doc_id, k)
        t["_uid"] = uid

        t["_raw_id"] = (t.get("id") or "")
        t["_headline"] = (t.get("headline") or "")
        unique_ids.append(uid)


    table_count = len(unique_ids)
    has_summary_table = any((t.get("id","") or "").strip().lower() == "summary" or
                            (t.get("headline","") or "").strip().lower() == "summary"
                            for t in tables)


    if not player_slug and source_url:
        m = re.search(r"/players/[a-z]/([a-z0-9]+)\.html", source_url, flags=re.I)
        if m: player_slug = m.group(1)


    text_name = normalize_spaces(name)
    meta_parts = [summary, personal, position, " ".join(jersey_numbers), " ".join(teams_played),
                  page_title, canonical_url, og_url, html_lang, player_slug]
    text_meta = normalize_spaces(" / ".join([x for x in meta_parts if x]))
    chunks = []
    for t in tables:
        chunks.append(str(t.get("headline","")))
        if t.get("header"): chunks.append(" ".join(t["header"]))
        for r in t.get("body", []): chunks.append(" ".join(r))
    text_tables = normalize_spaces(" / ".join(chunks))

    return Doc(
        id=doc_id, path=str(path), name=name, position=position, jersey_numbers=jersey_numbers,
        summary=summary, personal_info_text=personal, source_url=source_url, crawl_depth=crawl_depth,
        timestamp=timestamp, source_file=source_file, source_file_sha256=source_sha,
        teams_played=teams_played, page_title=page_title, canonical_url=canonical_url, og_url=og_url,
        html_lang=html_lang, player_slug=player_slug, table_count=table_count, table_ids=unique_ids,
        has_summary_table=has_summary_table, text_name=text_name, text_meta=text_meta, text_tables=text_tables
    ), meta, tables


def add_to_inv(inv: Dict[str, List[int]], doc_id: int, tokens: Iterable[str]):
    seen = set()
    for tok in tokens:
        if not tok or tok in seen: continue
        seen.add(tok)
        lst = inv.setdefault(tok, [])
        if not lst or lst[-1] != doc_id: lst.append(doc_id)



SEASON_RE = re.compile(r"^\d{4}-\d{2}$")

def to_num(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s: return None
    s = s.replace(",", "").replace("%", "")
    try: return float(s)
    except: return None

def to_int(s: str) -> Optional[int]:
    v = to_num(s)
    return int(v) if v is not None else None

def find_regular_per_game_table(tables: List[Dict]) -> Optional[Dict]:
    for t in tables:
        h = (t.get("headline") or "").strip().lower()
        if h.startswith("per game table") and "playoff" not in h:
            return t
    for t in tables:
        h = (t.get("headline") or "").strip().lower()
        if "per game" in h:
            return t
    return None

def extract_per_game_records(meta: Dict[str,str], tables: List[Dict], player_doc_id: int) -> List[Dict]:
    t = find_regular_per_game_table(tables)
    if not t or not t.get("header"): return []
    header = [ (c or "").strip() for c in t["header"] ]
    idx = {c: i for i, c in enumerate(header)}
    need = ["Season","Age","Team","Lg","Pos","G","PTS"]
    if not all(c in idx for c in need):
        return []
    out = []
    for row in t.get("body", []):
        if len(row) < len(header): row = row + [""]*(len(header)-len(row))
        season = (row[idx["Season"]] or "").strip()
        if not season or not SEASON_RE.match(season):
            continue
        team = (row[idx["Team"]] or "").strip().upper()
        if team in {"", "2TM"}:
            continue
        rec = {
            "player_doc_id": player_doc_id,
            "player_name": meta.get("name",""),
            "player_slug": meta.get("player_slug",""),
            "source_url": meta.get("source_url",""),
            "season": season,
            "team": team,
            "lg": (row[idx["Lg"]] or "").strip(),
            "pos": (row[idx["Pos"]] or "").strip(),
            "g": to_int(row[idx["G"]]),
            "age": to_int(row[idx["Age"]]),
            "pts": to_num(row[idx["PTS"]]),
        }
        for col in ["GS","MP","TRB","AST","STL","BLK","TOV","PF","FG%","3P%","FT%","TS%","eFG%"]:
            if col in idx:
                key = col.lower().replace("%","pct").replace("/","")
                rec[key] = to_num(row[idx[col]])
        out.append(rec)
    return out

def write_stats_jsonl(out_dir: Path, stats: List[Dict]):
    p = out_dir / "stats.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for rec in stats:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# -------------
# Build (text + stats)
# -------------

def build_index(src_dir: Path, out_dir: Path):
    t0 = time.time()
    files = sorted(src_dir.glob("**/*.tsv"))
    if not files:
        print("No TSV files found.", file=sys.stderr); return

    out_dir.mkdir(parents=True, exist_ok=True)

    docstore: List[LightDoc] = []
    texts_f = (out_dir / "texts.jsonl").open("w", encoding="utf-8")

    inv_full: Dict[str, List[int]] = {}
    inv_name: Dict[str, List[int]] = {}
    inv_meta: Dict[str, List[int]] = {}
    inv_tables: Dict[str, List[int]] = {}

    facets = { "teams": {}, "positions": {}, "jerseys": {}, "table_ids": {} }
    def facet_add(map_: Dict[str, List[int]], key: str, doc_id: int):
        if not key: return
        key = key.upper()
        lst = map_.setdefault(key, [])
        if not lst or lst[-1] != doc_id: lst.append(doc_id)

    stats_records: List[Dict] = []

    for i, path in enumerate(files):
        doc, meta, tables = doc_from_tsv(path, i)

        # docstore + texts
        docstore.append(doc_to_light(doc))
        texts_f.write(json.dumps({"id": doc.id, "name": doc.text_name,
                                  "meta": doc.text_meta, "tables": doc.text_tables},
                                 ensure_ascii=False) + "\n")

        # inverted indexes
        toks_name = tokenize(doc.text_name)
        toks_meta = tokenize(doc.text_meta)
        toks_tables = tokenize(doc.text_tables)
        add_to_inv(inv_name, doc.id, toks_name)
        add_to_inv(inv_meta, doc.id, toks_meta)
        add_to_inv(inv_tables, doc.id, toks_tables)
        add_to_inv(inv_full, doc.id, uniq_preserve(toks_name + toks_meta + toks_tables))

        # facets
        for tm in doc.teams_played: facet_add(facets["teams"], tm, doc.id)
        for j in doc.jersey_numbers: facet_add(facets["jerseys"], j, doc.id)
        for tok in uniq_preserve(tokenize(doc.position)): facet_add(facets["positions"], tok, doc.id)
        for tid in doc.table_ids: facet_add(facets["table_ids"], tid, doc.id)

        # per-game stats
        stats_records.extend(extract_per_game_records(meta, tables, doc.id))

        if i % 10 == 0:
            print(f"Processed {i} files")

    texts_f.close()

    # persist text index
    (out_dir / "docstore.json").write_text(
        json.dumps([asdict(d) for d in docstore], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # FAST PATH: sharded inverted indexes (search reads only touched buckets)
    write_sharded_inverted_index(out_dir, "full", inv_full)
    write_sharded_inverted_index(out_dir, "name", inv_name)
    write_sharded_inverted_index(out_dir, "meta", inv_meta)
    write_sharded_inverted_index(out_dir, "tables", inv_tables)


    # persist stats side index
    write_stats_jsonl(out_dir, stats_records)

    print(f"Indexed {len(files)} TSVs; stats rows: {len(stats_records)} -> {out_dir} in {time.time()-t0:.2f}s")

# -------------
# CLI (build only)
# -------------

def main():
    ap = argparse.ArgumentParser(description="Build custom TSV index (no DB) + Per-Game stats.")
    ap.add_argument("--src", required=True, help="Folder with TSV files")
    ap.add_argument("--out", required=True, help="Output index folder")
    args = ap.parse_args()
    build_index(Path(args.src), Path(args.out))

if __name__ == "__main__":
    main()
