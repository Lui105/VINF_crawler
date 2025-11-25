# index_players.py
# PyLucene 10.x
import argparse
import io
import os
import re
from pathlib import Path

import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField, StoredField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.store import FSDirectory


TABLE_START_RE = re.compile(r'^#\s*TABLE\s+([^\t\r\n]+)\s*(?:\t(.*))?$', re.I)

def _is_meta(line: str) -> bool:
    return line.startswith("# ") and "\t" in line and not line.lstrip().upper().startswith("# TABLE")

def _parse_meta(line: str):
    body = line[2:].rstrip("\n")
    k, v = body.split("\t", 1)
    return k.strip().lower(), v.strip()

def load_player_tsv(path: Path):

    meta = {}
    tables = []
    # Some files are big; use streaming
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        cur_table = None
        for raw in f:
            line = raw.rstrip("\n")


            if _is_meta(line):
                k, v = _parse_meta(line)
                meta[k] = v
                continue

            m = TABLE_START_RE.match(line)
            if m:
                if cur_table:
                    while cur_table["lines"] and not cur_table["lines"][-1].strip():
                        cur_table["lines"].pop()
                    tables.append({
                        "name": cur_table["name"],
                        "title": cur_table["title"],
                        "body": "\n".join(cur_table["lines"]).strip()
                    })
                    cur_table = None

                tbl_name = (m.group(1) or "").strip()
                tbl_title = (m.group(2) or "").strip()
                cur_table = {"name": tbl_name, "title": tbl_title, "lines": []}
                continue

            if cur_table is not None:
                cur_table["lines"].append(line)

        if cur_table:
            while cur_table["lines"] and not cur_table["lines"][-1].strip():
                cur_table["lines"].pop()
            tables.append({
                "name": cur_table["name"],
                "title": cur_table["title"],
                "body": "\n".join(cur_table["lines"]).strip()
            })

    return meta, tables




def build_doc(path: Path, meta: dict, tables: list, analyzer):
    doc = Document()

    doc.add(StringField("path", str(path), Field.Store.YES))
    doc.add(StringField("filename", path.name, Field.Store.YES))

    simple_string_fields = [
        "name", "position", "teams_played", "source_url",
        "jersey_numbers", "timestamp", "crawl_depth"
    ]
    for k in simple_string_fields:
        if k in meta and meta[k]:
            doc.add(StringField(k, meta[k], Field.Store.YES))


    text_keys = [
        "personal_info_text",
        "wiki_bio",
        "wiki_high_school",
        "wiki_college",
        "wiki_personal_life",
    ]
    all_text_buf = io.StringIO()

    for k in text_keys:
        val = meta.get(k, "")
        if val:
            doc.add(TextField(k, val, Field.Store.YES))
            all_text_buf.write("\n")
            all_text_buf.write(val)


    for t in tables:
        body = t.get("body", "")
        name = t.get("name", "")
        title = t.get("title", "")

        combined = f"{name}\n{title}\n{body}".strip()
        if combined:
            doc.add(TextField("table", combined, Field.Store.NO))

            stored_tbl_name = f"table::{name or title or 'unnamed'}"
            doc.add(StoredField(stored_tbl_name, combined))
            all_text_buf.write("\n")
            all_text_buf.write(combined)

    all_text = all_text_buf.getvalue().strip()
    if all_text:
        doc.add(TextField("all_text", all_text, Field.Store.NO))

    return doc


def index_folder(tsv_dir: Path, index_dir: Path, commit_every: int = 500):
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    directory = FSDirectory.open(Paths.get(str(index_dir)))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(directory, config)

    count = 0
    for p in tsv_dir.rglob("*.tsv"):
        meta, tables = load_player_tsv(p)

        has_content = bool(meta.get("name") or meta.get("personal_info_text") or tables)
        if not has_content:
            continue

        doc = build_doc(p, meta, tables, analyzer)
        writer.addDocument(doc)
        count += 1

        if commit_every and (count % commit_every == 0):
            writer.commit()
            print(f"[commit] indexed {count} files...")

    writer.commit()
    writer.close()
    print(f"Done. Indexed {count} TSV files into {index_dir}")


def main():
    ap = argparse.ArgumentParser(description="Index player TSVs (with tables + wiki fields) into a PyLucene index")
    ap.add_argument("--tsv-dir", required=True, help="Folder containing per-player .tsv files")
    ap.add_argument("--index-dir", required=True, help="Output Lucene index directory")
    ap.add_argument("--commit-every", type=int, default=500, help="Commit every N docs (default: 500)")
    args = ap.parse_args()

    tsv_dir = Path(args.tsv_dir).resolve()
    index_dir = Path(args.index_dir).resolve()
    index_dir.mkdir(parents=True, exist_ok=True)

    index_folder(tsv_dir, index_dir, args.commit_every)


if __name__ == "__main__":
    main()
