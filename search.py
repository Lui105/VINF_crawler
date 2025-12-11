

import argparse
import gzip
import hashlib
import html
import json
import math
import os
import sys
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Tuple
import orjson






NBSPS = {"\u00A0","\u202F","\u2007","\u2009","\u2002","\u2003","\u2004","\u2005","\u2006"}

def normalize_spaces(s: str) -> str:
    if not s: return ""
    s = html.unescape(s)
    s = unicodedata.normalize("NFKC", s)
    for ch in NBSPS:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s.strip()

_TOKEN_RE = None
def tokenize(text: str) -> List[str]:
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re
        _TOKEN_RE = re.compile(r"[A-Za-z0-9%]+")
    text = normalize_spaces(text).lower()
    return _TOKEN_RE.findall(text)


def sha1_first_byte(s: str) -> int:
    return hashlib.sha1(s.encode("utf-8")).digest()[0]

def load_sharded_postings(index_dir: str, tokens: List[str]) -> Dict[str, List[int]]:

    tokdir = os.path.join(index_dir, "full_tokens")
    if not os.path.isdir(tokdir):
        print(f"ERROR: token dir not found: {tokdir}", file=sys.stderr)
        return {}

    buckets: Dict[int, List[str]] = {}
    for t in tokens:
        buckets.setdefault(sha1_first_byte(t), []).append(t)

    out: Dict[str, List[int]] = {}
    for b, toks in buckets.items():
        shard_path = os.path.join(tokdir, f"bucket_{b:02x}.json.gz")
        if not os.path.exists(shard_path):
            continue
        with gzip.open(shard_path, "rb") as gz:
            shard = orjson.loads(gz.read())
        for t in toks:
            lst = shard.get(t)
            if lst:
                out[t] = lst
    return out



def load_docstore(index_dir: str) -> List[dict]:
    with open(os.path.join(index_dir, "docstore.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def load_manifest(index_dir: str) -> dict:
    p = os.path.join(index_dir, "manifest.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"doc_count": None}



def idf_smooth(N: int, df: int) -> float:
    return math.log((N + 1.0) / (df + 1.0)) + 1.0

def idf_bm25(N: int, df: int) -> float:
    return max(0.0, math.log((N - df + 0.5) / (df + 0.5) + 1.0))



def show_results(title: str, ranked: List[Tuple[int, float]], docs: List[dict], topk: int):
    k = min(topk, len(ranked))
    print(f"== {title} (top {k}) ==")
    for i, (did, score) in enumerate(ranked[:topk], 1):
        if did < 0 or did >= len(docs):
            continue
        d = docs[did]
        nm = d.get("name", "")
        pos = d.get("position", "")
        teams = ",".join(d.get("teams_played", []))
        url = d.get("source_url", "")
        print(f"{i:2d}. [{score:.3f}] {nm}  -- {pos}  (teams: {teams})")
        print(f"     id={did}  url={url}")



def run_query(index_dir: str, query: str, topk: int = 10, compare_idf: bool = False):
    q_tokens = tokenize(query)
    if not q_tokens:
        print("No query tokens after normalization.")
        return

    postings = load_sharded_postings(index_dir, q_tokens)
    if not postings:
        print("No hits for the given keywords.")
        return

    ordered = sorted(postings.items(), key=lambda kv: len(kv[1]))
    cands = set(ordered[0][1])
    for _, plist in ordered[1:]:
        if not cands:
            break
        cands &= set(plist)

    candidate_ids = sorted(cands)
    if not candidate_ids:
        print("No candidates after boolean match.")
        return

    docs = load_docstore(index_dir)
    N = len(docs) or (load_manifest(index_dir).get("doc_count") or 1)
    df_map = {t: len(postings.get(t, [])) for t in q_tokens}

    id_to_text: Dict[int, str] = {}
    texts_path = os.path.join(index_dir, "texts.jsonl")
    with open(texts_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            i = obj["id"]
            if i in candidate_ids:
                id_to_text[i] = " ".join([obj.get("name",""), obj.get("meta",""), obj.get("tables","")])
                if len(id_to_text) == len(candidate_ids):
                    break

    doc_tf = {i: Counter(tokenize(id_to_text.get(i, ""))) for i in candidate_ids}

    def score(doc_id: int, idf_func) -> float:
        tf = doc_tf.get(doc_id, {})
        s = 0.0
        for t in q_tokens:
            if t in tf:
                s += tf[t] * idf_func(N, df_map.get(t, 1))
        return s

    rank_smooth = sorted(
        ((d, score(d, idf_smooth)) for d in candidate_ids),
        key=lambda x: x[1],
        reverse=True
    )

    if not compare_idf:
        show_results("smooth IDF", rank_smooth, docs, topk)
        return

    rank_bm25 = sorted(
        ((d, score(d, idf_bm25)) for d in candidate_ids),
        key=lambda x: x[1],
        reverse=True
    )

    show_results("smooth IDF", rank_smooth, docs, topk)
    show_results("bm25 IDF",   rank_bm25,   docs, topk)




def main():
    ap = argparse.ArgumentParser(description="Keyword search over the custom sharded index (no DB).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("query", help="Run a keyword query (searches name+meta+tables)")
    q.add_argument("--index", required=True, help="Index folder")
    q.add_argument("--q", required=True, help="Query string")
    q.add_argument("--topk", type=int, default=10, help="How many results to show")
    q.add_argument("--compare-idf", action="store_true", help="Compare smooth vs BM25-style IDF ranking")
    q.set_defaults(func=lambda args: run_query(args.index, args.q, args.topk, args.compare_idf))

    args = ap.parse_args()
    if args.cmd == "query":
        args.func(args)
    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
