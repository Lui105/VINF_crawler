import argparse
import lucene

from java.nio.file import Paths

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanClause
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser



POSITION_SYNONYMS = {
    "pg": "point guard",
    "sg": "shooting guard",
    "sf": "small forward",
    "pf": "power forward",
    "c": "center",
    "g": "guard",
    "f": "forward",
}


def expand_position_tokens(tokens):
    expanded = []
    for t in tokens:
        lower = t.lower()
        if lower in POSITION_SYNONYMS:
            expanded.append(t)
            expanded.extend(POSITION_SYNONYMS[lower].split())
        else:
            expanded.append(t)
    return expanded



TEAM_DATA = [
    ("ATL", ["atlanta", "hawks"]),
    ("BOS", ["boston", "celtics"]),
    ("BRK", ["brooklyn", "nets"]),
    ("CHI", ["chicago", "bulls"]),
    ("CLE", ["cleveland", "cavaliers", "cavs"]),
    ("DAL", ["dallas", "mavericks", "mavs"]),
    ("DEN", ["denver", "nuggets"]),
    ("DET", ["detroit", "pistons"]),
    ("GSW", ["golden", "state", "warriors"]),
    ("HOU", ["houston", "rockets"]),
    ("IND", ["indiana", "pacers"]),
    ("LAC", ["los", "angeles", "clippers"]),
    ("LAL", ["los", "angeles", "lakers"]),
    ("MEM", ["memphis", "grizzlies"]),
    ("MIA", ["miami", "heat"]),
    ("MIL", ["milwaukee", "bucks"]),
    ("MIN", ["minnesota", "timberwolves", "wolves"]),
    ("NOP", ["new", "orleans", "pelicans"]),
    ("NYK", ["new", "york", "knicks"]),
    ("OKC", ["oklahoma", "city", "thunder"]),
    ("ORL", ["orlando", "magic"]),
    ("PHI", ["philadelphia", "sixers", "76ers", "philly"]),
    ("PHO", ["phoenix", "suns"]),
    ("POR", ["portland", "blazers", "trail", "trailblazers"]),
    ("SAC", ["sacramento", "kings"]),
    ("SAS", ["san", "antonio", "spurs"]),
    ("TOR", ["toronto", "raptors"]),
    ("UTA", ["utah", "jazz"]),
    ("WAS", ["washington", "wizards"]),
]

TEAM_BY_ABBR = {abbr: words for abbr, words in TEAM_DATA}

WORD_TO_TEAM_ABBR = {}
for abbr, words in TEAM_DATA:
    for w in words:
        WORD_TO_TEAM_ABBR.setdefault(w.lower(), set()).add(abbr)


def expand_team_tokens(tokens):
    expanded = []

    for t in tokens:
        lower = t.lower()
        added_synonyms = False

        if t.upper() in TEAM_BY_ABBR:
            abbr = t.upper()
            names = TEAM_BY_ABBR[abbr]

            expanded.append(t)
            expanded.append(abbr)

            for w in names:
                expanded.append(w.capitalize())

            added_synonyms = True

        elif lower in WORD_TO_TEAM_ABBR:
            expanded.append(t)

            for abbr in WORD_TO_TEAM_ABBR[lower]:
                expanded.append(abbr)
                for w in TEAM_BY_ABBR[abbr]:
                    expanded.append(w.capitalize())

            added_synonyms = True

        if not added_synonyms:
            expanded.append(t)

    return expanded









def make_all_terms_required(q: str) -> str:
    parts = [p for p in q.split() if p.strip()]
    if not parts:
        return q
    return " ".join(f"+{p}" for p in parts)


def preprocess_query_for_synonyms(q: str) -> str:

    if not q.strip():
        return q

    tokens = q.split()
    tokens = expand_position_tokens(tokens)
    tokens = expand_team_tokens(tokens)

    seen = set()
    deduped = []
    for t in tokens:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)

    return " ".join(deduped)




def open_searcher(index_dir):
    directory = FSDirectory.open(Paths.get(index_dir))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    analyzer = StandardAnalyzer()

    fields = [
        "name",
        "position",
        "teams_played",
        "personal_info_text",
        "wiki_bio",
        "wiki_high_school",
        "wiki_college",
        "wiki_personal_life",
        "all_text",
    ]

    return searcher, analyzer, fields, reader


def extract_terms_for_match_reporting(raw_q: str, expanded_q: str):

    parts = [p for p in expanded_q.split() if p.strip()]
    return [p.lower() for p in parts]


def run_query(searcher, analyzer, fields, query_str, top_k=10):
    SHOULD = BooleanClause.Occur.SHOULD
    flags = [SHOULD] * len(fields)

    raw_q = query_str.strip()

    expanded_q = preprocess_query_for_synonyms(raw_q)

    strict_q = make_all_terms_required(expanded_q)
    strict_query = MultiFieldQueryParser.parse(strict_q, fields, flags, analyzer)
    strict_hits = searcher.search(strict_query, top_k)

    match_terms = extract_terms_for_match_reporting(raw_q, expanded_q)

    if len(strict_hits.scoreDocs) > 0:
        return strict_query, strict_hits, match_terms

    relaxed_query = MultiFieldQueryParser.parse(expanded_q, fields, flags, analyzer)
    relaxed_hits = searcher.search(relaxed_query, top_k)

    return relaxed_query, relaxed_hits, match_terms


MATCH_REPORT_FIELDS = [
    "name",
    "position",
    "teams_played",
    "teams_played_raw",
    "personal_info_text",
    "wiki_bio",
    "wiki_high_school",
    "wiki_college",
    "wiki_personal_life",
]


def find_matched_fields(doc, terms):
    matched = []
    if not terms:
        return matched

    for field in MATCH_REPORT_FIELDS:
        val = doc.get(field)
        if not val:
            continue
        lower_val = val.lower()
        if any(term in lower_val for term in terms):
            matched.append(field)

    return matched


def format_hit_line(rank, score_doc, searcher, matched_fields):
    stored_fields = searcher.storedFields()
    doc = stored_fields.document(score_doc.doc)

    name = doc.get("name") or "(unknown name)"
    position = doc.get("position") or "(unknown position)"
    teams_raw = doc.get("teams_played_raw") or ""
    source_url = doc.get("source_url") or ""

    score_str = f"{score_doc.score:.3f}"
    rank_str = f"{rank:2d}."
    teams_part = f" (teams: {teams_raw})" if teams_raw else ""

    first_line = f"{rank_str} [{score_str}] {name}  -- {position}{teams_part}"

    second_parts = []
    if source_url:
        second_parts.append(f"url={source_url}")
    second_line = "     " + "  ".join(second_parts) if second_parts else ""

    third_line = ""
    if matched_fields:
        third_line = "     matched fields: " + ", ".join(matched_fields)

    return first_line, second_line, third_line




def main():
    ap = argparse.ArgumentParser(
        description="Search basketball player index (PyLucene 10, with synonyms, AND→OR behavior, and matched fields)"
    )
    ap.add_argument("--index-dir", required=True, help="Lucene index directory")
    ap.add_argument("--query", "-q", required=True, help="Query string")
    ap.add_argument(
        "--top-k", type=int, default=3, help="Number of hits to show (default: 3)"
    )
    args = ap.parse_args()

    lucene.initVM(vmargs=['-Djava.awt.headless=true'])

    searcher, analyzer, fields, reader = open_searcher(args.index_dir)

    try:
        query, hits, match_terms = run_query(
            searcher, analyzer, fields, args.query, top_k=args.top_k
        )
    except Exception as e:
        print(f"Error parsing query: {e}")
        reader.close()
        return


    stored_fields = searcher.storedFields()

    for i, sd in enumerate(hits.scoreDocs, start=1):
        doc = stored_fields.document(sd.doc)
        matched_fields = find_matched_fields(doc, match_terms)
        first_line, second_line, third_line = format_hit_line(i, sd, searcher, matched_fields)
        print(first_line)
        if second_line:
            print(second_line)
        if third_line:
            print(third_line)

    reader.close()


if __name__ == "__main__":
    main()
