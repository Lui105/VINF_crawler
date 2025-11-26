import argparse
import lucene

from java.nio.file import Paths

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import IndexSearcher, BooleanClause
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser


def open_searcher(index_dir):
    directory = FSDirectory.open(Paths.get(index_dir))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    analyzer = StandardAnalyzer()

    fields = [
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


def make_all_terms_required(q: str) -> str:
    parts = q.split()
    return " ".join(f"+{p}" for p in parts)

def run_query(searcher, analyzer, fields, query_str, top_k=3):
    SHOULD = BooleanClause.Occur.SHOULD
    flags = [SHOULD] * len(fields)

    forced = make_all_terms_required(query_str)
    query = MultiFieldQueryParser.parse(forced, fields, flags, analyzer)

    hits = searcher.search(query, top_k)
    return query, hits




def format_hit_line(rank, score_doc, searcher):
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

    second_line_parts = []
    if source_url:
        second_line_parts.append(f"url={source_url}")

    if second_line_parts:
        second_line = "     " + "  ".join(second_line_parts)
    else:
        second_line = ""

    return first_line, second_line


def main():
    ap = argparse.ArgumentParser(
        description="Search basketball player index (PyLucene 10, non-interactive)"
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
        query, hits = run_query(searcher, analyzer, fields, args.query, top_k=args.top_k)
    except Exception as e:
        print(f"Error parsing query: {e}")
        reader.close()
        return

    for i, sd in enumerate(hits.scoreDocs, start=1):
        first_line, second_line = format_hit_line(i, sd, searcher)
        print(first_line)
        if second_line:
            print(second_line)

    reader.close()


if __name__ == "__main__":
    main()
