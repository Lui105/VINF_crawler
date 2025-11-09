import os
import math
import lucene
from java.nio.file import Paths
from org.apache.lucene.store import FSDirectory
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.document import Document, Field, StringField, TextField
from org.apache.lucene.index import IndexWriterConfig, IndexWriter, DirectoryReader
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search.similarities import ClassicSimilarity


# =========================================
# CONFIG
# =========================================

MERGED_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\merged_tables"

GLOBAL_IDF_A_INDEX_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\lucene_indexes_global_custom\idf_a_index"
GLOBAL_IDF_B_INDEX_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\lucene_indexes_global_custom\idf_b_index"


# =========================================
# CUSTOM SIMILARITIES
# =========================================

class LogIDFSimilarity(ClassicSimilarity):
    """
    Custom IDF method A:
    idf = log10(N / df)

    - No smoothing
    - Strongly favors rare terms
    - Can produce 0 or negative values if df >= N (extreme edge case)
    """

    def idf(self, docFreq, docCount):
        # avoid division by zero
        if docFreq == 0:
            # if df = 0, theoretically term never appears;
            # you could return a huge number.
            # We'll just return 0 to be safe.
            return 0.0
        # docFreq and docCount come in as Java ints
        # convert to float in Python math
        return math.log10(float(docCount) / float(docFreq))


class CappedIDFSimilarity(ClassicSimilarity):
    """
    Custom IDF method B:
    idf = min( log(N/(df+1)) + 1, 4.0 )

    - Similar to Lucene classic, but capped at 4.0
    - Gives diminishing returns for extremely rare terms
    """

    def idf(self, docFreq, docCount):
        raw = math.log(float(docCount) / (float(docFreq) + 1.0)) + 1.0
        if raw > 4.0:
            raw = 4.0
        return raw


# =========================================
# PARSE MERGED FILES INTO TABLE BLOCKS
# =========================================

def parse_merged_all_tsv(path):
    """
    Parse one <base_stem>__ALL.tsv file and return a list of table dicts.

    Expected format (based on your latest merger script):

        ###PAGE_BEGIN###    <entity_id_from_page>

        ###TABLE_BEGIN###
        headline        Per Game — Playoffs
        source_file     giannis_per_game_playoffs__table2.tsv
        source_url      https://...
        crawl_depth     0
        timestamp       2025-10-28T20:15:03Z
        row_count       13
        col_count       29
        base_stem       giannis-antetokounmpo-milwaukee-bucks-profile
        Season  Age Team Lg Pos G GS MP PTS ...
        2019-20 25  MIL NBA PF ...
        ...
        ###TABLE_END###

    Rules:
    - From ###TABLE_BEGIN### until we read the line whose key is "base_stem", we treat
      lines as metadata key/value pairs.
    - The line with key "base_stem" is ALSO metadata.
    - After we see base_stem, the remaining lines (until ###TABLE_END###) are
      table rows (header + data).
    """

    tables = []

    current_table = None
    entity_id_from_page = None
    seen_base_stem_in_table = False
    table_rows_buf = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            parts = line.split("\t")

            # Capture page-level entity id
            if parts[0] == "###PAGE_BEGIN###":
                if len(parts) > 1:
                    entity_id_from_page = parts[1].strip()
                else:
                    entity_id_from_page = ""
                continue

            # Start of a new table
            if parts[0] == "###TABLE_BEGIN###":
                # flush previous if something weird happened
                if current_table is not None:
                    current_table["table_text"] = "\n".join(table_rows_buf)
                    current_table["headline"] = current_table["metadata"].get("headline", "")
                    current_table["source_file"] = current_table["metadata"].get("source_file", "")
                    current_table["base_stem"] = current_table["metadata"].get(
                        "base_stem",
                        current_table["entity_id"]
                    )
                    tables.append(current_table)

                current_table = {
                    "entity_id": entity_id_from_page or "",
                    "metadata": {},
                    "table_text": "",
                }
                seen_base_stem_in_table = False
                table_rows_buf = []
                continue

            # End of table
            if parts[0] == "###TABLE_END###":
                if current_table is not None:
                    current_table["table_text"] = "\n".join(table_rows_buf)
                    current_table["headline"] = current_table["metadata"].get("headline", "")
                    current_table["source_file"] = current_table["metadata"].get("source_file", "")
                    current_table["base_stem"] = current_table["metadata"].get(
                        "base_stem",
                        current_table["entity_id"]
                    )
                    tables.append(current_table)

                current_table = None
                seen_base_stem_in_table = False
                table_rows_buf = []
                continue

            # ignore lines outside tables
            if current_table is None:
                continue

            # still parsing metadata?
            if not seen_base_stem_in_table:
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = "\t".join(parts[1:]).strip()
                    current_table["metadata"][key] = val

                    if key == "base_stem":
                        # after this line we treat subsequent lines as actual data rows
                        seen_base_stem_in_table = True
                else:
                    # malformed (no tab) -> treat rest as table rows
                    seen_base_stem_in_table = True
                    if line.strip():
                        table_rows_buf.append(line)
            else:
                # after base_stem: these lines are the actual table content
                table_rows_buf.append(line)

    # finalize last table if file didn't close cleanly
    if current_table is not None:
        current_table["table_text"] = "\n".join(table_rows_buf)
        current_table["headline"] = current_table["metadata"].get("headline", "")
        current_table["source_file"] = current_table["metadata"].get("source_file", "")
        current_table["base_stem"] = current_table["metadata"].get(
            "base_stem",
            current_table["entity_id"]
        )
        tables.append(current_table)

    return tables


# =========================================
# TURN TABLE BLOCK INTO LUCENE DOCUMENT
# =========================================

def make_lucene_doc(table_block):
    """
    We store:
      entity_id     (page/entity id from ###PAGE_BEGIN###)
      base_stem     (explicitly from table metadata)
      headline      (ex: "Per Game — Playoffs")
      source_file   (original per-table TSV filename)
      metadata_text (flattened "key: value" pairs from metadata)
      table_text    (header+rows of stats)
      fulltext      (catch-all indexed field for searching)
    """

    metadata_items = [
        f"{k}: {v}"
        for (k, v) in table_block.get("metadata", {}).items()
    ]
    metadata_text = "\n".join(metadata_items)

    headline_val    = table_block.get("headline", "")
    source_file_val = table_block.get("source_file", "")
    entity_id_val   = table_block.get("entity_id", "")
    base_stem_val   = table_block.get("base_stem", "")
    table_text_val  = table_block.get("table_text", "")

    fulltext_val = " ".join([
        headline_val,
        entity_id_val,
        base_stem_val,
        source_file_val,
        metadata_text,
        table_text_val,
    ])

    doc = Document()

    # exact string fields we still want retrievable
    doc.add(StringField("entity_id", entity_id_val, Field.Store.YES))
    doc.add(StringField("base_stem", base_stem_val, Field.Store.YES))
    doc.add(StringField("source_file", source_file_val, Field.Store.YES))

    doc.add(StringField("headline_exact", headline_val, Field.Store.YES))
    doc.add(TextField("headline", headline_val, Field.Store.NO))

    # metadata and table body as retrievable text
    doc.add(TextField("metadata_text", metadata_text, Field.Store.YES))
    doc.add(TextField("table_text", table_text_val, Field.Store.YES))

    # catch-all for searching
    doc.add(TextField("fulltext", fulltext_val, Field.Store.NO))

    return doc


# =========================================
# GLOBAL INDEX BUILD USING CUSTOM IDF
# =========================================

def build_global_index(index_dir, similarity):
    """
    Build a single global index (all players, all teams, all games).
    Use the given custom similarity which defines how IDF is computed.
    """
    os.makedirs(index_dir, exist_ok=True)

    directory = FSDirectory.open(Paths.get(index_dir))
    analyzer = EnglishAnalyzer()

    config = IndexWriterConfig(analyzer)
    config.setSimilarity(similarity)
    writer = IndexWriter(directory, config)

    # loop through every <base_stem>__ALL.tsv in MERGED_DIR
    for fname in os.listdir(MERGED_DIR):
        if not fname.endswith("__ALL.tsv"):
            continue
        merged_path = os.path.join(MERGED_DIR, fname)

        tables = parse_merged_all_tsv(merged_path)
        for t in tables:
            doc = make_lucene_doc(t)
            writer.addDocument(doc)

    writer.close()
    directory.close()


def build_global_indexes_custom():
    """
    Build two global indexes, one per custom IDF method:
      - GLOBAL_IDF_A_INDEX_DIR uses LogIDFSimilarity
      - GLOBAL_IDF_B_INDEX_DIR uses CappedIDFSimilarity
    """
    lucene.initVM()

    print("[GLOBAL IDF A] Building global index with LogIDFSimilarity...")
    build_global_index(
        GLOBAL_IDF_A_INDEX_DIR,
        LogIDFSimilarity()
    )

    print("[GLOBAL IDF B] Building global index with CappedIDFSimilarity...")
    build_global_index(
        GLOBAL_IDF_B_INDEX_DIR,
        CappedIDFSimilarity()
    )

    print("Done building both custom-IDF global indexes.")


# =========================================
# SEARCH
# =========================================

def search_global(index_dir, query_str, similarity, limit=10):
    """
    Search any global index dir with the given similarity.
    Returns top hits for a query like "Playoffs AND MIL".
    """
    directory = FSDirectory.open(Paths.get(index_dir))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)

    searcher.setSimilarity(similarity)

    analyzer = EnglishAnalyzer()
    parser = QueryParser("fulltext", analyzer)
    query = parser.parse(query_str)

    hits = searcher.search(query, limit).scoreDocs

    for hit in hits:
        doc = searcher.doc(hit.doc)

        print("Score:", hit.score)
        print("Entity ID:", doc.get("entity_id"))
        print("Base stem:", doc.get("base_stem"))
        print("Headline:", doc.get("headline_exact"))
        print("Source file:", doc.get("source_file"))
        print("Metadata:")
        print(doc.get("metadata_text"))
        print("Snippet of table rows:")
        print((doc.get("table_text") or "")[:400], "...")
        print("=" * 80)

    reader.close()
    directory.close()


def demo_compare_search(query_str):
    """
    Run the SAME query across both global indexes
    (one using LogIDFSimilarity, one using CappedIDFSimilarity)
    so you can see how ranking differs.

    Example:
        demo_compare_search("Playoffs AND MIL")
        demo_compare_search("\"Boston Celtics\" AND 2024")
        demo_compare_search("WNBA AND Wings AND PTS")
    """

    print("\n=== Global search with LogIDFSimilarity ===")
    search_global(
        GLOBAL_IDF_A_INDEX_DIR,
        query_str,
        LogIDFSimilarity(),
        limit=10
    )

    print("\n=== Global search with CappedIDFSimilarity ===")
    search_global(
        GLOBAL_IDF_B_INDEX_DIR,
        query_str,
        CappedIDFSimilarity(),
        limit=10
    )


# =========================================
# MAIN
# =========================================

if __name__ == "__main__":
    # 1. build two global indexes using two custom IDF methods
    build_global_indexes_custom()

    # 2. run a test query across both and compare
    demo_compare_search("Playoffs AND MIL")
