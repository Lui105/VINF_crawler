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




MERGED_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\merged_tables"

GLOBAL_IDF_A_INDEX_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\lucene_indexes_global_custom\idf_a_index"
GLOBAL_IDF_B_INDEX_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\lucene_indexes_global_custom\idf_b_index"




class LogIDFSimilarity(ClassicSimilarity):


    def idf(self, docFreq, docCount):
        if docFreq == 0:

            return 0.0
        return math.log10(float(docCount) / float(docFreq))


class CappedIDFSimilarity(ClassicSimilarity):


    def idf(self, docFreq, docCount):
        raw = math.log(float(docCount) / (float(docFreq) + 1.0)) + 1.0
        if raw > 4.0:
            raw = 4.0
        return raw




def parse_merged_all_tsv(path):


    tables = []

    current_table = None
    entity_id_from_page = None
    seen_base_stem_in_table = False
    table_rows_buf = []

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            parts = line.split("\t")

            if parts[0] == "###PAGE_BEGIN###":
                if len(parts) > 1:
                    entity_id_from_page = parts[1].strip()
                else:
                    entity_id_from_page = ""
                continue

            if parts[0] == "###TABLE_BEGIN###":
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

            if current_table is None:
                continue

            if not seen_base_stem_in_table:
                if len(parts) >= 2:
                    key = parts[0].strip()
                    val = "\t".join(parts[1:]).strip()
                    current_table["metadata"][key] = val

                    if key == "base_stem":
                        seen_base_stem_in_table = True
                else:
                    seen_base_stem_in_table = True
                    if line.strip():
                        table_rows_buf.append(line)
            else:
                table_rows_buf.append(line)

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




def make_lucene_doc(table_block):


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

    doc.add(StringField("entity_id", entity_id_val, Field.Store.YES))
    doc.add(StringField("base_stem", base_stem_val, Field.Store.YES))
    doc.add(StringField("source_file", source_file_val, Field.Store.YES))

    doc.add(StringField("headline_exact", headline_val, Field.Store.YES))
    doc.add(TextField("headline", headline_val, Field.Store.NO))

    doc.add(TextField("metadata_text", metadata_text, Field.Store.YES))
    doc.add(TextField("table_text", table_text_val, Field.Store.YES))

    doc.add(TextField("fulltext", fulltext_val, Field.Store.NO))

    return doc




def build_global_index(index_dir, similarity):

    os.makedirs(index_dir, exist_ok=True)

    directory = FSDirectory.open(Paths.get(index_dir))
    analyzer = EnglishAnalyzer()

    config = IndexWriterConfig(analyzer)
    config.setSimilarity(similarity)
    writer = IndexWriter(directory, config)

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




def search_global(index_dir, query_str, similarity, limit=10):

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



if __name__ == "__main__":
    build_global_indexes_custom()

    demo_compare_search("Playoffs AND MIL")
