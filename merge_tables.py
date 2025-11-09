import os
import csv
import re
import glob

TABLES_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\out_dir_headings\tables"
OUT_DIR = r"C:\Users\lujvi\PycharmProjects\VINF_crawler\merged_tables"
os.makedirs(OUT_DIR, exist_ok=True)


def parse_single_tsv_for_page_merge(path):


    filename = os.path.basename(path)

    # Extract base_stem and headline slug from filename:
    # e.g.  some-page-slug__Per-Game-Playoffs__table2.tsv
    m = re.match(r"(.+?)__(.+?)__table\d+\.tsv$", filename)
    if m:
        base_stem = m.group(1)
        headline_slug = m.group(2)
    else:
        # fallback for unexpected filenames
        base_stem = filename
        headline_slug = "unknown"

    headline = ""
    header = None
    data_rows = []
    metadata = {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if not row:
                continue

            # metadata lines look like:
            # # key: value
            if row[0].startswith("#"):
                # Try to parse "# key: value"
                # Remove leading '# ' or '#'
                raw = row[0].lstrip("#").strip()
                # If row had multiple columns (usually it doesn't), join them
                if len(row) > 1:
                    # just in case metadata was split weirdly into columns
                    raw = " ".join([raw] + row[1:]).strip()

                # Pattern key: value
                m_meta = re.match(r"([^:]+):\s*(.*)$", raw)
                if m_meta:
                    key = m_meta.group(1).strip()
                    val = m_meta.group(2).strip()
                    metadata[key] = val
                continue

            # Title marker row:
            # __table_title__    Per Game — Playoffs
            if row[0] == "__table_title__":
                if len(row) > 1:
                    headline = row[1].strip()
                continue

            # First non-metadata, non-title row is the header
            if header is None:
                header = row
            else:
                data_rows.append(row)

    if not headline:
        headline = headline_slug.replace("-", " ")

    metadata.setdefault("source_file", filename)

    metadata.setdefault("base_stem", base_stem)

    metadata.setdefault("headline", headline)

    return {
        "base_stem": base_stem,
        "headline": headline,
        "header": header if header else [],
        "rows": data_rows,
        "source_file": filename,
        "metadata": metadata,
    }


def append_table_block(out_path, table_dict):

    file_exists = os.path.exists(out_path)

    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")

        if not file_exists:
            w.writerow(["###PAGE_BEGIN###", table_dict["base_stem"]])

        w.writerow(["###TABLE_BEGIN###"])

        w.writerow(["headline", table_dict["headline"]])
        w.writerow(["source_file", table_dict["source_file"]])


        for k, v in table_dict["metadata"].items():
            if k in ("headline", "source_file"):
                continue
            w.writerow([k, v])

        # Now the actual table content
        if table_dict["header"]:
            w.writerow(table_dict["header"])

        for row in table_dict["rows"]:
            w.writerow(row)

        # End marker
        w.writerow(["###TABLE_END###"])


def merge_by_page_streaming():
    pattern = os.path.join(TABLES_DIR, "*.tsv")

    for idx, path in enumerate(glob.iglob(pattern)):
        table_dict = parse_single_tsv_for_page_merge(path)

        out_name = f"{table_dict['base_stem']}__ALL.tsv"
        out_path = os.path.join(OUT_DIR, out_name)

        append_table_block(out_path, table_dict)

        if idx % 10000 == 0:
            print(f"Processed {idx} tables... (latest base_stem={table_dict['base_stem']})")

    print("Done.")


if __name__ == "__main__":
    merge_by_page_streaming()
