import argparse
import os
import shutil
import hashlib
from pathlib import Path

def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

def unique_name(dest_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = dest_dir / filename
    i = 1
    while candidate.exists():
        candidate = dest_dir / f"{base}_{i}{suffix}"
        i += 1
    return candidate

def build_dest_hash_set(dest: Path) -> set[str]:
    hashes = set()
    if not dest.exists():
        return hashes
    for root, _, files in os.walk(dest):
        for name in files:
            p = Path(root) / name
            try:
                hashes.add(file_hash(p))
            except Exception as e:
                print(f"Warning: couldn't hash {p}: {e}")
    return hashes

def main():
    parser = argparse.ArgumentParser(
        description="Copy all files whose names contain a pattern to another folder, skipping duplicates by content."
    )
    parser.add_argument("source", type=Path, help="Source folder to search (searched recursively)")
    parser.add_argument("dest", type=Path, help="Destination folder to copy files into")
    parser.add_argument("--pattern", default="-Height",
                        help="Substring to match in filenames (default: '-Height')")
    args = parser.parse_args()

    if not args.source.is_dir():
        parser.error(f"Source '{args.source}' is not a directory or doesn't exist.")

    args.dest.mkdir(parents=True, exist_ok=True)

    dest_hashes = set()
    existing_names = set()
    if args.name_only_duplicates:
        for root, _, files in os.walk(args.dest):
            for n in files:
                existing_names.add(n)
    else:
        dest_hashes = build_dest_hash_set(args.dest)

    matched = 0
    copied = 0
    skipped_duplicates = 0

    pat = args.pattern if not args.ignore_case else args.pattern.lower()

    for root, _, files in os.walk(args.source):
        for name in files:
            haystack = name if not args.ignore_case else name.lower()
            if pat in haystack:
                src_path = Path(root) / name

                is_duplicate = False
                if args.name_only_duplicates:
                    if name in existing_names:
                        is_duplicate = True
                else:
                    try:
                        h = file_hash(src_path)
                        if h in dest_hashes:
                            is_duplicate = True
                    except Exception as e:
                        print(f"Warning: couldn't hash {src_path}: {e}")

                if is_duplicate:
                    print(f"Skip (duplicate): {src_path}")
                    skipped_duplicates += 1
                    matched += 1
                    continue

                dst_path = (args.dest / name)
                if dst_path.exists():
                    dst_path = unique_name(args.dest, name)


                shutil.copy2(src_path, dst_path)
                print(f"Copied: {src_path} -> {dst_path}")
                copied += 1
                if args.name_only_duplicates:
                    existing_names.add(dst_path.name)
                else:
                    try:
                        if 'h' in locals():
                            dest_hashes.add(h)
                        else:
                            dest_hashes.add(file_hash(dst_path))
                    except Exception as e:
                        print(f"Warning: couldn't hash {dst_path}: {e}")

                matched += 1

    if matched == 0:
        print("No matching files found.")
    else:
        print(f"Done. Matched: {matched}, Copied: {copied}, Skipped as duplicates: {skipped_duplicates}.")

if __name__ == "__main__":
    main()
