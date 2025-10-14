import os
import re
import csv
import time
import html
import hashlib
import random
from turtledemo.penrose import start
import requests
from urllib.parse import urljoin, urlparse
from collections import deque


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^a-zA-Z0-9\-_]+", "", text)
    text = text.strip("-_")
    return text[:max_len] or "page"

def page_hash(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]

def clean_cell(s: str) -> str:
    s = re.sub(r"(?is)<script.*?</script>|<style.*?</style>", "", s)
    s = re.sub(r"(?is)<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s.strip()

def extract_tables(html_text: str):
    tables = []
    for tbl_html in re.findall(r"(?is)<table\b[^>]*>(.*?)</table>", html_text):
        rows_data = []
        rows = re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", tbl_html)
        for row in rows:
            cells = re.findall(r"(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>", row)
            if not cells:
                continue
            cleaned = [clean_cell(c) for c in cells]
            if any(cell for cell in cleaned):
                rows_data.append(cleaned)
        if rows_data:
            tables.append(rows_data)
    return tables

def write_tsv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

def write_html(path: str, html_text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)


class FullSiteCrawler:
    def __init__(
        self,
        start_url,
        out_dir,
        max_depth,
        delay,
        headers
    ):
        self.start_url = start_url
        u = urlparse(start_url)
        self.domain = u.netloc
        self.scheme = u.scheme or "https"
        self.out_dir = out_dir
        self.pages_dir = os.path.join(out_dir, "pages")
        self.tables_dir = os.path.join(out_dir, "tables")
        self.max_depth = max_depth
        self.delay = delay
        self.visited = set()

        self.headers = headers

        self.blocked_slugs = {
            "/wp-admin/", "/contact-us/", "/go/", "/beardeddragon/",
            "/detroitchicago/", "/ezoic/", "/porpoiseant/", "/parsonsmaize/",
            "/xmlrpc.php", "/wp-login.php", "/wp-cron.php", "/wp-json/",
            "/feed/", "/comments/"
        }

    def allowed(self, url: str) -> bool:
        pu = urlparse(url)
        if pu.netloc != self.domain:
            return False

        path = pu.path or "/"
        if pu.query:
            return False
        if re.search(r"\.(xlsx|pdf)(?:$|\?)", path, re.I):
            return False
        if any(path.startswith(s) for s in self.blocked_slugs):
            return False

        return True

    def fetch(self, url: str) -> str:
        if not self.allowed(url):
            print(f"Disallowed by policy: {url}")
            return ""
        try:
            print(f"Fetch {url}")
            r = requests.get(url, headers=self.headers, timeout=15)
            r.raise_for_status()
            ctype = r.headers.get("Content-Type", "")
            if not ctype.lower().startswith("text/html"):
                print(f"Skipping non-HTML ({ctype})")
                return ""
            time.sleep(random.uniform(*self.delay))
            return r.text
        except requests.RequestException as e:
            print(f"Fetch error: {e}")
            return ""

    def extract_links(self, html_text: str, base_url: str):
        hrefs = re.findall(r'(?i)href=["\'](.*?)["\']', html_text)
        links, seen = [], set()
        for h in hrefs:
            full = urljoin(base_url, h)
            if full in seen:
                continue
            seen.add(full)
            if self.allowed(full):
                links.append(full)
        return links

    def title_slug(self, html_text: str, url: str) -> str:
        m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html_text)
        if m:
            return slugify(clean_cell(m.group(1)))
        path_part = slugify(urlparse(url).path.strip("/").replace("/", "-")) or "page"
        return f"{path_part}-{page_hash(url)}"

    def save_page(self, url: str, html_text: str) -> str:
        base = self.title_slug(html_text, url)
        path = os.path.join(self.pages_dir, f"{base}.html")
        write_html(path, html_text)
        print(f"Saved HTML → {path}")
        return base

    def save_tables(self, base_stem: str, html_text: str) -> int:
        tables = extract_tables(html_text)
        if not tables:
            return 0

        count = 0
        for i, tbl in enumerate(tables, start=1):
            if len(tbl) < 3 or max(len(r) for r in tbl) < 3:
                continue
            width = max(len(r) for r in tbl)
            norm = [r + ([""] * (width - len(r))) for r in tbl]
            tsv_path = os.path.join(self.tables_dir, f"{base_stem}__table{i}.tsv")
            write_tsv(tsv_path, norm)
            count += 1
            print(f"Saved TSV ({len(norm)}×{width}) → {tsv_path}")
        return count

    def crawl(self):
        os.makedirs(self.pages_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

        q = deque([(self.start_url, 0)])
        total_tables = 0
        total_pages = 0

        while q:
            url, depth = q.popleft()
            if url in self.visited or depth > self.max_depth:
                continue
            self.visited.add(url)

            print(f"\n🔎 Depth {depth}: {url}")
            html_text = self.fetch(url)
            if not html_text:
                continue

            base_stem = self.save_page(url, html_text)
            total_pages += 1

            if re.search(r"(?is)<tr\b[^>]*>.*?</tr>", html_text):
                total_tables += self.save_tables(base_stem, html_text)


            if depth < self.max_depth:
                for link in self.extract_links(html_text, url):
                    if link not in self.visited:
                        q.append((link, depth + 1))

        print(f"\nSaved {total_pages} HTML pages and {total_tables} TSV table files.")
        return total_pages, total_tables


def main():
    start_url = "https://www.nbastuffer.com/"
    output_dir = "out_dir"
    depth = 3
    min_delay = 10.0
    max_delay = 20.0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/127.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


    crawler = FullSiteCrawler(
        start_url=start_url,
        out_dir=output_dir,
        max_depth=depth,
        delay=(min_delay, max_delay),
        headers=headers
    )
    crawler.crawl()



if __name__ == "__main__":
    main()

