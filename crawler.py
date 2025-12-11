import os
import re
import csv
import time
import html
import hashlib
import random
import requests
from urllib import robotparser
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode
from collections import deque
from datetime import datetime

def write_tsv_with_metadata(path: str, rows, metadata: dict):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# METADATA BEGIN\n")
        for key, val in metadata.items():
            f.write(f"# {key}: {val}\n")
        f.write("# METADATA END\n\n")

        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"\s+", "-", text.strip())
    text = re.sub(r"[^a-zA-Z0-9\-_]+", "", text)
    text = text.strip("-_")
    return text[:max_len] or "page"


def nearest_headline_above(html_text: str, table_start: int, lookback_chars: int = 90000) -> str:
    start = max(0, table_start - lookback_chars)
    ctx = html_text[start:table_start]


    wrapper_ids = re.findall(r'(?is)<div[^>]*id=["\']([^"\']+)["\'][^>]*>', ctx)
    season_mode = ""
    if wrapper_ids:
        wrapper_id = wrapper_ids[-1].lower()
        if ("post" in wrapper_id) or ("playoff" in wrapper_id):
            season_mode = "Playoffs"

    headings = re.findall(r"(?is)<h[1-6][^>]*>(.*?)</h[1-6]>", ctx)
    main_title = clean_cell(headings[-1]) if headings else ""


    if main_title and season_mode:
        return f"{main_title} — {season_mode}"
    if main_title:
        return main_title

    return ""


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


def extract_named_tables(html_text: str):

    results = []


    html_text = re.sub(r"(?is)<!--(.*?)-->", r"\1", html_text)

    for m in re.finditer(r"(?is)<table\b[^>]*>.*?</table>", html_text):
        table_html = m.group(0)
        inner_match = re.search(r"(?is)<table\b[^>]*>(.*)</table>", table_html)
        if not inner_match:
            continue

        inner_html = inner_match.group(1)

        rows_data = []
        for row_html in re.findall(r"(?is)<tr\b[^>]*>(.*?)</tr>", inner_html):
            cell_htmls = re.findall(r"(?is)<t[hd]\b[^>]*>(.*?)</t[hd]>", row_html)
            if not cell_htmls:
                continue
            cleaned_cells = [clean_cell(c) for c in cell_htmls]
            if any(c.strip() for c in cleaned_cells):
                rows_data.append(cleaned_cells)

        if not rows_data:
            continue

        title = nearest_headline_above(html_text, m.start())
        results.append({
            "headline": title,
            "rows": rows_data,
        })

    return results

def write_tsv(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

def write_html(path: str, html_text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)



def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    parsed = parsed._replace(fragment="")

    allowed_params = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        kl = k.lower()
        if kl.startswith("utm_"):
            continue
        if kl in ("fbclid", "gclid", "yclid", "ref", "source"):
            continue
        allowed_params.append((k, v))

    new_query = urlencode(allowed_params, doseq=True)

    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")

    cleaned = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path,
        query=new_query,
    )

    normalized = urlunparse(cleaned)

    if normalized.endswith("?"):
        normalized = normalized[:-1]

    return normalized



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

        self.user_agent: str = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        )

        self.blocked_slugs = {
            "/wp-admin/", "/contact-us/", "/go/", "/beardeddragon/",
            "/detroitchicago/", "/ezoic/", "/porpoiseant/", "/parsonsmaize/",
            "/xmlrpc.php", "/wp-login.php", "/wp-cron.php", "/wp-json/",
            "/feed/", "/comments/", "/blog", "/analytics-101"
        }
        self.rp = robotparser.RobotFileParser()
        self.rp.set_url(f"{self.scheme}://{self.domain}/robots.txt")
        try:
            self.rp.read()
        except Exception:
            pass

        self.explicit_deny_regexes = [
            re.compile(r"^/basketball/"),
            re.compile(r"^/blazers/"),
            re.compile(r"^/dump/"),
            re.compile(r"^/fc/"),
            re.compile(r"^/my/"),
            re.compile(r"^/7103"),
            re.compile(r"^/req/"),
            re.compile(r"^/short/"),
            re.compile(r"^/nocdn/"),

            re.compile(r"/play-index/[^/]*\.cgi(\?.*)?$", re.I),
            re.compile(r"/play-index/plus/[^/]*\.cgi(\?.*)?$", re.I),
            re.compile(r"/gamelog/"),
            re.compile(r"/splits/"),
            re.compile(r"/on-off/"),
            re.compile(r"/lineups/"),
            re.compile(r"/shooting/"),
        ]

        self.utility_block = {
            "/xmlrpc.php", "/wp-login.php", "/wp-cron.php", "/wp-json/", "/feed/", "/comments/",
        }

    def allowed_by_explicit(self, url: str) -> bool:
        pu = urlparse(url)
        path = pu.path or "/"
        if any(path.startswith(s) for s in self.utility_block):
            return False
        if re.search(r"\.(pdf|xlsx|docx?|pptx?)$", path, re.I):
            return False

        for rgx in self.explicit_deny_regexes:
            if rgx.search(path):
                return False
        return True

    def allowed(self, url: str) -> bool:
        pu = urlparse(url)
        if pu.netloc != self.domain:
            return False
        try:
            if not self.rp.can_fetch(self.user_agent, url):
                return False
        except Exception:
            pass
        return self.allowed_by_explicit(url)

    def fetch(self, url: str) -> str:
        if not self.allowed(url):
            print(f"Disallowed by policy: {url}")
            return ""
        try:
            print(f"Fetch {url}")
            r = requests.get(url, headers=self.headers, timeout=15)
            if r.status_code == 429:
                wait = random.uniform(60, 120)
                print(f"🚦 Hit 429. Sleeping for {wait:.1f}s...")
                time.sleep(wait)
                return ""
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
            full = normalize_url(full)
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

    def save_tables(self, base_stem: str, html_text: str, page_url: str = "", depth: int = 0) -> int:
        named_tables = extract_named_tables(html_text)
        if not named_tables:
            return 0

        count = 0
        for idx, tbl in enumerate(named_tables, start=1):
            headline = tbl["headline"].strip() or "unknown_table"
            rows = tbl["rows"]
            if not rows:
                continue

            if len(rows) < 3 or max(len(r) for r in rows) < 3:
                continue

            width = max(len(r) for r in rows)
            norm_rows = [r + [""] * (width - len(r)) for r in rows]
            decorated_rows = [["__table_title__", headline]] + norm_rows

            headline_slug = slugify(headline, max_len=60)
            tsv_name = f"{base_stem}__{headline_slug}__table{idx}.tsv"
            tsv_path = os.path.join(self.tables_dir, tsv_name)

            metadata = {
                "source_url": page_url,
                "headline": headline,
                "file_name": tsv_name,
                "crawl_depth": depth,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "row_count": len(norm_rows),
                "col_count": width,
            }

            write_tsv_with_metadata(tsv_path, decorated_rows, metadata)

            count += 1
            print(f"[TABLE] Saved {headline} ({len(norm_rows)}×{width}) → {tsv_path}")

        return count

    def crawl(self):
        os.makedirs(self.pages_dir, exist_ok=True)
        os.makedirs(self.tables_dir, exist_ok=True)

        q = deque([(self.start_url, 0)])
        total_tables = 0
        total_pages = 0

        while q:
            url, depth = q.popleft()
            url = normalize_url(url)
            if url in self.visited or depth > self.max_depth:
                continue
            self.visited.add(url)

            print(f"\n Depth {depth}: {url}")
            html_text = self.fetch(url)
            if not html_text:
                continue

            base_stem = self.save_page(url, html_text)
            total_pages += 1

            if re.search(r"(?is)<tr\b[^>]*>.*?</tr>", html_text):
                total_tables += self.save_tables(base_stem, html_text, page_url=url, depth=depth)

            if depth < self.max_depth:
                for link in self.extract_links(html_text, url):
                    link = normalize_url(link)
                    if link not in self.visited:
                        q.append((link, depth + 1))

        print(f"\nSaved {total_pages} HTML pages and {total_tables} TSV table files.")
        return total_pages, total_tables


def main():
    start_url = "https://www.basketball-reference.com/"
    output_dir = "out_dir_headings"
    depth = 3
    min_delay = 4.0
    max_delay = 6.0
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/127.0.0.0 Safari/537.36"
                        "contact: xvitarius@stuba.sk; purpose: academic-research)"
                       ),

        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
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

