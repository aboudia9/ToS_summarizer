"""Turn uploaded documents (.txt, .html, .docx, .pdf) into the plain text the pipeline
expects: one paragraph per line.

That matters because split_clauses works line by line (so headings stay on their own),
which means a sentence broken across lines gets chopped in half. HTML and PDFs are full
of those breaks: inline tags, and PDF text wrapped at the edge of every page. So each
format is put back into whole paragraphs here, and page clutter (menus, scripts, running
headers/footers, page numbers) is dropped before BERT ever sees it.

    text = extract_text(data, "terms.pdf")

These parse untrusted files. The web app runs them in a short-lived child process with
time and memory limits (webapp/extract_runner.py); this module just does the parsing.
"""
import io
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

SUPPORTED_TYPES = (".txt", ".html", ".htm", ".docx", ".pdf")
MAX_PDF_PAGES = 300
MAX_DOCX_XML_BYTES = 30_000_000   # uncompressed, so a zip bomb can't balloon in memory
MIN_TEXT_CHARS = 200


class ExtractionError(ValueError):
    """Something the user can act on; the message is shown to them as is."""


def extract_text(data, filename):
    ext = Path(filename).suffix.lower()
    if ext == ".txt":
        text = _decode(data)
    elif ext in (".html", ".htm"):
        text = html_to_text(data)
    elif ext == ".docx":
        text = docx_to_text(data)
    elif ext == ".pdf":
        text = pdf_to_text(data)
    else:
        raise ExtractionError(f"Can't read {ext or 'that'} files. Try .txt, .pdf, .docx or .html.")

    text = _normalize(text)
    if len(text) < MIN_TEXT_CHARS:
        raise ExtractionError("Couldn't find much text in that file. If it's a web page, try "
                              "copying the terms from your browser and pasting them instead.")
    return text


def _decode(data):
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _normalize(text):
    # NFKC turns PDF ligatures (ﬁ, ﬂ) and odd widths back into plain letters
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[­​‌‍⁠﻿]", "", text)   # soft hyphens, zero-widths
    lines = (re.sub(r"[ \t  -   　]+", " ", line).strip()
             for line in text.splitlines())
    return "\n".join(line for line in lines if line)


# ---------------- HTML ----------------

# Never part of the terms themselves
HTML_DROP = ["script", "style", "noscript", "template", "svg", "canvas", "iframe", "object",
             "nav", "header", "footer", "aside", "form", "button", "select", "label", "dialog",
             "menu"]
HTML_DROP_ROLES = {"navigation", "banner", "contentinfo", "search", "dialog", "alert", "menu"}
# Elements that start a new line in a browser; everything else is inline
HTML_BLOCKS = ["address", "article", "blockquote", "dd", "div", "dl", "dt", "figcaption",
               "figure", "h1", "h2", "h3", "h4", "h5", "h6", "hr", "li", "main", "ol", "p",
               "pre", "section", "table", "tr", "td", "th", "ul", "summary", "details"]


def html_to_text(data):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data, "html.parser")
    for tag in soup(HTML_DROP):
        tag.decompose()
    for tag in soup.find_all(_is_hidden):
        if not tag.decomposed:     # may already be gone with a hidden parent
            tag.decompose()

    body = soup.body or soup
    root = _main_content(body)

    # Put a line break around every block element and <br>, then read the text with no
    # separator, so inline tags (<a>, <b>, <span>) stay inside their sentence
    for br in root.find_all("br"):
        br.replace_with("\n")
    for block in root.find_all(HTML_BLOCKS):
        block.insert_before("\n")
        block.insert_after("\n")
    return root.get_text()


def _is_hidden(tag):
    style = (tag.get("style") or "").replace(" ", "").lower()
    return (tag.has_attr("hidden") or tag.get("aria-hidden") == "true"
            or "display:none" in style or "visibility:hidden" in style
            or tag.get("role") in HTML_DROP_ROLES)


def _main_content(body):
    # Prefer the page's main area when it holds most of the text, which drops sidebars
    # and cookie banners that live outside it
    total = len(body.get_text(" ", strip=True))
    for candidate in (body.find("main"), body.find(attrs={"role": "main"}), body.find("article")):
        if candidate and len(candidate.get_text(" ", strip=True)) >= 0.5 * total:
            return candidate
    return body


# ---------------- Word (.docx) ----------------

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def docx_to_text(data):
    # A .docx is a zip of XML files; the body text is in word/document.xml
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            info = z.getinfo("word/document.xml")
            if info.file_size > MAX_DOCX_XML_BYTES:
                raise ExtractionError("That Word document is too large to read.")
            xml = z.read(info)
    except (zipfile.BadZipFile, KeyError):
        raise ExtractionError("That doesn't look like a Word (.docx) file. Older .doc files "
                              "aren't supported; save it as .docx or .pdf first.")
    # Word never writes DTDs; refusing them rules out entity-expansion tricks
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise ExtractionError("That Word document couldn't be read.")

    paragraphs = []
    for p in ElementTree.fromstring(xml).iter(W + "p"):
        parts = []
        for node in p.iter():
            if node.tag == W + "t":
                parts.append(node.text or "")
            elif node.tag == W + "tab":
                parts.append(" ")
            elif node.tag in (W + "br", W + "cr"):
                parts.append("\n")
        paragraphs.append("".join(parts))
    return "\n".join(paragraphs)


# ---------------- PDF ----------------

PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.IGNORECASE)
BULLET = re.compile(r"^[•●▪◦‣·*–—-]\s")
# "1.", "(a)", "iv)" etc. Only treated as a new item after a comma-ended line, since
# "(200) cases" can just as well be the middle of a sentence that wrapped
NUMBERED = re.compile(r"^(\(?\d{1,2}(\.\d{1,2})*[.)]|\(?[a-zA-Z][.)]|\(?[ivxIVX]{1,5}[.)])\s")
# Lines vary in character count even when full (proportional fonts), so a line only
# counts as stopped early when it ends well short of a full one
FIT_SLACK = 0.75   # tuned on Chrome-printed PDFs of the example ToS pages
SENTENCE_END = re.compile(r"[.!?:;][\"'”’)\]]*$")


def pdf_to_text(data):
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("That PDF is password protected.")
        if len(reader.pages) > MAX_PDF_PAGES:
            raise ExtractionError(f"That PDF has over {MAX_PDF_PAGES} pages.")
        pages = [(page.extract_text() or "").splitlines() for page in reader.pages]
    except PdfReadError:
        raise ExtractionError("That PDF couldn't be read. It may be damaged.")

    if sum(len(line.strip()) for lines in pages for line in lines) < 20 * len(pages):
        raise ExtractionError("That PDF seems to be scanned images, with no text to read. "
                              "OCR isn't supported yet.")

    lines = [line for page in _strip_page_furniture(pages) for line in page]
    return "\n".join(_reflow(lines))


def _strip_page_furniture(pages, edge=3):
    # Running headers/footers repeat at the top or bottom of most pages, usually with a
    # changing page number, so compare lines with their digits masked out
    def key(line):
        return re.sub(r"\d+", "#", line.strip().lower())

    repeated = set()
    if len(pages) >= 3:
        counts = Counter(k for lines in pages
                         for k in {key(l) for l in lines[:edge] + lines[-edge:] if l.strip()})
        repeated = {k for k, n in counts.items() if n >= max(3, len(pages) / 2)}

    cleaned = []
    for lines in pages:
        lines = [l for l in lines if l.strip()]
        keep = []
        for i, line in enumerate(lines):
            at_edge = i < edge or i >= len(lines) - edge
            if at_edge and (key(line) in repeated or PAGE_NUMBER.match(line.strip())):
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned


def _reflow(lines):
    # Join lines that were only broken to fit the page width back into paragraphs.
    # A line break is kept when it was intentional: the line finishes a sentence, the
    # next line starts a bulleted item, or the next line's first word would have fit on
    # this line (so the text stopped early on purpose: a heading, a list item, the end of
    # a paragraph). Breaking at a sentence end is harmless, since the sentence splitter
    # splits there anyway. What has to be avoided is breaking *mid*-sentence
    lines = [l.strip() for l in lines if l.strip()]
    if not lines:
        return []
    widths = sorted(len(l) for l in lines)
    full_width = widths[int(0.9 * (len(widths) - 1))]
    # A hyphen at a line end is often a real one ("third-party", "service-specific"), so
    # it's only dropped when the document also spells the joined word without it
    words = set(re.findall(r"[a-z]+", " ".join(lines).lower()))

    paragraphs, current, prev = [], "", ""
    for line in lines:
        first_word = line.split(" ", 1)[0]
        # ...unless the next line starts lowercase, which almost always means it continues
        stopped_early = (len(prev) + 1 + len(first_word) < FIT_SLACK * full_width
                         and not line[:1].islower())
        if not current:
            current = line
        elif (BULLET.match(line) or SENTENCE_END.search(prev) or stopped_early
              or (NUMBERED.match(line) and prev.endswith(","))):
            paragraphs.append(current)
            current = line
        elif re.search(r"[A-Za-z]-$", prev) and line[:1].islower():
            head = re.search(r"([A-Za-z]+)-$", current).group(1)
            tail = re.match(r"[a-z]+", line).group(0)
            joined = (head + tail).lower() in words
            current = (current[:-1] if joined else current) + line   # word split across lines
        else:
            current += " " + line
        prev = line
    paragraphs.append(current)
    return paragraphs
