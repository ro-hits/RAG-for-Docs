"""
pdf_parser.py — Extract structured text from research paper PDFs
================================================================

Why this matters for RAG:
  Garbage in, garbage out. If your parser mangles tables, splits sentences
  across pages, or includes headers/footers as content, your chunks will
  be noisy and retrieval will suffer — no matter how good your embedding
  model is.

  Research papers are particularly hard because of:
  - Two-column layouts
  - Page headers/footers (journal name, page numbers)
  - Section numbering (1. Introduction, 2.1 Background...)
  - Figures, tables, equations mixed with text
  - References section (dense, differently formatted)

Dependencies:
  pip install pymupdf   # PyMuPDF — best open-source PDF parser for layout

Usage:
  from pdf_parser import parse_pdf
  doc = parse_pdf("attention_is_all_you_need.pdf")
  print(doc["sections"])
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ==================== DATA STRUCTURES ====================

@dataclass
class Section:
    """A logical section of the paper (e.g., Abstract, Introduction)."""
    title: str
    level: int          # 1 = top-level (Introduction), 2 = sub (2.1 Background)
    content: str        # the actual text
    page_start: int
    page_end: int

    def __repr__(self):
        preview = self.content[:80].replace('\n', ' ')
        return f"Section(title={self.title!r}, level={self.level}, chars={len(self.content)}, preview={preview!r}...)"


@dataclass
class ParsedDocument:
    """Full parsed document with metadata and sections."""
    filename: str
    title: str
    total_pages: int
    raw_text: str                       # full text, cleaned
    sections: list[Section] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def get_section(self, title_substr: str) -> Section | None:
        """Find section by partial title match."""
        title_substr = title_substr.lower()
        for s in self.sections:
            if title_substr in s.title.lower():
                return s
        return None

    def text_by_sections(self) -> str:
        """Reconstruct text with section markers."""
        parts = []
        for s in self.sections:
            marker = "#" * s.level
            parts.append(f"{marker} {s.title}\n\n{s.content}")
        return "\n\n".join(parts)


# ==================== CLEANING ====================

# Common header/footer patterns in research papers
HEADER_FOOTER_PATTERNS = [
    r'^[\d]+$',                                    # bare page numbers
    r'^page\s+\d+',                                # "Page 3"
    r'^\d+\s+of\s+\d+',                           # "3 of 12"
    r'^(proceedings|journal|conference|vol\.|arxiv)', # journal headers
    r'^preprint',                                   # arXiv preprints
    r'^under review',                               # submission headers
    r'^\w+\s+et\s+al\.',                           # "Smith et al." running headers
]
_header_footer_re = [re.compile(p, re.IGNORECASE) for p in HEADER_FOOTER_PATTERNS]


def _is_header_footer(line: str) -> bool:
    """Detect if a line is likely a page header or footer."""
    line = line.strip()
    if not line or len(line) > 150:  # headers/footers are short
        return False
    return any(p.match(line) for p in _header_footer_re)


def _clean_text(text: str) -> str:
    """Clean extracted text — fix common PDF extraction artifacts."""
    # Fix hyphenated line breaks: "algo-\nrithm" → "algorithm"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse multiple newlines but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix ligatures that PDF extractors sometimes miss
    replacements = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '--',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove isolated single characters on their own line (often column artifacts)
    text = re.sub(r'\n[a-zA-Z]\n', '\n', text)

    return text.strip()


# ==================== SECTION DETECTION ====================

# Patterns for section headers in research papers
SECTION_PATTERNS = [
    # Numbered: "1. Introduction", "2.1 Background", "3.2.1 Details"
    (re.compile(r'^(\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,80})$', re.MULTILINE), None),

    # Roman numerals: "I. Introduction", "II. Related Work"
    (re.compile(r'^((?:I{1,3}|IV|VI{0,3}|IX|X{0,3}))\.\s+([A-Z][^\n]{2,80})$', re.MULTILINE), None),

    # All-caps: "INTRODUCTION", "RELATED WORK" (common in IEEE format)
    (re.compile(r'^([A-Z][A-Z\s]{4,60})$', re.MULTILINE), "allcaps"),

    # Abstract (special case — not always numbered)
    (re.compile(r'^(Abstract|ABSTRACT)\s*$', re.MULTILINE), "abstract"),
]


def _detect_sections(text: str) -> list[dict]:
    """
    Find section boundaries in paper text.

    Returns list of {title, level, start_pos} sorted by position.
    """
    found = []

    for pattern, ptype in SECTION_PATTERNS:
        for match in pattern.finditer(text):
            if ptype == "allcaps":
                title = match.group(1).strip().title()
                level = 1
                start = match.start()
            elif ptype == "abstract":
                title = "Abstract"
                level = 1
                start = match.start()
            else:
                number = match.group(1)
                title = match.group(2).strip()
                level = number.count('.') + 1  # "2.1" → level 2
                title = f"{number} {title}"
                start = match.start()

            # Skip very short "sections" (likely false positives)
            if len(title) < 3:
                continue

            found.append({"title": title, "level": level, "start": start})

    # Deduplicate overlapping matches (within 10 chars of each other)
    found.sort(key=lambda x: x["start"])
    deduped = []
    for item in found:
        if deduped and abs(item["start"] - deduped[-1]["start"]) < 10:
            # Keep the one with more detail (longer title)
            if len(item["title"]) > len(deduped[-1]["title"]):
                deduped[-1] = item
        else:
            deduped.append(item)

    return deduped


# ==================== MAIN PARSER ====================

def parse_pdf(filepath: str | Path) -> ParsedDocument:
    """
    Parse a research paper PDF into structured sections.

    Args:
        filepath: path to PDF file

    Returns:
        ParsedDocument with extracted sections, metadata, and cleaned text

    What this does:
      1. Extract raw text page by page (PyMuPDF handles multi-column)
      2. Clean PDF artifacts (ligatures, hyphenation, headers/footers)
      3. Detect section boundaries using regex patterns
      4. Split text into Section objects with metadata
    """
    filepath = Path(filepath)
    if not HAS_PYMUPDF:
        raise ImportError("pip install pymupdf  # required for PDF parsing")
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")

    doc = fitz.open(str(filepath))

    # Step 1: Extract text page by page
    page_texts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        # "text" mode preserves reading order; "blocks" mode is layout-aware
        text = page.get_text("text")
        page_texts.append(text)

    # Step 2: Clean each page, remove headers/footers
    cleaned_pages = []
    for text in page_texts:
        lines = text.split('\n')
        # Remove first and last 2 lines if they look like headers/footers
        filtered = []
        for i, line in enumerate(lines):
            if (i < 2 or i >= len(lines) - 2) and _is_header_footer(line):
                continue
            filtered.append(line)
        cleaned_pages.append('\n'.join(filtered))

    # Step 3: Join all pages and clean
    full_text = _clean_text('\n\n'.join(cleaned_pages))

    # Step 4: Extract metadata
    pdf_meta = doc.metadata or {}
    title = pdf_meta.get("title", "")

    # Fallback: use first non-empty line as title if metadata is empty
    if not title:
        for line in full_text.split('\n'):
            line = line.strip()
            if len(line) > 10 and not line[0].isdigit():
                title = line
                break

    # Step 5: Detect and extract sections
    section_markers = _detect_sections(full_text)
    sections = []

    if section_markers:
        for i, marker in enumerate(section_markers):
            # Content = text from this marker to next marker (or end)
            start = marker["start"]
            end = section_markers[i + 1]["start"] if i + 1 < len(section_markers) else len(full_text)

            # Skip the header line itself
            content_start = full_text.index('\n', start) + 1 if '\n' in full_text[start:end] else start
            content = full_text[content_start:end].strip()

            # Figure out which pages this spans
            char_count = 0
            page_start = 0
            page_end = 0
            for pg_idx, pt in enumerate(cleaned_pages):
                if char_count + len(pt) >= start and page_start == 0:
                    page_start = pg_idx + 1
                if char_count + len(pt) >= end:
                    page_end = pg_idx + 1
                    break
                char_count += len(pt) + 2  # +2 for the \n\n join
            else:
                page_end = len(cleaned_pages)

            sections.append(Section(
                title=marker["title"],
                level=marker["level"],
                content=content,
                page_start=page_start,
                page_end=page_end,
            ))
    else:
        # No sections detected — treat entire text as one section
        sections.append(Section(
            title="Full Document",
            level=1,
            content=full_text,
            page_start=1,
            page_end=len(doc),
        ))

    doc.close()

    return ParsedDocument(
        filename=filepath.name,
        title=title,
        total_pages=len(page_texts),
        raw_text=full_text,
        sections=sections,
        metadata={
            "author": pdf_meta.get("author", ""),
            "subject": pdf_meta.get("subject", ""),
            "creator": pdf_meta.get("creator", ""),
            "keywords": pdf_meta.get("keywords", ""),
        },
    )


# ==================== CONVENIENCE ====================

def print_structure(doc: ParsedDocument):
    """Print document structure overview."""
    print(f"Document: {doc.title}")
    print(f"File: {doc.filename} ({doc.total_pages} pages)")
    print(f"Total chars: {len(doc.raw_text):,}")
    print(f"Sections: {len(doc.sections)}")
    print()
    for s in doc.sections:
        indent = "  " * (s.level - 1)
        print(f"  {indent}{s.title} ({len(s.content):,} chars, pp.{s.page_start}-{s.page_end})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <paper.pdf>")
        sys.exit(1)
    doc = parse_pdf(sys.argv[1])
    print_structure(doc)
