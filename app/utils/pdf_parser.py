import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _table_to_markdown(rows: list) -> str:
    """Convert a pdfplumber table (list of rows) to a markdown table string."""
    if not rows:
        return ""
    rows = [[str(cell).strip() if cell is not None else "" for cell in row] for row in rows]
    # Skip entirely empty tables
    if not any(any(cell for cell in row) for row in rows):
        return ""
    header    = "| " + " | ".join(rows[0]) + " |"
    separator = "| " + " | ".join("---" for _ in rows[0]) + " |"
    body      = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, separator] + body)


def parse_pdf(file_path: str, document_name: str) -> list:
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            parts = []

            # Locate table bounding boxes so we can exclude their cells from prose text
            table_bboxes = [t.bbox for t in page.find_tables()]

            if table_bboxes:
                def outside_tables(obj):
                    for x0, top, x1, bottom in table_bboxes:
                        if (obj.get("x0", 0) >= x0 - 1 and obj.get("top",    0) >= top    - 1 and
                                obj.get("x1", 0) <= x1 + 1 and obj.get("bottom", 0) <= bottom + 1):
                            return False
                    return True
                text = page.filter(outside_tables).extract_text() or ""
            else:
                text = page.extract_text() or ""

            if text.strip():
                parts.append(text.strip())

            # Append each table as a formatted markdown block
            for table_rows in page.extract_tables():
                md = _table_to_markdown(table_rows)
                if md:
                    parts.append(md)

            pages.append({
                "text": "\n\n".join(parts),
                "page_num": page_num,
                "document_name": document_name,
            })
    return pages


def chunk_pages(pages: list) -> list:
    # RecursiveCharacterTextSplitter is the standard choice for production RAG —
    # fast, predictable, and good enough for technical documents.
    # A higher-quality alternative is SemanticChunker (langchain_experimental),
    # which embeds every sentence to find topic boundaries, keeping register
    # descriptions and section content intact. Trade-off: ingest time grows
    # significantly for large documents (e.g. 30+ min for a 1000-page datasheet).
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for page in pages:
        text = page["text"].strip()
        if not text:
            continue
        for chunk in splitter.split_text(text):
            chunks.append({
                "text": chunk,
                "page_num": page["page_num"],
                "document_name": page["document_name"],
            })
    return chunks
