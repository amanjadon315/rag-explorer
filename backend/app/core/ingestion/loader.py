import io
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class LoadResult:
    text: str
    filename: str
    file_type: str
    char_count: int
    word_count: int


class DocumentLoader:
    """
    Extracts raw text from uploaded files.
    Supports: .txt, .md, .pdf, .docx, .html
    """

    SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".html", ".htm"}

    def load_bytes(self, content: bytes, filename: str) -> LoadResult:
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {self.SUPPORTED}")

        if suffix in (".txt", ".md"):
            text = content.decode("utf-8", errors="replace")
        elif suffix == ".pdf":
            text = self._load_pdf(content)
        elif suffix == ".docx":
            text = self._load_docx(content)
        elif suffix in (".html", ".htm"):
            text = self._load_html(content)
        else:
            text = content.decode("utf-8", errors="replace")

        return LoadResult(
            text=text,
            filename=filename,
            file_type=suffix.lstrip("."),
            char_count=len(text),
            word_count=len(text.split()),
        )

    def load_text(self, text: str, filename: str = "paste.txt") -> LoadResult:
        return LoadResult(
            text=text,
            filename=filename,
            file_type="txt",
            char_count=len(text),
            word_count=len(text.split()),
        )

    def _load_pdf(self, content: bytes) -> str:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)

    def _load_docx(self, content: bytes) -> str:
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _load_html(self, content: bytes) -> str:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")
