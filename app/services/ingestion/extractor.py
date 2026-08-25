"""Document text extraction for PDF, DOCX, and TXT files."""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import chardet
import pypdf
import pytesseract
from pdf2image import convert_from_path
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents extracted text from a single page/section."""

    page_number: int
    text: str
    source: str  # original filename


@dataclass
class ExtractionResult:
    """Full result from a document extraction run."""

    filename: str
    file_type: str
    total_pages: int
    pages: List[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


class TextExtractor:
    """Extracts and cleans text from PDF, DOCX, and TXT documents."""

    def extract(self, file_path: str | Path, original_filename: str) -> ExtractionResult:
        """Route to the appropriate extractor based on file extension."""
        path = Path(file_path)
        ext = path.suffix.lower().lstrip(".")

        logger.info(f"Extracting text from {original_filename} (type={ext})")

        if ext == "pdf":
            return self._extract_pdf(path, original_filename)
        elif ext == "docx":
            return self._extract_docx(path, original_filename)
        elif ext == "txt":
            return self._extract_txt(path, original_filename)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    def _extract_pdf(self, path: Path, filename: str) -> ExtractionResult:
        pages: List[PageContent] = []

        with open(path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                cleaned = self._clean_text(raw)
                if cleaned:
                    pages.append(PageContent(page_number=i, text=cleaned, source=filename))

        # Fallback: OCR for scanned/image-based PDFs
        if not pages:
            logger.info(f"No text found via pypdf — falling back to OCR for {filename}")
            pages = self._ocr_pdf(path, filename)

        logger.info(f"PDF extraction complete: {len(pages)} pages with text")
        return ExtractionResult(
            filename=filename,
            file_type="pdf",
            total_pages=len(pages),
            pages=pages,
        )

    def _ocr_pdf(self, path: Path, filename: str) -> List[PageContent]:
        """Convert each PDF page to an image and run Tesseract OCR."""
        pages: List[PageContent] = []
        try:
            images = convert_from_path(str(path), dpi=200)
            for i, image in enumerate(images, start=1):
                raw = pytesseract.image_to_string(image, lang="eng")
                cleaned = self._clean_text(raw)
                if cleaned:
                    pages.append(PageContent(page_number=i, text=cleaned, source=filename))
            logger.info(f"OCR complete: {len(pages)} pages extracted from {filename}")
        except Exception as exc:
            logger.error(f"OCR failed for {filename}: {exc}")
        return pages

    # ------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------
    def _extract_docx(self, path: Path, filename: str) -> ExtractionResult:
        doc = DocxDocument(str(path))
        pages: List[PageContent] = []
        section_texts: List[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                section_texts.append(text)

        # DOCX doesn't have real page breaks; treat each 20 paragraphs as a "page"
        chunk_size = 20
        for i in range(0, max(len(section_texts), 1), chunk_size):
            block = section_texts[i : i + chunk_size]
            combined = "\n".join(block)
            cleaned = self._clean_text(combined)
            if cleaned:
                page_num = (i // chunk_size) + 1
                pages.append(PageContent(page_number=page_num, text=cleaned, source=filename))

        logger.info(f"DOCX extraction complete: {len(pages)} sections")
        return ExtractionResult(
            filename=filename,
            file_type="docx",
            total_pages=len(pages),
            pages=pages,
        )

    # ------------------------------------------------------------------
    # TXT
    # ------------------------------------------------------------------
    def _extract_txt(self, path: Path, filename: str) -> ExtractionResult:
        raw_bytes = path.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

        try:
            raw_text = raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            raw_text = raw_bytes.decode("utf-8", errors="replace")

        cleaned = self._clean_text(raw_text)
        # Treat every 100 lines as a page
        lines = cleaned.splitlines()
        pages: List[PageContent] = []
        chunk_size = 100
        for i in range(0, max(len(lines), 1), chunk_size):
            block = "\n".join(lines[i : i + chunk_size])
            if block.strip():
                page_num = (i // chunk_size) + 1
                pages.append(PageContent(page_number=page_num, text=block.strip(), source=filename))

        logger.info(f"TXT extraction complete: {len(pages)} sections")
        return ExtractionResult(
            filename=filename,
            file_type="txt",
            total_pages=len(pages),
            pages=pages,
        )

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove noise while preserving meaningful content."""
        if not text:
            return ""
        # Normalise line endings
        text = re.sub(r"\r\n|\r", "\n", text)
        # Collapse horizontal whitespace (spaces/tabs) on each line
        text = re.sub(r"[ \t]+", " ", text)
        # Strip leading/trailing spaces on every line
        text = "\n".join(line.strip() for line in text.split("\n"))
        # Collapse 3+ consecutive blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
