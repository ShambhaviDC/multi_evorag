"""file_processor.py - Multimodal Document and Image Processing Service for EvoRAG Phase 7.

Supports lightweight CPU-friendly extraction for:
- PDF (.pdf) via pdfplumber / pypdf
- Word (.docx) via python-docx
- Text (.txt, .md, .json, .log)
- Spreadsheets (.csv, .xlsx) via pandas & openpyxl
- Images (.png, .jpg, .jpeg) via Pillow, OCR (pytesseract if available), and Gemini Vision base64 preparation
"""

import base64
import hashlib
import io
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Allowed file extensions and maximum limits
ALLOWED_EXTENSIONS = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain",
    "md": "text/markdown",
    "csv": "text/csv",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
}

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB per file
MAX_TOTAL_UPLOAD_BYTES = 60 * 1024 * 1024  # 60 MB total per query


@dataclass
class ProcessedFile:
    filename: str
    file_type: str  # "pdf", "docx", "txt", "csv", "xlsx", "image"
    mime_type: str
    size_bytes: int
    file_hash: str
    text_content: str = ""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    structured_data: Optional[Dict[str, Any]] = None  # for tabular data
    image_base64: Optional[str] = None  # for image payloads
    image_mime_type: Optional[str] = None
    ocr_text: Optional[str] = None
    summary_preview: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "filename": self.filename,
            "file_type": self.file_type,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "file_hash": self.file_hash,
            "text_content": self.text_content[:4000] if self.text_content else "",
            "tables_count": len(self.tables),
            "structured_data": self.structured_data,
            "ocr_text": self.ocr_text,
            "summary_preview": self.summary_preview,
            "has_image": bool(self.image_base64),
            "error": self.error,
        }


# In-memory document extraction cache (SHA256 -> ProcessedFile)
_EXTRACTION_CACHE: Dict[str, ProcessedFile] = {}


def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_cached_file(file_hash: str) -> Optional[ProcessedFile]:
    return _EXTRACTION_CACHE.get(file_hash)


def cache_file(processed: ProcessedFile) -> None:
    if processed.file_hash:
        _EXTRACTION_CACHE[processed.file_hash] = processed


# ---------------------------------------------------------------------------
# Individual Extractors
# ---------------------------------------------------------------------------

def extract_pdf(data: bytes, filename: str) -> ProcessedFile:
    """Extract text and tables from PDF using pdfplumber or pypdf fallback."""
    file_hash = _compute_hash(data)
    text_parts: List[str] = []
    tables: List[Dict[str, Any]] = []

    # Try pdfplumber first (rich text & table extraction)
    plumber_success = False
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text.strip()}")
                
                # Extract page tables
                extracted_tables = page.extract_tables() or []
                for t_idx, t in enumerate(extracted_tables):
                    if t and len(t) > 1:
                        headers = [str(col).strip() if col is not None else "" for col in t[0]]
                        rows = [
                            [str(cell).strip() if cell is not None else "" for cell in row]
                            for row in t[1:]
                        ]
                        tables.append({
                            "page": page_num,
                            "table_index": t_idx + 1,
                            "headers": headers,
                            "rows": rows[:100],  # cap rows per table
                        })
            plumber_success = True
    except Exception as exc:
        log.warning("pdfplumber failed for '%s' (%s), trying pypdf...", filename, exc)

    # Fallback to pypdf if pdfplumber didn't succeed
    if not plumber_success:
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"--- Page {page_num} ---\n{page_text.strip()}")
        except Exception as exc:
            return ProcessedFile(
                filename=filename,
                file_type="pdf",
                mime_type="application/pdf",
                size_bytes=len(data),
                file_hash=file_hash,
                error=f"Failed to read PDF: {exc}",
            )

    full_text = "\n\n".join(text_parts).strip()
    preview = full_text[:300].replace("\n", " ") + ("..." if len(full_text) > 300 else "")
    return ProcessedFile(
        filename=filename,
        file_type="pdf",
        mime_type="application/pdf",
        size_bytes=len(data),
        file_hash=file_hash,
        text_content=full_text,
        tables=tables,
        summary_preview=preview,
    )


def extract_docx(data: bytes, filename: str) -> ProcessedFile:
    """Extract paragraphs and tables from Word (.docx) documents."""
    file_hash = _compute_hash(data)
    try:
        import docx
        doc = docx.Document(io.BytesIO(data))
        text_parts: List[str] = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                text_parts.append(txt)

        tables: List[Dict[str, Any]] = []
        for t_idx, table in enumerate(doc.tables):
            table_rows: List[List[str]] = []
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                table_rows.append(cells)
            if table_rows:
                headers = table_rows[0]
                rows = table_rows[1:]
                tables.append({
                    "table_index": t_idx + 1,
                    "headers": headers,
                    "rows": rows[:100],
                })

        full_text = "\n\n".join(text_parts).strip()
        preview = full_text[:300].replace("\n", " ") + ("..." if len(full_text) > 300 else "")
        return ProcessedFile(
            filename=filename,
            file_type="docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size_bytes=len(data),
            file_hash=file_hash,
            text_content=full_text,
            tables=tables,
            summary_preview=preview,
        )
    except Exception as exc:
        return ProcessedFile(
            filename=filename,
            file_type="docx",
            mime_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            size_bytes=len(data),
            file_hash=file_hash,
            error=f"Failed to read DOCX: {exc}",
        )


def extract_txt(data: bytes, filename: str, mime_type: str = "text/plain") -> ProcessedFile:
    """Extract plain text with utf-8 and fallback encodings."""
    file_hash = _compute_hash(data)
    text = ""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if not text and len(data) > 0:
        return ProcessedFile(
            filename=filename,
            file_type="txt",
            mime_type=mime_type,
            size_bytes=len(data),
            file_hash=file_hash,
            error="Unable to decode file content with supported text encodings.",
        )

    preview = text[:300].replace("\n", " ") + ("..." if len(text) > 300 else "")
    return ProcessedFile(
        filename=filename,
        file_type="txt",
        mime_type=mime_type,
        size_bytes=len(data),
        file_hash=file_hash,
        text_content=text,
        summary_preview=preview,
    )


def extract_tabular(data: bytes, filename: str, is_excel: bool = False) -> ProcessedFile:
    """Extract tabular data (CSV or XLSX) into structured stats and records."""
    file_hash = _compute_hash(data)
    file_type = "xlsx" if is_excel else "csv"
    mime_type = ALLOWED_EXTENSIONS[file_type]

    try:
        import pandas as pd
        if is_excel:
            df = pd.read_excel(io.BytesIO(data))
        else:
            # Try utf-8 first, fallback to latin1
            try:
                df = pd.read_csv(io.BytesIO(data))
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(data), encoding="latin1")

        total_rows, total_cols = df.shape
        columns = [str(c) for c in df.columns]

        # Identify numeric columns
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
        categorical_cols = [c for c in columns if c not in numeric_cols]

        # Compute summary statistics for numeric columns
        stats_dict = {}
        if numeric_cols:
            describe_df = df[numeric_cols].describe().round(2)
            stats_dict = describe_df.to_dict()

        # Format sample data (top 20 rows) for LLM context
        sample_records = df.head(20).to_dict(orient="records")

        # Create structured representation
        structured_data = {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "columns": columns,
            "numerical_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "summary_statistics": stats_dict,
            "sample_rows": sample_records,
        }

        # Format textual description for prompts
        text_summary_lines = [
            f"Dataset: {filename} ({total_rows} rows, {total_cols} columns)",
            f"Columns: {', '.join(columns)}",
            f"Numerical Columns: {', '.join(numeric_cols) if numeric_cols else 'None'}",
            f"Categorical Columns: {', '.join(categorical_cols) if categorical_cols else 'None'}",
            "\nSample Data (First 5 Rows):",
        ]
        sample_table_str = df.head(5).to_string(index=False)
        text_summary_lines.append(sample_table_str)

        if numeric_cols:
            text_summary_lines.append("\nNumerical Summary Statistics:")
            text_summary_lines.append(df[numeric_cols].describe().to_string())

        full_text = "\n".join(text_summary_lines)
        preview = f"{total_rows} rows x {total_cols} cols | Cols: {', '.join(columns[:5])}"

        return ProcessedFile(
            filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            size_bytes=len(data),
            file_hash=file_hash,
            text_content=full_text,
            structured_data=structured_data,
            summary_preview=preview,
        )
    except Exception as exc:
        return ProcessedFile(
            filename=filename,
            file_type=file_type,
            mime_type=mime_type,
            size_bytes=len(data),
            file_hash=file_hash,
            error=f"Failed to parse tabular file: {exc}",
        )


def extract_image(data: bytes, filename: str) -> ProcessedFile:
    """Process image file: validate with Pillow, extract base64 & OCR text."""
    file_hash = _compute_hash(data)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
    mime_type = ALLOWED_EXTENSIONS.get(ext, "image/png")

    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img_format = img.format or ext.upper()
        width, height = img.size

        # Base64 encode for Gemini Vision API
        b64_str = base64.b64encode(data).decode("utf-8")

        # Attempt local OCR via pytesseract if installed and tesseract is available
        ocr_text = None
        try:
            import pytesseract
            # Limit execution time or catch binary not found
            extracted_ocr = pytesseract.image_to_string(img, timeout=5)
            if extracted_ocr and extracted_ocr.strip():
                ocr_text = extracted_ocr.strip()
        except Exception:
            # pytesseract binary not present on path or failed; Gemini Vision will handle
            pass

        text_content = f"Image Attachment: {filename} ({width}x{height} pixels, format {img_format})"
        if ocr_text:
            text_content += f"\nOCR Extracted Text:\n{ocr_text}"

        preview = f"Image ({width}x{height}px)" + (f" | OCR: {ocr_text[:60]}..." if ocr_text else "")

        return ProcessedFile(
            filename=filename,
            file_type="image",
            mime_type=mime_type,
            size_bytes=len(data),
            file_hash=file_hash,
            text_content=text_content,
            image_base64=b64_str,
            image_mime_type=mime_type,
            ocr_text=ocr_text,
            summary_preview=preview,
        )
    except Exception as exc:
        return ProcessedFile(
            filename=filename,
            file_type="image",
            mime_type=mime_type,
            size_bytes=len(data),
            file_hash=file_hash,
            error=f"Invalid or corrupted image: {exc}",
        )


# ---------------------------------------------------------------------------
# Main Unified Dispatcher
# ---------------------------------------------------------------------------

def process_uploaded_file(filename: str, data: bytes) -> ProcessedFile:
    """Validate and extract an uploaded file based on its extension."""
    if not filename:
        filename = "unnamed_attachment"

    file_size = len(data)
    file_hash = _compute_hash(data)

    # Check cache first (CPU optimization: avoid re-extracting)
    cached = get_cached_file(file_hash)
    if cached:
        log.info("Serving file extraction from cache: %s (%s)", filename, file_hash[:8])
        return cached

    # File size validation
    if file_size > MAX_FILE_SIZE_BYTES:
        err_file = ProcessedFile(
            filename=filename,
            file_type="unknown",
            mime_type="application/octet-stream",
            size_bytes=file_size,
            file_hash=file_hash,
            error=f"File exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB).",
        )
        return err_file

    if file_size == 0:
        return ProcessedFile(
            filename=filename,
            file_type="unknown",
            mime_type="application/octet-stream",
            size_bytes=0,
            file_hash=file_hash,
            error="File is empty (0 bytes).",
        )

    # Determine extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        return ProcessedFile(
            filename=filename,
            file_type="unsupported",
            mime_type="application/octet-stream",
            size_bytes=file_size,
            file_hash=file_hash,
            error=f"Unsupported file format '.{ext}'. Supported formats: PDF, DOCX, TXT, CSV, XLSX, PNG, JPG, JPEG.",
        )

    # Route to appropriate extractor
    if ext == "pdf":
        result = extract_pdf(data, filename)
    elif ext == "docx":
        result = extract_docx(data, filename)
    elif ext in ("txt", "md"):
        result = extract_txt(data, filename, ALLOWED_EXTENSIONS[ext])
    elif ext == "csv":
        result = extract_tabular(data, filename, is_excel=False)
    elif ext == "xlsx":
        result = extract_tabular(data, filename, is_excel=True)
    elif ext in ("png", "jpg", "jpeg"):
        result = extract_image(data, filename)
    else:
        result = ProcessedFile(
            filename=filename,
            file_type="unsupported",
            mime_type="application/octet-stream",
            size_bytes=file_size,
            file_hash=file_hash,
            error=f"Unsupported format: .{ext}",
        )

    # Save to memory cache if processing was clean
    if not result.error:
        cache_file(result)

    return result


def process_multiple_files(files_data: List[Tuple[str, bytes]]) -> List[ProcessedFile]:
    """Process multiple file attachments with total size guard."""
    total_bytes = sum(len(b) for _, b in files_data)
    if total_bytes > MAX_TOTAL_UPLOAD_BYTES:
        return [
            ProcessedFile(
                filename="total_upload",
                file_type="error",
                mime_type="application/octet-stream",
                size_bytes=total_bytes,
                file_hash="",
                error=f"Total upload size exceeds maximum allowed ({MAX_TOTAL_UPLOAD_BYTES // (1024 * 1024)}MB).",
            )
        ]

    results = []
    for filename, b in files_data:
        res = process_uploaded_file(filename, b)
        results.append(res)
    return results
