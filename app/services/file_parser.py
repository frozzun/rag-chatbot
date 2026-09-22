"""File text extraction parser supporting PDF, Markdown, and Text files."""

import io
import logging
from app.core.exceptions import InvalidRequestError

logger = logging.getLogger("app.parser")


def extract_text_from_pdf_bytes(content_bytes: bytes) -> str:
    """Extract text from PDF file bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        pages_text: list[str] = []

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages_text.append(f"--- [Page {idx + 1}] ---\n{text}")

        extracted = "\n\n".join(pages_text).strip()
        if not extracted:
            raise InvalidRequestError(
                "PDF 파일에서 텍스트를 추출할 수 없습니다. 텍스트가 없는 스캔 이미지 PDF일 수 있습니다."
            )
        return extracted
    except InvalidRequestError:
        raise
    except Exception as exc:
        logger.error(f"Failed to parse PDF bytes: {exc}", exc_info=True)
        raise InvalidRequestError(f"PDF 파일 파싱 중 오류가 발생했습니다: {exc}") from exc


def extract_text_from_plain_bytes(content_bytes: bytes) -> str:
    """Decode plain text bytes trying UTF-8, CP949 (Korean Windows), and Latin-1."""
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin-1"):
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise InvalidRequestError("지원되지 않는 파일 텍스트 인코딩입니다.")


def parse_file_content_sync(filename: str, content_bytes: bytes) -> str:
    """Synchronous CPU-bound file parser to be executed in a worker thread."""
    if not content_bytes:
        raise InvalidRequestError("업로드된 파일이 비어 있습니다.")

    lower_name = filename.lower()

    if lower_name.endswith(".pdf"):
        return extract_text_from_pdf_bytes(content_bytes)
    else:
        return extract_text_from_plain_bytes(content_bytes)

