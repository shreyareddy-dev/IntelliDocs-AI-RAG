from io import BytesIO

from pypdf import PdfReader


def extract_pages_from_pdf(file_bytes: bytes) -> list:
    """
    Extract text from a PDF while preserving page numbers.

    Returns:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]
    """

    pdf_reader = PdfReader(BytesIO(file_bytes))

    pages = []

    for page_number, page in enumerate(pdf_reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "page_number": page_number,
                    "text": text.strip()
                }
            )

    return pages