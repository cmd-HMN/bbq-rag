import os
import hashlib
import io
from typing import List
from PIL import Image


def compute_file_sha256_hash(filepath: str) -> str:
    """
    Computer hashing for filepath
    This will be stored in sqllite for faster lookup
    """
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file_stream:
        while chunk := file_stream.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_pdf_total_pages(pdf_filepath: str) -> int:
    """
    Returns the total number of pages in a PDF document without extracting page images.
    """
    import pymupdf

    target_path = pdf_filepath
    if not os.path.exists(target_path):
        abs_path = os.path.abspath(pdf_filepath)
        if os.path.exists(abs_path):
            target_path = abs_path
        else:
            raise FileNotFoundError(f"PDF file not found: {pdf_filepath}")

    document = pymupdf.open(target_path)
    total_pages = len(document)
    document.close()
    return total_pages


def extract_pdf_page_range_to_pil_images(
    pdf_filepath: str, start_page_idx: int, end_page_idx: int, dpi: int = 150
) -> List[Image.Image]:
    """
    Extracts a range of PDF pages [start_page_idx, end_page_idx) as PIL Images.
    Useful for batch processing large documents without exhausting RAM.
    """
    import pymupdf

    target_path = pdf_filepath
    if not os.path.exists(target_path):
        abs_path = os.path.abspath(pdf_filepath)
        if os.path.exists(abs_path):
            target_path = abs_path
        else:
            raise FileNotFoundError(f"PDF file not found: {pdf_filepath}")

    document = pymupdf.open(target_path)
    extracted_images: List[Image.Image] = []
    zoom: float = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    end_idx = min(end_page_idx, len(document))
    for page_index in range(start_page_idx, end_idx):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        extracted_images.append(pil_image)

    document.close()
    return extracted_images


def extract_pdf_pages_to_pil_images(
    pdf_filepath: str, dpi: int = 150
) -> List[Image.Image]:
    """
    Get all the pages of the PDF as PIL images
    """
    return extract_pdf_page_range_to_pil_images(
        pdf_filepath=pdf_filepath, start_page_idx=0, end_page_idx=100000, dpi=dpi
    )


def extract_single_pdf_page_image(
    pdf_filepath: str, page_number: int, dpi: int = 150
) -> Image.Image:
    """
    Extracts a specific PDF page (1-based index) as a PIL Image.
    """
    import pymupdf

    target_path = pdf_filepath
    if not os.path.exists(target_path):
        abs_path = os.path.abspath(pdf_filepath)
        if os.path.exists(abs_path):
            target_path = abs_path
        else:
            raise FileNotFoundError(f"PDF file not found at '{pdf_filepath}' or '{abs_path}'")

    document = pymupdf.open(target_path)
    page_idx = page_number - 1
    if page_idx < 0 or page_idx >= len(document):
        document.close()
        raise ValueError(f"Page number {page_number} out of range for PDF with {len(document)} pages.")

    page = document.load_page(page_idx)
    zoom: float = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    image_bytes = pixmap.tobytes("png")
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    document.close()
    return pil_image
