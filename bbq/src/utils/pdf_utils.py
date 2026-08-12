import hashlib
import io
from typing import List
import pymupdf
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


def extract_pdf_pages_to_pil_images(
    pdf_filepath: str, dpi: int = 150
) -> List[Image.Image]:
    """
    Get all the pages of the PDF as PIL images
    """
    document = pymupdf.open(pdf_filepath)
    extracted_images: List[Image.Image] = []
    # Zoom factor
    zoom: float = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        image_bytes = pixmap.tobytes("png")
        # process png 
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        extracted_images.append(pil_image)

    document.close()
    return extracted_images
