"""
Client module for interacting with the ColPali Document Retrieval Server,
retrieving document matches, and fetching/saving page images.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
import requests
from PIL import Image
import io

logger = logging.getLogger("bbq.client")

class BBQClient:
    """
    Client for querying the running ColPali RAG indexing server and fetching page images.
    """

    def __init__(self, server_url: str = "http://localhost:8000") -> None:
        self.server_url: str = server_url.rstrip("/")

    def get_status(self) -> Dict[str, Any]:
        """Retrieves server status information."""
        logger.info(f"Checking server status at {self.server_url}/status...")
        start_time = time.time()
        response = requests.get(f"{self.server_url}/status", timeout=10)
        elapsed = time.time() - start_time
        response.raise_for_status()
        logger.info(f"Server status retrieved in {elapsed:.3f}s (HTTP {response.status_code})")
        return response.json()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieves indexed document metadata records from server."""
        logger.info(f"Fetching document list from {self.server_url}/documents...")
        start_time = time.time()
        response = requests.get(f"{self.server_url}/documents", timeout=10)
        elapsed = time.time() - start_time
        response.raise_for_status()
        docs = response.json().get("documents", [])
        logger.info(f"Retrieved {len(docs)} document record(s) in {elapsed:.3f}s")
        return docs

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Sends a retrieval query to the server and returns the top matching PDF page results.

        Args:
            query_text (str): Search text query.
            top_k (int): Number of top matching PDF pages to retrieve.

        Returns:
            List[Dict[str, Any]]: List of matching results containing score, file_path, page_number, etc.
        """
        logger.info(f"Sending query request to {self.server_url}/query: '{query_text}' (top_k={top_k})")
        payload = {"query": query_text, "top_k": top_k}
        start_time = time.time()
        response = requests.post(f"{self.server_url}/query", json=payload, timeout=30)
        elapsed = time.time() - start_time
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        logger.info(f"Query completed in {elapsed:.3f}s (HTTP {response.status_code}): received {len(results)} match(es)")
        return results

    def get_page_image(
        self,
        file_path: str,
        page_number: int = 1,
        dpi: int = 150,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Fetches the rendered PNG image for a given PDF file and page number from the server.
        Optionally saves it to save_path.
        """
        logger.info(f"Requesting page image from server: '{file_path}' (page={page_number}, dpi={dpi})")
        params = {"file_path": file_path, "page_number": page_number, "dpi": dpi}
        start_time = time.time()
        response = requests.get(f"{self.server_url}/page_image", params=params, timeout=15)
        elapsed = time.time() - start_time
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        logger.info(f"Page image received in {elapsed:.3f}s ({len(response.content)} bytes)")

        if save_path:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            img.save(save_path, format="PNG")
            logger.info(f"Saved PDF page image to disk: {save_path}")
            print(f"Saved PDF page image to {save_path}")

        return img


def get_local_pdf_page_image(
    file_path: str, page_number: int = 1, dpi: int = 150, save_path: Optional[str] = None
) -> Image.Image:
    """
    Renders a PDF page image directly locally without needing an HTTP server.
    """
    from bbq.src.utils.pdf_utils import extract_single_pdf_page_image
    logger.info(f"Rendering local PDF page image: '{file_path}' (page={page_number}, dpi={dpi})")
    img = extract_single_pdf_page_image(pdf_filepath=file_path, page_number=page_number, dpi=dpi)

    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        img.save(save_path, format="PNG")
        logger.info(f"Saved local PDF page image to {save_path}")
        print(f"Saved local PDF page image to {save_path}")

    return img
