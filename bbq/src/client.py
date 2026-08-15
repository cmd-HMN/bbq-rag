"""
Client module for interacting with the ColPali Document Retrieval Server,
retrieving document matches, and fetching/saving page images.
"""

import sys
import os
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
import requests
from PIL import Image
import io

logger = logging.getLogger("bbq.client")

class ColPaliClient:
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


def main():
    parser = argparse.ArgumentParser(description="ColPali RAG Query & Image Viewer Client")
    parser.add_argument("query", type=str, help="Search query string to retrieve PDF parts")
    parser.add_argument("--server", type=str, default="http://localhost:8000", help="ColPali server URL")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top PDF page matches to retrieve")
    parser.add_argument("--save-images", action="store_true", help="Save retrieved PDF page images to disk")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose debug/info logging")

    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

    client = ColPaliClient(server_url=args.server)
    try:
        print(f"Sending query to {args.server}: '{args.query}' (top_k={args.top_k})...\n")
        results = client.query(query_text=args.query, top_k=args.top_k)

        if not results:
            print("No matching PDF pages found.")
            return

        print(f"Top {len(results)} Matching PDF Parts:\n" + "=" * 60)
        for i, res in enumerate(results, 1):
            print(f"Rank {i}:")
            print(f"  Score       : {res['score']:.4f}")
            print(f"  PDF File    : {res['file_path']}")
            print(f"  Page        : Page {res['page_number']} of {res['total_pages']}")
            print(f"  File Hash   : {res['file_hash'][:12]}")

            if args.save_images:
                out_img_path = f"retrieved_rank_{i}_page_{res['page_number']}.png"
                try:
                    client.get_page_image(
                        file_path=res["file_path"],
                        page_number=res["page_number"],
                        save_path=out_img_path,
                    )
                except Exception as img_err:
                    logger.error(f"Failed to fetch page image from server for rank {i}: {img_err}")
                    print(f"  [Failed to fetch page image from server: {img_err}]")

            print("-" * 60)

    except Exception as err:
        logger.error(f"Error querying ColPali server: {err}", exc_info=True)
        print(f"Error querying ColPali server: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
