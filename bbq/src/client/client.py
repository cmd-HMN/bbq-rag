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
    Client for querying the running ColPali RAG indexing server, generating Gemini multimodal answers,
    and fetching/saving page images.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        config: Optional[Any] = None,
    ) -> None:
        self.server_url: str = server_url.rstrip("/")
        self.config = config

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

    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Sends a retrieval query to the server and returns the top matching PDF page results.

        Args:
            query_text (str): Search text query.
            top_k (int): Number of top matching PDF pages to retrieve (default: 3).

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

    def query_and_answer(
        self,
        query_text: str,
        top_k: int = 3,
        gemini_api_key: Optional[str] = None,
        gemini_model: str = "gemini-3.6-flash",
        save_images: bool = False,
        images_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves top_k relevant pages (default: 3) and uses Google Gemini to generate
        a grounded answer from the page images. If no API key is given or Gemini fails,
        gracefully returns the retrieved pages.

        Args:
            query_text (str): The search query / question.
            top_k (int): Number of top pages to retrieve (default: 3).
            gemini_api_key (Optional[str]): Gemini API key. If omitted, uses GEMINI_API_KEY env.
            gemini_model (str): Gemini model name (default: 'gemini-3.6-flash' free-tier default).
            save_images (bool): Whether to save retrieved page images locally.
            images_output_dir (Optional[str]): Directory to save page images if save_images is True.

        Returns:
            Dict[str, Any]: Dictionary containing 'query', 'answer', 'sources', 'status', and 'engine'.
        """
        from bbq.src.client.gemini import GeminiClient

        # 1. Retrieve top-k matching pages
        results = self.query(query_text=query_text, top_k=top_k)
        if not results:
            return {
                "query": query_text,
                "answer": None,
                "sources": [],
                "status": "no_results",
                "engine": "none",
                "message": "No matching document pages found for query.",
            }

        # 2. Fetch page images for top retrieved matches
        page_images: List[Image.Image] = []
        for i, res in enumerate(results, 1):
            img = None
            save_path = None
            if save_images:
                out_dir = images_output_dir or "."
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f"retrieved_rank_{i}_page_{res['page_number']}.png")

            try:
                img = self.get_page_image(
                    file_path=res["file_path"],
                    page_number=res["page_number"],
                    save_path=save_path,
                )
            except Exception as e:
                logger.warning(f"Could not fetch server page image for {res['file_path']} page {res['page_number']}: {e}")
                # Fallback to local rendering if file exists locally
                if os.path.exists(res["file_path"]):
                    try:
                        img = get_local_pdf_page_image(
                            file_path=res["file_path"],
                            page_number=res["page_number"],
                            save_path=save_path,
                        )
                    except Exception as local_err:
                        logger.warning(f"Local PDF render failed: {local_err}")

            if img is not None:
                page_images.append(img)
                res["image_available"] = True
                if save_path:
                    res["saved_image_path"] = save_path
            else:
                res["image_available"] = False

        # Pull defaults from config if available
        if self.config is not None:
            if gemini_api_key is None and hasattr(self.config, "gemini_api_key"):
                gemini_api_key = self.config.gemini_api_key
            if gemini_model == "gemini-3.6-flash" and hasattr(self.config, "gemini_model"):
                gemini_model = self.config.gemini_model

        # 3. Check Gemini API client availability
        gemini = GeminiClient(api_key=gemini_api_key, model=gemini_model)

        if not gemini.is_available():
            logger.info("No Gemini API key provided. Returning retrieved pages only.")
            return {
                "query": query_text,
                "answer": None,
                "sources": results,
                "status": "fallback_pages_only",
                "engine": "retrieval_only",
                "fallback_reason": "No Gemini API key provided. Returning retrieved pages of the book.",
            }

        # 4. Attempt Gemini Multimodal Generation
        if page_images:
            answer = gemini.generate_answer(
                query=query_text,
                images=page_images,
                page_metadata=results,
            )
            if answer:
                return {
                    "query": query_text,
                    "answer": answer,
                    "sources": results,
                    "status": "success",
                    "engine": f"gemini ({gemini.model})",
                }
            else:
                logger.warning("Gemini API call failed. Falling back to returning retrieved pages.")
                return {
                    "query": query_text,
                    "answer": None,
                    "sources": results,
                    "status": "fallback_pages_only",
                    "engine": "retrieval_only",
                    "fallback_reason": "Gemini API request failed or rate limited. Returning retrieved pages of the book.",
                }

        return {
            "query": query_text,
            "answer": None,
            "sources": results,
            "status": "fallback_pages_only",
            "engine": "retrieval_only",
            "fallback_reason": "Could not load page images for Gemini multimodal analysis.",
        }

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
