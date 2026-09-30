"""
BBQ RAG Client module
"""

from __future__ import annotations

import os
import io
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
import requests
from PIL import Image

from bbq.src.terminal.tui import print_bbq

if TYPE_CHECKING:
    from bbq.src.config import Config

logger = logging.getLogger("bbq.client")


class BBQClient:
    """Client for querying the BBQ RAG server and multimodal answer generation."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        config: Optional[Config] = None,
        without_logo: bool = False,
    ) -> None:
        self.server_url: str = server_url.rstrip("/")
        self.config: Optional[Config] = config
        self.without_logo: bool = without_logo

    def print_logo(
        self,
        name: Optional[str] = None,
        tag: str = "CLIENT",
        server: Optional[Any] = None,
        top_k: Optional[int] = None,
    ) -> None:
        """Prints the BBQ banner unless without_logo is enabled."""
        if not self.without_logo:
            srv = server if server is not None else self.server_url
            print_bbq(name=name, tag=tag, server=srv, top_k=top_k)

    def get_status(self) -> Dict[str, Any]:
        """Retrieves server status information."""
        resp = requests.get(f"{self.server_url}/status", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def list_documents(self) -> List[Dict[str, Any]]:
        """Retrieves indexed document metadata records from server."""
        resp = requests.get(f"{self.server_url}/documents", timeout=10)
        resp.raise_for_status()
        return resp.json().get("documents", [])

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Sends a retrieval query to the server and returns the top matching results."""
        resp = requests.post(
            f"{self.server_url}/query",
            json={"query": query_text, "top_k": top_k},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])

    def get_page_image(
        self,
        file_path: str,
        page_number: int = 1,
        dpi: Optional[int] = None,
        save_path: Optional[str] = None,
    ) -> Image.Image:
        """Fetches rendered PNG page image from the server and optionally saves it."""
        render_dpi = (
            dpi
            if dpi is not None
            else (getattr(self.config, "pdf_render_dpi", 150) if self.config else 150)
        )
        resp = requests.get(
            f"{self.server_url}/page_image",
            params={
                "file_path": file_path,
                "page_number": page_number,
                "dpi": render_dpi,
            },
            timeout=15,
        )
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            img.save(save_path, format="PNG")
        return img

    def fetch_page_images(
        self,
        results: List[Dict[str, Any]],
        save_images: bool = False,
        images_output_dir: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Fetches page images for retrieved results and optionally saves them to disk.
        Default output directory is data/rr.
        Updates each result dict in results with 'image_available' and 'saved_image_path'.
        """
        out_dir = images_output_dir
        if not out_dir and self.config:
            out_dir = getattr(self.config, "images_output_dir", None)
        out_dir = out_dir or "data/rr"

        if save_images:
            os.makedirs(out_dir, exist_ok=True)

        page_images: List[Image.Image] = []
        for i, res in enumerate(results, 1):
            pnum = res.get("page_number", 1)
            save_path = (
                os.path.join(out_dir, f"rr_{i}_{pnum}.png") if save_images else None
            )
            img = None
            try:
                img = self.get_page_image(res["file_path"], pnum, save_path=save_path)
            except Exception as e:
                logger.debug(
                    f"Remote page image request failed for {res.get('file_path')}: {e}"
                )
                if os.path.exists(res.get("file_path", "")):
                    try:
                        dpi = (
                            getattr(self.config, "pdf_render_dpi", 150)
                            if self.config
                            else 150
                        )
                        img = get_local_pdf_page_image(
                            res["file_path"], pnum, dpi=dpi, save_path=save_path
                        )
                    except Exception as le:
                        logger.debug(
                            f"Local page image extraction failed for {res.get('file_path')}: {le}"
                        )
            if img:
                page_images.append(img)
                res["image_available"] = True
                if save_path:
                    res["saved_image_path"] = save_path
            else:
                res["image_available"] = False
        return page_images

    def save_page_images(
        self,
        results: List[Dict[str, Any]],
        images_output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Fetches and saves rendered page images for retrieved results to disk (default: data/rr).
        Returns a list of saved file paths.
        """
        self.fetch_page_images(
            results=results,
            save_images=True,
            images_output_dir=images_output_dir,
        )
        return [
            res["saved_image_path"] for res in results if res.get("saved_image_path")
        ]

    def save_page_images_threaded(
        self,
        results: List[Dict[str, Any]],
        images_output_dir: Optional[str] = None,
        on_progress: Optional[Callable[[int, int, str], None]] = None,
        on_complete: Optional[Callable[[List[str]], None]] = None,
    ) -> threading.Thread:
        """
        Launches a background daemon thread to fetch and save page images asynchronously.
        Calls on_progress(current_idx, total_count, saved_path) as each image is saved.
        Calls on_complete(saved_paths) when all images have finished saving.
        Returns the spawned background threading.Thread instance.
        """

        def _worker() -> None:
            out_dir = images_output_dir
            if not out_dir and self.config:
                out_dir = getattr(self.config, "images_output_dir", None)
            out_dir = out_dir or "data/rr"
            os.makedirs(out_dir, exist_ok=True)

            saved_paths: List[str] = []
            total = len(results)

            for i, res in enumerate(results, 1):
                pnum = res.get("page_number", 1)
                save_path = os.path.join(out_dir, f"rr_{i}_{pnum}.png")
                img = None
                try:
                    img = self.get_page_image(
                        res["file_path"], pnum, save_path=save_path
                    )
                except Exception as e:
                    logger.debug(
                        f"Remote page image request failed for {res.get('file_path')}: {e}"
                    )
                    if os.path.exists(res.get("file_path", "")):
                        try:
                            dpi = (
                                getattr(self.config, "pdf_render_dpi", 150)
                                if self.config
                                else 150
                            )
                            img = get_local_pdf_page_image(
                                res["file_path"], pnum, dpi=dpi, save_path=save_path
                            )
                        except Exception as le:
                            logger.debug(
                                f"Local page image extraction failed for {res.get('file_path')}: {le}"
                            )
                if img:
                    res["image_available"] = True
                    res["saved_image_path"] = save_path
                    saved_paths.append(save_path)
                    if on_progress:
                        try:
                            on_progress(i, total, save_path)
                        except Exception:
                            pass
                else:
                    res["image_available"] = False
                    if on_progress:
                        try:
                            on_progress(i, total, "")
                        except Exception:
                            pass

            if on_complete:
                try:
                    on_complete(saved_paths)
                except Exception:
                    pass

        thread = threading.Thread(
            target=_worker,
            name="BBQImageSaverThread",
            daemon=True,
        )
        thread.start()
        return thread

    def generate_answer_from_results(
        self,
        query_text: str,
        results: List[Dict[str, Any]],
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        save_images: bool = False,
        images_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Fetches page images for retrieved results and generates a grounded multimodal answer with Gemini.
        """
        page_images = self.fetch_page_images(
            results=results,
            save_images=save_images,
            images_output_dir=images_output_dir,
        )

        if self.config:
            gemini_api_key = gemini_api_key or getattr(
                self.config, "gemini_api_key", None
            )
            gemini_model = gemini_model or getattr(
                self.config, "gemini_model", "gemini-3.6-flash"
            )

        from bbq.src.client.gemini import GeminiClient

        gemini = GeminiClient(
            api_key=gemini_api_key, model=gemini_model or "gemini-3.6-flash"
        )

        answer = None
        fallback_reason = None
        if not gemini.is_available():
            fallback_reason = (
                "No Gemini API key provided. Returning retrieved pages of the book."
            )
        elif not page_images:
            fallback_reason = (
                "Could not load page images for Gemini multimodal analysis."
            )
        else:
            answer = gemini.generate_answer(
                query=query_text, images=page_images, page_metadata=results
            )
            if not answer:
                fallback_reason = "Gemini API request failed or rate limited. Returning retrieved pages of the book."

        if answer:
            return {
                "query": query_text,
                "answer": answer,
                "sources": results,
                "status": "success",
                "engine": f"gemini ({gemini.model})",
            }
        return {
            "query": query_text,
            "answer": None,
            "sources": results,
            "status": "fallback_pages_only",
            "engine": "retrieval_only",
            "fallback_reason": fallback_reason,
        }

    def query_and_answer(
        self,
        query_text: str,
        top_k: int = 10,
        use_llm: bool = True,
        gemini_api_key: Optional[str] = None,
        gemini_model: Optional[str] = None,
        save_images: bool = False,
        images_output_dir: Optional[str] = None,
        without_logo: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Retrieves top_k pages. If use_llm is True, passes retrieved pages to Gemini LLM.
        Otherwise, returns retrieved matches directly without calling LLM.
        If save_images is True, saves page images to disk (default: data/rr).
        """
        suppress_logo = self.without_logo if without_logo is None else without_logo
        if not suppress_logo:
            self.print_logo(top_k=top_k)

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

        if not use_llm:
            if save_images:
                self.save_page_images(
                    results=results, images_output_dir=images_output_dir
                )
            return {
                "query": query_text,
                "answer": None,
                "sources": results,
                "status": "retrieval_only",
                "engine": "none",
            }

        return self.generate_answer_from_results(
            query_text=query_text,
            results=results,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            save_images=save_images,
            images_output_dir=images_output_dir,
        )


def get_local_pdf_page_image(
    file_path: str,
    page_number: int = 1,
    dpi: int = 150,
    save_path: Optional[str] = None,
) -> Image.Image:
    """Renders a PDF page image directly locally."""
    from bbq.src.utils.pdf_utils import extract_single_pdf_page_image

    img = extract_single_pdf_page_image(
        pdf_filepath=file_path, page_number=page_number, dpi=dpi
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        img.save(save_path, format="PNG")
    return img


if __name__ == "__main__":
    from bbq.src.main import build_client_main_parser, run_client_query_command

    parser = build_client_main_parser()
    args = parser.parse_args()
    run_client_query_command(args)
