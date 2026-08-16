"""
Google Gemini Multimodal API Client for Visual Document RAG.
Supports free-tier defaults (gemini-3.6-flash / gemini-2.0-flash) with multimodal image inputs.
Gracefully handles API failures and missing keys with zero-crash fallbacks.
"""

import os
import io
import base64
import logging
from typing import List, Dict, Any, Optional
import requests
from PIL import Image

logger = logging.getLogger("bbq.client.gemini")

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

class GeminiClient:
    """
    Multimodal client for sending document page images and user queries to Google Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout: int = 30,
    ) -> None:
        self.api_key: Optional[str] = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.model: str = model
        self.timeout: int = timeout

    def is_available(self) -> bool:
        """Returns True if a valid API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    @staticmethod
    def _image_to_base64_part(image: Image.Image, format: str = "JPEG", quality: int = 85) -> Dict[str, Any]:
        """Converts a PIL Image into Gemini's inlineData base64 part."""
        buffered = io.BytesIO()
        # Convert RGBA to RGB for JPEG compatibility
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buffered, format=format, quality=quality)
        img_bytes = buffered.getvalue()
        b64_str = base64.b64encode(img_bytes).decode("utf-8")
        return {
            "inlineData": {
                "mimeType": f"image/{format.lower()}",
                "data": b64_str,
            }
        }

    def generate_answer(
        self,
        query: str,
        images: List[Image.Image],
        page_metadata: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """
        Generates an answer to the query using the retrieved document page images.

        Args:
            query (str): User's natural language question.
            images (List[Image.Image]): List of top retrieved PDF page images.
            page_metadata (Optional[List[Dict[str, Any]]]): Metadata for each page (filename, page_number).
            system_instruction (Optional[str]): Custom system instruction.

        Returns:
            Optional[str]: Generated text answer, or None if the API key is missing or call failed.
        """
        if not self.is_available():
            logger.warning("No Gemini API key provided. Skipping Gemini answer generation.")
            return None

        if not images:
            logger.warning("No document images provided to Gemini.")
            return None

        endpoint = f"{GEMINI_API_BASE_URL}/{self.model}:generateContent?key={self.api_key}"

        # Construct multimodal parts
        parts: List[Dict[str, Any]] = []

        default_sys_prompt = (
            "You are an expert visual document comprehension assistant. "
            "You are provided with images of relevant document pages from a book/document. "
            "Carefully analyze the text, diagrams, tables, and visuals in the provided page images to answer the user's question accurately. "
            "Always cite the relevant page numbers in your answer when referencing information."
        )
        instruction_text = system_instruction or default_sys_prompt
        parts.append({"text": instruction_text})

        # Add each retrieved page image with contextual label
        for idx, img in enumerate(images):
            meta_label = f"[Document Page {idx + 1}"
            if page_metadata and idx < len(page_metadata):
                meta = page_metadata[idx]
                fname = meta.get("filename") or os.path.basename(meta.get("file_path", ""))
                pnum = meta.get("page_number", idx + 1)
                meta_label += f" | File: {fname} | Page: {pnum}"
            meta_label += "]"

            parts.append({"text": meta_label})
            parts.append(self._image_to_base64_part(img))

        # Add the user query
        parts.append({"text": f"User Question: {query}\n\nPlease provide a clear, concise, and grounded answer based solely on the document pages above."})

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
            },
        }

        try:
            logger.info(f"Sending multimodal request to Gemini ({self.model}) with {len(images)} page image(s)...")
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )

            if response.status_code != 200:
                logger.warning(
                    f"Gemini API returned HTTP {response.status_code}: {response.text[:300]}"
                )
                return None

            resp_data = response.json()
            candidates = resp_data.get("candidates", [])
            if not candidates:
                logger.warning("Gemini API returned no response candidates.")
                return None

            content = candidates[0].get("content", {})
            resp_parts = content.get("parts", [])
            if not resp_parts:
                return None

            answer_text = "".join(part.get("text", "") for part in resp_parts).strip()
            logger.info(f"Successfully generated answer from Gemini ({len(answer_text)} chars).")
            return answer_text

        except requests.exceptions.RequestException as req_err:
            logger.warning(f"Network error communicating with Gemini API: {req_err}")
            return None
        except Exception as err:
            logger.warning(f"Unexpected error during Gemini generation: {err}")
            return None
