"""
Google Gemini Multimodal API Client for Visual Document RAG.
"""

from __future__ import annotations

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
    """Multimodal client for document page question answering via Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout: int = 90,
    ) -> None:
        self.api_key: Optional[str] = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        self.model: str = model
        self.timeout: int = timeout

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    @classmethod
    def from_config(cls, config: Any, timeout: int = 90) -> GeminiClient:
        return cls(
            api_key=getattr(config, "gemini_api_key", None),
            model=getattr(config, "gemini_model", DEFAULT_GEMINI_MODEL),
            timeout=timeout,
        )

    @staticmethod
    def _image_to_base64_part(
        image: Image.Image, format: str = "JPEG"
    ) -> Dict[str, Any]:
        buf = io.BytesIO()
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        image.save(buf, format=format, quality=85)
        return {
            "inlineData": {
                "mimeType": f"image/{format.lower()}",
                "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
            }
        }

    def generate_answer(
        self,
        query: str,
        images: List[Image.Image],
        page_metadata: Optional[List[Dict[str, Any]]] = None,
        system_instruction: Optional[str] = None,
    ) -> Optional[str]:
        if not self.is_available() or not images:
            return None

        prompt = system_instruction or (
            "You are an expert visual document comprehension assistant. "
            "Carefully analyze the text, diagrams, tables, and visuals in the provided page images to answer accurately. "
            "Always cite the relevant page numbers in your answer."
        )
        parts: List[Dict[str, Any]] = [{"text": prompt}]

        for idx, img in enumerate(images):
            meta = (
                page_metadata[idx] if page_metadata and idx < len(page_metadata) else {}
            )
            fname = meta.get("filename") or os.path.basename(meta.get("file_path", ""))
            pnum = meta.get("page_number", idx + 1)
            parts.append(
                {"text": f"[Document Page {idx + 1} | File: {fname} | Page: {pnum}]"}
            )
            parts.append(self._image_to_base64_part(img))

        parts.append(
            {
                "text": f"User Question: {query}\n\nPlease provide a clear, concise, and grounded answer based solely on the document pages above."
            }
        )

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
        }

        try:
            resp = requests.post(
                f"{GEMINI_API_BASE_URL}/{self.model}:generateContent?key={self.api_key}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                candidates = resp.json().get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    return "".join(p.get("text", "") for p in parts).strip() or None
            logger.warning(f"Gemini API error {resp.status_code}: {resp.text[:200]}")
        except Exception as err:
            logger.warning(f"Gemini request failed: {err}")
        return None
