"""
Utils regarding file management
"""
from __future__ import annotations

import os


def get_system_cache_dir(subfolder: str = "") -> str:
    """
    Returns the system default user cache directory for bbq.
    Respects XDG_CACHE_HOME if set, defaulting to ~/.cache/bbq.
    """
    cache_base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    bbq_cache_dir = os.path.join(cache_base, "bbq")
    return os.path.join(bbq_cache_dir, subfolder) if subfolder else bbq_cache_dir
