import os
import time
import logging
from typing import Callable, Tuple, Set
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers.api import BaseObserver
from watchdog.observers import Observer

logger = logging.getLogger(__name__)

class PDFWatchHandler(FileSystemEventHandler):
    """
    Watch handler for PDF files
    """
    def __init__(self, callback_on_pdf_ready: Callable[[str], None]) -> None:
        super().__init__()
        self.callback_on_pdf_ready: Callable[[str], None] = callback_on_pdf_ready
        self.recently_processed_files: Set[str] = set()

    def process_detected_pdf_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        filepath: str = str(event.src_path)
        if not filepath.lower().endswith(".pdf"):
            return

        if filepath in self.recently_processed_files:
            return

        if is_file_writing_complete(filepath):
            self.recently_processed_files.add(filepath)
            try:
                self.callback_on_pdf_ready(filepath)
            finally:
                # Wait for 0.5 seconds before removing from set
                time.sleep(0.5)
                self.recently_processed_files.discard(filepath)

    def on_created(self, event: FileSystemEvent) -> None:
        self.process_detected_pdf_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self.process_detected_pdf_event(event)



def is_file_writing_complete(filepath: str, wait_seconds: float = 1.0) -> bool:
    # Check if file exists
    if not os.path.exists(filepath):
        return False
    try:
        initial_size = os.path.getsize(filepath)
        time.sleep(wait_seconds)
        if not os.path.exists(filepath):
            return False
        secondary_size = os.path.getsize(filepath)
        return initial_size == secondary_size and initial_size > 0
    except OSError:
        return False


def start_pdf_folder_watcher(
    watch_directory_path: str,
    callback_on_pdf_ready: Callable[[str], None],
) -> Tuple[BaseObserver, PDFWatchHandler]:
    """
    Start watching a directory for PDF files
    """
    if not os.path.exists(watch_directory_path):
        os.makedirs(watch_directory_path, exist_ok=True)

    event_handler = PDFWatchHandler(callback_on_pdf_ready=callback_on_pdf_ready)
    observer = Observer()
    observer.schedule(event_handler, path=watch_directory_path, recursive=False)
    observer.start()
    return observer, event_handler
