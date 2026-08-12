"""
Contains all the database related stuff
"""


import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Any, Optional


class ProcessedFilesTracker:
    """
    Class for tracking the status of processed files


    Initializes a database connection and creates the necessary tables if they don't exist.
    """
    def __init__(self, db_filepath: str = "data/tracker.db") -> None:
        self.db_filepath: str = db_filepath
        parent_dir: str = os.path.dirname(os.path.abspath(db_filepath))
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        self.initialize_database_tables()

    def create_sqlite_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_filepath)

    def initialize_database_tables(self) -> None:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_files (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    num_pages INTEGER DEFAULT 0,
                    embedding_path TEXT,
                    processed_at TEXT
                )
                """
            )
            connection.commit()

    def fetch_file_record_by_hash(self, file_hash: str) -> Optional[Dict[str, Any]]:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT file_hash, file_path, status, error_message, num_pages, embedding_path, processed_at
                FROM processed_files
                WHERE file_hash = ?
                """,
                (file_hash,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "file_hash": row[0],
                "file_path": row[1],
                "status": row[2],
                "error_message": row[3],
                "num_pages": row[4],
                "embedding_path": row[5],
                "processed_at": row[6],
            }

    def record_initial_file_pending(self, file_hash: str, file_path: str) -> None:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            now_iso: str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                INSERT INTO processed_files (file_hash, file_path, status, processed_at)
                VALUES (?, ?, 'pending', ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    processed_at = excluded.processed_at
                WHERE status != 'done'
                """,
                (file_hash, file_path, now_iso),
            )
            connection.commit()

    def update_file_status_to_processing(self, file_hash: str, file_path: str) -> None:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            now_iso: str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                INSERT INTO processed_files (file_hash, file_path, status, processed_at)
                VALUES (?, ?, 'processing', ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = 'processing',
                    processed_at = excluded.processed_at
                """,
                (file_hash, file_path, now_iso),
            )
            connection.commit()

    def update_file_status_to_done(
        self, file_hash: str, file_path: str, num_pages: int, embedding_path: str
    ) -> None:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            now_iso: str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE processed_files
                SET file_path = ?, status = 'done', error_message = NULL, num_pages = ?, embedding_path = ?, processed_at = ?
                WHERE file_hash = ?
                """,
                (file_path, num_pages, embedding_path, now_iso, file_hash),
            )
            connection.commit()

    def update_file_status_to_failed(
        self, file_hash: str, file_path: str, error_message: str
    ) -> None:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            now_iso: str = datetime.now(timezone.utc).isoformat()
            cursor.execute(
                """
                UPDATE processed_files
                SET file_path = ?, status = 'failed', error_message = ?, processed_at = ?
                WHERE file_hash = ?
                """,
                (file_path, error_message, now_iso, file_hash),
            )
            connection.commit()

    def reset_in_progress_processing_to_pending(self) -> int:
        with self.create_sqlite_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                UPDATE processed_files
                SET status = 'pending'
                WHERE status = 'processing'
                """
            )
            count: int = cursor.rowcount
            connection.commit()
            return count
