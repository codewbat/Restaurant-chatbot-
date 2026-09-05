"""
OOP Table Manager for Dining Tables Management
"""
import sqlite3
from typing import List, Optional, Dict, Any
from pathlib import Path
from core.models import DiningTable

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "restaurant_crm.db"


class TableManager:
    """Encapsulates dining tables lifecycle, capacity lookups, seating, and release."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def get_all_tables(self) -> List[DiningTable]:
        """Fetch all tables with their status and section."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, table_number, capacity, section, status FROM dining_tables ORDER BY id ASC")
            rows = cur.fetchall()
            return [
                DiningTable(
                    id=r["id"],
                    table_number=r["table_number"],
                    capacity=r["capacity"],
                    section=r["section"],
                    status=r["status"],
                )
                for r in rows
            ]

    def get_table_by_number(self, table_number: str) -> Optional[DiningTable]:
        """Fetch a specific table by its table number (e.g., 'T-01', 'T-11')."""
        normalized = table_number.upper().strip()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, table_number, capacity, section, status FROM dining_tables WHERE UPPER(table_number) = ?",
                (normalized,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return DiningTable(
                id=row["id"],
                table_number=row["table_number"],
                capacity=row["capacity"],
                section=row["section"],
                status=row["status"],
            )

    def occupy_table(self, table_number: str) -> bool:
        """Mark table as occupied."""
        normalized = table_number.upper().strip()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE dining_tables SET status = 'occupied' WHERE UPPER(table_number) = ?",
                (normalized,),
            )
            conn.commit()
            return cur.rowcount > 0

    def release_table(self, table_number: str) -> bool:
        """Release table back to available."""
        normalized = table_number.upper().strip()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE dining_tables SET status = 'available' WHERE UPPER(table_number) = ?",
                (normalized,),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_available_tables(self, min_capacity: int = 1) -> List[DiningTable]:
        """Fetch available tables suitable for a given guest count."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT id, table_number, capacity, section, status FROM dining_tables WHERE status = 'available' AND capacity >= ? ORDER BY capacity ASC",
                (min_capacity,),
            )
            rows = cur.fetchall()
            return [
                DiningTable(
                    id=r["id"],
                    table_number=r["table_number"],
                    capacity=r["capacity"],
                    section=r["section"],
                    status=r["status"],
                )
                for r in rows
            ]
