"""
OOP Session Manager for Customer Authentication, Session Restoration, and Chat History
"""
import sqlite3
import json
import re
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from core.models import Customer, Order
from core.table_manager import TableManager
from core.order_manager import OrderManager

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "restaurant_crm.db"


class SessionManager:
    """Encapsulates Customer Identification by Phone, Chat History Persistence, and State Recovery across restarts."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.table_manager = TableManager(db_path=db_path)
        self.order_manager = OrderManager(db_path=db_path)
        self._ensure_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _ensure_tables(self):
        """Ensures that chat_history and active_sessions exist in the DB."""
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    phone TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS active_sessions (
                    phone TEXT PRIMARY KEY,
                    customer_id INTEGER,
                    table_number TEXT,
                    last_active TEXT NOT NULL,
                    metadata_json TEXT,
                    FOREIGN KEY (customer_id) REFERENCES customers(id)
                );
                """
            )
            conn.commit()

    @staticmethod
    def normalize_phone(phone: str) -> str:
        """Extracts 10-digit clean phone string."""
        digits = re.sub(r"\D", "", phone)
        if len(digits) > 10:
            digits = digits[-10:]
        return digits if len(digits) >= 10 else phone.strip()

    def get_or_create_customer(self, phone: str, name: Optional[str] = None) -> Customer:
        """Looks up existing customer by phone number or creates a new profile."""
        clean_phone = self.normalize_phone(phone)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, name, phone, email, loyalty_points, vip_status, food_preference, total_visits, last_visit_date
                FROM customers
                WHERE phone = ? OR phone LIKE ?
                LIMIT 1
                """,
                (clean_phone, f"%{clean_phone}%"),
            )
            row = cur.fetchone()

            if row:
                return Customer(
                    id=row["id"],
                    name=row["name"],
                    phone=row["phone"],
                    email=row["email"],
                    loyalty_points=row["loyalty_points"] or 0,
                    vip_status=row["vip_status"] or "regular",
                    food_preference=row["food_preference"],
                    total_visits=row["total_visits"] or 1,
                    last_visit_date=row["last_visit_date"],
                )

            # Create new customer record
            cust_name = name.strip() if (name and name.strip()) else f"Customer {clean_phone[-4:]}"
            today_str = datetime.now().strftime("%Y-%m-%d")
            cur.execute(
                """
                INSERT INTO customers (name, phone, loyalty_points, vip_status, total_visits, last_visit_date)
                VALUES (?, ?, 10, 'regular', 1, ?)
                """,
                (cust_name, clean_phone, today_str),
            )
            cust_id = cur.lastrowid
            conn.commit()

            return Customer(
                id=cust_id,
                name=cust_name,
                phone=clean_phone,
                loyalty_points=10,
                vip_status="regular",
                total_visits=1,
                last_visit_date=today_str,
            )

    def save_message(self, session_id: str, phone: str, role: str, content: str) -> None:
        """Persists a single chat message to chat_history table."""
        clean_phone = self.normalize_phone(phone)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO chat_history (session_id, phone, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, clean_phone, role, content, now_str),
            )
            conn.commit()

    def get_chat_history(self, phone: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieves recent chat history for a customer phone number."""
        clean_phone = self.normalize_phone(phone)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT role, content, created_at
                FROM chat_history
                WHERE phone = ? OR phone LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_phone, f"%{clean_phone}%", limit),
            )
            rows = cur.fetchall()
            # Reverse so oldest in the limit comes first
            return [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in reversed(rows)]

    def update_active_session(
        self, phone: str, customer_id: int, table_number: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Updates or registers active session table."""
        clean_phone = self.normalize_phone(phone)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_str = json.dumps(metadata or {})
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO active_sessions (phone, customer_id, table_number, last_active, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(phone) DO UPDATE SET
                    customer_id = excluded.customer_id,
                    table_number = COALESCE(excluded.table_number, active_sessions.table_number),
                    last_active = excluded.last_active,
                    metadata_json = excluded.metadata_json
                """,
                (clean_phone, customer_id, table_number, now_str, meta_str),
            )
            conn.commit()

    def restore_customer_session(self, phone: str) -> Dict[str, Any]:
        """
        Production session recovery:
        1. Identifies customer profile (Name, Loyalty, VIP).
        2. Discovers any active occupied table or pending KOT orders for this customer.
        3. Retrieves recent chat history.
        4. Returns consolidated restored session state.
        """
        clean_phone = self.normalize_phone(phone)
        customer = self.get_or_create_customer(clean_phone)

        # Check if customer has an active table in active_sessions or orders
        active_table = None
        with self._get_connection() as conn:
            cur = conn.cursor()
            # Check active_sessions table
            cur.execute("SELECT table_number, metadata_json FROM active_sessions WHERE phone = ?", (clean_phone,))
            sess_row = cur.fetchone()
            if sess_row and sess_row["table_number"]:
                active_table = sess_row["table_number"]

            # Also check if customer has active orders with a table_id
            if not active_table:
                cur.execute(
                    """
                    SELECT d.table_number
                    FROM orders o
                    JOIN dining_tables d ON o.table_id = d.id
                    WHERE o.customer_id = ? AND o.status IN ('cooking', 'pending', 'served')
                    ORDER BY o.id DESC LIMIT 1
                    """,
                    (customer.id,),
                )
                ord_tbl = cur.fetchone()
                if ord_tbl:
                    active_table = ord_tbl["table_number"]

        # Fetch active orders for this table/customer
        active_orders: List[Order] = []
        if active_table:
            active_orders = self.order_manager.get_active_orders_for_table(active_table)
            # Update session record
            self.update_active_session(clean_phone, customer.id, active_table)

        recent_chats = self.get_chat_history(clean_phone, limit=6)

        return {
            "customer": customer,
            "phone": clean_phone,
            "session_id": f"phone_{clean_phone}",
            "active_table": active_table,
            "active_orders": active_orders,
            "chat_history": recent_chats,
        }
