"""
OOP Order Manager for KOTs, Billing, and Payments
"""
import sqlite3
import time
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from core.models import Order, OrderItem, Bill
from core.table_manager import TableManager

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "restaurant_crm.db"


class OrderManager:
    """Encapsulates Multi-round KOT Orders, Kitchen Lifecycle, Consolidated Billing, and Payment Settlement."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.table_manager = TableManager(db_path=db_path)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def create_kot_order(
        self,
        table_number: str,
        items: List[Dict[str, Any]],
        customer_id: Optional[int] = None,
        order_type: str = "dine-in",
        notes: str = "",
    ) -> Order:
        """
        Creates a new KOT order in DB for the specified table and items.
        Items format: [{'name': 'Hot Chocolate', 'quantity': 2, 'unit_price': 150.0, 'menu_item_id': 1}, ...]
        """
        table = self.table_manager.get_table_by_number(table_number)
        table_id = table.id if table else None

        # Calculate subtotal & tax
        subtotal = sum(it["quantity"] * it["unit_price"] for it in items)
        tax_amount = round(subtotal * 0.05, 2)
        net_amount = round(subtotal + tax_amount, 2)

        # Unique KOT order number with ms precision to avoid collision
        order_number = f"ORD-2026-{int(time.time() * 1000) % 1000000:06d}"
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO orders (
                    order_number, table_id, customer_id, order_type,
                    total_amount, discount_amount, tax_amount, net_amount,
                    status, payment_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, 0.0, ?, ?, 'cooking', 'unpaid', ?)
                """,
                (order_number, table_id, customer_id, order_type, subtotal, tax_amount, net_amount, created_at),
            )
            order_id = cur.lastrowid

            order_items_objs: List[OrderItem] = []
            for it in items:
                menu_item_id = it.get("menu_item_id")
                if not menu_item_id:
                    # Lookup menu_item_id by name
                    cur.execute("SELECT id FROM menu_items WHERE LOWER(name) = LOWER(?) LIMIT 1", (it["name"],))
                    row = cur.fetchone()
                    menu_item_id = row["id"] if row else 1

                it_total = round(it["quantity"] * it["unit_price"], 2)
                cur.execute(
                    """
                    INSERT INTO order_items (order_id, menu_item_id, quantity, unit_price, total_price, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (order_id, menu_item_id, it["quantity"], it["unit_price"], it_total, notes),
                )
                order_items_objs.append(
                    OrderItem(
                        menu_item_id=menu_item_id,
                        name=it["name"],
                        quantity=it["quantity"],
                        unit_price=it["unit_price"],
                        total_price=it_total,
                        notes=notes,
                    )
                )

            # Ensure table is marked occupied
            if table_number:
                cur.execute(
                    "UPDATE dining_tables SET status = 'occupied' WHERE UPPER(table_number) = ?",
                    (table_number.upper().strip(),),
                )

            conn.commit()

        return Order(
            id=order_id,
            order_number=order_number,
            table_id=table_id,
            table_number=table_number,
            customer_id=customer_id,
            order_type=order_type,
            total_amount=subtotal,
            tax_amount=tax_amount,
            net_amount=net_amount,
            status="cooking",
            payment_mode="unpaid",
            created_at=created_at,
            items=order_items_objs,
        )

    def get_active_orders_for_table(self, table_number: str) -> List[Order]:
        """Fetch all active KOT orders (cooking, pending, served) for a table."""
        table = self.table_manager.get_table_by_number(table_number)
        if not table:
            return []

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, order_number, table_id, customer_id, order_type,
                       total_amount, discount_amount, tax_amount, net_amount,
                       status, payment_mode, created_at
                FROM orders
                WHERE table_id = ? AND status IN ('cooking', 'pending', 'served')
                ORDER BY id ASC
                """,
                (table.id,),
            )
            order_rows = cur.fetchall()

            orders: List[Order] = []
            for o in order_rows:
                cur.execute(
                    """
                    SELECT oi.menu_item_id, m.name, oi.quantity, oi.unit_price, oi.total_price, oi.notes
                    FROM order_items oi
                    JOIN menu_items m ON oi.menu_item_id = m.id
                    WHERE oi.order_id = ?
                    """,
                    (o["id"],),
                )
                item_rows = cur.fetchall()
                items = [
                    OrderItem(
                        menu_item_id=ir["menu_item_id"],
                        name=ir["name"],
                        quantity=ir["quantity"],
                        unit_price=ir["unit_price"],
                        total_price=ir["total_price"],
                        notes=ir["notes"],
                    )
                    for ir in item_rows
                ]

                orders.append(
                    Order(
                        id=o["id"],
                        order_number=o["order_number"],
                        table_id=o["table_id"],
                        table_number=table_number,
                        customer_id=o["customer_id"],
                        order_type=o["order_type"],
                        total_amount=o["total_amount"],
                        discount_amount=o["discount_amount"],
                        tax_amount=o["tax_amount"],
                        net_amount=o["net_amount"],
                        status=o["status"],
                        payment_mode=o["payment_mode"],
                        created_at=o["created_at"],
                        items=items,
                    )
                )

            return orders

    def generate_consolidated_bill(self, table_number: str) -> Bill:
        """Consolidate all active KOT orders for a table into a unified Bill with dynamic UPI QR details."""
        active_orders = self.get_active_orders_for_table(table_number)

        # Aggregate items across rounds
        aggregated_items: Dict[str, Dict[str, Any]] = {}
        for o in active_orders:
            for it in o.items:
                if it.name in aggregated_items:
                    aggregated_items[it.name]["quantity"] += it.quantity
                    aggregated_items[it.name]["total_price"] += it.total_price
                else:
                    aggregated_items[it.name] = {
                        "menu_item_id": it.menu_item_id,
                        "name": it.name,
                        "quantity": it.quantity,
                        "unit_price": it.unit_price,
                        "total_price": it.total_price,
                        "notes": it.notes,
                    }

        item_objs = [
            OrderItem(
                menu_item_id=v["menu_item_id"],
                name=v["name"],
                quantity=v["quantity"],
                unit_price=v["unit_price"],
                total_price=round(v["total_price"], 2),
                notes=v.get("notes"),
            )
            for v in aggregated_items.values()
        ]

        subtotal = round(sum(it.total_price for it in item_objs), 2)
        tax_rate = 0.05
        tax_amount = round(subtotal * tax_rate, 2)
        net_total = round(subtotal + tax_amount, 2)

        return Bill(
            table_number=table_number,
            orders=active_orders,
            items=item_objs,
            subtotal=subtotal,
            tax_rate=tax_rate,
            tax_amount=tax_amount,
            net_total=net_total,
            upi_id="9660888489@axl",
            payee_name="Umaid Haveli",
        )

    def settle_table_bill(self, table_number: str, payment_mode: str = "upi") -> Tuple[bool, str]:
        """Mark all active orders for table as completed, record payment mode, and release the table."""
        table = self.table_manager.get_table_by_number(table_number)
        if not table:
            return False, f"Table {table_number} not found."

        mode = payment_mode.lower().strip()
        if mode not in ["cash", "card", "upi"]:
            mode = "upi"

        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE orders
                SET status = 'completed', payment_mode = ?
                WHERE table_id = ? AND status IN ('cooking', 'pending', 'served')
                """,
                (mode, table.id),
            )
            updated_count = cur.rowcount

            # Release table
            cur.execute(
                "UPDATE dining_tables SET status = 'available' WHERE UPPER(table_number) = ?",
                (table_number.upper().strip(),),
            )
            conn.commit()

        return True, f"Successfully settled {updated_count} order(s) for Table {table_number} via {mode.upper()}."

    VALID_TRANSITIONS = {
        "pending": ["cooking", "served", "cancelled"],
        "cooking": ["served", "cancelled"],
        "in_kitchen": ["served", "cancelled"],
        "served": ["completed"],  # Cannot go back to cooking/pending, and CANNOT be cancelled!
        "completed": [],          # Terminal state: cannot go back or cancel
        "cancelled": [],          # Terminal state: cannot go back or modify
    }

    def update_order_status(self, order_id: int, new_status: str) -> Tuple[bool, str]:
        """
        Updates order status following strict unidirectional restaurant state lifecycle:
        - Once 'completed', order cannot go back to any state.
        - Once 'served', order cannot go back to cooking/pending and CANNOT be cancelled (can only move to completed).
        - Once 'cancelled', order is in terminal state.
        """
        target_status = new_status.lower().strip()
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, order_number, status FROM orders WHERE id = ?", (order_id,))
            row = cur.fetchone()
            if not row:
                return False, f"Order #{order_id} not found."

            current_status = row["status"].lower()
            order_number = row["order_number"]

            if current_status == target_status:
                return True, f"Order #{order_number} is already '{current_status}'."

            # Rule 1: Completed is final
            if current_status == "completed":
                return False, f"Order #{order_number} is already COMPLETED and paid. State cannot be reversed."

            # Rule 2: Cancelled is final
            if current_status == "cancelled":
                return False, f"Order #{order_number} is already CANCELLED. State cannot be modified."

            # Rule 3: Served cannot be cancelled or moved back to cooking
            if current_status == "served":
                if target_status == "cancelled":
                    return False, f"Order #{order_number} has already been SERVED to the guest. It CANNOT be cancelled."
                if target_status in ["cooking", "pending", "in_kitchen"]:
                    return False, f"Order #{order_number} is already SERVED. It cannot go back to '{target_status}'."
                if target_status != "completed":
                    return False, f"Order #{order_number} is served. Allowed next state is only 'completed' via billing."

            # Rule 4: General valid transition check
            allowed = self.VALID_TRANSITIONS.get(current_status, ["completed", "cancelled"])
            if target_status not in allowed:
                return False, f"Invalid transition from '{current_status}' to '{target_status}' for Order #{order_number}."

            cur.execute("UPDATE orders SET status = ? WHERE id = ?", (target_status, order_id))
            conn.commit()
            return True, f"Order #{order_number} status updated from '{current_status}' to '{target_status}'."

