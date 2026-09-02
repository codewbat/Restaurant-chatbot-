import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
except ImportError:  # pragma: no cover
    psycopg2 = None
    sql = None
    RealDictCursor = None


class RestaurantMenuConfig:
    """Load and query the restaurant menu JSON file, including inventory data."""

    def __init__(self, file_path: Optional[str] = None):
        default_path = file_path or os.getenv("MENU_DATABASE_PATH") or str(
            Path(__file__).resolve().parent / "menu_database.json"
        )
        self.file_path = Path(default_path)
        self.data = self._load_data()

    def _load_data(self) -> Dict[str, Any]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Menu file not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    @property
    def restaurant(self) -> Dict[str, Any]:
        return self.data.get("restaurant", {})

    @property
    def categories(self) -> List[Dict[str, Any]]:
        return self.data.get("categories", [])

    @property
    def menu_items(self) -> List[Dict[str, Any]]:
        return self.data.get("menu_items", [])

    @property
    def inventory(self) -> List[Dict[str, Any]]:
        return self.data.get("inventory", [])

    def get_category_name(self, category_id: int) -> str:
        for category in self.categories:
            if category.get("id") == category_id:
                return category.get("name", "Unknown")
        return "Unknown"

    def get_items_by_category(self, category_id: int) -> List[Dict[str, Any]]:
        return [
            item for item in self.menu_items if item.get("category_id") == category_id
        ]

    def get_item_by_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        for item in self.menu_items:
            if item.get("id") == item_id:
                return item
        return None

    def get_inventory_by_item_id(self, item_id: int) -> Optional[Dict[str, Any]]:
        for item in self.inventory:
            if item.get("item_id") == item_id:
                return item
        return None

    def get_available_items(self) -> List[Dict[str, Any]]:
        return [item for item in self.inventory if item.get("available") is True]

    def get_low_stock_items(self) -> List[Dict[str, Any]]:
        return [
            item
            for item in self.inventory
            if item.get("stock", 0) <= item.get("reorder_level", 0)
        ]

    def search_items(self, query: str) -> List[Dict[str, Any]]:
        keyword = query.lower().strip()
        if not keyword:
            return []

        return [
            item
            for item in self.menu_items
            if keyword in item.get("name", "").lower()
        ]

    def search_inventory(self, query: str) -> List[Dict[str, Any]]:
        keyword = query.lower().strip()
        if not keyword:
            return []

        return [
            item for item in self.inventory if keyword in item.get("name", "").lower()
        ]

    def total_categories(self) -> int:
        return len(self.categories)

    def total_menu_items(self) -> int:
        return len(self.menu_items)

    def total_inventory_items(self) -> int:
        return len(self.inventory)


class RestaurantDB:
    """PostgreSQL access layer for production database-backed menu queries."""

    def __init__(self, db_url: Optional[str] = None, connection=None):
        if psycopg2 is None:
            raise ImportError(
                "psycopg2 is not installed. Install it with: pip install psycopg2-binary"
            )

        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.conn = connection

    def connect(self):
        if self.conn is None:
            if not self.db_url:
                raise ValueError("Database URL is missing. Set DATABASE_URL or pass db_url.")
            self.conn = psycopg2.connect(self.db_url)
        return self.conn

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        conn = self.connect()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()

    def get_menu_items(self, limit: int = 50) -> List[Dict[str, Any]]:
        query = "SELECT id, name, price, category_id FROM menu_items LIMIT %s;"
        return self.execute_query(query, (limit,))

    def get_inventory(self, available_only: bool = True) -> List[Dict[str, Any]]:
        query = "SELECT item_id, name, stock, available, unit, price, category FROM inventory"
        params: tuple = ()
        if available_only:
            query += " WHERE available = %s"
            params = (True,)
        query += " ORDER BY name LIMIT 50;"
        return self.execute_query(query, params)

    def search_items(self, search_term: str) -> List[Dict[str, Any]]:
        if sql is None:
            raise ImportError("psycopg2 SQL support is unavailable.")

        query = sql.SQL(
            """
            SELECT id, name, price, category_id
            FROM menu_items
            WHERE name ILIKE %s
            LIMIT 10;
            """
        )
        with self.connect().cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (f"%{search_term}%",))
            return cur.fetchall()

    def get_item_with_category(self, item_name: str) -> List[Dict[str, Any]]:
        if sql is None:
            raise ImportError("psycopg2 SQL support is unavailable.")

        query = sql.SQL(
            """
            SELECT m.id, m.name, m.price, c.name as category
            FROM menu_items m
            JOIN categories c ON m.category_id = c.id
            WHERE m.name ILIKE %s
            LIMIT 10;
            """
        )
        with self.connect().cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (f"%{item_name}%",))
            return cur.fetchall()

    def get_low_stock(self) -> List[Dict[str, Any]]:
        query = """
            SELECT name, stock, reorder_level, unit
            FROM inventory
            WHERE stock <= reorder_level
            ORDER BY stock ASC;
        """
        return self.execute_query(query)

    def get_timings(self) -> Dict[str, Any]:
        query = "SELECT meal_timings, kitchen_timings FROM restaurant LIMIT 1;"
        result = self.execute_query(query)
        return result[0] if result else {}


import sqlite3

class RestaurantCRMDatabase:
    """SQLite access layer for the complete restaurant CRM database."""

    def __init__(self, db_path: Optional[str] = None):
        default_path = db_path or os.getenv("CRM_DATABASE_PATH") or str(
            Path(__file__).resolve().parent / "restaurant_crm.db"
        )
        self.db_path = Path(default_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"CRM database not found at {self.db_path}. Please run init_db.py first.")

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_tables(self) -> List[str]:
        query = "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        results = self.execute_query(query)
        return [row["name"] for row in results]

    def get_table_schema(self, table_name: str) -> str:
        query = "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?;"
        results = self.execute_query(query, (table_name,))
        if results and results[0]["sql"]:
            return results[0]["sql"]
        return ""

    def get_all_schemas(self) -> Dict[str, str]:
        tables = self.get_tables()
        return {table: self.get_table_schema(table) for table in tables}


class RestaurantDataStore:
    """Unified access layer supporting JSON, SQLite, and PostgreSQL backends."""

    def __init__(self, source: str = "sqlite", file_path: Optional[str] = None, db_url: Optional[str] = None):
        self.source = source.lower()
        self.json_store: Optional[RestaurantMenuConfig] = None
        self.db_store: Optional[RestaurantDB] = None
        self.crm_db: Optional[RestaurantCRMDatabase] = None

        if self.source == "json":
            self.json_store = RestaurantMenuConfig(file_path)
        elif self.source == "postgres":
            self.db_store = RestaurantDB(db_url=db_url)
        elif self.source == "sqlite":
            self.crm_db = RestaurantCRMDatabase(db_path=file_path)
        else:
            raise ValueError("source must be 'json', 'sqlite', or 'postgres'.")

    def search(self, query: str) -> List[Dict[str, Any]]:
        if self.source == "json":
            return self.json_store.search_items(query)
        elif self.source == "postgres":
            return self.db_store.search_items(query)
        else:
            sql_q = "SELECT id, name, price, category_id FROM menu_items WHERE name LIKE ? LIMIT 10;"
            return self.crm_db.execute_query(sql_q, (f"%{query}%",))

    def get_available(self) -> List[Dict[str, Any]]:
        if self.source == "json":
            return self.json_store.get_available_items()
        elif self.source == "postgres":
            return self.db_store.get_inventory(available_only=True)
        else:
            sql_q = "SELECT item_id, name, stock, available, unit, price, category FROM inventory WHERE available = 1 LIMIT 50;"
            return self.crm_db.execute_query(sql_q)

    def get_low_stock(self) -> List[Dict[str, Any]]:
        if self.source == "json":
            return self.json_store.get_low_stock_items()
        elif self.source == "postgres":
            return self.db_store.get_low_stock()
        else:
            sql_q = "SELECT name, stock, reorder_level, unit FROM inventory WHERE stock <= reorder_level ORDER BY stock ASC;"
            return self.crm_db.execute_query(sql_q)

    def get_restaurant(self) -> Dict[str, Any]:
        if self.source == "json":
            return self.json_store.restaurant
        elif self.source == "sqlite":
            rows = self.crm_db.execute_query("SELECT * FROM restaurant LIMIT 1;")
            return rows[0] if rows else {}
        return {}


if __name__ == "__main__":
    crm = RestaurantCRMDatabase()
    print("Available tables in CRM DB:", crm.get_tables())
    
    # Test sample CRM queries
    rahul_att = crm.execute_query("""
        SELECT e.name, a.date, a.status, a.check_in, a.check_out
        FROM attendance a
        JOIN employees e ON e.id = a.employee_id
        WHERE e.name LIKE '%Rahul%' AND a.date LIKE '2026-08%'
        LIMIT 5;
    """)
    print("\nSample Rahul Attendance:", rahul_att)

    orders = crm.execute_query("SELECT COUNT(*) as active_orders FROM orders WHERE status IN ('pending', 'cooking', 'served');")
    print("\nActive Orders:", orders)

    coffee = crm.execute_query("SELECT name, price FROM menu_items WHERE name LIKE '%Coffee%' AND price <= 200;")
    print("\nCoffee under 200:", coffee)

