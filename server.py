import os
import time
import sqlite3
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database.menu_config import RestaurantCRMDatabase
from agent.graph import restaurant_agent
from agent.restaurant_agent_service import RestaurantAgentService
from core.session_manager import SessionManager
from core.order_manager import OrderManager
from core.table_manager import TableManager

load_dotenv()

app = FastAPI(title="Umaid Haveli Restaurant CRM & AI Agent Dashboard")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

crm_db = RestaurantCRMDatabase()
agent_service = RestaurantAgentService()
session_manager = SessionManager()
order_manager = OrderManager()
table_manager = TableManager()

# Request / Response Schemas
class ChatRequest(BaseModel):
    message: str
    phone: Optional[str] = "9660888489"
    session_id: Optional[str] = "web_session_default"
    role: Optional[str] = "customer"

class CustomerLoginRequest(BaseModel):
    phone: str
    name: Optional[str] = None

class TableStatusUpdate(BaseModel):
    table_number: str
    status: str  # available, occupied, reserved

class TableBillSettlement(BaseModel):
    table_number: str
    payment_mode: str = "upi"

class OrderStatusUpdate(BaseModel):
    order_id: int
    status: str  # cooking, in_kitchen, served, completed, cancelled

class InventoryUpdate(BaseModel):
    item_id: int
    stock: Optional[int] = None
    available: Optional[int] = None

# =============================================================================
# REST API ENDPOINTS
# =============================================================================

@app.post("/api/customer/login")
def login_customer(payload: CustomerLoginRequest):
    """Logs in or registers a customer by phone and returns their profile, active table, orders & chat history."""
    try:
        session_info = agent_service.login_customer(payload.phone, payload.name)
        customer = session_info["customer"]
        active_orders = session_info.get("active_orders", [])
        return {
            "success": True,
            "customer": customer.to_dict(),
            "phone": session_info["phone"],
            "session_id": session_info["session_id"],
            "active_table": session_info.get("active_table"),
            "active_orders": [o.to_dict() for o in active_orders],
            "chat_history": session_info.get("chat_history", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chat/history")
def get_chat_history(phone: str = "9660888489", limit: int = 20):
    """Retrieves persistent chat history for a customer mobile number."""
    try:
        history = session_manager.get_chat_history(phone, limit=limit)
        return {"success": True, "phone": phone, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kpi")
def get_kpi():
    """Returns top KPI statistics for the restaurant CRM dashboard."""
    try:
        conn = crm_db.get_connection()
        cursor = conn.cursor()

        # 1. Active orders
        cursor.execute("SELECT COUNT(*) FROM orders WHERE status NOT IN ('completed', 'cancelled')")
        active_orders = cursor.fetchone()[0]

        # 2. Total Tables & Available
        cursor.execute("SELECT COUNT(*), SUM(CASE WHEN status = 'available' THEN 1 ELSE 0 END) FROM dining_tables")
        total_tables, available_tables = cursor.fetchone()
        occupied_tables = total_tables - available_tables if total_tables else 0

        # 3. Total revenue today
        cursor.execute("SELECT SUM(net_amount) FROM orders WHERE status != 'cancelled'")
        total_revenue = cursor.fetchone()[0] or 0.0

        # 4. Low stock items
        cursor.execute("SELECT COUNT(*) FROM inventory WHERE stock <= reorder_level OR available = 0")
        low_stock_count = cursor.fetchone()[0]

        return {
            "success": True,
            "data": {
                "active_orders": active_orders,
                "total_tables": total_tables,
                "available_tables": available_tables,
                "occupied_tables": occupied_tables,
                "total_revenue": round(total_revenue, 2),
                "low_stock_count": low_stock_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/orders")
def get_orders():
    """Lists all restaurant orders with itemized breakdown."""
    try:
        conn = crm_db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT o.id, o.order_number, o.table_id, t.table_number, t.section,
                   o.order_type, o.total_amount, o.tax_amount, o.net_amount,
                   o.status, o.created_at
            FROM orders o
            LEFT JOIN dining_tables t ON o.table_id = t.id
            ORDER BY o.id DESC
            LIMIT 50;
            """
        )
        orders = [dict(r) for r in cursor.fetchall()]

        # Fetch items for each order
        for ord_dict in orders:
            cursor.execute(
                """
                SELECT oi.quantity, oi.unit_price, oi.total_price, m.name as item_name
                FROM order_items oi
                LEFT JOIN menu_items m ON oi.menu_item_id = m.id
                WHERE oi.order_id = ?;
                """,
                (ord_dict["id"],)
            )
            ord_dict["items"] = [dict(r) for r in cursor.fetchall()]

        return {"success": True, "orders": orders}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orders/update_status")
def update_order_status(payload: OrderStatusUpdate):
    """Updates order status adhering to strict unidirectional restaurant lifecycle rules."""
    try:
        success, msg = order_manager.update_order_status(payload.order_id, payload.status)
        if not success:
            raise HTTPException(status_code=400, detail=msg)
        return {"success": True, "message": msg}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/tables")
def get_tables():
    """Lists all dining tables with current occupancy status."""
    try:
        conn = crm_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, table_number, capacity, section, status FROM dining_tables ORDER BY id ASC;")
        tables = [dict(r) for r in cursor.fetchall()]
        return {"success": True, "tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tables/update_status")
def update_table_status(payload: TableStatusUpdate):
    """Updates table status (available, occupied, reserved)."""
    try:
        conn = crm_db.get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE dining_tables SET status = ? WHERE UPPER(table_number) = UPPER(?)",
                (payload.status, payload.table_number.strip())
            )
        return {"success": True, "message": f"Table {payload.table_number} status updated to {payload.status}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class TableBillSettlement(BaseModel):
    table_number: str
    payment_mode: str = "cash"  # cash, upi, card


@app.get("/api/tables/{table_number}/bill")
def get_table_bill(table_number: str):
    """Returns consolidated itemized active bill for a table."""
    try:
        conn = crm_db.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT o.id, o.order_number, o.status, o.total_amount, o.tax_amount, o.net_amount, o.created_at,
                   oi.quantity, oi.unit_price, oi.total_price, m.name as item_name
            FROM orders o
            JOIN dining_tables t ON o.table_id = t.id
            JOIN order_items oi ON o.id = oi.order_id
            JOIN menu_items m ON oi.menu_item_id = m.id
            WHERE UPPER(t.table_number) = UPPER(?) AND o.status NOT IN ('completed', 'cancelled')
            ORDER BY o.id ASC;
            """,
            (table_number.strip(),)
        )
        rows = [dict(r) for r in cursor.fetchall()]

        if not rows:
            return {
                "success": True,
                "has_active_orders": False,
                "table_number": table_number,
                "message": f"No active unpaid orders found for Table {table_number}."
            }

        kots = list(set(r["order_number"] for r in rows))
        aggregated_items = {}
        for r in rows:
            name = r["item_name"]
            qty = r["quantity"]
            unit_p = r["unit_price"]
            if name not in aggregated_items:
                aggregated_items[name] = {"name": name, "quantity": 0, "unit_price": unit_p, "total_price": 0.0}
            aggregated_items[name]["quantity"] += qty
            aggregated_items[name]["total_price"] += (unit_p * qty)

        items_list = list(aggregated_items.values())
        subtotal = sum(it["total_price"] for it in items_list)
        tax = round(subtotal * 0.05, 2)
        net_total = round(subtotal + tax, 2)

        upi_id = "9660888489@axl"
        qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=240x240&data=upi%3A%2F%2Fpay%3Fpa%3D{upi_id}%26pn%3DUmaid%2520Haveli%26am%3D{net_total:.2f}%26cu%3DINR%26tn%3DTable%2520{table_number}%2520Bill"
        upi_link = f"upi://pay?pa={upi_id}&pn=Umaid%20Haveli&am={net_total:.2f}&cu=INR&tn=Table%20{table_number}%20Bill"

        return {
            "success": True,
            "has_active_orders": True,
            "table_number": table_number,
            "kots": kots,
            "items": items_list,
            "subtotal": subtotal,
            "tax": tax,
            "net_total": net_total,
            "upi_id": upi_id,
            "upi_qr_url": qr_url,
            "upi_link": upi_link
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tables/settle_bill")
def settle_table_bill(payload: TableBillSettlement):
    """Marks all active table orders as completed and frees the table to available."""
    try:
        conn = crm_db.get_connection()
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM dining_tables WHERE UPPER(table_number) = UPPER(?) LIMIT 1", (payload.table_number.strip(),))
            t_row = cursor.fetchone()
            if not t_row:
                raise HTTPException(status_code=404, detail=f"Table {payload.table_number} not found.")
            
            tbl_id = t_row["id"]
            mode = payload.payment_mode.lower() if payload.payment_mode.lower() in ["cash", "card", "upi"] else "cash"

            # Update all active orders for this table
            cursor.execute(
                "UPDATE orders SET status = 'completed', payment_mode = ? WHERE table_id = ? AND status NOT IN ('completed', 'cancelled')",
                (mode, tbl_id)
            )
            # Free table to available
            cursor.execute(
                "UPDATE dining_tables SET status = 'available' WHERE id = ?",
                (tbl_id,)
            )

        return {
            "success": True,
            "message": f"Bill settled for Table {payload.table_number} via {mode.upper()}. Table is now AVAILABLE."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/inventory")
def get_inventory(q: Optional[str] = None):
    """Lists all inventory stock items with optional search filter."""
    try:
        conn = crm_db.get_connection()
        cursor = conn.cursor()
        if q and q.strip():
            cursor.execute(
                "SELECT item_id, name, stock, unit, available, reorder_level, price, category FROM inventory WHERE name LIKE ? ORDER BY name ASC;",
                (f"%{q.strip()}%",)
            )
        else:
            cursor.execute(
                "SELECT item_id, name, stock, unit, available, reorder_level, price, category FROM inventory ORDER BY name ASC;"
            )
        items = [dict(r) for r in cursor.fetchall()]
        return {"success": True, "inventory": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/inventory/update_stock")
def update_inventory_stock(payload: InventoryUpdate):
    """Updates inventory stock level or available flag."""
    try:
        conn = crm_db.get_connection()
        with conn:
            cursor = conn.cursor()
            if payload.stock is not None and payload.available is not None:
                cursor.execute(
                    "UPDATE inventory SET stock = ?, available = ? WHERE item_id = ?",
                    (payload.stock, payload.available, payload.item_id)
                )
            elif payload.stock is not None:
                cursor.execute(
                    "UPDATE inventory SET stock = ?, available = CASE WHEN ? > 0 THEN 1 ELSE 0 END WHERE item_id = ?",
                    (payload.stock, payload.stock, payload.item_id)
                )
            elif payload.available is not None:
                cursor.execute(
                    "UPDATE inventory SET available = ? WHERE item_id = ?",
                    (payload.available, payload.item_id)
                )
        return {"success": True, "message": f"Inventory item #{payload.item_id} updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
def chat_with_agent(payload: ChatRequest):
    """Invokes the OOP Restaurant Agent Service and returns response, active table, orders & tokens."""
    try:
        phone = payload.phone or "9660888489"
        role = payload.role or "manager"
        result = agent_service.chat(
            user_input=payload.message,
            phone=phone,
            user_role=role
        )

        return {
            "success": True,
            "response": result.get("response"),
            "intent": result.get("intent"),
            "tokens": result.get("token_usage"),
            "sql_query": result.get("sql_query"),
            "table_number": result.get("table_number"),
            "customer": result.get("customer"),
            "active_orders": result.get("active_orders")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount static frontend directory
static_path = Path(__file__).resolve().parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_path), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(">> Starting Umaid Haveli Restaurant CRM Server on http://127.0.0.1:8000 ...")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)

