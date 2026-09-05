"""
OOP Restaurant Agent Service Layer
"""
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from core.models import Customer, Order, Bill
from core.session_manager import SessionManager
from core.table_manager import TableManager
from core.order_manager import OrderManager
from agent.graph import restaurant_agent


class RestaurantAgentService:
    """
    Object-Oriented Service wrapping the AI Restaurant Agent, LangGraph Execution,
    Customer Phone Authentication, Conversation Persistence, and State Recovery.
    """

    def __init__(self):
        self.session_manager = SessionManager()
        self.table_manager = TableManager()
        self.order_manager = OrderManager()
        self.graph = restaurant_agent

    def login_customer(self, phone: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Authenticate customer by phone number, restoring active table, pending orders, and chat history.
        """
        return self.session_manager.restore_customer_session(phone)

    def chat(
        self,
        user_input: str,
        phone: str,
        user_role: str = "customer",
        extra_slots: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Executes a turn of conversation with full context persistence and session restoration.
        """
        clean_phone = self.session_manager.normalize_phone(phone)
        session_info = self.session_manager.restore_customer_session(clean_phone)
        customer: Customer = session_info["customer"]
        active_table = session_info.get("active_table")

        thread_id = f"phone_{clean_phone}"
        config = {"configurable": {"thread_id": thread_id}}

        # Retrieve existing graph state to preserve pending confirmations
        existing_state = self.graph.get_state(config)
        merged_slots = {}
        if existing_state and existing_state.values:
            merged_slots = dict(existing_state.values.get("active_slots") or {})

        # Merge customer and table context
        merged_slots["customer_id"] = customer.id
        merged_slots["customer_name"] = customer.name
        merged_slots["vip_status"] = customer.vip_status
        merged_slots["phone"] = clean_phone
        if active_table and "table_number" not in merged_slots:
            merged_slots["table_number"] = active_table
        if extra_slots:
            merged_slots.update(extra_slots)

        initial_state = {
            "user_input": user_input,
            "user_role": user_role,
            "active_slots": merged_slots,
            "retry_count": 0,
            "max_retries": 2,
        }

        # Invoke LangGraph
        result = self.graph.invoke(initial_state, config=config)

        # Clean AI response
        raw_response = result.get("response", "") or ""
        clean_response = re.sub(r"<think>.*?(?:</think>|$)", "", raw_response, flags=re.DOTALL).strip()
        clean_response = re.sub(r"(?i)here's a thinking process:.*$", "", clean_response, flags=re.DOTALL).strip()
        clean_response = re.sub(r"(?i)thinking process:.*$", "", clean_response, flags=re.DOTALL).strip()
        if not clean_response:
            clean_response = "Aapki request process ho gayi hai."

        # Persist messages to SQLite
        self.session_manager.save_message(thread_id, clean_phone, "user", user_input)
        self.session_manager.save_message(thread_id, clean_phone, "assistant", clean_response)

        # Check if table slot was updated during invocation or present in response
        res_slots = result.get("active_slots", {})
        assigned_table = res_slots.get("table_number") or active_table
        if not assigned_table:
            tbl_match = re.search(r"\b(T-\d{1,2})\b", clean_response, re.IGNORECASE)
            if tbl_match:
                assigned_table = tbl_match.group(1).upper()

        if assigned_table:
            # If bill was settled, table is freed
            if "Payment Received" in clean_response or "Settled" in clean_response:
                self.session_manager.update_active_session(
                    phone=clean_phone,
                    customer_id=customer.id,
                    table_number=None,
                    metadata={"last_intent": "settlement"},
                )
                assigned_table = None
            else:
                self.session_manager.update_active_session(
                    phone=clean_phone,
                    customer_id=customer.id,
                    table_number=assigned_table,
                    metadata={"last_intent": result.get("intent")},
                )

        # Get updated active orders
        updated_active_orders = []
        if assigned_table:
            updated_active_orders = self.order_manager.get_active_orders_for_table(assigned_table)

        return {
            "customer": customer.to_dict(),
            "phone": clean_phone,
            "response": clean_response,
            "intent": result.get("intent"),
            "table_number": assigned_table,
            "active_orders": [o.to_dict() for o in updated_active_orders],
            "token_usage": result.get("token_usage", {}),
            "sql_query": result.get("sql_query"),
        }

    def get_table_bill(self, table_number: str) -> Bill:
        """Fetch consolidated bill with UPI details for a table."""
        return self.order_manager.generate_consolidated_bill(table_number)

    def settle_bill(self, table_number: str, payment_mode: str = "upi") -> Tuple[bool, str]:
        """Settle bill and free table."""
        return self.order_manager.settle_table_bill(table_number, payment_mode)
