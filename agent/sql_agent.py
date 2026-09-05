import os
import re
import json
import difflib
import sqlite3
from typing import Dict, Any, List, Tuple, Optional, Set
from pydantic import BaseModel, Field
from dotenv import load_dotenv


from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from agent.guardrails import SQLGuardrail
from agent.prompts import SCHEMA_PRUNER_PROMPT, build_situation_sql_messages
from database.menu_config import RestaurantCRMDatabase

load_dotenv()
api_key = os.getenv("groq_key") or os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=api_key,
    temperature=0.1,
    max_tokens=500,
    max_retries=3,
)

crm_db = RestaurantCRMDatabase()
ALL_DB_TABLES = set(crm_db.get_tables())

# Live entity cache
DB_EMPLOYEE_LIST: List[str] = []
DB_EMPLOYEE_NAMES: Set[str] = set()
DB_MENU_ITEMS: List[str] = []
DB_CATEGORIES: List[str] = []

try:
    _emp_rows = crm_db.execute_query("SELECT name FROM employees")
    DB_EMPLOYEE_LIST = [r["name"] for r in _emp_rows]
    DB_EMPLOYEE_NAMES = {w.lower() for r in _emp_rows for w in r["name"].split() if len(w) > 2}
except Exception:
    DB_EMPLOYEE_LIST, DB_EMPLOYEE_NAMES = [], set()

try:
    _menu_rows = crm_db.execute_query("SELECT name FROM menu_items")
    DB_MENU_ITEMS = [r["name"] for r in _menu_rows]
except Exception:
    DB_MENU_ITEMS = []

try:
    _cat_rows = crm_db.execute_query("SELECT name FROM categories")
    DB_CATEGORIES = [r["name"] for r in _cat_rows]
except Exception:
    DB_CATEGORIES = []

# Pydantic Output Schemas
class TableSelectionOutput(BaseModel):
    selected_tables: List[str] = Field(
        default_factory=list,
        description="List of 2 to 4 database table names needed for the query"
    )

class SQLGenerationOutput(BaseModel):
    sql: str = Field(description="Strictly valid SQLite SELECT or WITH query")

table_parser = JsonOutputParser(pydantic_object=TableSelectionOutput)
sql_parser = JsonOutputParser(pydantic_object=SQLGenerationOutput)


# Auto-created SQLite table for dynamically learned food synonyms (Zero Developer Maintenance)
DYNAMIC_SYNONYMS: Dict[str, List[str]] = {}
try:
    crm_db.execute_query("""
    CREATE TABLE IF NOT EXISTS auto_synonyms (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alias TEXT UNIQUE,
        target_term TEXT
    );
    """)
    _syn_rows = crm_db.execute_query("SELECT alias, target_term FROM auto_synonyms")
    for r in _syn_rows:
        DYNAMIC_SYNONYMS[r["alias"].lower()] = [t.strip() for t in r["target_term"].lower().split(",")]
except Exception:
    DYNAMIC_SYNONYMS = {}

class SQLAgent:
    """Production Text-to-SQL Agent with Dynamic Fuzzy Grounding, Schema Pruning, and Guardrails."""

    @staticmethod
    def auto_resolve_synonyms(query: str, raw_words: Optional[List[str]] = None) -> List[str]:
        """
        Dynamically and automatically resolves Hindi/Regional/Hinglish terms to English menu concepts.
        Learns new terms on the fly and saves them into SQLite DB for instant 0-token future lookups.
        """
        if raw_words is None:
            raw_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
        expanded = list(raw_words)
        ignored_words = {
            "apne", "apna", "kya", "pass", "batao", "dikhao", "chahiye", "milta", "rate", "cost",
            "price", "kitna", "wali", "wala", "mein", "menu", "dishes", "item", "items", "food",
            "order", "hoga", "hogi", "bhi", "hai", "hain", "karo", "kardo", "please"
        }

        for w in raw_words:
            if w in DYNAMIC_SYNONYMS:
                expanded.extend(DYNAMIC_SYNONYMS[w])
            else:
                matches_existing = any(w in dish.lower() for dish in DB_MENU_ITEMS) or any(w in cat.lower() for cat in DB_CATEGORIES)
                if not matches_existing and len(w) >= 3 and w not in ignored_words:
                    try:
                        prompt_text = f'Translate the Hindi food term to English JSON: "{w}". Example output format: {{"{w}": "english_word"}}'
                        resp = llm.invoke(prompt_text)
                        content = re.sub(r"<think>.*?(?:</think>|$)", "", resp.content, flags=re.DOTALL).strip()
                        match = re.search(r"\{[^{}]*\}", content)
                        if match:
                            parsed = json.loads(match.group(0))
                            for k, v in parsed.items():
                                k_clean = k.lower().strip()
                                v_clean = str(v).lower().strip()
                                if k_clean and v_clean and k_clean != v_clean:
                                    DYNAMIC_SYNONYMS[k_clean] = [v_clean]
                                    expanded.append(v_clean)
                                    try:
                                        crm_db.execute_query(
                                            f"INSERT OR REPLACE INTO auto_synonyms (alias, target_term) VALUES ('{k_clean}', '{v_clean}');"
                                        )
                                    except Exception:
                                        pass
                    except Exception:
                        pass

        return list(set(expanded))

    @staticmethod
    def resolve_fuzzy_entities(query: str) -> Dict[str, Any]:
        """
        Dynamically matches user typos/variations and auto-learned synonyms to DB records (0 tokens after discovery).
        """
        q_lower = query.lower()
        raw_words = [w for w in re.findall(r"\w+", q_lower) if len(w) > 2]
        
        # Expand words with auto-learned synonyms dynamically (chai -> tea, chawal -> rice, murg -> chicken)
        words = SQLAgent.auto_resolve_synonyms(query, raw_words)

        matched_dishes: Set[str] = set()
        matched_categories: Set[str] = set()
        matched_employees: Set[str] = set()

        # 1. Menu item fuzzy matching
        for item in DB_MENU_ITEMS:
            item_lower = item.lower()
            item_tokens = item_lower.split()
            if any(w == token or w in item_tokens for w in words for token in item_tokens):
                matched_dishes.add(item)
            elif any(w in item_lower and len(w) >= 4 for w in words):
                matched_dishes.add(item)
            else:
                for w in words:
                    close = difflib.get_close_matches(w, item_tokens, n=1, cutoff=0.80)
                    if close:
                        matched_dishes.add(item)

        # 2. Category matching
        for cat in DB_CATEGORIES:
            cat_lower = cat.lower()
            if any(w in cat_lower for w in words):
                matched_categories.add(cat)

        # 3. Employee matching
        for emp in DB_EMPLOYEE_LIST:
            emp_lower = emp.lower()
            if any(w in emp_lower for w in words):
                matched_employees.add(emp)
            else:
                for w in words:
                    close = difflib.get_close_matches(w, emp_lower.split(), n=1, cutoff=0.80)
                    if close:
                        matched_employees.add(emp)

        hints = []
        if matched_dishes:
            hints.append(f"Matching Menu Dishes in DB: {sorted(list(matched_dishes))[:4]}")
        if matched_categories:
            hints.append(f"Matching Categories in DB: {sorted(list(matched_categories))[:3]}")
        if matched_employees:
            hints.append(f"Matching Staff/Employees in DB: {sorted(list(matched_employees))[:3]}")

        return {
            "dishes": list(matched_dishes),
            "categories": list(matched_categories),
            "employees": list(matched_employees),
            "hint_text": "\n".join(hints)
        }



    @staticmethod
    def prune_schema(query: str) -> List[str]:
        """High-speed deterministic table selector based on user query keywords and matched entities (0 LLM tokens)."""
        q = query.lower()
        selected = set()

        # 1. Staff & Attendance (Dynamic matching against database employee names)
        words_in_query = set(re.findall(r"\w+", q))
        has_emp_name = bool(words_in_query.intersection(DB_EMPLOYEE_NAMES))
        is_attendance = any(w in q for w in ["attendance", "present", "absent", "leave", "half day", "check in", "check out", "hours", "kaam kiya"])

        if is_attendance:
            selected.update(["employees", "attendance"])
        elif has_emp_name or any(w in q for w in ["employee", "staff", "waiter", "chef", "manager", "captain", "worker", "roster", "shift", "salary", "phone", "hire"]):
            selected.update(["employees"])

        # 2. Orders & Kitchen status
        elif any(w in q for w in ["order", "bill", "cooking", "pending", "served", "table"]):
            selected.update(["orders", "order_items", "menu_items", "dining_tables"])

        # 3. Stock & Raw Material Inventory (only if specifically asking for stock/inventory)
        elif any(w in q for w in ["stock", "inventory", "raw material", "bottles", "scoops", "reorder", "godown"]):
            selected.update(["inventory"])

        # 4. Customer Profiles, VIP Tiers, Loyalty Points
        elif any(w in q for w in ["customer", "customers", "guest", "guests", "vip", "loyalty", "points"]):
            selected.update(["customers"])

        # 5. Advance Table Reservations
        elif any(w in q for w in ["reservation", "reservations", "booking", "booked", "advance booking"]):
            selected.update(["reservations", "customers", "dining_tables"])

        # 6. Table Seating & Sections
        elif any(w in q for w in ["seating", "capacity", "section", "rooftop", "ac hall", "poolside", "garden"]):
            selected.update(["dining_tables"])

        # 7. Customer Feedback & Ratings
        elif any(w in q for w in ["rating", "review", "feedback", "comment"]):
            selected.update(["feedback", "customers"])

        # 8. Menu Items & Categories (Default for dishes, prices, drinks, jain, veg, spicy, etc.)
        else:
            selected.update(["menu_items", "categories"])

        return sorted(list(selected))

    PRECOMPILED_QUERIES = {
        "menu": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "menu dikhao": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "show menu": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "menu card": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "full menu": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "menu items": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "menu list": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "kya kya hai menu me": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "khana dikhao": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "dishes": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "dishes list": "SELECT m.name, m.price, c.name as category FROM menu_items m JOIN categories c ON m.category_id = c.id LIMIT 10 OFFSET 0;",
        "active orders": "SELECT COUNT(*) as active_kitchen_orders FROM orders WHERE status = 'cooking';",
        "active orders count": "SELECT COUNT(*) as active_kitchen_orders FROM orders WHERE status = 'cooking';",
        "live orders": "SELECT COUNT(*) as active_kitchen_orders FROM orders WHERE status = 'cooking';",
        "vip customers": "SELECT name, phone, vip_status, loyalty_points FROM customers WHERE LOWER(vip_status) IN ('gold', 'platinum') LIMIT 10 OFFSET 0;",
        "vip guests": "SELECT name, phone, vip_status, loyalty_points FROM customers WHERE LOWER(vip_status) IN ('gold', 'platinum') LIMIT 10 OFFSET 0;",
    }

    @staticmethod
    def generate_sql(
        query: str,
        tables: List[str],
        user_role: str = "customer",
        previous_error: Optional[str] = None
    ) -> str:
        """Generates SQLite-compliant read-only SQL query via Situation-Based Dynamic Prompting with Fuzzy Grounding."""
        clean_q = re.sub(r"[?!.,]", "", query.lower()).strip()
        
        # 0. Deterministic Fast-Path Lookup (0 LLM Tokens)
        if not previous_error and clean_q in SQLAgent.PRECOMPILED_QUERIES:
            SQLAgent.last_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return SQLAgent.PRECOMPILED_QUERIES[clean_q]

        entity_res = SQLAgent.resolve_fuzzy_entities(query)
        hints = entity_res.get("hint_text", "")

        # Normalize query with auto-learned synonyms (e.g., chai -> tea, chawal -> rice, murg -> chicken)
        normalized_query = query
        for alias, targets in DYNAMIC_SYNONYMS.items():
            if targets and re.search(rf"\b{alias}\b", normalized_query, re.IGNORECASE):
                normalized_query = re.sub(rf"\b{alias}\b", targets[0], normalized_query, flags=re.IGNORECASE)

        messages = build_situation_sql_messages(
            tables=tables,
            query=normalized_query,
            error_context=previous_error or "",
            entity_hints=hints
        )


        raw_sql = ""
        response = None
        try:
            response = llm.invoke(messages)
            content = re.sub(r"<think>.*?(?:</think>|$)", "", response.content, flags=re.DOTALL).strip()
            parsed = sql_parser.parse(content)
            raw_sql = parsed.get("sql", "").strip()
        except Exception:
            # Fallback extraction in case of partial JSON
            if response is not None and hasattr(response, "content"):
                raw_sql = re.sub(r"<think>.*?(?:</think>|$)", "", response.content, flags=re.DOTALL).strip()
                match = re.search(r'"sql"\s*:\s*"([^"]+)"', raw_sql)
                if match:
                    raw_sql = match.group(1)
                else:
                    raw_sql = re.sub(r"^```(?:sql|json)?\s*", "", raw_sql)
                    raw_sql = re.sub(r"\s*```$", "", raw_sql)

        # Unescape escaped characters like \n or \" if any
        raw_sql = raw_sql.replace(r'\"', '"').replace(r"\n", " ").strip()
        if response and hasattr(response, "response_metadata"):
            SQLAgent.last_token_usage = response.response_metadata.get("token_usage", {})
        else:
            SQLAgent.last_token_usage = {}
        return raw_sql


    @staticmethod
    def execute_with_guardrail(
        query: str,
        user_role: str = "customer"
    ) -> Tuple[bool, Optional[List[Dict[str, Any]]], Optional[str], str]:
        """
        Validates through Guardrails and executes on SQLite database.
        Returns: (success, results, error_message, executed_sql)
        """
        is_safe, sanitized_sql, guardrail_err = SQLGuardrail.validate_sql(
            query,
            user_role=user_role,
            allowed_tables=ALL_DB_TABLES
        )
        if not is_safe:
            return False, None, guardrail_err, query

        try:
            results = crm_db.execute_query(sanitized_sql)
            return True, results, None, sanitized_sql
        except sqlite3.Error as e:
            return False, None, f"SQLite Execution Error: {str(e)}", sanitized_sql
        except Exception as e:
            return False, None, f"Database Error: {str(e)}", sanitized_sql

