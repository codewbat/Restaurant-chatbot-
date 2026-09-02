import os
import json
import re
import sqlite3
from typing import Dict, Any, List, Tuple, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser

from agent.guardrails import SQLGuardrail
from agent.prompts import SCHEMA_PRUNER_PROMPT, SQL_GENERATOR_PROMPT
from database.menu_config import RestaurantCRMDatabase

load_dotenv()
api_key = os.getenv("groq_key") or os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=api_key,
    temperature=0.1,
    max_tokens=1536,
    reasoning_format="hidden",
    max_retries=3,
)

crm_db = RestaurantCRMDatabase()
ALL_DB_TABLES = set(crm_db.get_tables())

# Pydantic Output Schemas
class TableSelectionOutput(BaseModel):
    selected_tables: List[str] = Field(
        default_factory=list,
        description="List of 2 to 4 database table names needed for the query"
    )

class SQLGenerationOutput(BaseModel):
    sql: str = Field(description="Strictly valid SQLite SELECT or WITH query")
    tables_used: List[str] = Field(default_factory=list, description="Tables used in query")
    reasoning: str = Field(default="", description="1-line reasoning for query structure")

table_parser = JsonOutputParser(pydantic_object=TableSelectionOutput)
sql_parser = JsonOutputParser(pydantic_object=SQLGenerationOutput)

class SQLAgent:
    """Production Text-to-SQL Agent with LLM-Driven Schema Pruning and JSON Structured Output."""

    @staticmethod
    def prune_schema(query: str) -> List[str]:
        """High-speed deterministic table selector based on user query keywords (0 LLM tokens)."""
        q = query.lower()
        selected = set()

        # 1. Staff & Attendance
        if any(w in q for w in ["attendance", "hours", "present", "absent", "leave", "half day", "check in", "check out"]):
            selected.update(["employees", "attendance"])
        elif any(w in q for w in ["employee", "staff", "waiter", "chef", "manager", "captain", "worker", "roster", "shift", "salary", "phone", "hire"]):
            selected.update(["employees"])

        # 2. Stock & Inventory
        elif any(w in q for w in ["stock", "inventory", "available", "bottles", "scoops"]):
            selected.update(["inventory"])

        # 3. Active Orders & Kitchen status
        elif any(w in q for w in ["order", "bill", "cooking", "pending", "served", "table 1", "table 2", "table 3", "table 4", "table 5"]):
            selected.update(["orders", "order_items", "menu_items", "dining_tables"])

        # 4. Customer Feedback
        elif any(w in q for w in ["rating", "review", "feedback", "comment"]):
            selected.update(["feedback", "customers"])

        # 5. Menu Items & Categories (Default for dishes, prices, drinks, etc.)
        else:
            selected.update(["menu_items", "categories"])

        return sorted(list(selected))

    @staticmethod
    def generate_sql(
        query: str,
        tables: List[str],
        user_role: str = "customer",
        previous_error: Optional[str] = None
    ) -> str:
        """Generates SQLite-compliant read-only SQL query via JsonOutputParser."""
        schema_text = "\n\n".join([f"Table {t}:\n{crm_db.get_table_schema(t)}" for t in tables])

        error_context = ""
        if previous_error:
            error_context = (
                f"\n<self_correction_error>\n"
                f"Previous query failed with error: '{previous_error}'. "
                f"Fix syntax and column references using the exact schema.\n"
                f"</self_correction_error>"
            )

        messages = SQL_GENERATOR_PROMPT.format_messages(
            schema=schema_text,
            error_context=error_context,
            query=query
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
                # Attempt regex capture of SQL
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
        is_safe, sanitized_sql, guardrail_err = SQLGuardrail.validate_sql(query, user_role=user_role)
        if not is_safe:
            return False, None, guardrail_err, query

        try:
            results = crm_db.execute_query(sanitized_sql)
            return True, results, None, sanitized_sql
        except sqlite3.Error as e:
            return False, None, f"SQLite Execution Error: {str(e)}", sanitized_sql
        except Exception as e:
            return False, None, f"Database Error: {str(e)}", sanitized_sql
