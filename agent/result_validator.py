import re
from typing import List, Dict, Any, Tuple, Literal, Optional

class ResultValidator:
    """Validates and categorizes database query execution results."""

    @staticmethod
    def validate(
        query: str,
        results: Optional[List[Dict[str, Any]]],
        error: Optional[str]
    ) -> Tuple[Literal["valid", "empty", "suspicious", "error"], Optional[str]]:
        """
        Validates query results and flags empty or suspicious returns.
        """
        if error:
            return "error", error

        if results is None:
            return "empty", "No results returned from database."

        if len(results) == 0:
            return "empty", "Query executed successfully, but 0 matching records were found."

        # Suspicious check: e.g. COUNT or SUM returned None
        first_row = results[0]
        for k, v in first_row.items():
            if ("count" in k.lower() or "sum" in k.lower() or "avg" in k.lower()) and v is None:
                return "empty", f"Aggregation for {k} found no records."

        return "valid", None


class ResponseFormatter:
    """
    Response Presentation Layer:
    Filters out internal database IDs and technical fields before passing to LLM.
    Enforces clean, professional presentation without data leaks.
    """

    # Always banned from user presentation
    INTERNAL_FIELDS = {
        "id", "category_id", "created_at", "updated_at",
        "employee_id", "customer_id", "table_id", "order_id"
    }

    @classmethod
    def sanitize_for_presentation(cls, query: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters raw database records based on user query intent.
        Menu defaults: Item, Price, Category.
        Only keeps diet/spice/prep_time if the user explicitly asked about them.
        """
        if not rows:
            return []

        q = query.lower()
        wants_diet = any(w in q for w in ["veg", "jain", "non-veg", "non veg", "diet", "shakahari", "mansahari"])
        wants_spice = any(w in q for w in ["spice", "spicy", "tikha", "mirch", "mild", "medium"])
        wants_time = any(w in q for w in ["time", "timing", "prep", "kitna time", "jaldi", "fast", "minute", "mins"])

        sanitized_rows = []
        for row in rows:
            clean_row = {}
            for k, v in row.items():
                k_lower = k.lower()

                # 1. Strip internal identifiers and database keys
                if k_lower in cls.INTERNAL_FIELDS:
                    continue

                # 2. Conditional dietary fields
                if k_lower in ["is_veg", "is_jain"]:
                    if not wants_diet:
                        continue
                    if k_lower == "is_veg":
                        clean_row["Type"] = "Veg" if v == 1 else "Non-Veg"
                        continue
                    if k_lower == "is_jain":
                        clean_row["Jain Friendly"] = "Yes" if v == 1 else "No"
                        continue

                # 3. Conditional spice level
                if k_lower == "spice_level":
                    if not wants_spice:
                        continue
                    clean_row["Spice Level"] = str(v).capitalize()
                    continue

                # 4. Conditional prep time
                if k_lower in ["prep_time_mins", "prep_time"]:
                    if not wants_time:
                        continue
                    clean_row["Prep Time"] = f"{v} mins"
                    continue

                # 5. Standard user-facing columns (name -> Item, price -> Price, category -> Category)
                if k_lower == "name":
                    clean_row["Item"] = v
                elif k_lower == "price":
                    clean_row["Price"] = f"₹{int(v) if isinstance(v, (int, float)) and v == int(v) else v}"
                elif k_lower == "category":
                    clean_row["Category"] = v
                else:
                    clean_key = k.replace("_", " ").title()
                    clean_row[clean_key] = v

            sanitized_rows.append(clean_row if clean_row else row)

        return sanitized_rows

    @classmethod
    def format_direct_lookup(cls, query: str, rows: List[Dict[str, Any]]) -> Optional[str]:
        """
        Formats single-item or single-entity lookups into a conversational, grounded response (0 LLM tokens).
        """
        if not rows or len(rows) != 1:
            return None

        row = rows[0]
        q_lower = query.lower()

        # 1. Single Dish lookup
        if "name" in row and "price" in row:
            name = row.get("name")
            price = row.get("price")
            price_str = f"₹{int(price) if isinstance(price, (int, float)) and price == int(price) else price}"
            category = row.get("category", row.get("category_id", ""))
            cat_str = f" ({category})" if category and not str(category).isdigit() else ""
            spice = f", Spice: {row.get('spice_level')}" if "spice_level" in row and row.get("spice_level") else ""
            prep = f", Prep Time: {row.get('prep_time_mins')} mins" if "prep_time_mins" in row and row.get("prep_time_mins") else ""
            return f"🍽️ **{name}**{cat_str} hamare menu mein available hai.\n- **Price:** {price_str}{spice}{prep}"

        # 2. Single Table lookup
        if "table_number" in row:
            tbl = row.get("table_number")
            section = row.get("section", "Main")
            capacity = row.get("capacity", "")
            status = row.get("status", "available")
            cap_str = f" ({capacity} Seater)" if capacity else ""
            return f"🪑 **Table {tbl}** [{section}{cap_str}]: Abhi **{status.upper()}** hai."

        # 3. Aggregation/Count single result (e.g. active orders count, total bill)
        for k, v in row.items():
            k_lower = k.lower()
            if "count" in k_lower or "active" in k_lower:
                return f"📊 Filhal total **{v}** active/matching records hain."
            if "total" in k_lower or "sum" in k_lower or "bill" in k_lower:
                val_str = f"₹{int(v) if isinstance(v, (int, float)) and v == int(v) else v}" if isinstance(v, (int, float)) else str(v)
                return f"💰 Total amount: **{val_str}**"

        return None

    @staticmethod
    def format_markdown_table(rows: List[Dict[str, Any]]) -> str:
        """Generates a clean markdown table directly from sanitized records."""
        if not rows:
            return ""
        headers = list(rows[0].keys())
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + "|"
        ]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
        return "\n".join(lines)

