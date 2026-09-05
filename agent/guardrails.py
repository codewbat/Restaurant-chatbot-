import re
from typing import Tuple, Optional, Set, Iterable

DISALLOWED_KEYWORDS = [
    r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bDROP\b",
    r"\bALTER\b", r"\bTRUNCATE\b", r"\bCREATE\b", r"\bREPLACE\b",
    r"\bATTACH\b", r"\bDETACH\b", r"\bPRAGMA\b", r"\bEXEC\b", r"\bEXECUTE\b",
    r"\bGRANT\b", r"\bREVOKE\b", r"\bLOAD_EXTENSION\b", r"\bWRITEFILE\b"
]

SENSITIVE_COLUMNS = {
    "salary": ["manager", "admin"],
    "phone": ["manager", "admin", "waiter"],  # Customers cannot scrape staff phone numbers
}

KNOWN_DB_TABLES = {
    "dining_tables", "customers", "employees", "attendance", "orders",
    "order_items", "menu_items", "categories", "inventory", "reservations",
    "feedback", "restaurant"
}

class SQLGuardrail:
    """Strict SQL Safety and Permissions Validator with AST and Catalog Verification."""

    @staticmethod
    def validate_sql(
        query: str,
        user_role: str = "customer",
        allowed_tables: Optional[Iterable[str]] = None
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Validates SQL query for read-only safety, formatting, and permissions.
        Returns: (is_valid, sanitized_query_or_error, error_message)
        """
        if not query or not query.strip():
            return False, None, "Generated SQL query is empty."

        clean_query = query.strip()
        
        # 0. Strip markdown code fences ```sql and ```
        clean_query = re.sub(r"^```(?:sql|json)?\s*", "", clean_query, flags=re.IGNORECASE)
        clean_query = re.sub(r"\s*```$", "", clean_query)
        
        # 1. Strip SQL Comments (both /* ... */ and -- ...) to prevent comment-based filter bypasses
        clean_query = re.sub(r"/\*.*?\*/", " ", clean_query, flags=re.DOTALL)
        clean_query = re.sub(r"--.*?$", " ", clean_query, flags=re.MULTILINE)
        clean_query = re.sub(r"\s+", " ", clean_query).strip()

        # 2. Anti-chaining check (Disallow multiple statements separated by semicolon)
        statements = [stmt.strip() for stmt in clean_query.split(";") if stmt.strip()]
        if len(statements) > 1:
            return False, None, "Security Violation: Multi-statement SQL queries are not permitted."

        single_query = statements[0] if statements else clean_query

        # 3. Strict Read-Only Operation Check (Only SELECT or WITH allowed at start)
        first_word = single_query.split()[0].upper() if single_query.split() else ""
        if first_word not in ("SELECT", "WITH"):
            return False, None, f"Security Violation: Only SELECT and WITH statements are allowed. Got '{first_word}'."

        # 4. Disallowed Keywords & Mutation Check
        for pattern in DISALLOWED_KEYWORDS:
            if re.search(pattern, single_query, re.IGNORECASE):
                keyword = pattern.replace(r"\b", "")
                return False, None, f"Security Violation: Mutating SQL keyword '{keyword}' is strictly prohibited."

        # 5. Block internal database system tables
        if re.search(r"\b(sqlite_master|sqlite_sequence|sqlite_stat\w*|information_schema)\b", single_query, re.IGNORECASE):
            return False, None, "Security Violation: Querying internal database system tables is forbidden."

        # 6. Catalog Table Validation (Detect and block hallucinated non-existent tables)
        valid_catalog = set(allowed_tables) if allowed_tables else KNOWN_DB_TABLES
        from_join_tables = re.findall(r"\b(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)", single_query, re.IGNORECASE)
        for tbl in from_join_tables:
            tbl_lower = tbl.lower()
            if tbl_lower not in valid_catalog and tbl_lower not in ("select", "with", "as", "lateral"):
                return False, None, f"Hallucination Blocked: Table '{tbl}' does not exist in the restaurant database catalog."

        # 7. RBAC Column Check
        query_lower = single_query.lower()
        for col, allowed_roles in SENSITIVE_COLUMNS.items():
            col_pattern = rf"\b{col}\b"
            if re.search(col_pattern, query_lower):
                if user_role not in allowed_roles:
                    return False, None, f"Access Denied: Role '{user_role}' is not authorized to query column '{col}'."

        # 8. Auto-LIMIT injection and capping
        limit_match = re.search(r"\bLIMIT\s+(\d+)", single_query, re.IGNORECASE)
        if limit_match:
            limit_val = int(limit_match.group(1))
            if limit_val > 100:
                # Cap to 100
                single_query = re.sub(r"\bLIMIT\s+\d+", "LIMIT 100", single_query, flags=re.IGNORECASE)
        else:
            # Force safe default LIMIT 50 if no aggregate
            if not re.search(r"\b(COUNT|AVG|SUM|MIN|MAX)\s*\(", single_query, re.IGNORECASE):
                single_query = f"{single_query} LIMIT 50"

        return True, single_query, None

