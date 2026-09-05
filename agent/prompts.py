from langchain_core.prompts import ChatPromptTemplate

# 1. Contextual Query Rewriter Prompt Template (Streamlined for ~75% Token Reduction)
REWRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Contextual Query Rewriter for Umaid Haveli Restaurant CRM.
Rewrite the user's latest follow-up into a self-contained natural language query.
Active slots: {active_slots}

Rules:
1. Retain previously discussed entity (dish/category/employee/table) when user modifies only budget, date, or filter (e.g. 'coffee' discussed -> '300 nahi 200' -> 'Show coffee items from menu where price <= 200').
2. Resolve pronouns (iska, usme, inme) to the exact entity from history.
3. Classify intent strictly into: 'sql' (database records, staff, menu, orders), 'rag' (policies, timings), 'order', 'pagination', 'clarification'.
4. Return strictly JSON: {{"rewritten_query": "...", "intent": "sql", "confidence": 1.0}}"""),
    ("human", """<chat_history>
{chat_history}
</chat_history>
User: {user_input}""")
])


# 2. Dynamic Schema Pruner Prompt Template (LLM Table Selection)
SCHEMA_PRUNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """<role>
You are an intelligent Database Architect for a Restaurant CRM system.
Your job is to select strictly the 2 to 4 most relevant database tables needed to answer the user's business query.
</role>

<available_tables>
1. dining_tables: Seating capacity, table numbers (T-01..T-15), section (AC Hall, Rooftop, Garden, Private Dining, Poolside), table status (available, occupied, reserved, maintenance).
2. customers: Guest profiles, phone, email, loyalty points, VIP tier (regular, gold, platinum), dietary preferences, total visits.
3. employees: Staff roster, names (including Rahul Sharma), roles (waiter, chef, manager, captain), shifts, phone, salary.
4. attendance: Daily staff attendance logs (date, check_in, check_out, status: present, absent, half_day, leave, week_off).
5. orders: Guest orders, order numbers, table_id, customer_id, amounts, discounts, GST tax, order status (pending, cooking, served, completed), payment modes.
6. order_items: Detailed items per order (menu_item_id, quantity, unit_price, total_price, special notes).
7. menu_items: Food and beverage dishes, names, prices, category_id, is_veg, is_jain, spice_level, prep_time.
8. categories: Menu categories (Hot Beverages, Cold Beverages, Chinese, Veg Specialities, Tandoori, Desserts, etc.).
9. inventory: Raw stock levels, unit, reorder_level, available status (in-stock vs out-of-stock).
10. reservations: Advance table bookings, reservation date/time, guest count, special requests, status.
11. feedback: Customer ratings (food, service, ambiance 1-5) and review comments.
12. restaurant: Restaurant operating hours, meal timings (breakfast, lunch, dinner), policies.
</available_tables>

<rules>
- Return strictly a JSON object: {{"selected_tables": ["table1", "table2"]}}
- Select ONLY tables that exist in <available_tables>.
- Do not select unnecessary tables. Typically 1 to 3 tables are sufficient.
</rules>"""),
    ("human", "<user_query>{query}</user_query>")
])


from typing import List, Any
from langchain_core.messages import SystemMessage, HumanMessage

COMPACT_SCHEMAS = {
    "menu_items": "menu_items(id, name, price, category_id, is_veg, is_jain, spice_level, prep_time_mins)",
    "categories": "categories(id, name)",
    "employees": "employees(id, name, role, phone, salary, shift)",
    "attendance": "attendance(id, employee_id, date, status, check_in, check_out)",
    "orders": "orders(id, order_number, table_id, customer_id, net_amount, status)",
    "order_items": "order_items(id, order_id, menu_item_id, quantity, unit_price, total_price)",
    "dining_tables": "dining_tables(id, table_number, capacity, section, status)",
    "inventory": "inventory(id, name, stock, unit, available, price)",
    "customers": "customers(id, name, phone, loyalty_points, vip_status, total_visits)",
    "feedback": "feedback(id, order_id, customer_id, food_rating, service_rating, comments)",
    "reservations": "reservations(id, customer_id, table_id, guest_count, reservation_date, status)",
    "restaurant": "restaurant(id, name, contact_phone, opening_time, closing_time)"
}

DOMAIN_RULES_AND_EXAMPLES = {
    "menu": "- Search: ALWAYS use word matching like `(m.name LIKE '% ' || 'word' || '%' OR m.name LIKE 'word%' OR m.name LIKE '% ' || 'word')` or `(m.name LIKE '% Tea%' OR m.name LIKE 'Tea%')` so 'tea' never matches 'Steam'. Join: menu_items m JOIN categories c ON m.category_id = c.id. Select: m.name, m.price, c.name as category.",
    "attendance": "- Dates: a.date LIKE '2026-08%'. Staff matching: ALWAYS use e.name LIKE '%name%'. Join: attendance a JOIN employees e ON a.employee_id = e.id. Select: a.date, a.status, a.check_in, a.check_out.",

    "orders": "- Join: orders o JOIN order_items oi ON o.id = oi.order_id JOIN menu_items m ON oi.menu_item_id = m.id. Active: status != 'completed'.",
    "inventory": "- Select: name, stock, available, price, unit FROM inventory.",
    "customers": "- VIP: LOWER(vip_status) IN ('gold', 'platinum'). Select: name, phone, vip_status, loyalty_points.",
    "tables": "- Columns: table_number, capacity, section, status FROM dining_tables.",
    "feedback": "- Join: feedback f JOIN customers c ON f.customer_id = c.id."
}


def build_situation_sql_messages(
    tables: List[str],
    query: str,
    error_context: str = "",
    entity_hints: str = ""
) -> List[Any]:
    """Ultra-lean micro-prompt for SQL generation (under 120 prompt tokens)."""
    schema_lines = [COMPACT_SCHEMAS.get(t, f"{t}(...)") for t in tables]
    schema_text = "; ".join(schema_lines)

    if any(t in tables for t in ["attendance", "employees"]):
        domain = "attendance"
    elif any(t in tables for t in ["orders", "order_items"]):
        domain = "orders"
    elif "inventory" in tables:
        domain = "inventory"
    elif "customers" in tables:
        domain = "customers"
    elif "dining_tables" in tables:
        domain = "tables"
    elif "feedback" in tables:
        domain = "feedback"
    else:
        domain = "menu"

    rule = DOMAIN_RULES_AND_EXAMPLES.get(domain, DOMAIN_RULES_AND_EXAMPLES["menu"])
    err_str = f" Error: {error_context}" if error_context else ""
    hint_str = f" DB Hints: {entity_hints}" if entity_hints else ""

    sys_prompt = f"""Generate SQLite SELECT query. Output strictly JSON: {{"sql": "SELECT ..."}}
Schema: {schema_text}
Rule: {rule}{hint_str}{err_str}"""

    return [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Query: {query}")
    ]




# Backward compatibility alias
SQL_GENERATOR_PROMPT = None


# 4. Response Synthesizer Prompt Template (Zero-Fluff, Token-Efficient)
SYNTHESIZER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """<role>
You are the AI Assistant for Umaid Haveli Restaurant, Jaipur.
Provide direct, concise, and clean answers strictly grounded in the provided Database Results and Policy Documents.
</role>

<guidelines>
1. Zero Filler & High Conciseness: Be crisp and direct. NEVER write repetitive greetings ("Namaste! Umaid Haveli mein swagat hai..."), conversational filler ("hum taaza bana kar serve karenge", "hume khushi hogi"), or category ID disclaimers.
2. Presentation Schema:
   - For menu listings, present a clean markdown table with strictly: `Item`, `Price`, `Category`.
   - NEVER output database internal IDs, `#` numbers, or Category IDs (e.g. do NOT write `1 (Beverages)`, write only `Beverages`).
   - Include conditional columns (`Diet`, `Spice`, `Prep Time`) ONLY if they are explicitly present in the provided Context Data.
3. No Emojis: Maintain a clean, professional, and emoji-free tone.
4. Grounding: Output ONLY facts from Context Data. Mention prices formatted as `₹<amount>`.
5. Pagination: Present exactly and ONLY the items in the current Context Data. Do not invent or repeat other items.
</guidelines>"""),
    ("human", """<user_question>
{query}
</user_question>

<context_data>
{context_data}
</context_data>

<task>
Synthesize the final natural response for the guest/staff member:
</task>""")
])


# 5. Dynamic Clarification & Follow-Up Prompt Template (Streamlined for ~65% Token Reduction)
DYNAMIC_FOLLOWUP_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the AI Assistant for Umaid Haveli Restaurant, Jaipur.
The requested item/record was not found, or the query is ambiguous.
Generate a polite 1-2 sentence response in Hindi/Hinglish:
- If not found: politely mention it, suggest 2 real alternatives (Tandoori Starters, Paneer Specials, Hot/Cold Beverages, Desserts), and ask a follow-up.
- If ambiguous: ask a focused follow-up question specifically for that entity.
No robotic lists, no emojis, crisp and warm."""),
    ("human", """User Query: {user_query}
Situation: {situation}
Generate a 1-2 sentence response:""")
])
