import os
import re
import json
import time
from typing import Dict, Any, List, Optional, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.output_parsers import StrOutputParser

from agent.state import AgentState
from agent.rewriter import analyze_and_rewrite, fast_filter_classify
from agent.sql_agent import SQLAgent
from agent.rag_agent import RAGAgent
from agent.result_validator import ResultValidator, ResponseFormatter
from agent.prompts import SYNTHESIZER_PROMPT, DYNAMIC_FOLLOWUP_PROMPT
from database.menu_config import RestaurantCRMDatabase

load_dotenv()
api_key = os.getenv("groq_key") or os.getenv("GROQ_API_KEY")

str_parser = StrOutputParser()
crm_db = RestaurantCRMDatabase()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=api_key,
    temperature=0.2,
    max_tokens=500,
    max_retries=3,
)

rag_agent = RAGAgent()

# ----------------- NODE IMPLEMENTATIONS ----------------- #

def fast_filter_node(state: AgentState) -> Dict[str, Any]:
    """Node 0 (Entry Point): Deterministic triage for greetings, pagination, orders, and standalone queries."""
    user_input = state["user_input"]
    active_slots = state.get("active_slots", {})
    has_history = len(state.get("messages", [])) > 0

    analysis = fast_filter_classify(
        user_input=user_input,
        active_slots=active_slots,
        has_history=has_history
    )

    cur_tokens = state.get("token_usage") or {"prompt": 0, "completion": 0, "total": 0}
    return {
        "needs_context": analysis.get("needs_context", False),
        "intent": analysis.get("intent", "sql"),
        "confidence": analysis.get("confidence", 1.0),
        "rewritten_query": analysis.get("rewritten_query", user_input),
        "active_slots": analysis.get("updated_slots", active_slots),
        "response": None,
        "token_usage": cur_tokens
    }

def rewriter_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Sliding Window Contextual Rewriter + NLP Ambiguity Detector (Only for context follow-ups)."""
    user_input = state["user_input"]
    chat_history = state.get("messages", [])
    active_slots = state.get("active_slots", {})

    analysis = analyze_and_rewrite(
        user_input=user_input,
        chat_history=chat_history,
        active_slots=active_slots
    )

    rw_tokens = analysis.get("tokens") or {}
    cur_tokens = state.get("token_usage") or {"prompt": 0, "completion": 0, "total": 0}
    updated_tokens = {
        "prompt": cur_tokens.get("prompt", 0) + rw_tokens.get("prompt_tokens", 0),
        "completion": cur_tokens.get("completion", 0) + rw_tokens.get("completion_tokens", 0),
        "total": cur_tokens.get("total", 0) + rw_tokens.get("total_tokens", 0)
    }

    return {
        "rewritten_query": analysis.get("rewritten_query", user_input),
        "intent": analysis.get("intent", "sql"),
        "confidence": analysis.get("confidence", 1.0),
        "clarification_question": analysis.get("clarification_question"),
        "active_slots": analysis.get("updated_slots", active_slots),
        "response": None,
        "token_usage": updated_tokens
    }

def greeting_node(state: AgentState) -> Dict[str, Any]:
    """Fast exit node for simple greetings."""
    reply = "Namaste! Umaid Haveli Restaurant CRM Assistant mein aapka swagat hai. Main menu, orders, table reservations, staff attendance, aur restaurant timings ke sawal answer kar sakta hoon. Aaj main aapki kya madad karoon?"
    return {
        "response": reply,
        "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
    }

def clarification_node(state: AgentState) -> Dict[str, Any]:
    """Dynamic Clarification Node: Formulates context-aware follow-up question based on user query."""
    user_input = state["user_input"]
    question = state.get("clarification_question")
    cur_tokens = state.get("token_usage") or {"prompt": 0, "completion": 0, "total": 0}

    if not question:
        try:
            messages = DYNAMIC_FOLLOWUP_PROMPT.format_messages(
                user_query=user_input,
                situation="The user query was ambiguous, incomplete, or had low confidence. Formulate a polite 1-2 sentence clarification question in Hindi/Hinglish."
            )
            llm_resp = llm.invoke(messages)
            clean_q = re.sub(r"<think>.*?(?:</think>|$)", "", llm_resp.content, flags=re.DOTALL).strip()
            clean_q = re.sub(r"(?i)here's a thinking process:.*$", "", clean_q, flags=re.DOTALL).strip()
            clean_q = re.sub(r"(?i)thinking process:.*$", "", clean_q, flags=re.DOTALL).strip()
            question = str_parser.parse(clean_q).strip()

            tok = getattr(llm_resp, "response_metadata", {}).get("token_usage", {})
            cur_tokens = {
                "prompt": cur_tokens.get("prompt", 0) + tok.get("prompt_tokens", 0),
                "completion": cur_tokens.get("completion", 0) + tok.get("completion_tokens", 0),
                "total": cur_tokens.get("total", 0) + tok.get("total_tokens", 0)
            }
        except Exception:
            question = (
                "Maaf kijiye, main aapka sawal poori tarah samajh nahi paaya. "
                "Kya aap menu, table orders, ya staff attendance ke baare mein kuch specific dekhna chahte hain?"
            )

    return {
        "response": question,
        "token_usage": cur_tokens,
        "messages": state.get("messages", []) + [HumanMessage(content=user_input), AIMessage(content=question)]
    }

def order_node(state: AgentState) -> Dict[str, Any]:
    """
    Human-in-the-Loop (HITL) Node for Food Ordering and Table Reservations.
    - Stage 1: Itemized breakdown, bill preview & asks for confirmation.
    - Stage 2: Commits to SQLite DB upon user confirmation ('haan' / 'yes') or cancels ('nahi' / 'no').
    - Stage 3: Handles quantity updates (e.g. '5 order karni hai') dynamically on pending orders.
    """
    raw_input = state["user_input"]
    clean_text = raw_input.strip().lower()
    active_slots = dict(state.get("active_slots", {}))
    pending_conf = active_slots.get("pending_confirmation")

    # Word-to-number mapping
    word_num_map = {
        "ek": 1, "do": 2, "teen": 3, "tin": 3, "char": 4, "chaar": 4,
        "paanch": 5, "panch": 5, "chhe": 6, "che": 6, "saat": 7, "aath": 8,
        "nau": 9, "das": 10, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5
    }

    # 0. Check if user is asking about their assigned table status
    is_my_table_query = bool(re.search(r"\b(meri|mera|merko|merki|mujhe|my)\b.*\b(table|booking|seat)\b", clean_text)) or any(p in clean_text for p in [
        "konsi table", "koun si table", "kounsi table", "assigned table", "my table", "mera table", "meri table", "konsa table", "kounsa table"
    ])
    if is_my_table_query and not any(w in clean_text for w in ["order", "mangwa", "cup", "plate", "dish", "pizza", "coffee", "tea", "paneer", "chocolate"]):
        assigned_t = active_slots.get("assigned_table")
        if assigned_t:
            t_rows = SQLAgent.execute_with_guardrail(f"SELECT * FROM dining_tables WHERE UPPER(table_number) = '{assigned_t.upper()}' LIMIT 1;")[1]
            if t_rows:
                t_info = t_rows[0]
                reply = f"🪑 Aapko **Table {t_info['table_number']}** ({t_info['section']}, {t_info['capacity']} Seater) assign / reserve hai. Ab aap apna food order de sakte hain!"
            else:
                reply = f"🪑 Aapko **Table {assigned_t}** assign ki gayi hai. Ab aap apna food order de sakte hain!"
        else:
            reply = (
                "⚠️ Aapko abhi tak koi Table assign nahi hui hai.\n\n"
                "👉 Agar aap restaurant mein baithe hain, toh apna **Table Number (jaise T-02 ya T-08)** batayein.\n"
                "👉 Ya phir batayein aap **kitne logo ke liye Table book** karna chahte hain!"
            )
        return {
            "response": reply,
            "active_slots": active_slots,
            "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
        }

    # 0B. Check if user is querying their active order status or a specific order number
    ord_num_match = re.search(r"\b(ORD-[\w\-]+)\b", raw_input, re.IGNORECASE)
    is_my_order_status_query = (bool(ord_num_match) or any(p in clean_text for p in [
        "order complete", "order status", "order kab", "order ban gya", "order bana",
        "mera order", "status kya hai", "order ka status", "order ready", "khana kab aayega"
    ])) and not any(w in clean_text for w in ["pizza", "coffee", "tea", "paneer", "chocolate", "plate", "cup", "mangwa", "order kardo", "order kar do", "order karni hai", "order karna hai"])

    if is_my_order_status_query and not pending_conf:
        target_ord_num = ord_num_match.group(1).upper() if ord_num_match else active_slots.get("last_order_number")
        assigned_t = active_slots.get("assigned_table")
        
        ord_row = None
        if target_ord_num:
            res = SQLAgent.execute_with_guardrail(
                f"SELECT o.id, o.order_number, o.status, o.total_amount, o.tax_amount, o.net_amount, t.table_number, t.section "
                f"FROM orders o LEFT JOIN dining_tables t ON o.table_id = t.id "
                f"WHERE UPPER(o.order_number) = '{target_ord_num}' LIMIT 1;"
            )[1]
            if res:
                ord_row = res[0]
        elif assigned_t:
            res = SQLAgent.execute_with_guardrail(
                f"SELECT o.id, o.order_number, o.status, o.total_amount, o.tax_amount, o.net_amount, t.table_number, t.section "
                f"FROM orders o LEFT JOIN dining_tables t ON o.table_id = t.id "
                f"WHERE UPPER(t.table_number) = '{assigned_t.upper()}' ORDER BY o.id DESC LIMIT 1;"
            )[1]
            if res:
                ord_row = res[0]

        if ord_row:
            items = SQLAgent.execute_with_guardrail(
                f"SELECT oi.quantity, oi.unit_price, oi.total_price, m.name as item_name "
                f"FROM order_items oi LEFT JOIN menu_items m ON oi.menu_item_id = m.id "
                f"WHERE oi.order_id = {ord_row['id']};"
            )[1]
            items_str = "\n".join([f"• **{it['item_name']}** × {it['quantity']} = ₹{it['total_price']}" for it in items]) if items else "• Items in kitchen"
            status_desc = {
                "cooking": "👨‍🍳 Kitchen mein prepare ho raha hai (Cooking)",
                "in_kitchen": "👨‍🍳 Kitchen mein prepare ho raha hai",
                "served": "🍽️ Table par serve ho chuka hai (Served)",
                "completed": "✅ Complete & Paid",
                "cancelled": "❌ Cancelled"
            }.get(ord_row["status"].lower(), ord_row["status"])

            reply = (
                f"📋 **Order Status Details:**\n"
                f"─────────────────────────────────────────\n"
                f"🧾 **Order Number:** #{ord_row['order_number']}\n"
                f"🪑 **Table:** {ord_row.get('table_number', assigned_t or 'T-??')}\n"
                f"⏱️ **Current Status:** {status_desc}\n\n"
                f"**Items:**\n{items_str}\n"
                f"─────────────────────────────────────────\n"
                f"💳 **Total Amount:** ₹{ord_row['net_amount'] or ord_row['total_amount']}\n"
            )
        else:
            reply = (
                "⚠️ Aapka koi active order nahi mila.\n\n"
                "👉 Agar aap order dena chahte hain, toh apna manpasand item batayein (jaise *'2 Cold Coffee order kardo'*)."
            )

        return {
            "response": reply,
            "active_slots": active_slots,
            "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
        }

    # 0C. Check for Consolidated Table Bill Generation & Checkout Intent
    is_bill_request = any(p in clean_text for p in [
        "bill bana do", "bill banado", "bill banao", "bill kitna", "mera bill", "total bill",
        "bill do", "bill de do", "bill chahiye", "bill mangwa", "bill le aao", "check out", "checkout",
        "hisaab", "hisab", "payment", "pay karna", "settle", "bill ready"
    ]) or (
        any(w in clean_text for w in ["bill", "checkout", "hisaab", "hisab", "payment", "settle"]) and
        (bool(re.search(r"\b(T-\d+|T\d+|table\s*\d+)\b", clean_text, re.IGNORECASE)) or bool(active_slots.get("assigned_table")))
    )

    # Detect payment done directly
    is_direct_payment = any(p in clean_text for p in [
        "upi se", "cash de", "card se", "payment done", "pay kar diya", "paid",
        "bill pay", "paise de diye", "online pay", "payment kardi", "pay kar dia", "done payment"
    ])

    # Subcase: User wants to generate/see their Consolidated Table Bill
    if is_bill_request and not pending_conf and not is_direct_payment:
        tbl_match_bill = re.search(r"\b(T-\d+|T\d+|table\s*\d+)\b", clean_text, re.IGNORECASE)
        target_tbl = None
        if tbl_match_bill:
            raw_t = tbl_match_bill.group(1).upper().replace("TABLE", "T-").replace(" ", "").strip()
            if not raw_t.startswith("T-") and raw_t.startswith("T"):
                raw_t = f"T-{raw_t[1:]}"
            target_tbl = raw_t
        else:
            target_tbl = active_slots.get("assigned_table")

        if not target_tbl:
            reply = (
                "🧾 **Bill Generate karne ke liye Table Number batayein:**\n\n"
                "👉 Kripya apna **Table Number** (jaise *'Table T-02 ka bill bana do'*) batayein, ya pehle food order karein!"
            )
            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

        # Query all active / running orders for this table
        res_orders = SQLAgent.execute_with_guardrail(
            f"SELECT o.id, o.order_number, o.status, o.total_amount, o.tax_amount, o.net_amount, o.created_at, "
            f"       oi.quantity, oi.unit_price, oi.total_price, m.name as item_name "
            f"FROM orders o "
            f"JOIN dining_tables t ON o.table_id = t.id "
            f"JOIN order_items oi ON o.id = oi.order_id "
            f"JOIN menu_items m ON oi.menu_item_id = m.id "
            f"WHERE UPPER(t.table_number) = UPPER('{target_tbl}') AND o.status NOT IN ('completed', 'cancelled') "
            f"ORDER BY o.id ASC;"
        )[1]

        if not res_orders:
            # Check if there was any recent completed order today
            past_res = SQLAgent.execute_with_guardrail(
                f"SELECT o.order_number, o.net_amount, o.status FROM orders o "
                f"JOIN dining_tables t ON o.table_id = t.id "
                f"WHERE UPPER(t.table_number) = UPPER('{target_tbl}') ORDER BY o.id DESC LIMIT 1;"
            )[1]
            if past_res:
                reply = (
                    f"ℹ️ **Table {target_tbl}** par koi active/pending order nahi hai.\n"
                    f"Pichhla order (#{past_res[0]['order_number']}) pehle hi settle/complete ho chuka hai. ✅"
                )
            else:
                reply = f"⚠️ **Table {target_tbl}** par abhi koi active food order nahi mila."
            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

        # Aggregate items across multiple KOTs
        kots = list(set(r["order_number"] for r in res_orders))
        kot_list_str = ", ".join([f"#{k}" for k in kots])
        
        aggregated_items = {}
        for r in res_orders:
            name = r["item_name"]
            qty = r["quantity"]
            unit_p = r["unit_price"]
            if name not in aggregated_items:
                aggregated_items[name] = {"qty": 0, "unit_price": unit_p, "total_price": 0.0}
            aggregated_items[name]["qty"] += qty
            aggregated_items[name]["total_price"] += (unit_p * qty)

        subtotal = sum(it["total_price"] for it in aggregated_items.values())
        tax = round(subtotal * 0.05, 2)
        net_total = round(subtotal + tax, 2)

        active_slots["pending_confirmation"] = {
            "type": "bill_settlement",
            "table_number": target_tbl,
            "subtotal": subtotal,
            "tax": tax,
            "net_total": net_total
        }

        items_lines = []
        for i, (name, it) in enumerate(aggregated_items.items()):
            items_lines.append(f"{i+1}. **{name}** × {it['qty']} = ₹{int(it['total_price'])}")
        items_preview = "\n".join(items_lines)

        upi_id = "9660888489@axl"
        qr_image_url = f"https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=upi%3A%2F%2Fpay%3Fpa%3D{upi_id}%26pn%3DUmaid%2520Haveli%26am%3D{net_total:.2f}%26cu%3DINR%26tn%3DTable%2520{target_tbl}%2520Bill"

        reply = (
            f"🧾 **UMAID HAVELI - CONSOLIDATED TABLE BILL** 🍽️\n"
            f"─────────────────────────────────────────\n"
            f"🪑 **Table:** {target_tbl}\n"
            f"📋 **Active KOT Orders:** {kot_list_str}\n"
            f"─────────────────────────────────────────\n"
            f"**Itemized Invoice:**\n{items_preview}\n"
            f"─────────────────────────────────────────\n"
            f"💰 **Food Subtotal:** ₹{int(subtotal)}\n"
            f"🧾 **GST (5%):** ₹{tax:.2f}\n"
            f"💳 **Grand Total Payable:** ₹{net_total:.2f}\n"
            f"─────────────────────────────────────────\n\n"
            f"📱 **Instant UPI QR Payment:**\n"
            f"• **UPI ID:** `{upi_id}` (Umaid Haveli)\n"
            f"• **Payable Amount:** ₹{net_total:.2f}\n"
            f"• **Scan QR with GPay / PhonePe / Paytm:**\n"
            f"![UPI QR Code]({qr_image_url})\n\n"
            f"👉 Payment karne ke baad batayein: *'UPI se pay kar diya'* ya *'Cash de diya'*!"
        )
        return {
            "response": reply,
            "active_slots": active_slots,
            "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
        }

    # =========================================================================
    # CASE 1: User is Responding to an Active HITL Confirmation Prompt or Payment Settlement
    # =========================================================================
    if pending_conf or is_direct_payment:
        affirmative_words = [
            "yes", "haan", "ha", "haa", "haji", "ha ji", "haanji", "haan ji", "confirm",
            "kardo", "kar do", "karo", "ok", "okay", "sure", "done", "theek hai", "thik hai",
            "bilkul", "yep", "yeah", "book kardo", "order kardo", "order kar do", "book kar do",
            "upi", "cash", "card"
        ]
        negative_words = [
            "no", "nahi", "nhi", "nah", "cancel", "mat karo", "rehne do", "rahne do",
            "abort", "not now", "don't", "dont", "nope", "mat karna"
        ]
        is_yes = any(clean_text == w or clean_text.startswith(w + " ") for w in affirmative_words) or is_direct_payment
        is_no = any(clean_text == w or clean_text.startswith(w + " ") for w in negative_words)

        # 1A. Settle Bill Payment & Vacate Table
        conf_type = (pending_conf or {}).get("type")
        if conf_type == "bill_settlement" or (is_direct_payment and active_slots.get("assigned_table")):
            if is_yes:
                target_tbl = (pending_conf or {}).get("table_number", active_slots.get("assigned_table", "T-01"))
                total_amt = (pending_conf or {}).get("net_total")

                # Detect payment mode
                pay_mode = "upi"
                if "cash" in clean_text:
                    pay_mode = "cash"
                elif "card" in clean_text:
                    pay_mode = "card"

                # Commit Settlement to Database
                try:
                    conn = crm_db.get_connection()
                    with conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT id FROM dining_tables WHERE UPPER(table_number) = UPPER(?) LIMIT 1", (target_tbl,))
                        t_row = cursor.fetchone()
                        tbl_id = t_row["id"] if t_row else None

                        if tbl_id:
                            # Update all active orders for this table to completed and paid
                            cursor.execute(
                                "UPDATE orders SET status = 'completed', payment_mode = ? WHERE table_id = ? AND status NOT IN ('completed', 'cancelled')",
                                (pay_mode, tbl_id)
                            )
                            # Release table to available
                            cursor.execute(
                                "UPDATE dining_tables SET status = 'available' WHERE id = ?",
                                (tbl_id,)
                            )
                except Exception:
                    pass

                active_slots.pop("pending_confirmation", None)
                active_slots.pop("assigned_table", None)
                active_slots.pop("last_order_number", None)

                pay_mode_display = {"upi": "📱 UPI Payment", "cash": "💵 Cash", "card": "💳 Card"}.get(pay_mode, pay_mode.upper())
                amt_str = f"₹{total_amt:.2f}" if total_amt else "Amount"

                reply = (
                    f"🎉 **Payment Received & Bill Settled Successfully!** ✅\n"
                    f"─────────────────────────────────────────\n"
                    f"🪑 **Table:** {target_tbl} (Ab Available/Free ho chuki hai)\n"
                    f"💳 **Payment Mode:** {pay_mode_display}\n"
                    f"🧾 **Invoice Status:** Paid & Completed\n"
                    f"─────────────────────────────────────────\n\n"
                    f"🙏 **Umaid Haveli mein padharne ke liye aapka bohot-bohot Dhanyawad!**\n"
                    f"Aasha hai aapko hamara khana aur sewa pasand aayi. Phir zaroor padhariye! ✨"
                )
                return {
                    "response": reply,
                    "active_slots": active_slots,
                    "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
                }
            elif is_no:
                active_slots.pop("pending_confirmation", None)
                reply = "❌ Bill settlement cancel kar di gayi hai. Jab aap tayar hon tab checkout ke liye batayein."
                return {
                    "response": reply,
                    "active_slots": active_slots,
                    "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
                }

        if is_yes and pending_conf:
            conf_type = pending_conf.get("type")
            if conf_type == "table_booking":
                tbl_num = pending_conf.get("table_number", "T-01")
                guests = pending_conf.get("guests", 2)
                section = pending_conf.get("section", "Main Dining")

                # Commit to DB: Update table status to reserved
                try:
                    conn = crm_db.get_connection()
                    with conn:
                        cursor = conn.cursor()
                        cursor.execute("UPDATE dining_tables SET status = 'reserved' WHERE UPPER(table_number) = UPPER(?)", (tbl_num,))
                except Exception:
                    pass

                active_slots.pop("pending_confirmation", None)
                active_slots["assigned_table"] = tbl_num

                # If user had a pending food order before booking table, immediately bring up bill preview
                pending_order_items = active_slots.pop("pending_order_items", None)
                if pending_order_items:
                    subtotal = sum(it["subtotal"] for it in pending_order_items)
                    tax = round(subtotal * 0.05, 2)
                    net_total = round(subtotal + tax, 2)
                    active_slots["pending_confirmation"] = {
                        "type": "food_order",
                        "items": pending_order_items,
                        "subtotal": subtotal,
                        "tax": tax,
                        "net_total": net_total,
                        "table_number": tbl_num
                    }
                    items_preview = "\n".join([f"{i+1}. **{it['name']}** × {it['qty']} = ₹{int(it['subtotal'])}" for i, it in enumerate(pending_order_items)])
                    reply = (
                        f"🎉 **Table {tbl_num} ({section}) Reserve ho gayi!** ✅\n"
                        f"─────────────────────────────────────────\n"
                        f"Ab aapka pending food order preview:\n\n"
                        f"🧾 **Order Summary & Bill Preview:**\n"
                        f"─────────────────────────────────────────\n"
                        f"{items_preview}\n"
                        f"─────────────────────────────────────────\n"
                        f"💰 **Subtotal:** ₹{int(subtotal)}\n"
                        f"🧾 **GST (5%):** ₹{tax:.2f}\n"
                        f"💳 **Total Payable Amount:** ₹{net_total:.2f}\n"
                        f"🍽️ **Service:** Dine-in (Table {tbl_num})\n"
                        f"─────────────────────────────────────────\n\n"
                        f"⚠️ **Kya main Table {tbl_num} par yeh order confirm karke kitchen mein bhej doon?**\n"
                        f"*(Kripya 'Haan' / 'Yes' ya 'Nahi' / 'Cancel' bole)*"
                    )
                else:
                    reply = (
                        f"🎉 **Table Reservation Confirmed!** ✅\n"
                        f"─────────────────────────────────────────\n"
                        f"🪑 **Table:** {tbl_num} ({section})\n"
                        f"👥 **Guests:** {guests} Persons\n"
                        f"💳 **Booking Fee:** ₹0 (Complimentary)\n"
                        f"⏱️ **Status:** Confirmed & Reserved\n\n"
                        f"Aapki **Table {tbl_num}** reserve ho chuki hai. Umaid Haveli mein aapka swagat hai!"
                    )
            else:
                # Food Order Commitment
                items = pending_conf.get("items", [])
                subtotal = pending_conf.get("subtotal", 0.0)
                tax = pending_conf.get("tax", 0.0)
                net_total = pending_conf.get("net_total", subtotal + tax)
                tbl_num = pending_conf.get("table_number", active_slots.get("assigned_table", "T-02"))

                # Generate Order Number and commit to orders & order_items
                order_num = f"ORD-2026-{int(time.time() * 1000) % 1000000:06d}"
                try:
                    conn = crm_db.get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT id FROM dining_tables WHERE UPPER(table_number) = UPPER(?) LIMIT 1", (tbl_num,))
                    t_row = cursor.fetchone()
                    tbl_id = t_row["id"] if t_row else 1

                    cursor.execute(
                        """
                        INSERT INTO orders (order_number, table_id, customer_id, order_type, total_amount, discount_amount, tax_amount, net_amount, status, payment_mode, created_at)
                        VALUES (?, ?, 1, 'dine-in', ?, 0.0, ?, ?, 'cooking', 'unpaid', datetime('now'))
                        """,
                        (order_num, tbl_id, subtotal, tax, net_total)
                    )
                    order_id = cursor.lastrowid

                    for it in items:
                        cursor.execute(
                            """
                            INSERT INTO order_items (order_id, menu_item_id, quantity, unit_price, total_price)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (order_id, it.get("item_id", 1), it.get("qty", 1), it.get("price", 0.0), it.get("subtotal", 0.0))
                        )
                    
                    cursor.execute("UPDATE dining_tables SET status = 'occupied' WHERE id = ?", (tbl_id,))
                    conn.commit()
                    cursor.close()
                    conn.close()
                except Exception as e:
                    print(f"Error committing food order: {e}")

                active_slots.pop("pending_confirmation", None)
                active_slots["last_order_number"] = order_num

                items_formatted = "\n".join([f"- **{it['name']}** × {it['qty']} = ₹{int(it['subtotal'])}" for it in items])
                reply = (
                    f"🎉 **Order Confirmed & Sent to Kitchen!** 👨‍🍳\n"
                    f"─────────────────────────────────────────\n"
                    f"📋 **Order Number:** #{order_num}\n"
                    f"🪑 **Table:** {tbl_num}\n\n"
                    f"**Items Ordered:**\n{items_formatted}\n"
                    f"─────────────────────────────────────────\n"
                    f"💰 **Subtotal:** ₹{int(subtotal)}\n"
                    f"🧾 **GST (5%):** ₹{tax:.2f}\n"
                    f"💳 **Total Amount:** ₹{net_total:.2f}\n"
                    f"⏱️ **Status:** 👨‍🍳 Kitchen mein prepare ho raha hai\n\n"
                    f"Aapka khana jaldi hi serve kar diya jayega. Dhanyawad!"
                )

            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

        elif is_no:
            active_slots.pop("pending_confirmation", None)
            reply = "❌ Request cancel kar di gayi hai. Agar aap kuch aur dekhna ya order karna chahte hain toh batayein!"
            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

        # Check if user is updating quantity (e.g. "5 order karni hai", "5 cup kar do", "3 plate kardo")
        qty_update_match = re.search(r"\b(\d+)\s*(?:plate|plates|portion|portions|cup|cups|glass|glasses|piece|pieces|bottle|bottles)?\b", clean_text)
        qty_word_val = None
        if not qty_update_match:
            for w, n in word_num_map.items():
                # Avoid false matching verb 'do' as number 2 unless explicit
                if w == "do" and re.search(r"\b(kar\s*do|kardo|dijiye)\b", clean_text):
                    continue
                if re.search(rf"\b{w}\b", clean_text):
                    qty_word_val = n
                    break
        new_qty = int(qty_update_match.group(1)) if qty_update_match else qty_word_val

        # If quantity update is requested without mentioning a new dish
        dish_indicator_words = ["paneer", "chicken", "coffee", "tea", "chai", "soup", "rice", "naan", "roti", "paratha", "pizza", "burger", "papad", "sandwich", "dal", "mutton", "fish"]
        has_new_dish = any(w in clean_text for w in dish_indicator_words)

        if new_qty and not has_new_dish and pending_conf.get("type") == "food_order":
            # Update quantity for items in pending order
            items = pending_conf.get("items", [])
            for it in items:
                it["qty"] = new_qty
                it["subtotal"] = it["price"] * new_qty

            subtotal = sum(it["subtotal"] for it in items)
            tax = round(subtotal * 0.05, 2)
            net_total = round(subtotal + tax, 2)

            pending_conf["items"] = items
            pending_conf["subtotal"] = subtotal
            pending_conf["tax"] = tax
            pending_conf["net_total"] = net_total
            active_slots["pending_confirmation"] = pending_conf

            items_preview = "\n".join([f"{i+1}. **{it['name']}** × {it['qty']} = ₹{int(it['subtotal'])}" for i, it in enumerate(items)])
            reply = (
                f"🧾 **Quantity Update ({new_qty} units) & Bill Preview:**\n"
                f"─────────────────────────────────────────\n"
                f"{items_preview}\n"
                f"─────────────────────────────────────────\n"
                f"💰 **Subtotal:** ₹{int(subtotal)}\n"
                f"🧾 **GST (5%):** ₹{tax:.2f}\n"
                f"💳 **Total Payable Amount:** ₹{net_total:.2f}\n"
                f"🍽️ **Service:** Dine-in (Table {pending_conf.get('table_number', 'T-02')})\n"
                f"─────────────────────────────────────────\n\n"
                f"⚠️ **Aapka Total ₹{net_total:.2f} hota hai. Kya main yeh order confirm karke kitchen mein bhej doon?**\n"
                f"*(Kripya 'Haan' / 'Yes' ya 'Nahi' / 'Cancel' bole)*"
            )
            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

    # =========================================================================
    # CASE 2: Table Availability / Status Queries & Table Reservation
    # =========================================================================
    tbl_match = re.search(r"\b(T-\d+|T\d+|table\s*\d+)\b", clean_text, re.IGNORECASE)
    guest_match = re.search(r"(\d+)\s*(?:person|people|guest|guests|pax|log|members|vyakti)", clean_text, re.IGNORECASE)
    
    is_avail_query = any(w in clean_text for w in [
        "available", "avaible", "availble", "khali", "free", "vacant", "status", "hai kya",
        "konsi", "koun si", "kounsi", "batao", "dikhao", "check", "mil sakti"
    ]) and not any(w in clean_text for w in ["order", "mangwa", "le aao"])

    is_food_order_intent = any(w in clean_text for w in [
        "order", "mangwa", "mangwana", "le aao", "parcel", "pack", "cup", "plate", "plates", "portion", "dish"
    ])

    is_booking_request = any(w in clean_text for w in [
        "book kardo", "book kar do", "booking karni", "booking kardo", "table book",
        "seat book", "reserve kardo", "reserve kar do", "reservation", "book karna", "reserve karna"
    ]) or (any(w in clean_text for w in ["book", "reserve", "booking"]) and not is_food_order_intent)

    food_keywords = ["pizza", "paneer", "chicken", "coffee", "tea", "chai", "dal", "soup", "plate", "dish", "rice", "naan", "roti", "paratha", "lassi", "mutton", "fish", "sandwich", "papad", "chocolate", "milk"]
    has_food_item = any(w in clean_text for w in food_keywords) or is_food_order_intent

    # Subcase 2A: Table Availability / Status Query (e.g. "T-11 available hai kya", "Garden me table khali hai kya")
    if (tbl_match or "table" in clean_text or "seat" in clean_text) and is_avail_query and not is_booking_request and not has_food_item:
        tbl_num = tbl_match.group(1).upper() if tbl_match else None
        if tbl_num:
            clean_tbl = tbl_num.replace("TABLE", "T-").replace(" ", "").strip()
            if not clean_tbl.startswith("T-") and clean_tbl.startswith("T"):
                clean_tbl = f"T-{clean_tbl[1:]}"
            tbl_num = clean_tbl

        if tbl_num:
            t_rows = SQLAgent.execute_with_guardrail(f"SELECT * FROM dining_tables WHERE UPPER(table_number) = '{tbl_num}' LIMIT 1;")[1]
            if t_rows:
                t_info = t_rows[0]
                status_str = t_info.get("status", "available").upper()
                sec = t_info.get("section", "Main")
                cap = t_info.get("capacity", 4)
                
                if status_str == "AVAILABLE":
                    reply = (
                        f"🪑 **Table {tbl_num} ({sec} • {cap} Seater)** abhi **AVAILABLE** (Khali) hai! ✅\n\n"
                        f"👉 Kya aap is table ko book karna chahte hain? (Aap bol sakte hain: *'{tbl_num} book kardo'*)"
                    )
                elif status_str == "RESERVED":
                    reply = (
                        f"🪑 **Table {tbl_num} ({sec} • {cap} Seater)** abhi **RESERVED** (Booked) hai. 🔒\n\n"
                        f"👉 Hamare paas {sec} section mein doosri tables available hain. Kya aap doosri table check karna chahte hain?"
                    )
                else:
                    reply = (
                        f"🪑 **Table {tbl_num} ({sec} • {cap} Seater)** abhi **OCCUPIED** (Guests seated) hai. 👥\n\n"
                        f"👉 Kya aap koi doosri available table dekhna chahte hain?"
                    )
            else:
                reply = f"Maaf kijiye, Table '{tbl_num}' hamare restaurant record mein nahi mili. Total tables T-01 se T-15 tak hain."

            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }
        else:
            # Section filter (e.g. Garden, Rooftop, AC Hall)
            sec_filter = ""
            for s in ["AC Hall", "Rooftop", "Garden", "Poolside", "Private Dining"]:
                if s.lower() in clean_text:
                    sec_filter = f" AND LOWER(section) = '{s.lower()}'"
                    break
            
            avail_tables = SQLAgent.execute_with_guardrail(f"SELECT table_number, capacity, section, status FROM dining_tables WHERE status = 'available'{sec_filter} ORDER BY id ASC;")[1]
            if avail_tables:
                rows_text = "\n".join([f"• **{t['table_number']}** — {t['section']} ({t['capacity']} Seater)" for t in avail_tables])
                reply = (
                    f"🪑 **Available Tables List:**\n"
                    f"─────────────────────────────────────────\n"
                    f"{rows_text}\n"
                    f"─────────────────────────────────────────\n"
                    f"👉 Aap inme se kisi bhi table ko book karne ke liye bol sakte hain (jaise *'T-10 book kardo'*)."
                )
            else:
                reply = "Abhi is section mein koi table available nahi hai. Hamare staff se floor par sampark karein."

            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

    # If user is simply providing an already occupied table number without booking intent (e.g. "Table T-05 par hu")
    is_pure_table_spec = tbl_match and not is_booking_request and not guest_match and not has_food_item
    if is_pure_table_spec:
        raw_t = tbl_match.group(1).upper().replace("TABLE", "T-").replace(" ", "").strip()
        if not raw_t.startswith("T-") and raw_t.startswith("T"):
            raw_t = f"T-{raw_t[1:]}"
        active_slots["assigned_table"] = raw_t

        pending_order_items = active_slots.pop("pending_order_items", None)
        if pending_order_items:
            subtotal = sum(it["subtotal"] for it in pending_order_items)
            tax = round(subtotal * 0.05, 2)
            net_total = round(subtotal + tax, 2)
            active_slots["pending_confirmation"] = {
                "type": "food_order",
                "items": pending_order_items,
                "subtotal": subtotal,
                "tax": tax,
                "net_total": net_total,
                "table_number": raw_t
            }
            items_preview = "\n".join([f"{i+1}. **{it['name']}** × {it['qty']} = ₹{int(it['subtotal'])}" for i, it in enumerate(pending_order_items)])
            reply = (
                f"🪑 **Table {raw_t} Assign ho gaya!**\n\n"
                f"🧾 **Order Summary & Bill Preview:**\n"
                f"─────────────────────────────────────────\n"
                f"{items_preview}\n"
                f"─────────────────────────────────────────\n"
                f"💰 **Subtotal:** ₹{int(subtotal)}\n"
                f"🧾 **GST (5%):** ₹{tax:.2f}\n"
                f"💳 **Total Payable Amount:** ₹{net_total:.2f}\n"
                f"🍽️ **Service:** Dine-in (Table {raw_t})\n"
                f"─────────────────────────────────────────\n\n"
                f"⚠️ **Aapka Total ₹{net_total:.2f} hota hai. Kya main Table {raw_t} par yeh order confirm karke kitchen mein bhej doon?**\n"
                f"*(Kripya 'Haan' / 'Yes' ya 'Nahi' / 'Cancel' bole)*"
            )
        else:
            reply = f"🪑 **Table {raw_t}** aapko assign kar di gayi hai! Ab aap apna order de sakte hain."

        return {
            "response": reply,
            "active_slots": active_slots,
            "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
        }

    # Subcase 2B: Table Booking Intent (HITL Confirmation)
    is_table_booking = (is_booking_request or (guest_match and any(w in clean_text for w in ["table", "seat", "seating"]))) and not is_food_order_intent
    if is_table_booking:
        tbl_num = tbl_match.group(1).upper() if tbl_match else None
        if tbl_num:
            clean_tbl = tbl_num.replace("TABLE", "T-").replace(" ", "").strip()
            if not clean_tbl.startswith("T-") and clean_tbl.startswith("T"):
                clean_tbl = f"T-{clean_tbl[1:]}"
            tbl_num = clean_tbl

        table_info = None
        if tbl_num:
            t_rows = SQLAgent.execute_with_guardrail(f"SELECT * FROM dining_tables WHERE UPPER(table_number) = '{tbl_num}' LIMIT 1;")[1]
            if t_rows:
                table_info = t_rows[0]
                if table_info.get("status") != "available":
                    status_val = table_info.get("status", "occupied")
                    reply = (
                        f"⚠️ **Maaf kijiye, Table {tbl_num} abhi '{status_val.upper()}' hai.**\n\n"
                        f"Aap kisi doosri available table (jaise T-01, T-02, T-08, T-10) par book kar sakte hain."
                    )
                    return {
                        "response": reply,
                        "active_slots": active_slots,
                        "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
                    }
        else:
            # Pick first available table matching capacity
            guests_count = int(guest_match.group(1)) if guest_match else 4
            t_rows = SQLAgent.execute_with_guardrail(f"SELECT * FROM dining_tables WHERE capacity >= {guests_count} AND status = 'available' LIMIT 1;")[1]
            if t_rows:
                table_info = t_rows[0]
                tbl_num = table_info["table_number"]

        if table_info:
            guests_count = int(guest_match.group(1)) if guest_match else table_info.get("capacity", 4)
            active_slots["pending_confirmation"] = {
                "type": "table_booking",
                "table_number": table_info["table_number"],
                "section": table_info["section"],
                "capacity": table_info["capacity"],
                "guests": guests_count
            }
            reply = (
                f"📋 **Table Booking Review & Details:**\n"
                f"─────────────────────────────────────────\n"
                f"🪑 **Table Number:** {table_info['table_number']}\n"
                f"📍 **Section:** {table_info['section']}\n"
                f"👥 **Capacity:** {table_info['capacity']} Seater (For {guests_count} Guests)\n"
                f"💳 **Reservation Fee:** ₹0 (Free Booking)\n"
                f"─────────────────────────────────────────\n\n"
                f"⚠️ **Kya main Table {table_info['table_number']} aapke liye confirm book kar doon?**\n"
                f"*(Kripya 'Haan' / 'Yes' ya 'Nahi' / 'Cancel' bole)*"
            )
        else:
            guests_count = int(guest_match.group(1)) if guest_match else 4
            reply = (
                f"Maaf kijiye, {guests_count} guests ke liye abhi koi table available nahi mila. "
                f"Kya aap kisi doosre section ya timings par check karna chahte hain?"
            )

        return {
            "response": reply,
            "active_slots": active_slots,
            "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
        }

    # =========================================================================
    # CASE 3: New Food Dish Ordering Intent
    # =========================================================================
    # 1. Translate synonyms (e.g. 'chai' -> 'tea')
    normalized_text = clean_text
    try:
        from agent.sql_agent import DYNAMIC_SYNONYMS
        for alias, targets in DYNAMIC_SYNONYMS.items():
            if alias in normalized_text and targets:
                normalized_text = re.sub(rf"\b{re.escape(alias)}\b", targets[0], normalized_text)
    except Exception:
        pass

    # Check if table number is specified in the order text itself (e.g. "Table T-05 par 2 Cold Coffee order kardo")
    if tbl_match:
        raw_t = tbl_match.group(1).upper().replace("TABLE", "T-").replace(" ", "").strip()
        if not raw_t.startswith("T-") and raw_t.startswith("T"):
            raw_t = f"T-{raw_t[1:]}"
        active_slots["assigned_table"] = raw_t

    # Split into item segments if multiple items ordered (e.g. '2 Butter Paneer Masala aur 4 Naan')
    segments = re.split(r"\b(?:aur|and|\+|comma|,)\b", normalized_text)
    matched_items = []

    # Fetch all menu items from DB for fast in-memory ranking
    all_menu_items = []
    try:
        conn = crm_db.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT m.id, m.name, m.price, c.name as category, i.stock, i.available
            FROM menu_items m
            LEFT JOIN categories c ON m.category_id = c.id
            LEFT JOIN inventory i ON m.name = i.name
            """
        )
        all_menu_items = [dict(r) for r in cursor.fetchall()]
    except Exception:
        all_menu_items = []

    stop_words = {
        "order", "karni", "karna", "hai", "karo", "merko", "mujhe", "chahiye", "mangwa", "batao",
        "please", "plate", "plates", "scoop", "ek", "do", "tin", "teen", "char", "paanch",
        "1", "2", "3", "4", "5", "lagao", "le", "aao", "ye", "yeh", "lao", "bhejo",
        "toh", "to", "bhi", "kuch", "wali", "wala", "waley", "wale", "me", "mein", "ka", "ki", "ke",
        "liye", "cup", "cups", "glass", "glasses", "portion", "portions", "item", "items", "dish", "table", "seat"
    }

    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue

        # Strip table references (e.g. 'Table T-08') from segment before extracting quantity and words
        clean_seg = re.sub(r"\b(T-\d+|T\d+|table\s*\d+)\b", "", seg, flags=re.IGNORECASE).strip()

        # Extract quantity (digits or words)
        qty = 1
        digit_match = re.search(r"\b(\d+)\s*(?:plate|plates|portion|portions|cup|cups|glass|glasses|piece|pieces|bottle|bottles)?\b", clean_seg)
        if digit_match:
            qty = max(1, int(digit_match.group(1)))
        else:
            for w, n in word_num_map.items():
                if re.search(rf"\b{w}\b", clean_seg):
                    qty = n
                    break

        words = [w for w in re.findall(r"\w+", clean_seg) if len(w) > 2 and w not in stop_words]
        if not words:
            continue

        # Smart Overlap & Substring Scoring against all menu items
        best_match = None
        best_score = 0.0

        for item in all_menu_items:
            item_name_lower = item["name"].lower()
            item_tokens = item_name_lower.split()

            # Exact token matches
            overlap = sum(1 for w in words if w in item_tokens)
            # Partial substring matches
            substring_matches = sum(0.5 for w in words if w in item_name_lower and w not in item_tokens)
            score = overlap * 2.0 + substring_matches

            # Bonus if phrase is directly contiguous in name (e.g. 'single pot masala tea')
            search_phrase = " ".join(words)
            if search_phrase in item_name_lower:
                score += 5.0

            if score > best_score and score >= 1.5:
                best_score = score
                best_match = item

        if best_match:
            price = float(best_match.get("price", 0.0))
            matched_items.append({
                "item_id": best_match.get("id", 1),
                "name": best_match.get("name"),
                "qty": qty,
                "price": price,
                "subtotal": price * qty,
                "category": best_match.get("category", "Menu"),
                "stock": best_match.get("stock"),
                "available": best_match.get("available", 1)
            })

    # Contextual Follow-up: If no dish matched in this turn, check last_dish from previous turn (e.g. 'ha mhuje 10 cup order karna hai')
    if not matched_items and active_slots.get("last_dish"):
        last_d = active_slots.get("last_dish")
        clean_followup = re.sub(r"\b(T-\d+|T\d+|table\s*\d+)\b", "", clean_text, flags=re.IGNORECASE).strip()
        qty = 1
        digit_match = re.search(r"\b(\d+)\s*(?:plate|plates|portion|portions|cup|cups|glass|glasses|piece|pieces|bottle|bottles)?\b", clean_followup)
        if digit_match:
            qty = max(1, int(digit_match.group(1)))
        else:
            for w, n in word_num_map.items():
                if re.search(rf"\b{w}\b", clean_followup):
                    qty = n
                    break

        price = float(last_d.get("price", 0.0))
        matched_items.append({
            "item_id": last_d.get("id", 1),
            "name": last_d.get("name"),
            "qty": qty,
            "price": price,
            "subtotal": price * qty,
            "category": last_d.get("category", "Menu"),
            "stock": last_d.get("stock"),
            "available": last_d.get("available", 1)
        })

    if matched_items:
        # Check stock availability
        out_of_stock = [it["name"] for it in matched_items if it.get("stock") is not None and (it["stock"] <= 0 or it["available"] == 0)]
        if out_of_stock:
            reply = f"Kshama karein, '{', '.join(out_of_stock)}' abhi out of stock hai. Kya aap iski jagah koi doosri dish pasand karenge?"
            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

        # MANDATORY CHECK: Table must be assigned before accepting order!
        assigned_table = active_slots.get("assigned_table")
        if not assigned_table:
            # Save items in active_slots so user doesn't have to re-type them
            active_slots["pending_order_items"] = matched_items
            items_names = ", ".join([f"{it['qty']}x {it['name']}" for it in matched_items])
            reply = (
                f"⚠️ **Order lene se pehle Table Assign karna zaroori hai!**\n"
                f"─────────────────────────────────────────\n"
                f"Aapki abhi koi Table assign nahi hai (Selected Items: {items_names}).\n\n"
                f"👉 Kripya apna **Table Number (jaise T-02 ya T-08)** batayein,\n"
                f"👉 Ya phir batayein aap **kitne logo ke liye Table book** karna chahte hain?\n\n"
                f"*Table assign hote hi aapka order confirmation ke liye process ho jayega!*"
            )
            return {
                "response": reply,
                "active_slots": active_slots,
                "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
            }

        subtotal = sum(it["subtotal"] for it in matched_items)
        tax = round(subtotal * 0.05, 2)
        net_total = round(subtotal + tax, 2)

        active_slots["pending_confirmation"] = {
            "type": "food_order",
            "items": matched_items,
            "subtotal": subtotal,
            "tax": tax,
            "net_total": net_total,
            "table_number": assigned_table
        }

        items_preview = "\n".join([f"{i+1}. **{it['name']}** × {it['qty']} = ₹{int(it['subtotal'])}" for i, it in enumerate(matched_items)])
        reply = (
            f"🧾 **Order Summary & Bill Preview:**\n"
            f"─────────────────────────────────────────\n"
            f"{items_preview}\n"
            f"─────────────────────────────────────────\n"
            f"💰 **Subtotal:** ₹{int(subtotal)}\n"
            f"🧾 **GST (5%):** ₹{tax:.2f}\n"
            f"💳 **Total Payable Amount:** ₹{net_total:.2f}\n"
            f"🍽️ **Service:** Dine-in (Table {assigned_table})\n"
            f"─────────────────────────────────────────\n\n"
            f"⚠️ **Aapka Total ₹{net_total:.2f} hota hai. Kya main Table {assigned_table} par yeh order confirm karke kitchen mein bhej doon?**\n"
            f"*(Kripya 'Haan' / 'Yes' ya 'Nahi' / 'Cancel' bole)*"
        )
    else:
        reply = (
            "Aapne jis item ke liye order request kiya hai, kripya uska poora naam batayein (jaise 'Butter Paneer Masala 2 plate' ya 'Single Pot Masala Tea'). "
            "Aap hamara menu dekh kar bhi chun sakte hain!"
        )

    return {
        "response": reply,
        "active_slots": active_slots,
        "messages": state.get("messages", []) + [HumanMessage(content=raw_input), AIMessage(content=reply)]
    }

def sql_node(state: AgentState) -> Dict[str, Any]:
    """SQL Sub-Agent: Schema Pruner + Generator + Guardrails + DB Execution + Self-Healing."""
    query = state.get("rewritten_query", state["user_input"])
    user_role = state.get("user_role", "customer")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    previous_error = state.get("error")

    # Step A: Schema Pruner
    tables = SQLAgent.prune_schema(query)

    # Step B: Generate SQL
    generated_sql = SQLAgent.generate_sql(
        query=query,
        tables=tables,
        user_role=user_role,
        previous_error=previous_error
    )

    if not generated_sql or not generated_sql.strip():
        return {
            "tables": tables,
            "sql_query": None,
            "query_result": None,
            "result_status": "error",
            "error": "Failed to generate valid SQL query.",
            "retry_count": retry_count + 1
        }

    # Clean base SQL for pagination (remove any LIMIT / OFFSET and trailing semicolons)
    clean_sql = generated_sql.strip().rstrip(";").strip()
    base_sql = re.sub(r"\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*$", "", clean_sql, flags=re.IGNORECASE).strip().rstrip(";").strip()
    page_size = 10

    # If it's a multi-row query without aggregates, execute with LIMIT 10 OFFSET 0
    if not re.search(r"\b(COUNT|AVG|SUM|MIN|MAX)\s*\(", base_sql, re.IGNORECASE):
        paged_sql = f"{base_sql} LIMIT {page_size} OFFSET 0;"
    else:
        paged_sql = generated_sql

    # Step C & D: Guardrail Validation & DB Execution
    success, results, error_msg, executed_sql = SQLAgent.execute_with_guardrail(
        query=paged_sql,
        user_role=user_role
    )

    if not success and retry_count < max_retries:
        # Trigger self-correction retry
        return {
            "tables": tables,
            "sql_query": executed_sql,
            "query_result": None,
            "error": error_msg,
            "retry_count": retry_count + 1,
            "result_status": "error"
        }

    # Step E: Clean base SQL for pagination (remove LIMIT / OFFSET)
    base_sql = re.sub(r"\s+LIMIT\s+\d+(\s+OFFSET\s+\d+)?\s*;?$", "", executed_sql, flags=re.IGNORECASE).strip()
    page_size = 10
    has_more = len(results) >= page_size if results else False

    cur_tokens = state.get("token_usage") or {"prompt": 0, "completion": 0, "total": 0}
    sql_tok = getattr(SQLAgent, "last_token_usage", {}) or {}
    updated_tokens = {
        "prompt": cur_tokens.get("prompt", 0) + sql_tok.get("prompt_tokens", 0),
        "completion": cur_tokens.get("completion", 0) + sql_tok.get("completion_tokens", 0),
        "total": cur_tokens.get("total", 0) + sql_tok.get("total_tokens", 0)
    }

    # Save last dish in active_slots for contextual follow-up orders
    active_slots = dict(state.get("active_slots", {}))
    if results and any(t in tables for t in ["menu_items", "categories"]):
        first_row = results[0]
        if "name" in first_row and ("price" in first_row or "category" in first_row):
            active_slots["last_dish"] = first_row

    return {
        "tables": tables,
        "sql_query": executed_sql,
        "query_result": results,
        "base_sql": base_sql,
        "page": 1,
        "page_size": page_size,
        "has_more": has_more,
        "doc_context": None,
        "token_usage": updated_tokens,
        "active_slots": active_slots,
        "error": error_msg,
        "retry_count": retry_count
    }

def result_validator_node(state: AgentState) -> Dict[str, Any]:
    """Node 6b: Dedicated Result Validator inspecting database execution output."""
    sql_query = state.get("sql_query", "")
    results = state.get("query_result")
    error_msg = state.get("error")

    res_status, val_msg = ResultValidator.validate(sql_query, results, error_msg)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    new_retry = retry_count + 1 if res_status == "error" else retry_count

    return {
        "result_status": res_status,
        "error": val_msg if res_status == "error" else error_msg,
        "retry_count": new_retry
    }

def rag_node(state: AgentState) -> Dict[str, Any]:
    """RAG Sub-Agent: Retrieves restaurant timings, policies, and guidelines."""
    query = state.get("rewritten_query", state["user_input"])
    docs = rag_agent.retrieve(query, top_k=2)
    return {
        "doc_context": docs,
        "query_result": None,
        "sql_query": None,
        "result_status": "valid" if docs else "empty"
    }

def hybrid_node(state: AgentState) -> Dict[str, Any]:
    """Hybrid Node: Coordinates both SQL and RAG retrieval."""
    sql_res = sql_node(state)
    rag_res = rag_node(state)
    return {**sql_res, **rag_res}

def pagination_node(state: AgentState) -> Dict[str, Any]:
    """Dedicated Pagination Node: executes next OFFSET on base_sql without LLM generation."""
    base_sql = state.get("base_sql")
    page_size = state.get("page_size", 12)
    current_page = state.get("page", 1)

    if not base_sql:
        reply = "Aapne pehle koi list ya menu search nahi kiya hai. Kripya pehle batayein aap kya dekhna chahte hain (jaise 'menu me kya hai' ya 'Rahul ka attendance')."
        return {
            "response": reply,
            "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
        }

    clean_in = state["user_input"].lower().strip()
    is_prev = (state.get("active_slots", {}).get("pagination_direction") == "prev") or any(
        w in clean_in for w in ["prev", "previous", "back", "piche", "peeche", "pichla", "pichhla", "pehle"]
    )

    if is_prev:
        if current_page <= 1:
            reply = "Aap pehle se hi Page 1 par hain. Isse pehle koi page nahi hai."
            return {
                "response": reply,
                "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
            }
        target_page = current_page - 1
    else:
        target_page = current_page + 1

    offset = (target_page - 1) * page_size
    paged_sql = f"{base_sql.rstrip('; ')} LIMIT {page_size} OFFSET {offset};"

    success, results, error_msg, executed_sql = SQLAgent.execute_with_guardrail(
        query=paged_sql,
        user_role=state.get("user_role", "customer")
    )

    if not success or not results:
        reply = f"Yeh is list ka aakhiri page tha (Page {current_page}). Iske aage aur koi items nahi hain. Kya aap kuch aur search karna chahte hain?"
        return {
            "response": reply,
            "has_more": False,
            "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
        }

    res_status, val_msg = ResultValidator.validate(executed_sql, results, error_msg)

    return {
        "sql_query": executed_sql,
        "query_result": results,
        "result_status": res_status,
        "page": target_page,
        "page_size": page_size,
        "has_more": len(results) >= page_size,
        "response": None,
        "error": None
    }

def formatter_node(state: AgentState) -> Dict[str, Any]:
    """Node 9a: Zero-Token Deterministic Markdown Table Formatter (Bypasses Synthesizer LLM)."""
    query = state.get("rewritten_query", state["user_input"])
    results = state.get("query_result", [])
    sql_query = state.get("sql_query")

    if not results:
        final_text = "Maaf kijiye, koi record nahi mila."
    else:
        # Check if it's a single-entity direct conversational lookup
        direct_lookup_text = ResponseFormatter.format_direct_lookup(query, results)
        if direct_lookup_text:
            final_text = direct_lookup_text
        else:
            sanitized_rows = ResponseFormatter.sanitize_for_presentation(query, results[:15])
            if len(sanitized_rows) == 1 and len(sanitized_rows[0]) <= 2:
                items = [f"**{k}:** {v}" for k, v in sanitized_rows[0].items()]
                final_text = "\n".join(items)
            else:
                final_text = ResponseFormatter.format_markdown_table(sanitized_rows)


    current_page = state.get("page", 1)
    if state.get("has_more"):
        if current_page > 1:
            final_text += f"\n\n*(Page {current_page} • Agle ke liye 'next' ya pichhle ke liye 'previous' bole)*"
        else:
            final_text += f"\n\n*(Page 1 • Aage dekhne ke liye 'next item do' ya 'aur dikhao' bole)*"
    elif current_page > 1:
        final_text += f"\n\n*(Yeh aakhiri page hai - Page {current_page} • Pichhle ke liye 'previous' bole)*"

    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["user_input"]),
        AIMessage(content=final_text)
    ]

    audit_data = {
        "timestamp": time.time(),
        "user_role": state.get("user_role", "customer"),
        "raw_input": state["user_input"],
        "rewritten_query": query,
        "intent": state.get("intent"),
        "sql": sql_query,
        "rows_count": len(results) if results else 0,
        "status": "success",
        "formatter": "direct_deterministic_bypass"
    }

    return {
        "response": final_text,
        "messages": new_messages,
        "audit_log": audit_data
    }

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Node 9b: Synthesizes responses for RAG policies, empty fallbacks, and error messages."""
    query = state.get("rewritten_query", state["user_input"])
    results = state.get("query_result")
    result_status = state.get("result_status")
    doc_context = state.get("doc_context")
    sql_query = state.get("sql_query")
    error = state.get("error")

    # If execution encountered a fatal error
    if result_status == "error" or error:
        reply = f"Kshama karein, request process karte waqt ek issue aaya: {error}. Kripya thodi der baad dobara koshish karein."
        return {
            "response": reply,
            "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
        }

    # If no data found - Dynamically formulate a helpful follow-up / alternative suggestion
    if result_status == "empty" and not doc_context:
        cur_tokens = state.get("token_usage") or {"prompt": 0, "completion": 0, "total": 0}
        q_lower = query.lower()
        tables_used = state.get("tables") or []
        is_staff_query = any(w in q_lower for w in ["staff", "employee", "waiter", "chef", "manager", "attendance", "shift", "salary", "hours", "phone", "suresh", "rahul", "vikash", "dinesh"]) or any(t in tables_used for t in ["employees", "attendance"])
        is_inventory_query = "inventory" in tables_used or "stock" in q_lower

        if is_staff_query:
            situation = (
                f"The database search for staff/employee '{query}' returned 0 matching records. "
                "Explain politely in Hindi/Hinglish that this specific staff member was not found in our employee roster. "
                "Ask if they want to check details for another staff member (like Rahul Sharma, Suresh Meena, Vikash Mehra, Mohan Lal, Dinesh Gurjar) or verify spelling. "
                "CRITICAL: Do NOT suggest food, dishes, or menu items!"
            )
        elif is_inventory_query:
            situation = (
                f"The inventory search for '{query}' returned 0 matching stock records. "
                "Explain politely that this raw item is not tracked or currently not available in store inventory."
            )
        else:
            situation = (
                "The database executed the query successfully, but returned 0 matching records. "
                "Explain politely that this specific dish was not found, suggest 2 relevant available alternatives from our royal Rajasthani/North Indian & Continental restaurant menu, and ask a dynamic follow-up question."
            )

        try:
            messages = DYNAMIC_FOLLOWUP_PROMPT.format_messages(
                user_query=query,
                situation=situation
            )
            llm_resp = llm.invoke(messages)
            clean_reply = re.sub(r"<think>.*?(?:</think>|$)", "", llm_resp.content, flags=re.DOTALL).strip()
            clean_reply = re.sub(r"(?i)here's a thinking process:.*$", "", clean_reply, flags=re.DOTALL).strip()
            clean_reply = re.sub(r"(?i)thinking process:.*$", "", clean_reply, flags=re.DOTALL).strip()
            reply = str_parser.parse(clean_reply).strip()

            tok = getattr(llm_resp, "response_metadata", {}).get("token_usage", {})
            final_tokens = {
                "prompt": cur_tokens.get("prompt", 0) + tok.get("prompt_tokens", 0),
                "completion": cur_tokens.get("completion", 0) + tok.get("completion_tokens", 0),
                "total": cur_tokens.get("total", 0) + tok.get("total_tokens", 0)
            }
        except Exception:
            reply = f"Maaf kijiye, '{query}' ke mutabik koi record nahi mila. Hamare paas North Indian, Tandoori Starters aur Beverages available hain. Kya aap unke options dekhna chahenge?"
            final_tokens = cur_tokens

        return {
            "response": reply,
            "token_usage": final_tokens,
            "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
        }

    # Prepare grounding prompt for LLM (only for RAG / Policy / Hybrid queries)
    context_data = ""
    if results is not None:
        sanitized_rows = ResponseFormatter.sanitize_for_presentation(query, results[:15])
        context_data += f"\nDatabase Query Results ({len(results)} rows):\n{json.dumps(sanitized_rows, indent=1)}\n"
    if doc_context:
        context_data += f"\nRestaurant Policy & Operational Documents:\n{json.dumps(doc_context, indent=1)}\n"

    messages = SYNTHESIZER_PROMPT.format_messages(
        query=query,
        context_data=context_data if context_data else "No specific context available."
    )

    llm_resp = llm.invoke(messages)
    clean_content = re.sub(r"<think>.*?(?:</think>|$)", "", llm_resp.content, flags=re.DOTALL).strip()
    clean_content = re.sub(r"(?i)here's a thinking process:.*$", "", clean_content, flags=re.DOTALL).strip()
    clean_content = re.sub(r"(?i)thinking process:.*$", "", clean_content, flags=re.DOTALL).strip()
    final_text = str_parser.parse(clean_content).strip()

    # Update state messages
    new_messages = state.get("messages", []) + [
        HumanMessage(content=state["user_input"]),
        AIMessage(content=final_text)
    ]

    # Audit log
    audit_data = {
        "timestamp": time.time(),
        "user_role": state.get("user_role", "customer"),
        "raw_input": state["user_input"],
        "rewritten_query": query,
        "intent": state.get("intent"),
        "sql": sql_query,
        "rows_count": len(results) if results else 0,
        "status": "success"
    }

    cur_tokens = state.get("token_usage") or {"prompt": 0, "completion": 0, "total": 0}
    syn_tok = getattr(llm_resp, "response_metadata", {}).get("token_usage", {}) if llm_resp is not None else {}
    final_tokens = {
        "prompt": cur_tokens.get("prompt", 0) + syn_tok.get("prompt_tokens", 0),
        "completion": cur_tokens.get("completion", 0) + syn_tok.get("completion_tokens", 0),
        "total": cur_tokens.get("total", 0) + syn_tok.get("total_tokens", 0)
    }

    return {
        "response": final_text,
        "messages": new_messages,
        "base_sql": state.get("base_sql"),
        "page": state.get("page", 1),
        "page_size": state.get("page_size", 10),
        "has_more": state.get("has_more", False),
        "token_usage": final_tokens,
        "audit_log": audit_data
    }

# ----------------- CONDITIONAL ROUTING ----------------- #

def route_after_fast_filter(state: AgentState) -> str:
    """Node 0 Routing: Dispatches greetings, pagination, orders, RAG, rewriter, or direct SQL."""
    intent = state.get("intent", "sql")
    needs_context = state.get("needs_context", False)

    if intent == "greeting":
        return "greeting"
    elif intent == "pagination":
        return "pagination"
    elif intent == "order":
        return "order"
    elif intent == "rag":
        return "rag"
    elif needs_context:
        return "rewriter"
    else:
        return "sql"

def route_after_rewriter(state: AgentState) -> str:
    """Routes based on intent detected by rewriter node."""
    intent = state.get("intent", "sql")
    confidence = state.get("confidence", 1.0)
    
    if intent == "greeting":
        return "greeting"
    elif intent == "clarification" or confidence < 0.6:
        return "clarification"
    elif intent == "pagination":
        return "pagination"
    elif intent == "order":
        return "order"
    elif intent == "rag":
        return "rag"
    elif intent == "hybrid":
        return "hybrid"
    else:
        return "sql"

def route_after_validation(state: AgentState) -> str:
    """Routes based on ResultValidator: self-healing retry, 0-token formatter, or synthesizer fallback."""
    res_status = state.get("result_status")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if res_status == "error" and retry_count < max_retries:
        return "retry_sql"
    elif res_status == "valid":
        return "formatter"
    else:
        return "synthesizer"

def route_after_pagination(state: AgentState) -> str:
    """If pagination node produced a direct response (e.g. end of list), exit; otherwise format with 0 tokens."""
    if state.get("query_result") is None:
        return END
    return "formatter"

# ----------------- GRAPH ASSEMBLY ----------------- #

def create_restaurant_agent_graph():
    """Builds and compiles the complete LangGraph conversational graph with fast-path triaging."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("fast_filter", fast_filter_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("order", order_node)
    workflow.add_node("pagination", pagination_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("result_validator", result_validator_node)
    workflow.add_node("formatter", formatter_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("hybrid", hybrid_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Entry point is now the Deterministic Fast Filter
    workflow.set_entry_point("fast_filter")

    # Conditional edge from fast_filter
    workflow.add_conditional_edges(
        "fast_filter",
        route_after_fast_filter,
        {
            "greeting": "greeting",
            "pagination": "pagination",
            "order": "order",
            "rag": "rag",
            "rewriter": "rewriter",
            "sql": "sql"
        }
    )

    # Conditional edge from rewriter (for contextual follow-ups)
    workflow.add_conditional_edges(
        "rewriter",
        route_after_rewriter,
        {
            "greeting": "greeting",
            "clarification": "clarification",
            "pagination": "pagination",
            "order": "order",
            "sql": "sql",
            "rag": "rag",
            "hybrid": "hybrid"
        }
    )

    # Direct exits
    workflow.add_edge("greeting", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("order", END)
    workflow.add_edge("formatter", END)

    # SQL node passes directly to dedicated result_validator
    workflow.add_edge("sql", "result_validator")

    # Result validator routes to self-heal (sql), 0-token formatter, or synthesizer
    workflow.add_conditional_edges(
        "result_validator",
        route_after_validation,
        {
            "retry_sql": "sql",
            "formatter": "formatter",
            "synthesizer": "synthesizer"
        }
    )

    # Pagination routes directly to 0-token formatter or END
    workflow.add_conditional_edges(
        "pagination",
        route_after_pagination,
        {
            "formatter": "formatter",
            END: END
        }
    )

    # RAG & Hybrid routes to synthesizer
    workflow.add_edge("rag", "synthesizer")
    workflow.add_edge("hybrid", "synthesizer")
    workflow.add_edge("synthesizer", END)

    # Checkpointer for session persistence
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    return compiled_graph

# Global compiled graph instance
restaurant_agent = create_restaurant_agent_graph()
