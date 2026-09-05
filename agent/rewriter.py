import os
import json
import re
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from agent.prompts import REWRITER_PROMPT

load_dotenv()
api_key = os.getenv("groq_key") or os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=api_key,
    temperature=0.1,
    max_tokens=400,
    max_retries=3,
)

class RewriterAnalysis(BaseModel):
    is_greeting: bool = Field(default=False, description="True if input is pure small talk or greeting")
    needs_clarification: bool = Field(default=False, description="True if NLP understanding is ambiguous or input is too vague")
    clarification_question: Optional[str] = Field(default=None, description="Polite question asking user for clarification with 2-3 options")
    rewritten_query: str = Field(description="Self-contained standalone query resolving pronouns and historical filters")
    intent: Literal["greeting", "clarification", "sql", "rag", "hybrid", "pagination", "order"] = Field(default="sql", description="Routing destination")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0")
    updated_slots: Dict[str, Any] = Field(default_factory=dict, description="Active session slots e.g. budget, category, table_id")

parser = JsonOutputParser(pydantic_object=RewriterAnalysis)

def fast_filter_classify(
    user_input: str,
    active_slots: Optional[Dict[str, Any]] = None,
    has_history: bool = False
) -> Dict[str, Any]:
    """
    Deterministic Fast-Filter (0 LLM Tokens):
    Filters commands (greeting, pagination, order, policy) and standalone business queries
    away from the Rewriter LLM, saving ~1,000 prompt tokens per request.
    """
    slots = active_slots or {}
    clean_input = user_input.strip().lower()

    # 1. Check if user is confirming, cancelling, or modifying an active HITL Pending Confirmation or assigning Table
    pending_conf = slots.get("pending_confirmation")
    pending_items = slots.get("pending_order_items")
    tbl_pattern = bool(re.search(r"\b(T-\d+|T\d+|table\s*\d+)\b", clean_input, re.IGNORECASE))

    # Check if user is asking about their assigned table status or any table availability / status (0 Tokens)
    is_my_table_query = any(p in clean_input for p in [
        "konsi table assign", "koun si table assign", "meri table", "mera table",
        "assigned table", "my table", "mera table number", "meri table konsi", "konsa table assign", "kounsa table assign", "meri booking"
    ])
    is_table_avail_query = (
        tbl_pattern or
        any(w in clean_input for w in ["table", "tables", "seat", "seating", "seater"])
    ) and any(w in clean_input for w in [
        "available", "availble", "avaible", "khali", "free", "vacant", "status", "hai kya",
        "dikhao", "batao", "mil sakti", "book", "reserve", "booking", "garden hai", "rooftop hai", "ac hall"
    ])

    ord_num_pattern = bool(re.search(r"\b(ORD-[\w\-]+)\b", user_input, re.IGNORECASE))
    is_my_order_status_query = ord_num_pattern or any(p in clean_input for p in [
        "order complete", "order status", "order kab", "order ban gya", "order bana",
        "mera order", "status kya hai", "order ka status", "order ready", "khana kab aayega"
    ]) and not any(w in clean_input for w in ["pizza", "coffee", "tea", "paneer", "chocolate", "plate", "cup", "mangwa", "order kardo", "order kar do", "order karni hai", "order karna hai"])

    # 0C. Check for Table Bill Generation, Checkout, QR Code, or Payment Settlement (0 LLM Tokens -> order node)
    is_bill_or_checkout_query = any(p in clean_input for p in [
        "bill bana do", "bill banado", "bill banao", "bill kitna", "mera bill", "total bill",
        "bill do", "bill de do", "bill chahiye", "bill mangwa", "bill le aao", "check out", "checkout",
        "hisaab", "hisab", "payment", "pay karna", "settle", "bill ready", "qr", "qr code", "upi qr", "scanner", "scan", "barcode"
    ]) or (
        any(w in clean_input for w in ["bill", "checkout", "hisaab", "hisab", "payment", "settle", "qr", "upi"]) and
        (tbl_pattern or bool(slots.get("assigned_table")))
    )

    is_payment_done_phrase = any(p in clean_input for p in [
        "upi se", "cash de", "card se", "payment done", "pay kar diya", "paid",
        "bill pay", "paise de diye", "online pay", "payment kardi", "pay kar dia", "done payment"
    ])

    if is_bill_or_checkout_query or is_payment_done_phrase:
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "order",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    is_table_spec_phrase = tbl_pattern and any(w in clean_input for w in [
        "baithe", "baitha", "par", "pe", "assign", "seat", "seating", "table", "hu", "hain"
    ])

    if is_my_table_query or is_table_avail_query or is_my_order_status_query or is_table_spec_phrase:
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "order",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    if pending_items and tbl_pattern:
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "order",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    if pending_conf:
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
        is_yes = any(clean_input == w or clean_input.startswith(w + " ") for w in affirmative_words)
        is_no = any(clean_input == w or clean_input.startswith(w + " ") for w in negative_words)
        is_qty_update = bool(re.search(r"\b(\d+|ek|do|teen|tin|char|chaar|paanch|panch)\s*(?:order|plate|plates|cup|cups|glass|portion|piece|kardo|karni)?\b", clean_input))

        if is_yes or is_no or is_qty_update or is_payment_done_phrase:
            return {
                "needs_context": False,
                "is_fast_exit": True,
                "intent": "order",
                "confidence": 1.0,
                "rewritten_query": user_input,
                "updated_slots": slots,
                "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    # 2. Greetings & Capabilities Queries (0 LLM Tokens -> greeting node)
    capabilities_phrases = [
        "what do you provide", "what do u provide", "what can you do", "who are you",
        "tum kya kar sakte ho", "aap kya kar sakte ho", "kya kya service hai", "help", "features", "capabilities",
        "kya karte ho", "kya kya karte ho", "batao kya karte ho", "batao kya kar sakte ho", "kya help kar sakte ho",
        "services kya hai", "apne baare me batao", "intro", "introduction", "kya kaam karte ho"
    ]
    is_greeting = clean_input in [
        "hi", "hello", "hey", "namaste", "namaskar", "pranam", "good morning", "good evening",
        "thanks", "thank you", "dhanyawad", "shukriya", "kya haal hai", "kaise ho"
    ]
    is_capability = any(p in clean_input for p in capabilities_phrases)

    if is_greeting or is_capability:
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "greeting",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 3. Food & Drink Ordering & Table Booking Action (0 LLM Tokens -> order node)
    is_order_intent = (
        any(w in clean_input for w in ["order", "book", "reserve", "booking", "mangwa", "le aao", "pack kar", "parcel", "chai mangwa", "coffee mangwa"]) and
        not any(w in clean_input for w in ["active order", "orders count", "status", "kitne order", "kitchen order", "live order", "order list", "kitne hai", "kya order hai"])
    )
    if is_order_intent:
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "order",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }


    # 3. Pagination Commands (0 LLM Tokens -> pagination node)
    prev_keywords = [
        "previous", "prev", "back", "piche", "peeche", "pichla", "pichhla", "pehle wala",
        "previous page", "pichhla page", "pichhe ka", "previous dikhao", "pichle items"
    ]
    if any(clean_input == k or clean_input.startswith(k) for k in prev_keywords):
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "pagination",
            "confidence": 1.0,
            "rewritten_query": "Show previous page of previous results",
            "updated_slots": {**slots, "pagination_direction": "prev"},
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    next_keywords = [
        "next", "next page", "next item do", "next item", "next items", "next do", "aur dikhao",
        "aur batao", "more", "show more", "aage ka", "aage dikhao", "baaki dishes",
        "baaki items", "next 10", "next records", "agle items", "agle page"
    ]
    if any(clean_input == k or clean_input.startswith(k) for k in next_keywords):
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "pagination",
            "confidence": 1.0,
            "rewritten_query": "Show next page of previous results",
            "updated_slots": {**slots, "pagination_direction": "next"},
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 4. Direct Operational Policy Queries (0 LLM Tokens -> RAG)
    policy_keywords = [
        "policy", "policies", "rule", "rules", "guideline", "guidelines",
        "outside food", "alcohol", "liquor", "wine", "beer", "drink allowed",
        "cancellation", "cancel", "refund", "dress code", "pet", "pets",
        "parking", "smoking", "wifi", "children", "kids", "allowed", "permission",
        "timing", "timings", "hours", "open", "close", "start hota", "shuru hota", "khatam",
        "breakfast timing", "lunch timing", "dinner timing", "tandoor timing", "buffet timing", "break timing"
    ]
    if any(k in clean_input for k in policy_keywords):
        return {
            "needs_context": False,
            "is_fast_exit": True,
            "intent": "rag",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # 5. Check if query requires Context Resolution (Follow-ups, Corrections & Pronouns)
    context_indicators = [
        "inme", "inmein", "iska", "iske", "iski", "unka", "unke", "usme", "usmein",
        "ye wale", "wo wale", "aur dikhao", "or dikhao", "isme", "isme se", "unme se",
        "nahi", "nhi", "not", "instead", "badal", "change", "sirf", "only", "tak chalega", "bhi"
    ]
    has_context_cue = any(p in clean_input for p in context_indicators)

    if has_history and has_context_cue:
        return {
            "needs_context": True,
            "is_fast_exit": False,
            "intent": "sql",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # Standalone CRM business query without pronouns (0 Rewriter LLM Tokens -> SQL node)
    crm_keywords = [
        "menu", "dish", "dishes", "food", "coffee", "tea", "chai", "chay", "chaay", "doodh", "dudh",
        "paani", "pani", "water", "chawal", "bhaat", "rice", "drink", "drinks", "beverage", "beverages",
        "shake", "shakes", "juice", "juices", "price", "rate", "cost", "paisa", "rupaye", "rs", "kitne ka",
        "veg", "non-veg", "non veg", "jain", "spicy", "teekha", "sweet", "meetha", "mithai", "halwa",
        "pizza", "burger", "ice cream", "dessert", "desserts", "lassi", "momo", "dimsum", "rolls", "chaat",
        "bread", "roti", "naan", "paratha", "starter", "starters", "soup", "chilly", "paneer", "chicken", "dal",
        "waiter", "waiters", "chef", "head chef", "salary", "attendance", "hours", "kaam kiya", "shift",
        "employee", "employees", "staff", "table", "active order", "active orders",
        "bill", "kitchen", "cooking", "served", "pending", "stock", "inventory", "available", "milta hai", "hai kya"
    ]
    if not has_context_cue and any(k in clean_input for k in crm_keywords):

        return {
            "needs_context": False,
            "is_fast_exit": False,
            "intent": "sql",
            "confidence": 1.0,
            "rewritten_query": user_input,
            "updated_slots": slots,
            "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

    # If has_history and pronouns/follow-up detected, requires LLM Rewriter
    return {
        "needs_context": True,
        "is_fast_exit": False,
        "intent": "sql",
        "confidence": 1.0,
        "rewritten_query": user_input,
        "updated_slots": slots,
        "tokens": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


def analyze_and_rewrite(
    user_input: str,
    chat_history: List[BaseMessage],
    active_slots: Dict[str, Any]
) -> Dict[str, Any]:
    """
    LCEL Chain with ChatPromptTemplate and JsonOutputParser for robust context resolution.
    Only called when needs_context is True!
    """
    clean_input = user_input.strip().lower()

    # 4. Compact sliding window of last 3 messages
    window_messages = chat_history[-3:] if len(chat_history) > 3 else chat_history
    formatted_history = []
    for msg in window_messages:
        if isinstance(msg, HumanMessage):
            formatted_history.append(f"User: {msg.content}")
        else:
            # Strip large tables and verbose boilerplate to save ~75% prompt tokens
            clean_lines = [l.strip() for l in str(msg.content).split("\n") if l.strip() and not l.strip().startswith("|")]
            short_content = " ".join(clean_lines[:2])
            short_content = short_content[:120] if len(short_content) > 120 else short_content
            formatted_history.append(f"Assistant: {short_content}")
    history_str = "\n".join(formatted_history) if formatted_history else "No previous history."

    # 3. Format prompt using ChatPromptTemplate
    messages = REWRITER_PROMPT.format_messages(
        active_slots=json.dumps(active_slots or {}),
        chat_history=history_str,
        user_input=user_input
    )

    try:
        response = llm.invoke(messages)
        content = re.sub(r"<think>.*?(?:</think>|$)", "", response.content, flags=re.DOTALL).strip()
        parsed = parser.parse(content)

        # Ensure employee and staff queries are always routed to SQL database
        if any(w in clean_input for w in ["employee", "staff", "waiter", "chef", "manager", "captain"]):
            parsed["intent"] = "sql"
            parsed["confidence"] = 1.0
            parsed["needs_clarification"] = False

        # Merge updated slots with persistent active_slots
        merged_slots = dict(active_slots or {})
        for k, v in parsed.get("updated_slots", {}).items():
            if v is not None:
                merged_slots[k] = v
        parsed["updated_slots"] = merged_slots
        parsed["tokens"] = response.response_metadata.get("token_usage", {})
        return parsed

    except Exception as e:
        # Fallback to safe defaults if parsing fails
        return {
            "is_greeting": False,
            "needs_clarification": False,
            "clarification_question": None,
            "rewritten_query": user_input,
            "intent": "sql",
            "confidence": 0.5,
            "updated_slots": active_slots
        }
