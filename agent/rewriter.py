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

    # 1. Greetings & Capabilities Queries (0 LLM Tokens -> greeting node)
    capabilities_phrases = [
        "what do you provide", "what do u provide", "what can you do", "who are you",
        "tum kya kar sakte ho", "aap kya kar sakte ho", "kya kya service hai", "help", "features", "capabilities"
    ]
    is_greeting = clean_input in ["hi", "hello", "hey", "namaste", "namaskar", "pranam", "good morning", "good evening", "thanks", "thank you", "dhanyawad"]
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

    # 2. Food & Drink Ordering Action (0 LLM Tokens -> order node)
    order_phrases = [
        "order karni hai", "order karna hai", "order lena hai", "order lagao", "order book karo",
        "order chahiye", "ye mangwa do", "mangwa do", "le aao", "pack kar do", "parcel kar do", "order please"
    ]
    if any(p in clean_input for p in order_phrases):
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
        "menu", "dish", "dishes", "food", "coffee", "tea", "drink", "drinks", "beverage", "beverages",
        "shake", "shakes", "juice", "juices", "price", "rate", "cost",
        "veg", "non-veg", "non veg", "jain", "spicy", "teekha", "sweet", "meetha", "mithai", "halwa",
        "pizza", "burger", "ice cream", "dessert", "desserts", "lassi", "momo", "dimsum", "rolls", "chaat",
        "bread", "roti", "naan", "paratha", "starter", "starters", "soup", "chilly", "paneer", "chicken", "dal",
        "waiter", "waiters", "chef", "head chef", "salary", "attendance", "hours", "kaam kiya", "shift",
        "employee", "employees", "staff", "table", "active order", "active orders",
        "bill", "kitchen", "cooking", "served", "pending", "stock", "inventory", "available"
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
