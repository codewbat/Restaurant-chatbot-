from typing import TypedDict, List, Dict, Any, Optional, Literal
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """Unified state maintained across the LangGraph conversational agent."""
    # Conversation & Memory
    messages: List[BaseMessage]
    user_input: str
    rewritten_query: Optional[str]
    active_slots: Dict[str, Any]  # Persistent slots: e.g. {"table_id": 4, "category": "coffee", "budget": 200, "employee": "Rahul"}
    
    # NLP Understanding & Routing
    intent: Optional[Literal["greeting", "clarification", "sql", "rag", "hybrid", "general", "pagination", "order"]]
    confidence: float
    clarification_question: Optional[str]
    
    # SQL Sub-Agent Pipeline
    tables: List[str]
    sql_query: Optional[str]
    query_result: Optional[List[Dict[str, Any]]]
    result_status: Optional[Literal["valid", "empty", "suspicious", "error"]]
    
    # Pagination State
    base_sql: Optional[str]  # Saved query without LIMIT/OFFSET for pagination
    page: int                # Current page number (1, 2, 3...)
    page_size: int           # Default page size (e.g. 10 or 12)
    has_more: bool           # True if more records exist on next page
    
    # RAG Knowledge Pipeline
    doc_context: Optional[List[Dict[str, Any]]]
    
    # Synthesis & Audit
    response: Optional[str]
    error: Optional[str]
    retry_count: int
    max_retries: int
    user_role: str  # customer, waiter, chef, manager, admin
    token_usage: Optional[Dict[str, int]]  # {'prompt': int, 'completion': int, 'total': int}
    audit_log: Optional[Dict[str, Any]]
