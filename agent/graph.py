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
from agent.rewriter import analyze_and_rewrite
from agent.sql_agent import SQLAgent
from agent.rag_agent import RAGAgent
from agent.result_validator import ResultValidator, ResponseFormatter
from agent.prompts import SYNTHESIZER_PROMPT

load_dotenv()
api_key = os.getenv("groq_key") or os.getenv("GROQ_API_KEY")

str_parser = StrOutputParser()

llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key=api_key,
    temperature=0.2,
    max_tokens=1536,
    reasoning_format="hidden",
    max_retries=3,
)

rag_agent = RAGAgent()

# ----------------- NODE IMPLEMENTATIONS ----------------- #

def rewriter_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Sliding Window Contextual Rewriter + NLP Ambiguity Detector."""
    user_input = state["user_input"]
    chat_history = state.get("messages", [])
    active_slots = state.get("active_slots", {})

    analysis = analyze_and_rewrite(
        user_input=user_input,
        chat_history=chat_history,
        active_slots=active_slots
    )

    rw_tokens = analysis.get("tokens") or {}
    return {
        "rewritten_query": analysis.get("rewritten_query", user_input),
        "intent": analysis.get("intent", "sql"),
        "confidence": analysis.get("confidence", 1.0),
        "clarification_question": analysis.get("clarification_question"),
        "active_slots": analysis.get("updated_slots", active_slots),
        "response": None,
        "token_usage": {
            "prompt": rw_tokens.get("prompt_tokens", 0),
            "completion": rw_tokens.get("completion_tokens", 0),
            "total": rw_tokens.get("total_tokens", 0)
        }
    }

def greeting_node(state: AgentState) -> Dict[str, Any]:
    """Fast exit node for simple greetings."""
    reply = "Namaste! Umaid Haveli Restaurant CRM Assistant mein aapka swagat hai. Main menu, orders, table reservations, staff attendance, aur restaurant timings ke sawal answer kar sakta hoon. Aaj main aapki kya madad karoon?"
    return {
        "response": reply,
        "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
    }

def clarification_node(state: AgentState) -> Dict[str, Any]:
    """Fallback node when NLP is ambiguous or user intent is not clear."""
    question = state.get("clarification_question")
    if not question:
        question = (
            "Maaf kijiye, main aapka sawal poori tarah samajh nahi paaya. Kya aap:\n"
            "1. Restaurant ka menu ya prices dekhna chahte hain?\n"
            "2. Table ka active order ya bill check karna chahte hain?\n"
            "3. Staff attendance ya restaurant timings janna chahte hain?"
        )
    return {
        "response": question,
        "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=question)]
    }

def order_node(state: AgentState) -> Dict[str, Any]:
    """Handles food and drink ordering intent, checks availability, price, stock, and guides the guest."""
    raw_input = state["user_input"]
    clean_text = raw_input.lower()

    # Extract food keywords by removing common intent words
    stop_words = {"order", "karni", "karna", "hai", "karo", "merko", "mujhe", "chahiye", "mangwa", "batao", "please", "plate", "scoop", "ek", "do", "tin", "1", "2", "3", "lagao", "le", "aao", "ye", "yeh"}
    words = [w for w in re.findall(r"\w+", clean_text) if len(w) > 2 and w not in stop_words]

    matched_items = []
    if words:
        # 1. Try strict matching (all keywords present)
        and_clauses = " AND ".join([f"m.name LIKE '%{w}%'" for w in words])
        query = f"""
        SELECT m.name, m.price, c.name as category, i.stock, i.available
        FROM menu_items m
        LEFT JOIN categories c ON m.category_id = c.id
        LEFT JOIN inventory i ON m.name = i.name
        WHERE {and_clauses}
        LIMIT 3;
        """
        success, rows, err, _ = SQLAgent.execute_with_guardrail(query, user_role="manager")
        if success and rows:
            matched_items = rows
        else:
            # 2. Smart relevance scoring fallback (prioritize items matching the most keywords)
            valid_words = [w for w in words if len(w) >= 3]
            if valid_words:
                score_expr = " + ".join([f"CASE WHEN m.name LIKE '%{w}%' THEN 1 ELSE 0 END" for w in valid_words])
                or_expr = " OR ".join([f"m.name LIKE '%{w}%'" for w in valid_words])
                query_fallback = f"""
                SELECT m.name, m.price, c.name as category, i.stock, i.available,
                       ({score_expr}) as match_score
                FROM menu_items m
                LEFT JOIN categories c ON m.category_id = c.id
                LEFT JOIN inventory i ON m.name = i.name
                WHERE {or_expr}
                ORDER BY match_score DESC, m.price ASC
                LIMIT 3;
                """
                s2, r2, _, _ = SQLAgent.execute_with_guardrail(query_fallback, user_role="manager")
                if s2 and r2:
                    matched_items = r2

    if matched_items:
        item = matched_items[0]
        name = item.get("name")
        price = item.get("price")
        stock = item.get("stock")
        avail = item.get("available", 1)

        if stock is not None and (stock <= 0 or avail == 0):
            reply = f"Kshama karein, '{name}' abhi out of stock hai. Kya aap koi doosri dish ya dessert pasand karenge?"
        else:
            stock_info = f" (Stock: {stock} units available)" if stock is not None else ""
            reply = (
                f"Haan ji! '{name}' hamare paas available hai.\n"
                f"- Price: ₹{int(price) if isinstance(price, (int, float)) and price == int(price) else price}\n"
                f"- Category: {item.get('category', 'Menu')}{stock_info}\n\n"
                f"Aap ise Table par serve karwana chahte hain ya Takeaway? Kripya apna Table Number aur Quantity batayein taaki main order request kitchen aur captain ko inform kar sakoon!"
            )
    else:
        reply = (
            "Aapne jis item ke liye order request kiya hai, kripya uska poora naam batayein (jaise 'Ice Cream (Chocolate)' ya 'Butter Chicken'). "
            "Aap hamara menu dekh kar bhi chun sakte hain!"
        )

    return {
        "response": reply,
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

    # Step E: Validate Output
    res_status, val_msg = ResultValidator.validate(executed_sql, results, error_msg)

    # Clean base SQL for pagination (remove LIMIT / OFFSET)
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

    return {
        "tables": tables,
        "sql_query": executed_sql,
        "query_result": results,
        "result_status": res_status,
        "base_sql": base_sql,
        "page": 1,
        "page_size": page_size,
        "has_more": has_more,
        "doc_context": None,
        "token_usage": updated_tokens,
        "error": val_msg if res_status == "error" else None,
        "retry_count": retry_count
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

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """Synthesizes factual, polite response in Hindi/Hinglish grounded in DB rows and docs."""
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

    # If no data found
    if result_status == "empty" and not doc_context:
        q_lower = query.lower()
        if any(w in q_lower for w in ["employee", "waiter", "staff", "chef", "manager", "captain"]):
            reply = f"Maaf kijiye, '{query}' ke mutabik koi staff record nahi mila."
        elif any(w in q_lower for w in ["order", "bill", "table", "kitchen"]):
            reply = f"Maaf kijiye, '{query}' ke mutabik koi order ya table record nahi mila."
        else:
            reply = f"Maaf kijiye, '{query}' ke mutabik koi record nahi mila. Hamare menu mein Pizza, Sandwiches, Chinese, Paneer aur Beverages available hain. Kya aap inke options dekhna chahenge?"
        return {
            "response": reply,
            "messages": state.get("messages", []) + [HumanMessage(content=state["user_input"]), AIMessage(content=reply)]
        }

    # If pure database results (no RAG docs), format directly to save 2,500 LLM tokens!
    if results is not None and not doc_context:
        sanitized_rows = ResponseFormatter.sanitize_for_presentation(query, results[:15])
        if len(sanitized_rows) == 1 and len(sanitized_rows[0]) <= 2:
            items = [f"**{k}:** {v}" for k, v in sanitized_rows[0].items()]
            final_text = "\n".join(items)
        else:
            final_text = ResponseFormatter.format_markdown_table(sanitized_rows)
        # 0 tokens spent on synthesizer!
    else:
        # Prepare grounding prompt for LLM (only for RAG / Policy queries)
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

        if llm_resp and hasattr(llm_resp, "response_metadata"):
            tu = llm_resp.response_metadata.get("token_usage", {})
            current_tokens["prompt"] += tu.get("prompt_tokens", 0)
            current_tokens["completion"] += tu.get("completion_tokens", 0)
            current_tokens["total"] += tu.get("total_tokens", 0)

        # Fallback to direct clean table if synthesizer was empty or hijacked by thinking
        if not final_text and results:
            final_text = ResponseFormatter.format_markdown_table(sanitized_rows)

    # Append pagination navigation hint if more records exist
    current_page = state.get("page", 1)
    if state.get("has_more"):
        if current_page > 1:
            final_text += f"\n\n*(Page {current_page} • Agle ke liye 'next' ya pichhle ke liye 'previous' bole)*"
        else:
            final_text += f"\n\n*(Page 1 • Aage dekhne ke liye 'next item do' ya 'aur dikhao' bole)*"
    elif current_page > 1:
        final_text += f"\n\n*(Yeh aakhiri page hai - Page {current_page} • Pichhle ke liye 'previous' bole)*"

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
    syn_tok = getattr(llm_resp, "response_metadata", {}).get("token_usage", {}) or {}
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

def route_after_sql(state: AgentState) -> str:
    """Checks if SQL node needs self-correction retry or moves to synthesizer."""
    if state.get("result_status") == "error" and state.get("retry_count", 0) < state.get("max_retries", 2):
        return "retry_sql"
    return "synthesizer"

def route_after_pagination(state: AgentState) -> str:
    """If pagination node produced a direct response (e.g. end of list), exit; otherwise synthesize."""
    if state.get("query_result") is None:
        return END
    return "synthesizer"

# ----------------- GRAPH ASSEMBLY ----------------- #

def create_restaurant_agent_graph():
    """Builds and compiles the complete LangGraph conversational graph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("clarification", clarification_node)
    workflow.add_node("order", order_node)
    workflow.add_node("pagination", pagination_node)
    workflow.add_node("sql", sql_node)
    workflow.add_node("rag", rag_node)
    workflow.add_node("hybrid", hybrid_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # Entry point
    workflow.set_entry_point("rewriter")

    # Conditional edge from rewriter
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

    # Order node exits to END
    workflow.add_edge("order", END)

    # SQL node loop or advance
    workflow.add_conditional_edges(
        "sql",
        route_after_sql,
        {
            "retry_sql": "sql",
            "synthesizer": "synthesizer"
        }
    )

    # Pagination node route
    workflow.add_conditional_edges(
        "pagination",
        route_after_pagination,
        {
            "synthesizer": "synthesizer",
            END: END
        }
    )

    # Direct edges
    workflow.add_edge("greeting", END)
    workflow.add_edge("clarification", END)
    workflow.add_edge("rag", "synthesizer")
    workflow.add_edge("hybrid", "synthesizer")
    workflow.add_edge("synthesizer", END)

    # Checkpointer for session persistence
    memory = MemorySaver()
    compiled_graph = workflow.compile(checkpointer=memory)
    return compiled_graph

# Global compiled graph instance
restaurant_agent = create_restaurant_agent_graph()
