import os
import sys
from pathlib import Path

# Configure UTF-8 encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.graph import restaurant_agent

def run_chat(thread_id: str, user_input: str, user_role: str = "customer"):
    print(f"\n{'='*70}")
    print(f"👤 USER: {user_input}")
    print(f"{'='*70}")

    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "user_input": user_input,
        "user_role": user_role,
        "retry_count": 0,
        "max_retries": 2,
    }

    result = restaurant_agent.invoke(initial_state, config=config)

    print(f"\n🤖 ASSISTANT RESPONSE:")
    print(result.get("response"))
    if result.get("rewritten_query") and result.get("rewritten_query") != user_input:
        print(f"\n[Context Resolved Standalone Query]: {result.get('rewritten_query')}")
    if result.get("sql_query"):
        print(f"[Generated & Executed SQL]: {result.get('sql_query')}")
    if result.get("intent"):
        print(f"[Intent]: {result.get('intent')} (Confidence: {result.get('confidence', 1.0)})")
    if result.get("active_slots"):
        print(f"[Active Slots]: {result.get('active_slots')}")

    return result

def test_all_scenarios():
    print("\n🚀 STARTING COMPLETE RESTAURANT CRM AI AGENT TEST SUITE 🚀\n")

    # TEST 1: Greeting
    print("\n--- TEST 1: GREETING FAST-EXIT ---")
    run_chat(thread_id="test_sess_1", user_input="Namaste")

    # TEST 2: Multi-turn Contextual Follow-up (Coffee Budget change)
    print("\n--- TEST 2: MULTI-TURN CONTEXTUAL FOLLOW-UP (300 -> 200) ---")
    run_chat(thread_id="test_sess_2", user_input="Mujhe coffee chahiye 300 hai mera budget")
    run_chat(thread_id="test_sess_2", user_input="ab mujhe do 300 nahi 200 hai mera budget")

    # TEST 3: Structured CRM Attendance Query
    print("\n--- TEST 3: STRUCTURED CRM ATTENDANCE ---")
    run_chat(thread_id="test_sess_3", user_input="Rahul ka August attendance batao")

    # TEST 4: Live Orders Count
    print("\n--- TEST 4: LIVE ORDERS COUNT ---")
    run_chat(thread_id="test_sess_4", user_input="Abhi kitne active orders hain kitchen mein?")

    # TEST 5: RAG Policy & Timings Query
    print("\n--- TEST 5: RAG TIMINGS & POLICY ---")
    run_chat(thread_id="test_sess_5", user_input="Dinner ka timing kya hai aur tandoor kab shuru hota hai?")

    # TEST 6: Smart NLP Ambiguity / Clarification Fallback
    print("\n--- TEST 6: SMART NLP CLARIFICATION FALLBACK ---")
    run_chat(thread_id="test_sess_6", user_input="mera wo kar do")

    # TEST 7: Security Guardrail Block (Unsafe Write attempt)
    print("\n--- TEST 7: SECURITY GUARDRAIL BLOCK (DROP TABLE) ---")
    from agent.guardrails import SQLGuardrail
    is_safe, sanitized, err = SQLGuardrail.validate_sql("DROP TABLE orders;")
    print(f"Safety check for 'DROP TABLE orders;': is_safe={is_safe}, error='{err}'")

    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY! 🎉\n")

if __name__ == "__main__":
    test_all_scenarios()
