import os
import sys
import re
import uuid
from pathlib import Path

# Configure UTF-8 encoding for Windows terminals (CMD and PowerShell)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stdin, "reconfigure"):
    sys.stdin.reconfigure(encoding="utf-8")

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent.graph import restaurant_agent

def start_terminal_chat():
    session_id = f"cli_session_{uuid.uuid4().hex[:6]}"
    config = {"configurable": {"thread_id": session_id}}

    print("=" * 70)
    print("🍽️  UMAID HAVELI RESTAURANT CRM AI AGENT - TERMINAL INTERFACE  🍽️")
    print("=" * 70)
    print("• You can ask about:")
    print("  - Menu & Prices (e.g. 'Coffee chahiye 200 budget hai', 'Veg specialities')")
    print("  - Staff Attendance (e.g. 'Rahul ka August attendance batao')")
    print("  - Live Kitchen Orders (e.g. 'Abhi kitne active orders hain?')")
    print("  - Timings & Policies (e.g. 'Dinner timing kya hai aur tandoor kab start hota hai?')")
    print("• Type 'exit', 'quit', or 'q' to end the session.")
    print("• Session ID:", session_id)
    print("=" * 70)
    last_tokens = {"prompt": 0, "completion": 0, "total": 0}

    while True:
        try:
            user_input = input("\n👤 Aap (User): ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q", "bye"]:
                print("\n🙏 Dhanyawad! Have a great day!")
                break

            initial_state = {
                "user_input": user_input,
                "user_role": "manager",  # default to manager for full read-access
                "retry_count": 0,
                "max_retries": 2,
            }

            print("⏳ Processing...", end="\r", flush=True)
            result = restaurant_agent.invoke(initial_state, config=config)

            # Clear processing line
            print(" " * 30, end="\r")

            print("\n🤖 AI Assistant:")
            resp_text = result.get("response", "No response generated.") or ""
            resp_text = re.sub(r"<think>.*?(?:</think>|$)", "", resp_text, flags=re.DOTALL).strip()
            resp_text = re.sub(r"(?i)here's a thinking process:.*$", "", resp_text, flags=re.DOTALL).strip()
            resp_text = re.sub(r"(?i)thinking process:.*$", "", resp_text, flags=re.DOTALL).strip()
            print(resp_text if resp_text else "Order / query details processed.")

            # Print debug info only if SQL was executed in THIS turn
            if result.get("intent") in ["sql", "pagination"]:
                if result.get("tables"):
                    print(f"[Dynamic Tables Selected]: {result.get('tables')}")
                if result.get("sql_query"):
                    print(f"[Executed SQL]: {result.get('sql_query')}")
            if result.get("token_usage"):
                tu = result.get("token_usage")
                delta_p = tu.get("prompt", 0) - last_tokens.get("prompt", 0)
                delta_c = tu.get("completion", 0) - last_tokens.get("completion", 0)
                delta_t = tu.get("total", 0) - last_tokens.get("total", 0)
                last_tokens = dict(tu)
                print(f"[This Query Tokens]: Prompt: {delta_p} | Completion: {delta_c} | Total: {delta_t}  (Session Total: {tu.get('total', 0)})")

        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    start_terminal_chat()
