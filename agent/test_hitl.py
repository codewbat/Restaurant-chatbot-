import sys
import io

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
from agent.graph import restaurant_agent

config = {"configurable": {"thread_id": "table_status_test_session"}}

def chat(msg):
    print(f"\n==================================================")
    print(f"👤 USER: {msg}")
    print(f"==================================================")
    res = restaurant_agent.invoke({"user_input": msg}, config=config)
    print(f"🤖 ASSISTANT:\n{res.get('response')}\n")
    print(f"[Assigned Table]: {res.get('active_slots', {}).get('assigned_table')}")

print("=== 1. CHECK TABLE BEFORE BOOKING ===")
chat("merko konsi table assign hai")

print("\n=== 2. BOOK TABLE T-08 ===")
chat("T-08 book kardo 5 person ke liye")
chat("yes")

print("\n=== 3. CHECK TABLE AFTER BOOKING ===")
chat("merko konsi table assign hai")
