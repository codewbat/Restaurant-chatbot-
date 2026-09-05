import os
import sys
import re
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

from agent.restaurant_agent_service import RestaurantAgentService


def start_terminal_chat():
    service = RestaurantAgentService()

    print("=" * 72)
    print("🍽️  UMAID HAVELI RESTAURANT CRM AI AGENT - PRODUCTION CLI INTERFACE  🍽️")
    print("=" * 72)
    print("• Intelligent Dining Lifecycle, Menu Assistant, Billing & Staff Support")
    print("• Persistent Sessions: State & chat history are saved by Mobile Number.")
    print("=" * 72)

    # Prompt for Mobile Number Login
    phone_input = input("📱 Enter Mobile Number to Login (Default: 9660888489): ").strip()
    phone = phone_input if phone_input else "9660888489"

    print(f"\n🔄 Logging in & restoring session for phone: {phone}...")
    session_data = service.login_customer(phone)
    customer = session_data["customer"]
    active_table = session_data.get("active_table")
    active_orders = session_data.get("active_orders", [])
    chat_history = session_data.get("chat_history", [])

    print("-" * 72)
    print(f"👤 Customer Profile: {customer.name} | {customer.badge}")
    print(f"📞 Phone: {customer.phone} | 🏆 Loyalty Points: {customer.loyalty_points}")

    if active_table:
        print(f"🪑 Current Assigned Table: {active_table}")
    if active_orders:
        kot_list = ", ".join([f"#{o.order_number} ({o.status.upper()})" for o in active_orders])
        print(f"👨‍🍳 Active Kitchen Orders (KOT): {kot_list}")

    if chat_history:
        print("\n📜 --- Recent Chat History ---")
        for msg in chat_history[-4:]:
            sender = "👤 You" if msg["role"] == "user" else "🤖 AI"
            preview = msg["content"].split("\n")[0]
            print(f"  {sender}: {preview}")
        print("-------------------------------")

    print("\n💡 Type your message below (e.g. 'Table T-03 book karo', '2 Dal Makhani', 'Bill chahiye')")
    print("• Type 'exit', 'quit', or 'q' to end the session.")
    print("=" * 72)

    last_tokens = {"prompt": 0, "completion": 0, "total": 0}

    while True:
        try:
            user_input = input(f"\n👤 [{customer.name}]: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q", "bye"]:
                print(f"\n🙏 Dhanyawad {customer.name} ji! Aapka session save kar liya gaya hai. Have a great day!")
                break

            print("⏳ Processing...", end="\r", flush=True)
            result = service.chat(user_input=user_input, phone=phone, user_role="manager")

            # Clear processing line
            print(" " * 30, end="\r")

            print("\n🤖 AI Assistant:")
            print(result["response"])

            # Token usage & SQL telemetry
            if result.get("sql_query"):
                print(f"\n[Executed SQL]: {result['sql_query']}")
            if result.get("token_usage"):
                tu = result.get("token_usage")
                delta_p = tu.get("prompt", 0) - last_tokens.get("prompt", 0)
                delta_c = tu.get("completion", 0) - last_tokens.get("completion", 0)
                delta_t = tu.get("total", 0) - last_tokens.get("total", 0)
                last_tokens = dict(tu)
                print(f"[Tokens]: Prompt: {delta_p} | Completion: {delta_c} | Total: {delta_t}")

        except KeyboardInterrupt:
            print(f"\n\nSession saved for {customer.name}. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    start_terminal_chat()
