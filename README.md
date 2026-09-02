# 🍽️ Umaid Haveli Restaurant CRM AI Agent

A state-of-the-art Restaurant CRM & Guest Assistant AI agent built with **LangGraph**, **LangChain**, and **Groq Cloud**. It autonomously queries restaurant menus, live orders, kitchen statuses, table billing, inventory levels, staff rosters, attendance logs, and operational guidelines.

---

## 🌟 Key Features

- **Dynamic Text-to-SQL Engine:** Autonomous schema pruning and safe SQLite query generation for menus, inventory, active orders, and staff attendance.
- **Bi-Directional Pagination:** Interactive terminal pagination (`next` / `previous`) with zero redundant generation.
- **Order Intent Node:** Handles food orders with automatic typo tolerance and stock/price verification.
- **RAG Policy Knowledge:** Retrieves restaurant operational timings (buffet, tandoor), smoking guidelines, and dining policies.
- **Security Guardrails:** Read-only SQL validator preventing injection, table drops, and data corruption.
- **Response Presentation Schema:** Clean, executive markdown tables without database internal IDs.

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/codewbat/Restaurant-chatbot-.git
cd Restaurant-chatbot-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment variables
Create a `.env` file in the project root:
```env
groq_key=your_groq_api_key_here
```

### 4. Initialize the SQLite CRM Database
```bash
python database/init_db.py
```

### 5. Run the Interactive CLI
```bash
python agent/cli.py
```

---

## 🏗️ Architecture

```
reva-restaurant-agent/
├── agent/
│   ├── cli.py               # Interactive terminal interface
│   ├── graph.py             # LangGraph state machine & workflows
│   ├── guardrails.py        # SQL security guardrails
│   ├── prompts.py           # Few-shot prompts & guidelines
│   ├── rag_agent.py         # Static policy retrieval
│   ├── result_validator.py  # Presentation sanitization & table formatting
│   ├── rewriter.py          # Intent analysis & context rewriting
│   ├── sql_agent.py         # Text-to-SQL generation & schema pruning
│   ├── state.py             # AgentState definitions
│   └── test_agent.py        # Automated test suite
├── database/
│   ├── init_db.py           # Database seeder & schema initialization
│   ├── menu_config.py       # Menu categorization & seed definitions
│   ├── restaurant_crm.json  # Comprehensive CRM seed dataset
│   └── restaurant_crm.db    # SQLite relational database
├── .env.example
├── .gitignore
└── requirements.txt
```
