import os
import json
import sqlite3
from datetime import date
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MENU_JSON_PATH = BASE_DIR / "menu_database.json"
DB_PATH = BASE_DIR / "restaurant_crm.db"
CRM_JSON_PATH = BASE_DIR / "restaurant_crm.json"

def init_crm_database():
    print(f"Loading base menu data from {MENU_JSON_PATH}...")
    with open(MENU_JSON_PATH, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    # Remove existing DB file if it exists
    if DB_PATH.exists():
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    cursor = conn.cursor()

    print("Creating CRM database tables...")

    # 1. Restaurant table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS restaurant (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        location TEXT NOT NULL,
        currency TEXT DEFAULT 'INR',
        taxes TEXT,
        breakfast_timing TEXT,
        lunch_timing TEXT,
        snacks_timing TEXT,
        dinner_timing TEXT,
        kitchen_closed_afternoon TEXT,
        kitchen_closed_night TEXT,
        notes TEXT
    );
    """)

    # 2. Categories table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL
    );
    """)

    # 3. Menu items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS menu_items (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        price REAL NOT NULL,
        category_id INTEGER NOT NULL,
        is_veg INTEGER DEFAULT 1,
        is_jain INTEGER DEFAULT 0,
        spice_level TEXT DEFAULT 'mild',
        prep_time_mins INTEGER DEFAULT 15,
        FOREIGN KEY (category_id) REFERENCES categories(id)
    );
    """)

    # 4. Inventory table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inventory (
        item_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        available INTEGER NOT NULL,
        stock INTEGER NOT NULL,
        unit TEXT NOT NULL,
        reorder_level INTEGER NOT NULL,
        max_stock INTEGER NOT NULL,
        price REAL NOT NULL,
        category TEXT NOT NULL
    );
    """)

    # 5. Dining tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS dining_tables (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        table_number TEXT UNIQUE NOT NULL,
        capacity INTEGER NOT NULL,
        section TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('available', 'occupied', 'reserved', 'maintenance'))
    );
    """)

    # 6. Customers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        phone TEXT UNIQUE NOT NULL,
        email TEXT,
        loyalty_points INTEGER DEFAULT 0,
        vip_status TEXT DEFAULT 'regular' CHECK (vip_status IN ('regular', 'gold', 'platinum')),
        food_preference TEXT,
        total_visits INTEGER DEFAULT 1,
        last_visit_date TEXT
    );
    """)

    # 7. Employees table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        role TEXT NOT NULL,
        phone TEXT NOT NULL,
        salary REAL NOT NULL,
        shift TEXT NOT NULL CHECK (shift IN ('morning', 'evening', 'night', 'full_day')),
        hire_date TEXT NOT NULL
    );
    """)

    # 8. Attendance table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        employee_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        check_in TEXT,
        check_out TEXT,
        status TEXT NOT NULL CHECK (status IN ('present', 'absent', 'half_day', 'leave', 'week_off')),
        FOREIGN KEY (employee_id) REFERENCES employees(id),
        UNIQUE(employee_id, date)
    );
    """)

    # 9. Reservations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reservations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        table_id INTEGER NOT NULL,
        guest_count INTEGER NOT NULL,
        reservation_date TEXT NOT NULL,
        reservation_time TEXT NOT NULL,
        special_request TEXT,
        status TEXT NOT NULL CHECK (status IN ('confirmed', 'seated', 'cancelled', 'completed')),
        FOREIGN KEY (customer_id) REFERENCES customers(id),
        FOREIGN KEY (table_id) REFERENCES dining_tables(id)
    );
    """)

    # 10. Orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_number TEXT UNIQUE NOT NULL,
        table_id INTEGER,
        customer_id INTEGER,
        order_type TEXT NOT NULL CHECK (order_type IN ('dine-in', 'takeaway', 'room-service')),
        total_amount REAL NOT NULL,
        discount_amount REAL DEFAULT 0.0,
        tax_amount REAL NOT NULL,
        net_amount REAL NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('pending', 'cooking', 'served', 'completed', 'cancelled')),
        payment_mode TEXT CHECK (payment_mode IN ('cash', 'card', 'upi', 'unpaid')),
        created_at TEXT NOT NULL,
        FOREIGN KEY (table_id) REFERENCES dining_tables(id),
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """)

    # 11. Order items table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER NOT NULL,
        menu_item_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        total_price REAL NOT NULL,
        notes TEXT,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (menu_item_id) REFERENCES menu_items(id)
    );
    """)

    # 12. Feedback table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id INTEGER,
        customer_id INTEGER,
        food_rating INTEGER CHECK (food_rating BETWEEN 1 AND 5),
        service_rating INTEGER CHECK (service_rating BETWEEN 1 AND 5),
        ambiance_rating INTEGER CHECK (ambiance_rating BETWEEN 1 AND 5),
        comments TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(id),
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    );
    """)

    print("Populating restaurant, categories, menu items, and inventory...")

    # Insert restaurant info
    rest = base_data.get("restaurant", {})
    meal_t = rest.get("meal_timings", {})
    kitch_t = rest.get("kitchen_timings", {})
    notes_str = " | ".join(rest.get("notes", []))
    cursor.execute("""
    INSERT INTO restaurant (
        name, location, currency, taxes,
        breakfast_timing, lunch_timing, snacks_timing, dinner_timing,
        kitchen_closed_afternoon, kitchen_closed_night, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rest.get("name", "UMAID HAVELI"),
        rest.get("location", "Jaipur"),
        rest.get("currency", "INR"),
        rest.get("taxes", "Extra as applicable"),
        meal_t.get("breakfast", "07:00 to 10:30"),
        meal_t.get("lunch", "12:30 to 16:00"),
        meal_t.get("snacks", "17:30 to 18:30"),
        meal_t.get("dinner", "18:30 to 22:00"),
        kitch_t.get("closed_from", "16:00 to 17:30"),
        kitch_t.get("closed_from_night", "22:30 to 07:00"),
        notes_str
    ))

    # Insert categories
    categories = base_data.get("categories", [])
    for cat in categories:
        cursor.execute("INSERT INTO categories (id, name) VALUES (?, ?)", (cat["id"], cat["name"]))

    # Insert menu_items with enhanced tags
    non_veg_cat_ids = {6, 9, 15, 16, 21}  # Non-veg categories
    non_veg_keywords = ["chicken", "mutton", "fish", "prawn", "egg", "non veg", "murg", "sula"]

    menu_items = base_data.get("menu_items", [])
    for item in menu_items:
        name_lower = item["name"].lower()
        is_nv = (item["category_id"] in non_veg_cat_ids) or any(k in name_lower for k in non_veg_keywords)
        is_veg = 0 if is_nv else 1
        
        # Jain friendly check
        is_jain = 1 if is_veg and not any(k in name_lower for k in ["onion", "garlic", "potato", "aloo", "ginger", "lahsun"]) else 0
        
        # Spice level
        if any(k in name_lower for k in ["tikka", "chilli", "kadai", "kadhai", "masala", "curry", "65", "hydrabadi"]):
            spice_level = "spicy"
        elif any(k in name_lower for k in ["plain", "milk", "tea", "coffee", "juice", "ice cream", "sweet", "dessert", "kheer", "halwa"]):
            spice_level = "mild"
        else:
            spice_level = "medium"

        # Prep time mins
        if item["category_id"] in [1, 2]:
            prep_time = 7
        elif item["category_id"] in [3, 4]:
            prep_time = 12
        elif item["category_id"] in [20, 21, 15, 16]:
            prep_time = 25
        else:
            prep_time = 18

        cursor.execute("""
        INSERT INTO menu_items (id, name, price, category_id, is_veg, is_jain, spice_level, prep_time_mins)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (item["id"], item["name"], item["price"], item["category_id"], is_veg, is_jain, spice_level, prep_time))

    # Insert inventory
    inventory = base_data.get("inventory", [])
    for inv in inventory:
        cursor.execute("""
        INSERT INTO inventory (item_id, name, available, stock, unit, reorder_level, max_stock, price, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inv["item_id"], inv["name"], 1 if inv["available"] else 0,
            inv["stock"], inv["unit"], inv["reorder_level"], inv["max_stock"],
            inv["price"], inv["category"]
        ))

    print("Seeding dining tables...")
    tables_data = [
        ("T-01", 2, "AC Hall", "occupied"),
        ("T-02", 4, "AC Hall", "available"),
        ("T-03", 4, "AC Hall", "available"),
        ("T-04", 6, "AC Hall", "occupied"),
        ("T-05", 2, "Rooftop", "available"),
        ("T-06", 4, "Rooftop", "occupied"),
        ("T-07", 6, "Rooftop", "reserved"),
        ("T-08", 8, "Rooftop", "available"),
        ("T-09", 4, "Garden", "available"),
        ("T-10", 4, "Garden", "occupied"),
        ("T-11", 6, "Garden", "available"),
        ("T-12", 8, "Private Dining", "reserved"),
        ("T-13", 12, "Private Dining", "available"),
        ("T-14", 4, "Poolside", "available"),
        ("T-15", 4, "Poolside", "maintenance"),
    ]
    cursor.executemany("INSERT INTO dining_tables (table_number, capacity, section, status) VALUES (?, ?, ?, ?)", tables_data)

    print("Seeding customers...")
    customers_data = [
        ("Rahul Verma", "+919876543210", "rahul.verma@example.com", 320, "gold", "Non-Veg", 8, "2026-08-28"),
        ("Priya Sharma", "+919876543211", "priya.sharma@example.com", 750, "platinum", "Pure Veg", 15, "2026-09-01"),
        ("Amit Patel", "+919876543212", "amit.patel@example.com", 120, "regular", "Jain", 3, "2026-08-20"),
        ("Sneha Kulkarni", "+919876543213", "sneha.k@example.com", 450, "gold", "Veg", 9, "2026-08-31"),
        ("Vikram Singh", "+919876543214", "vikram.singh@example.com", 890, "platinum", "Non-Veg", 22, "2026-09-02"),
        ("Ananya Roy", "+919876543215", "ananya.roy@example.com", 90, "regular", "Non-Veg", 2, "2026-07-15"),
        ("Rohan Gupta", "+919876543216", "rohan.g@example.com", 210, "regular", "Pure Veg", 5, "2026-08-25"),
        ("Meera Joshi", "+919876543217", "meera.j@example.com", 580, "gold", "Jain", 11, "2026-08-29"),
        ("Rajesh Agarwal", "+919876543218", "rajesh.a@example.com", 1150, "platinum", "Pure Veg", 28, "2026-09-01"),
        ("Neha Mehta", "+919876543219", "neha.mehta@example.com", 140, "regular", "Veg", 4, "2026-08-14"),
        ("Siddharth Jain", "+919876543220", "sid.jain@example.com", 380, "gold", "Jain", 7, "2026-08-27"),
        ("Pooja Choudhary", "+919876543221", "pooja.c@example.com", 60, "regular", "Non-Veg", 1, "2026-06-10"),
        ("Karan Kapoor", "+919876543222", "karan.k@example.com", 620, "gold", "Non-Veg", 14, "2026-08-30"),
        ("Sunita Yadav", "+919876543223", "sunita.y@example.com", 400, "gold", "Pure Veg", 8, "2026-08-22"),
        ("Manish Tiwari", "+919876543224", "manish.t@example.com", 180, "regular", "Non-Veg", 4, "2026-08-18"),
        ("Deepak Mittal", "+919876543225", "deepak.m@example.com", 820, "platinum", "Pure Veg", 19, "2026-09-02"),
        ("Shweta Sen", "+919876543226", "shweta.sen@example.com", 95, "regular", "Veg", 2, "2026-07-28"),
        ("Tarun Rathore", "+919876543227", "tarun.r@example.com", 340, "gold", "Non-Veg", 6, "2026-08-26"),
        ("Kavita Rao", "+919876543228", "kavita.rao@example.com", 110, "regular", "Pure Veg", 3, "2026-08-09"),
        ("Gaurav Chauhan", "+919876543229", "gaurav.c@example.com", 510, "gold", "Non-Veg", 10, "2026-08-31"),
        ("Nisha Bansal", "+919876543230", "nisha.b@example.com", 700, "platinum", "Jain", 16, "2026-09-01"),
        ("Arjun Nair", "+919876543231", "arjun.nair@example.com", 130, "regular", "Non-Veg", 3, "2026-08-11"),
        ("Divya Solanki", "+919876543232", "divya.s@example.com", 290, "regular", "Pure Veg", 6, "2026-08-23"),
        ("Harshvardhan Goel", "+919876543233", "harsh.goel@example.com", 980, "platinum", "Pure Veg", 24, "2026-08-30"),
        ("Preeti Saxena", "+919876543234", "preeti.s@example.com", 160, "regular", "Veg", 3, "2026-08-17"),
    ]
    cursor.executemany("""
    INSERT INTO customers (name, phone, email, loyalty_points, vip_status, food_preference, total_visits, last_visit_date)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, customers_data)

    print("Seeding employees & staff...")
    # Include Rahul specifically as employee 1
    employees_data = [
        ("Rahul Sharma", "Senior Captain", "+919811100001", 38000.0, "evening", "2023-04-15"),
        ("Suresh Meena", "Head Chef", "+919811100002", 55000.0, "morning", "2022-01-10"),
        ("Mohan Lal", "Sous Chef (Tandoor)", "+919811100003", 35000.0, "evening", "2023-08-01"),
        ("Ramesh Kumar", "Waiter", "+919811100004", 22000.0, "morning", "2024-02-15"),
        ("Vikash Mehra", "Waiter", "+919811100005", 22000.0, "evening", "2024-03-01"),
        ("Dinesh Gurjar", "Waiter", "+919811100006", 21000.0, "evening", "2024-05-10"),
        ("Sunil Jangid", "Bartender / Beverage Lead", "+919811100007", 32000.0, "evening", "2023-11-20"),
        ("Kavita Pareek", "Hostess & Reservation Desk", "+919811100008", 26000.0, "morning", "2024-01-05"),
        ("Ajay Shekhawat", "General Manager", "+919811100009", 75000.0, "full_day", "2021-06-01"),
        ("Mahesh Saini", "Kitchen Helper / Dishwasher", "+919811100010", 18000.0, "morning", "2024-04-12"),
        ("Kamal Rathore", "Cashier & Billing Officer", "+919811100011", 28000.0, "full_day", "2023-09-15"),
        ("Pankaj Verma", "Store & Inventory Manager", "+919811100012", 34000.0, "morning", "2023-03-22")
    ]
    cursor.executemany("""
    INSERT INTO employees (name, role, phone, salary, shift, hire_date)
    VALUES (?, ?, ?, ?, ?, ?)
    """, employees_data)

    print("Seeding August 2026 attendance for all employees (including Rahul)...")
    # August 2026 has 31 days
    for emp_id in range(1, len(employees_data) + 1):
        for day in range(1, 32):
            dt_str = f"2026-08-{day:02d}"
            curr_date = date(2026, 8, day)
            weekday = curr_date.weekday() # 6 is Sunday

            if emp_id == 1: # Rahul Sharma
                if weekday == 6: # Sunday week-off
                    status = "week_off"
                    cin, cout = None, None
                elif day in [12, 23]:
                    status = "leave"
                    cin, cout = None, None
                elif day in [7, 19]:
                    status = "half_day"
                    cin, cout = "17:30", "20:30"
                else:
                    status = "present"
                    cin, cout = "17:15", "23:45"
            else:
                if weekday == 6:
                    status = "week_off"
                    cin, cout = None, None
                elif (day + emp_id) % 15 == 0:
                    status = "leave"
                    cin, cout = None, None
                elif (day + emp_id) % 9 == 0:
                    status = "half_day"
                    cin, cout = "10:30", "15:00"
                else:
                    status = "present"
                    cin, cout = "10:00", "22:00"

            cursor.execute("""
            INSERT INTO attendance (employee_id, date, check_in, check_out, status)
            VALUES (?, ?, ?, ?, ?)
            """, (emp_id, dt_str, cin, cout, status))

    print("Seeding reservations...")
    reservations_data = [
        (1, 7, 6, "2026-09-02", "19:30", "Rooftop candlelight seating with anniversary flowers", "confirmed"),
        (2, 12, 8, "2026-09-02", "20:00", "VIP family dinner, pure veg food priority", "confirmed"),
        (5, 6, 4, "2026-09-02", "20:30", "Window side table", "seated"),
        (9, 8, 8, "2026-09-03", "13:00", "Business lunch meeting with client", "confirmed"),
        (8, 5, 2, "2026-09-03", "19:00", "Quiet corner table", "confirmed"),
        (11, 13, 10, "2026-09-04", "20:00", "Birthday celebration with cake table", "confirmed"),
        (16, 2, 4, "2026-09-01", "20:00", "Dinner with guests", "completed"),
        (21, 9, 4, "2026-08-30", "19:30", "Garden seating", "completed"),
        (4, 1, 2, "2026-08-28", "21:00", "Late dinner", "cancelled"),
        (13, 4, 6, "2026-08-25", "13:30", "Family lunch", "completed")
    ]
    cursor.executemany("""
    INSERT INTO reservations (customer_id, table_id, guest_count, reservation_date, reservation_time, special_request, status)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, reservations_data)

    print("Seeding orders and order_items...")
    # Create 30 realistic orders
    order_samples = [
        ("ORD-20260902-001", 1, 1, "dine-in", "cooking", "unpaid", "2026-09-02 11:15:00", [(6, 2), (32, 1), (47, 1)]),
        ("ORD-20260902-002", 4, 5, "dine-in", "pending", "unpaid", "2026-09-02 11:20:00", [(12, 1), (40, 2)]),
        ("ORD-20260902-003", 6, 8, "dine-in", "served", "unpaid", "2026-09-02 10:45:00", [(4, 2), (85, 1)]),
        ("ORD-20260902-004", 10, 15, "dine-in", "cooking", "unpaid", "2026-09-02 11:18:00", [(19, 2), (46, 1)]),
        ("ORD-20260902-005", None, 3, "takeaway", "pending", "unpaid", "2026-09-02 11:25:00", [(221, 2), (227, 2)]),
        ("ORD-20260901-001", 2, 2, "dine-in", "completed", "upi", "2026-09-01 13:10:00", [(227, 2), (222, 1), (6, 2)]),
        ("ORD-20260901-002", 7, 9, "dine-in", "completed", "card", "2026-09-01 13:45:00", [(228, 4), (275, 2), (15, 4)]),
        ("ORD-20260901-003", 3, 11, "dine-in", "completed", "cash", "2026-09-01 14:15:00", [(220, 2), (221, 1), (273, 2)]),
        ("ORD-20260901-004", 8, 16, "dine-in", "completed", "card", "2026-09-01 20:00:00", [(250, 2), (237, 1), (221, 2)]),
        ("ORD-20260901-005", 5, 14, "dine-in", "completed", "upi", "2026-09-01 20:30:00", [(244, 2), (222, 1), (270, 2)]),
        ("ORD-20260831-001", 1, 4, "dine-in", "completed", "card", "2026-08-31 19:40:00", [(235, 1), (251, 1), (12, 2)]),
        ("ORD-20260831-002", 6, 20, "dine-in", "completed", "upi", "2026-08-31 20:15:00", [(253, 2), (221, 1), (19, 2)]),
        ("ORD-20260830-001", 9, 21, "dine-in", "completed", "card", "2026-08-30 13:20:00", [(227, 3), (274, 3)]),
        ("ORD-20260830-002", 12, 24, "dine-in", "completed", "card", "2026-08-30 20:10:00", [(228, 6), (244, 3), (275, 4)]),
        ("ORD-20260829-001", 2, 8, "dine-in", "completed", "upi", "2026-08-29 19:30:00", [(220, 2), (244, 1), (268, 2)]),
        ("ORD-20260828-001", 1, 1, "dine-in", "completed", "card", "2026-08-28 20:50:00", [(255, 1), (250, 1), (19, 2)]),
        ("ORD-20260827-001", 5, 11, "dine-in", "completed", "cash", "2026-08-27 13:00:00", [(221, 2), (273, 1)]),
        ("ORD-20260826-001", 3, 18, "dine-in", "completed", "upi", "2026-08-26 19:45:00", [(236, 2), (253, 1)]),
        ("ORD-20260825-001", 7, 7, "dine-in", "completed", "card", "2026-08-25 14:10:00", [(227, 2), (275, 2)]),
        ("ORD-20260824-001", 4, 13, "dine-in", "completed", "card", "2026-08-24 20:15:00", [(250, 1), (237, 1), (221, 1)]),
        ("ORD-20260823-001", 11, 23, "dine-in", "completed", "upi", "2026-08-23 13:30:00", [(227, 4), (244, 2)]),
        ("ORD-20260822-001", 8, 14, "dine-in", "completed", "card", "2026-08-22 19:50:00", [(228, 3), (270, 2)]),
        ("ORD-20260821-001", 2, 5, "dine-in", "completed", "card", "2026-08-21 20:30:00", [(258, 2), (221, 1)]),
        ("ORD-20260820-001", 6, 3, "dine-in", "completed", "cash", "2026-08-20 13:15:00", [(221, 2), (274, 1)]),
        ("ORD-20260819-001", 10, 15, "dine-in", "completed", "upi", "2026-08-19 20:00:00", [(255, 1), (250, 1)]),
        ("ORD-20260818-001", 1, 15, "dine-in", "completed", "card", "2026-08-18 19:40:00", [(250, 1), (235, 1)]),
        ("ORD-20260817-001", 3, 25, "dine-in", "completed", "upi", "2026-08-17 13:00:00", [(227, 2), (12, 2)]),
        ("ORD-20260816-001", 4, 2, "dine-in", "completed", "card", "2026-08-16 20:10:00", [(228, 3), (244, 1)]),
        ("ORD-20260815-001", 7, 9, "dine-in", "completed", "card", "2026-08-15 13:30:00", [(228, 5), (275, 4)]),
        ("ORD-20260814-001", 5, 10, "dine-in", "completed", "cash", "2026-08-14 19:50:00", [(222, 2), (270, 1)]),
    ]

    menu_price_map = {item["id"]: item["price"] for item in menu_items}

    for ord_info in order_samples:
        order_num, tbl_id, cust_id, otype, ostatus, paymode, created_at, items = ord_info
        
        subtotal = 0.0
        calculated_items = []
        for mid, qty in items:
            uprice = menu_price_map.get(mid, 200.0)
            tprice = uprice * qty
            subtotal += tprice
            calculated_items.append((mid, qty, uprice, tprice))
        
        discount = round(subtotal * 0.10, 2) if (cust_id and cust_id % 3 == 0) else 0.0
        tax = round((subtotal - discount) * 0.05, 2) # 5% GST
        net_amount = round((subtotal - discount) + tax, 2)

        cursor.execute("""
        INSERT INTO orders (order_number, table_id, customer_id, order_type, total_amount, discount_amount, tax_amount, net_amount, status, payment_mode, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (order_num, tbl_id, cust_id, otype, subtotal, discount, tax, net_amount, ostatus, paymode, created_at))
        
        order_id = cursor.lastrowid
        for mid, qty, uprice, tprice in calculated_items:
            cursor.execute("""
            INSERT INTO order_items (order_id, menu_item_id, quantity, unit_price, total_price, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (order_id, mid, qty, uprice, tprice, "Standard preparation"))

    print("Seeding customer feedback...")
    feedbacks_data = [
        (6, 2, 5, 5, 5, "Amazing Dal Makhani and Rajasthan Thali! Loved the authentic royal heritage feel.", "2026-09-01 14:30:00"),
        (7, 9, 5, 4, 5, "Excellent service by Rahul. Food was fresh and served hot.", "2026-09-01 15:00:00"),
        (8, 11, 4, 4, 4, "Great Jain options available. Very cooperative staff.", "2026-09-01 15:15:00"),
        (9, 16, 5, 5, 5, "Best Butter Chicken in Jaipur! Tandoori roti was crisp.", "2026-09-01 21:30:00"),
        (10, 14, 4, 4, 4, "Nice rooftop atmosphere with pleasant live instrumental music.", "2026-09-01 22:00:00"),
        (11, 4, 5, 5, 4, "Chicken Seekh kabab was delicious.", "2026-08-31 21:00:00"),
        (12, 20, 4, 3, 4, "Food was good but order took slightly longer than 30 mins.", "2026-08-31 21:30:00"),
        (13, 21, 5, 5, 5, "Rajasthan Thali is a must try! Very fulfilling.", "2026-08-30 14:45:00"),
        (14, 24, 5, 4, 5, "Celebrated birthday in Private Dining. Wonderful arrangement.", "2026-08-30 22:00:00"),
        (15, 8, 4, 5, 4, "Paneer tikka was extremely tender and fresh.", "2026-08-29 21:00:00")
    ]
    cursor.executemany("""
    INSERT INTO feedback (order_id, customer_id, food_rating, service_rating, ambiance_rating, comments, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, feedbacks_data)

    conn.commit()
    print("Database committed successfully.")

    # Export full JSON dump
    print("Exporting combined data to JSON format...")
    crm_json_export = {
        "restaurant": rest,
        "categories": categories,
        "menu_items": [
            dict(zip(["id", "name", "price", "category_id", "is_veg", "is_jain", "spice_level", "prep_time_mins"], row))
            for row in cursor.execute("SELECT id, name, price, category_id, is_veg, is_jain, spice_level, prep_time_mins FROM menu_items").fetchall()
        ],
        "inventory": inventory,
        "dining_tables": [
            dict(zip(["id", "table_number", "capacity", "section", "status"], row))
            for row in cursor.execute("SELECT id, table_number, capacity, section, status FROM dining_tables").fetchall()
        ],
        "customers": [
            dict(zip(["id", "name", "phone", "email", "loyalty_points", "vip_status", "food_preference", "total_visits", "last_visit_date"], row))
            for row in cursor.execute("SELECT id, name, phone, email, loyalty_points, vip_status, food_preference, total_visits, last_visit_date FROM customers").fetchall()
        ],
        "employees": [
            dict(zip(["id", "name", "role", "phone", "salary", "shift", "hire_date"], row))
            for row in cursor.execute("SELECT id, name, role, phone, salary, shift, hire_date FROM employees").fetchall()
        ],
        "reservations": [
            dict(zip(["id", "customer_id", "table_id", "guest_count", "reservation_date", "reservation_time", "special_request", "status"], row))
            for row in cursor.execute("SELECT id, customer_id, table_id, guest_count, reservation_date, reservation_time, special_request, status FROM reservations").fetchall()
        ],
        "orders": [
            dict(zip(["id", "order_number", "table_id", "customer_id", "order_type", "total_amount", "discount_amount", "tax_amount", "net_amount", "status", "payment_mode", "created_at"], row))
            for row in cursor.execute("SELECT id, order_number, table_id, customer_id, order_type, total_amount, discount_amount, tax_amount, net_amount, status, payment_mode, created_at FROM orders").fetchall()
        ],
        "order_items": [
            dict(zip(["id", "order_id", "menu_item_id", "quantity", "unit_price", "total_price", "notes"], row))
            for row in cursor.execute("SELECT id, order_id, menu_item_id, quantity, unit_price, total_price, notes FROM order_items").fetchall()
        ],
        "feedback": [
            dict(zip(["id", "order_id", "customer_id", "food_rating", "service_rating", "ambiance_rating", "comments", "created_at"], row))
            for row in cursor.execute("SELECT id, order_id, customer_id, food_rating, service_rating, ambiance_rating, comments, created_at FROM feedback").fetchall()
        ],
        "orders_summary": {
            "total_orders": cursor.execute("SELECT COUNT(*) FROM orders").fetchone()[0],
            "active_orders": cursor.execute("SELECT COUNT(*) FROM orders WHERE status IN ('pending', 'cooking', 'served')").fetchone()[0],
            "total_revenue": cursor.execute("SELECT SUM(net_amount) FROM orders WHERE status = 'completed'").fetchone()[0]
        }
    }

    with open(CRM_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(crm_json_export, f, indent=2)

    conn.close()
    print(f"CRM SQLite database created at: {DB_PATH}")
    print(f"CRM JSON exported to: {CRM_JSON_PATH}")

if __name__ == "__main__":
    init_crm_database()
