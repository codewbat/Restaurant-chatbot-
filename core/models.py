"""
OOP Domain Models for Restaurant CRM System
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import urllib.parse
from datetime import datetime


@dataclass
class Customer:
    id: Optional[int] = None
    name: str = "Guest"
    phone: str = ""
    email: Optional[str] = None
    loyalty_points: int = 0
    vip_status: str = "regular"  # regular, gold, platinum
    food_preference: Optional[str] = None
    total_visits: int = 1
    last_visit_date: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "phone": self.phone,
            "email": self.email,
            "loyalty_points": self.loyalty_points,
            "vip_status": self.vip_status,
            "food_preference": self.food_preference,
            "total_visits": self.total_visits,
            "last_visit_date": self.last_visit_date,
        }

    @property
    def badge(self) -> str:
        if self.vip_status == "platinum":
            return "💎 Platinum VIP"
        elif self.vip_status == "gold":
            return "⭐ Gold Member"
        return "👤 Regular Customer"


@dataclass
class DiningTable:
    id: Optional[int] = None
    table_number: str = ""
    capacity: int = 4
    section: str = "main_hall"
    status: str = "available"  # available, occupied, reserved, maintenance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "table_number": self.table_number,
            "capacity": self.capacity,
            "section": self.section,
            "status": self.status,
        }


@dataclass
class OrderItem:
    menu_item_id: int
    name: str
    quantity: int
    unit_price: float
    total_price: float
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "menu_item_id": self.menu_item_id,
            "name": self.name,
            "quantity": self.quantity,
            "unit_price": self.unit_price,
            "total_price": self.total_price,
            "notes": self.notes,
        }


@dataclass
class Order:
    id: Optional[int] = None
    order_number: str = ""
    table_id: Optional[int] = None
    table_number: Optional[str] = None
    customer_id: Optional[int] = None
    order_type: str = "dine-in"
    total_amount: float = 0.0
    discount_amount: float = 0.0
    tax_amount: float = 0.0
    net_amount: float = 0.0
    status: str = "cooking"  # pending, cooking, served, completed, cancelled
    payment_mode: str = "unpaid"  # unpaid, cash, card, upi
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    items: List[OrderItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "order_number": self.order_number,
            "table_id": self.table_id,
            "table_number": self.table_number,
            "customer_id": self.customer_id,
            "order_type": self.order_type,
            "total_amount": self.total_amount,
            "discount_amount": self.discount_amount,
            "tax_amount": self.tax_amount,
            "net_amount": self.net_amount,
            "status": self.status,
            "payment_mode": self.payment_mode,
            "created_at": self.created_at,
            "items": [it.to_dict() for it in self.items],
        }


@dataclass
class Bill:
    table_number: str
    orders: List[Order] = field(default_factory=list)
    items: List[OrderItem] = field(default_factory=list)
    subtotal: float = 0.0
    tax_rate: float = 0.05
    tax_amount: float = 0.0
    net_total: float = 0.0
    upi_id: str = "9660888489@axl"
    payee_name: str = "Umaid Haveli"

    @property
    def upi_link(self) -> str:
        return f"upi://pay?pa={self.upi_id}&pn={urllib.parse.quote(self.payee_name)}&am={self.net_total:.2f}&cu=INR&tn={urllib.parse.quote(f'Table {self.table_number} Bill')}"

    @property
    def upi_qr_url(self) -> str:
        encoded_link = urllib.parse.quote(self.upi_link)
        return f"https://api.qrserver.com/v1/create-qr-code/?size=240x240&data={encoded_link}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "table_number": self.table_number,
            "orders": [o.order_number for o in self.orders],
            "items": [it.to_dict() for it in self.items],
            "subtotal": self.subtotal,
            "tax_rate": self.tax_rate,
            "tax_amount": self.tax_amount,
            "net_total": self.net_total,
            "upi_id": self.upi_id,
            "upi_qr_url": self.upi_qr_url,
            "upi_link": self.upi_link,
        }
