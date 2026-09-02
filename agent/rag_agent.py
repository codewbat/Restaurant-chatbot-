import json
import re
from typing import List, Dict, Any
from database.menu_config import RestaurantCRMDatabase

crm_db = RestaurantCRMDatabase()

class RAGAgent:
    """Knowledge retriever for restaurant timings, meal schedules, and policy rules."""

    def __init__(self):
        self.documents = self._build_knowledge_base()

    def _build_knowledge_base(self) -> List[Dict[str, Any]]:
        """Constructs knowledge chunks from restaurant profile and operational guidelines."""
        rest_info = crm_db.execute_query("SELECT * FROM restaurant LIMIT 1")
        if not rest_info:
            return []

        row = rest_info[0]
        notes_list = [n.strip() for n in row["notes"].split("|") if n.strip()]

        docs = [
            {
                "section": "Meal Timings",
                "title": "Restaurant Operating & Meal Hours",
                "content": f"Breakfast: {row['breakfast_timing']}. Lunch: {row['lunch_timing']}. Snacks: {row['snacks_timing']}. Dinner: {row['dinner_timing']}.",
                "keywords": ["meal", "breakfast", "lunch", "dinner", "snacks", "timing", "open", "close", "hours"]
            },
            {
                "section": "Kitchen Hours & Breaks",
                "title": "Kitchen Afternoon Break & Night Closing",
                "content": f"The kitchen is closed in the afternoon from {row['kitchen_closed_afternoon']}, and closed for the night from {row['kitchen_closed_night']}. Orders cannot be prepared during kitchen closure.",
                "keywords": ["kitchen", "closed", "break", "afternoon", "night", "last order", "timing"]
            },
            {
                "section": "Buffet Policy",
                "title": "Buffet Breakfast Pricing & Rules",
                "content": "Buffet breakfast is available at Rs. 365/- per person (taxes extra as applicable). Only available items will be served.",
                "keywords": ["buffet", "breakfast", "rate", "price", "cost", "per person", "charge"]
            },
            {
                "section": "Tandoor & Barbeque",
                "title": "Tandoor Operating Schedule",
                "content": "Tandoor and Barbeque items are served exclusively from 19:00 to 22:30 (7:00 PM to 10:30 PM).",
                "keywords": ["tandoor", "tandoori", "barbeque", "tikka", "kabab", "evening", "timing"]
            },
            {
                "section": "Order Preparation Policy",
                "title": "Advance Order Requirement",
                "content": "Lunch and Dinner orders require a minimum of 30 minutes advance placement for preparation.",
                "keywords": ["advance", "preparation", "wait time", "30 minutes", "order time"]
            },
            {
                "section": "General Rules & Jurisdiction",
                "title": "Smoking, Allergies & Legal Notes",
                "content": "Smoking is legally prohibited on restaurant premises. Guests are requested to inform the captain in advance if allergic to any food ingredients. All disputes subject to Jaipur jurisdiction.",
                "keywords": ["smoking", "smoke", "allergy", "allergic", "ingredients", "jaipur", "dispute", "rules"]
            }
        ]

        return docs

    def retrieve(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Keyword & semantic overlap retriever for policy documentation."""
        q_words = set(re.findall(r"\w+", query.lower()))
        scored_docs = []

        for doc in self.documents:
            # Score based on keyword overlap and content match
            score = 0
            for kw in doc["keywords"]:
                if kw in q_words or kw in query.lower():
                    score += 2

            content_words = set(re.findall(r"\w+", doc["content"].lower()))
            overlap = len(q_words.intersection(content_words))
            score += overlap

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
