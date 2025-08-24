# services/customer_service.py
import sqlite3

class CustomerService:
    def __init__(self, db_path):
        self.db_path = db_path

    def get_customers(self, limit, offset):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM customers LIMIT ? OFFSET ?", (limit, offset))
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            items = [dict(zip(columns, row)) for row in rows]
            total = cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
            next_offset = offset + limit if offset + limit < total else None
            return {"items": items, "total": total, "offset": offset, "limit": limit, "next_offset": next_offset}
        finally:
            conn.close()