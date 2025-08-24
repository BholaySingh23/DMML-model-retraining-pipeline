# utils/auth.py
from fastapi import HTTPException
import os

API_KEY = os.environ.get('CHURN_API_KEY', 'demo-key')

def auth_check(key: str):
    print("key check", key)
    if key != API_KEY:
        raise HTTPException(status_code=401, detail='Unauthorized')