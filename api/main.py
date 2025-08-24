
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Query, Depends
from api.kaggle_api import router
from pydantic import BaseModel
from services.customer_services import CustomerService
from utils.auth import auth_check

ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCES = os.path.join(ROOT, 'sources')
API_KEY = os.environ.get('CHURN_API_KEY', 'demo-key')

DB_PATH = os.path.join(ROOT, 'churn.db')
customer_service = CustomerService(DB_PATH)
def get_customer_service():
    return customer_service

app = FastAPI(title='Churn Data Source API', version='1.0.0')
app.include_router(router)

class Page(BaseModel):
    items: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    next_offset: Optional[int]

def paginate(df: pd.DataFrame, limit: int, offset: int) -> Page:
    total = len(df)
    chunk = df.iloc[offset: offset+limit]
    next_offset = offset + limit if offset + limit < total else None
    return Page(items=chunk.to_dict(orient='records'), total=total, offset=offset, limit=limit, next_offset=next_offset)

def date_filter(df: pd.DataFrame, col: str, since: Optional[str]):
    if since:
        try:
            ts = pd.to_datetime(since)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid 'since' format: {since}")
        s = pd.to_datetime(df[col], errors='coerce')
        df = df.loc[s >= ts]
    return df

@app.get('/health')
def health():
    return {'status': 'ok', 'time': datetime.utcnow().isoformat()}

@app.get('/customers')
def customers_ep(limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0), since: Optional[str] = None, x_api_key: str = Header(None)):
    print('getting customers ', x_api_key)
    print('limit', limit)
    print('offset', offset)
    auth_check(x_api_key)
    auth_check(x_api_key)
    df = pd.read_csv(os.path.join(SOURCES,'customers.csv'))
    df = date_filter(df, 'signup_date', since)
    return paginate(df, limit, offset)

@app.get('/transactions')
def transactions_ep(limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0), since: Optional[str] = None, x_api_key: str = Header(None)):
    auth_check(x_api_key)
    df = pd.read_csv(os.path.join(SOURCES,'transactions.csv'))
    df = date_filter(df, 'txn_date', since)
    return paginate(df, limit, offset)

@app.get('/web_logs')
def weblogs_ep(limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0), since: Optional[str] = None, x_api_key: str = Header(None)):
    auth_check(x_api_key)
    df = pd.read_csv(os.path.join(SOURCES,'web_logs.csv'))
    df = date_filter(df, 'event_time', since)
    return paginate(df, limit, offset)

@app.get('/labels')
def labels_ep(limit: int = Query(500, ge=1, le=5000), offset: int = Query(0, ge=0), since: Optional[str] = None, x_api_key: str = Header(None)):
    auth_check(x_api_key)
    df = pd.read_csv(os.path.join(SOURCES,'labels.csv'))
    if since:
        df = df[df['churn_date'].notna() & (df['churn_date']!='')]
    df = date_filter(df, 'churn_date', since)
    return paginate(df, limit, offset)

@app.get('/db_customers')
def db_customers_ep(
    limit: int = Query(500, ge=1, le=5000),
    offset: int = Query(0, ge=0),
    x_api_key: str = Header(None, alias="x-api-key"),
    service: CustomerService = Depends(get_customer_service)
):
    auth_check(x_api_key)
    return service.get_customers(limit, offset)
