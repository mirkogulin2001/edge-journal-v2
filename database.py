import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
from datetime import datetime
import json

# --- CONEXIÓN A SUPABASE (PostgreSQL) ---
def get_connection():
    # Usamos st.secrets para no exponer la contraseña en GitHub
    return psycopg2.connect(
        host=st.secrets["supabase"]["host"],
        database=st.secrets["supabase"]["dbname"],
        user=st.secrets["supabase"]["user"],
        password=st.secrets["supabase"]["password"],
        port=st.secrets["supabase"]["port"]
    )

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    # 1. Tabla de Usuarios (Sintaxis Postgres: SERIAL en vez de AUTOINCREMENT)
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    name TEXT,
                    created_at TEXT,
                    initial_balance REAL DEFAULT 10000.0,
                    strategy_config TEXT DEFAULT '{}'
                )''')
    
    # 2. Tabla de Trades (Sintaxis Postgres)
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    user_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    entry_price REAL,
                    quantity INTEGER,
                    entry_date TEXT,
                    notes TEXT,
                    initial_stop_loss REAL,
                    current_stop_loss REAL,
                    take_profit REAL,
                    tags TEXT,
                    status TEXT DEFAULT 'OPEN',
                    exit_price REAL,
                    exit_date TEXT,
                    pnl REAL,
                    result_type TEXT,
                    initial_quantity INTEGER,
                    partial_realized_pnl REAL DEFAULT 0.0
                )''')
    conn.commit()
    conn.close()

# --- GESTIÓN DE USUARIOS ---

def create_user(username, password, name):
    conn = get_connection()
    c = conn.cursor()
    try:
        # Postgres usa %s como placeholder, no ?
        c.execute('''INSERT INTO users (username, password, name, created_at, initial_balance, strategy_config) 
                     VALUES (%s, %s, %s, %s, %s, %s)''', 
                     (username, password, name, datetime.now(), 10000.0, "{}"))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error creando usuario: {e}")
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_connection()
    c = conn.cursor()
    # Postgres usa %s
    c.execute('SELECT id, password, name, created_at, initial_balance, strategy_config FROM users WHERE username = %s', (username,))
    data = c.fetchone()
    conn.close()
    return data

def update_initial_balance(username, new_balance):
    conn = get_connection()
    c = conn.cursor()
    c.execute('UPDATE users SET initial_balance = %s WHERE username = %s', (new_balance, username))
    conn.commit()
    conn.close()

def update_strategy_config(username, config_dict):
    conn = get_connection()
    c = conn.cursor()
    config_json = json.dumps(config_dict)
    try:
        c.execute('UPDATE users SET strategy_config = %s WHERE username = %s', (config_json, username))
        conn.commit()
        return True
    except:
        return False
    finally:
        conn.close()

# --- GESTIÓN DE TRADES ---

def open_new_trade(user_id, symbol, side, price, qty, date_in, notes, sl, current_sl, tags_dict):
    conn = get_connection()
    c = conn.cursor()
    tags_json = json.dumps(tags_dict)
    # Sintaxis Postgres: %s
    c.execute('''INSERT INTO trades (user_id, symbol, side, entry_price, quantity, entry_date, notes, 
                 initial_stop_loss, current_stop_loss, tags, status, initial_quantity) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s)''',
              (user_id, symbol, side, price, qty, date_in, notes, sl, current_sl, tags_json, qty))
    conn.commit()
    conn.close()

def get_open_trades(user_id):
    import pandas as pd
    conn = get_connection()
    # pandas.read_sql funciona bien con conexiones psycopg2, pero necesita %s
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id = %s AND status = 'OPEN'", conn, params=(user_id,))
    conn.close()
    return df

def get_closed_trades(user_id):
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id = %s AND status = 'CLOSED'", conn, params=(user_id,))
    conn.close()
    return df

def get_all_trades_for_analytics(user_id):
    import pandas as pd
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id = %s", conn, params=(user_id,))
    conn.close()
    return df

def close_trade(trade_id, exit_price, exit_date, pnl, result_type):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''UPDATE trades SET status='CLOSED', exit_price=%s, exit_date=%s, pnl=%s, result_type=%s 
                 WHERE id=%s''', (exit_price, exit_date, pnl, result_type, trade_id))
    conn.commit()
    conn.close()

def update_stop_loss(trade_id, new_sl):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE trades SET current_stop_loss = %s WHERE id = %s", (new_sl, trade_id))
    conn.commit()
    conn.close()

def delete_trade(trade_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE id = %s", (trade_id,))
    conn.commit()
    conn.close()

def delete_all_trades(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("DELETE FROM trades WHERE user_id = %s", (user_id,))
    conn.commit()
    conn.close()
    return True

def execute_partial_close(trade_id, qty_to_sell, sell_price, realized_pnl):
    conn = get_connection()
    c = conn.cursor()
    
    c.execute("SELECT quantity, partial_realized_pnl FROM trades WHERE id=%s", (trade_id,))
    row = c.fetchone()
    if not row: return False
    
    current_qty = row[0]
    # En Postgres row[1] puede venir como Decimal, asegurar float
    current_banked = float(row[1]) if row[1] else 0.0
    
    new_qty = current_qty - qty_to_sell
    new_banked = current_banked + realized_pnl
    
    c.execute('''UPDATE trades SET quantity = %s, partial_realized_pnl = %s 
                 WHERE id = %s''', (new_qty, new_banked, trade_id))
    
    conn.commit()
    conn.close()
    return True

def import_batch_trades(user_id, df):
    conn = get_connection()
    c = conn.cursor()
    try:
        for _, row in df.iterrows():
            symbol = row.get('Ticker', row.get('Symbol', 'UNKNOWN'))
            pnl = row.get('PnL', row.get('Net Profit', 0))
            
            c.execute('''INSERT INTO trades (user_id, symbol, pnl, status, result_type, exit_date, entry_date) 
                         VALUES (%s, %s, %s, 'CLOSED', %s, %s, %s)''',
                      (user_id, symbol, pnl, 
                       'WIN' if pnl > 0 else 'LOSS', 
                       datetime.now(), datetime.now()))
        conn.commit()
        return True
    except Exception as e:
        print(e)
        return False
    finally:
        conn.close()
