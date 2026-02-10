import psycopg2
import streamlit as st
from datetime import datetime
import json
import pandas as pd

# --- CONEXIÓN ---
def get_connection():
    return psycopg2.connect(
        host=st.secrets["supabase"]["host"],
        database=st.secrets["supabase"]["dbname"],
        user=st.secrets["supabase"]["user"],
        password=st.secrets["supabase"]["password"],
        port=st.secrets["supabase"]["port"],
        sslmode='require'
    )

def init_db():
    # La creación de tablas ya la hicimos en SQL directo en el Paso 1
    # Pero dejamos esto por si acaso para verificar conexión al inicio
    conn = get_connection()
    conn.close()

# --- USUARIOS ---
def create_user(username, password, name):
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute('''INSERT INTO users (username, password, name, created_at, initial_balance, strategy_config) 
                     VALUES (%s, %s, %s, %s, %s, %s)''', 
                     (username, password, name, datetime.now(), 10000.0, "{}"))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error create_user: {e}")
        return False
    finally:
        conn.close()

def get_user(username):
    conn = get_connection()
    c = conn.cursor()
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
    try:
        c.execute('UPDATE users SET strategy_config = %s WHERE username = %s', (json.dumps(config_dict), username))
        conn.commit()
        return True
    except: return False
    finally: conn.close()

# --- TRADES ---
def open_new_trade(user_id, symbol, side, price, qty, date_in, notes, sl, current_sl, tags_dict):
    conn = get_connection()
    c = conn.cursor()
    c.execute('''INSERT INTO trades (user_id, symbol, side, entry_price, quantity, entry_date, notes, 
                 initial_stop_loss, current_stop_loss, tags, status, initial_quantity) 
                 VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'OPEN', %s)''',
              (user_id, symbol, side, price, qty, date_in, notes, sl, current_sl, json.dumps(tags_dict), qty))
    conn.commit()
    conn.close()

def get_open_trades(user_id):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id = %s AND status = 'OPEN'", conn, params=(user_id,))
    conn.close()
    return df

def get_closed_trades(user_id):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id = %s AND status = 'CLOSED'", conn, params=(user_id,))
    conn.close()
    return df

def get_all_trades_for_analytics(user_id):
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM trades WHERE user_id = %s", conn, params=(user_id,))
    conn.close()
    return df

def close_trade(trade_id, exit_price, exit_date, pnl, result_type):
    conn = get_connection()
    c = conn.cursor()
    c.execute("UPDATE trades SET status='CLOSED', exit_price=%s, exit_date=%s, pnl=%s, result_type=%s WHERE id=%s", 
              (exit_price, exit_date, pnl, result_type, trade_id))
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
    if row:
        new_qty = row[0] - qty_to_sell
        current_banked = float(row[1]) if row[1] else 0.0
        new_banked = current_banked + realized_pnl
        c.execute("UPDATE trades SET quantity = %s, partial_realized_pnl = %s WHERE id = %s", (new_qty, new_banked, trade_id))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def import_batch_trades(user_id, df):
    conn = get_connection()
    c = conn.cursor()
    try:
        for _, row in df.iterrows():
            sym = row.get('Ticker', row.get('Symbol', 'UNKNOWN'))
            pnl = row.get('PnL', row.get('Net Profit', 0))
            c.execute('''INSERT INTO trades (user_id, symbol, pnl, status, result_type, exit_date, entry_date) 
                         VALUES (%s, %s, %s, 'CLOSED', %s, %s, %s)''',
                      (user_id, sym, pnl, 'WIN' if pnl > 0 else 'LOSS', datetime.now(), datetime.now()))
        conn.commit()
        return True
    except: return False
    finally: conn.close()
