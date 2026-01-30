import pandas as pd
import streamlit as st
import psycopg2

def get_connection():
    try:
        return psycopg2.connect(st.secrets["DB_URL"], sslmode='require')
    except Exception as e:
        st.error(f"Error de conexión: {e}")
        return None

def init_db():
    pass 

# --- USUARIOS ---
def create_user(username, password, name):
    conn = get_connection()
    if not conn: return False
    try:
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, name) VALUES (%s, %s, %s)", (username, password, name))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def get_user(username):
    conn = get_connection()
    if not conn: return None
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = c.fetchone()
    conn.close()
    return user

# --- TRADES ---

def open_new_trade(username, symbol, side, price, quantity, date, notes):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO trades (username, symbol, side, entry_price, quantity, entry_date, notes, exit_price, pnl)
            VALUES (%s, %s, %s, %s, %s, %s, %s, NULL, NULL)
        ''', (username, symbol, side, price, quantity, date, notes))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def close_trade(trade_id, exit_price, exit_date, pnl):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE trades 
            SET exit_price = %s, pnl = %s, notes = notes || %s 
            WHERE id = %s
        ''', (exit_price, pnl, f" | Cerrado el {exit_date}", trade_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def delete_trade(trade_id):
    """Elimina definitivamente un trade de la base de datos"""
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM trades WHERE id = %s", (trade_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al eliminar: {e}")
        return False

def get_open_trades(username):
    conn = get_connection()
    query = "SELECT id, symbol, side, entry_price, quantity, entry_date, notes FROM trades WHERE username = %s AND (exit_price IS NULL OR exit_price = 0)"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_closed_trades(username):
    conn = get_connection()
    query = "SELECT id, symbol, side, entry_price, exit_price, quantity, entry_date, pnl, notes FROM trades WHERE username = %s AND exit_price > 0 ORDER BY entry_date DESC"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_all_trades_for_analytics(username):
    conn = get_connection()
    query = "SELECT * FROM trades WHERE username = %s"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df
