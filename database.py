import pandas as pd
import streamlit as st
import psycopg2
import json
import numpy as np

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
        default_config = json.dumps({
            "Setup": ["Principal (SUP)", "Secundario (SS)", "Fin Movimiento (SFM)", "Acumulación"],
            "Grado": ["Mayor", "Menor"],
            "Fibonacci": ["Prob. Acumulada", "Prob. Maxima", "Prob. Extendida"]
        })
        c.execute("INSERT INTO users (username, password, name, initial_balance, strategy_config) VALUES (%s, %s, %s, 10000, %s)", 
                  (username, password, name, default_config))
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

def update_initial_balance(username, new_balance):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("UPDATE users SET initial_balance = %s WHERE username = %s", (new_balance, username))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

def update_strategy_config(username, new_config_dict):
    try:
        conn = get_connection()
        c = conn.cursor()
        json_config = json.dumps(new_config_dict)
        c.execute("UPDATE users SET strategy_config = %s WHERE username = %s", (json_config, username))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        return False

# --- TRADES ---

def open_new_trade(username, symbol, side, price, quantity, date, notes, sl_init, sl_curr, tags_dict):
    try:
        conn = get_connection()
        c = conn.cursor()
        tags_json = json.dumps(tags_dict)
        c.execute('''
            INSERT INTO trades (username, symbol, side, entry_price, quantity, entry_date, notes, initial_stop_loss, current_stop_loss, tags, exit_price, pnl)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL)
        ''', (username, symbol, side, price, quantity, date, notes, sl_init, sl_curr, tags_json))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al abrir trade: {e}")
        return False

def update_stop_loss(trade_id, new_sl):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("UPDATE trades SET current_stop_loss = %s WHERE id = %s", (new_sl, trade_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def close_trade(trade_id, exit_price, exit_date, pnl, result_type):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute('''
            UPDATE trades 
            SET exit_price = %s, pnl = %s, result_type = %s, exit_date = %s, notes = notes || %s 
            WHERE id = %s
        ''', (exit_price, pnl, result_type, exit_date, f" | Cerrado el {exit_date}", trade_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error cerrando: {e}")
        return False

def delete_trade(trade_id):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM trades WHERE id = %s", (trade_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

# --- IMPORTACIÓN OPTIMIZADA PARA TU EXCEL ---
def import_batch_trades(username, df):
    conn = get_connection()
    if not conn: return False
    try:
        c = conn.cursor()
        
        # Normalizamos nombres de columnas (todo minúscula y sin espacios extra)
        # Ejemplo: "Entry Date " -> "entry date"
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        count = 0
        for _, row in df.iterrows():
            # 1. Datos Esenciales (con valores por defecto seguros)
            symbol = str(row.get('symbol', 'UNKNOWN')).upper()
            qty = float(row.get('qty', 0))
            
            # SIDE (Ahora lo leemos directo)
            side = str(row.get('side', 'LONG')).upper().strip()
            
            # Precios
            entry_price = float(row.get('entry price', 0))
            exit_price = float(row.get('exit price', 0))
            
            # Fechas
            entry_date = pd.to_datetime(row.get('entry date', date.today())).strftime('%Y-%m-%d')
            exit_date = pd.to_datetime(row.get('exit date', date.today())).strftime('%Y-%m-%d')
            
            # Stop Loss Inicial (Columna vital para calculos de riesgo)
            sl_val = float(row.get('stop loss inicial', entry_price))
            
            # PnL y Status
            pnl = float(row.get('pnl', 0))
            
            # Status: Mapeamos lo que venga en el excel a WIN/LOSS/BE
            status_raw = str(row.get('status', 'BE')).upper()
            if 'WIN' in status_raw: result_type = 'WIN'
            elif 'LOSS' in status_raw: result_type = 'LOSS'
            else: result_type = 'BE'
            
            # 2. Construcción de TAGS (Estrategia)
            tags_dict = {}
            
            if 'setup' in row and pd.notna(row['setup']):
                tags_dict['Setup'] = str(row['setup']).strip()
            
            if 'grado' in row and pd.notna(row['grado']):
                tags_dict['Grado'] = str(row['grado']).strip()
                
            if 'prob' in row and pd.notna(row['prob']):
                tags_dict['Fibonacci'] = str(row['prob']).strip() # Mapeamos 'prob' -> 'Fibonacci'
            
            tags_json = json.dumps(tags_dict)
            
            # 3. Notas (Incluimos el RR aquí para referencia)
            rr_val = row.get('rr', '')
            notes = f"Importado. RR Realizado: {rr_val}"

            # 4. Inserción SQL
            c.execute('''
                INSERT INTO trades (
                    username, symbol, side, entry_price, quantity, entry_date, 
                    exit_price, exit_date, pnl, result_type, notes, 
                    initial_stop_loss, current_stop_loss, tags
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', (
                username, symbol, side, entry_price, qty, entry_date,
                exit_price, exit_date, pnl, result_type, notes,
                sl_val, sl_val, tags_json
            ))
            count += 1
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error importando fila {count+1}: {e}")
        return False

def get_open_trades(username):
    conn = get_connection()
    query = "SELECT id, symbol, side, entry_price, quantity, entry_date, notes, initial_stop_loss, current_stop_loss, tags FROM trades WHERE username = %s AND (exit_price IS NULL OR exit_price = 0)"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_closed_trades(username):
    conn = get_connection()
    query = "SELECT id, symbol, side, entry_price, exit_price, quantity, entry_date, exit_date, pnl, notes, initial_stop_loss, tags, result_type FROM trades WHERE username = %s AND exit_price > 0 ORDER BY entry_date DESC"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_all_trades_for_analytics(username):
    conn = get_connection()
    query = "SELECT * FROM trades WHERE username = %s"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df
