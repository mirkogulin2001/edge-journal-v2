import pandas as pd
import streamlit as st
import psycopg2
import json
import numpy as np
from datetime import date

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
        # Inicializamos initial_quantity igual a quantity
        c.execute('''
            INSERT INTO trades (
                username, symbol, side, entry_price, quantity, initial_quantity, 
                entry_date, notes, initial_stop_loss, current_stop_loss, tags, 
                exit_price, pnl, partial_realized_pnl, total_exit_value
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL, 0.0, 0.0)
        ''', (username, symbol, side, price, quantity, quantity, date, notes, sl_init, sl_curr, tags_json))
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

# --- NUEVO: CIERRE PARCIAL ---
def execute_partial_close(trade_id, qty_to_close, exit_price, partial_pnl):
    try:
        conn = get_connection()
        c = conn.cursor()
        
        # 1. Reducimos la cantidad activa (quantity)
        # 2. Sumamos el PnL a la bolsa (partial_realized_pnl)
        # 3. Sumamos el valor de salida para el promedio final (total_exit_value)
        c.execute('''
            UPDATE trades 
            SET quantity = quantity - %s,
                partial_realized_pnl = partial_realized_pnl + %s,
                total_exit_value = total_exit_value + (%s * %s)
            WHERE id = %s
        ''', (qty_to_close, partial_pnl, exit_price, qty_to_close, trade_id))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error en parcial: {e}")
        return False

# --- CIERRE FINAL (MODIFICADO PARA PROMEDIAR) ---
def close_trade(trade_id, exit_price, exit_date, final_chunk_pnl, result_type):
    try:
        conn = get_connection()
        c = conn.cursor()
        
        # Primero necesitamos leer los acumulados para hacer el promedio
        c.execute("SELECT initial_quantity, quantity, partial_realized_pnl, total_exit_value FROM trades WHERE id = %s", (trade_id,))
        row = c.fetchone()
        
        if row:
            init_qty = float(row[0])
            last_chunk_qty = float(row[1]) # Lo que quedaba
            past_pnl = float(row[2])
            past_value = float(row[3])
            
            # Matemáticas del promedio
            final_chunk_value = last_chunk_qty * exit_price
            total_value_generated = past_value + final_chunk_value
            
            # Precio Promedio Ponderado = Total Generado / Cantidad Original
            weighted_avg_exit_price = total_value_generated / init_qty if init_qty > 0 else exit_price
            
            # PnL Total = Lo que ya cobré + Lo que cobro ahora
            total_final_pnl = past_pnl + final_chunk_pnl

            # Actualizamos: Ponemos quantity = initial_quantity para que el historial se vea "completo"
            c.execute('''
                UPDATE trades 
                SET exit_price = %s, 
                    pnl = %s, 
                    result_type = %s, 
                    exit_date = %s, 
                    quantity = initial_quantity,
                    notes = notes || %s 
                WHERE id = %s
            ''', (weighted_avg_exit_price, total_final_pnl, result_type, exit_date, f" | Cerrado el {exit_date}", trade_id))
            
            conn.commit()
            conn.close()
            return True
        return False
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

def delete_all_trades(username):
    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("DELETE FROM trades WHERE username = %s", (username,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error al borrar todo: {e}")
        return False

# --- IMPORTACIÓN ---
def clean_number(val):
    if isinstance(val, (int, float)): return float(val)
    if pd.isna(val) or str(val).strip() == '': return 0.0
    s = str(val).strip().replace('$', '').replace(' ', '').replace('.', '').replace(',', '.')
    try: return float(s)
    except: return 0.0

def import_batch_trades(username, df):
    conn = get_connection()
    if not conn: return False
    try:
        c = conn.cursor()
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        col_status = 'status' if 'status' in df.columns else 'satus'
        col_pnl = 'pnl $' if 'pnl $' in df.columns else 'pnl'

        count = 0
        for _, row in df.iterrows():
            symbol = str(row.get('symbol', 'UNKNOWN')).upper()
            qty = clean_number(row.get('qty', 0))
            entry_price = clean_number(row.get('entry price', 0))
            exit_price = clean_number(row.get('exit price', 0))
            sl_val = clean_number(row.get('stop loss inicial', entry_price))
            if sl_val == 0: sl_val = entry_price
            pnl = clean_number(row.get(col_pnl, 0))
            
            raw_side = str(row.get('side', 'L')).upper().strip()
            side = 'SHORT' if raw_side.startswith('S') else 'LONG'
            
            try: entry_date = pd.to_datetime(row.get('entry date'), dayfirst=True).strftime('%Y-%m-%d')
            except: entry_date = date.today()
            
            try: exit_date = pd.to_datetime(row.get('exit date'), dayfirst=True).strftime('%Y-%m-%d')
            except: exit_date = date.today()
            
            status_raw = str(row.get(col_status, 'BE')).upper()
            if 'WIN' in status_raw: result_type = 'WIN'
            elif 'LOSS' in status_raw: result_type = 'LOSS'
            else: result_type = 'BE'
            
            tags_dict = {}
            if 'setup' in row and pd.notna(row['setup']): tags_dict['Setup'] = str(row['setup']).strip()
            if 'grado' in row and pd.notna(row['grado']): tags_dict['Grado'] = str(row['grado']).strip()
            col_prob = next((c for c in df.columns if 'prob' in c), None)
            if col_prob and pd.notna(row[col_prob]): tags_dict['Fibonacci'] = str(row[col_prob]).strip()
            
            tags_json = json.dumps(tags_dict)
            rr_val = row.get('rr', '')
            notes = f"Importado. RR: {rr_val}"

            # Insertamos con initial_quantity = qty y sin parciales (asumimos trade cerrado completo)
            c.execute('''
                INSERT INTO trades (
                    username, symbol, side, entry_price, quantity, initial_quantity,
                    entry_date, exit_price, exit_date, pnl, result_type, notes, 
                    initial_stop_loss, current_stop_loss, tags, partial_realized_pnl, total_exit_value
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 0.0, 0.0)
            ''', (username, symbol, side, entry_price, qty, qty, entry_date, exit_price, exit_date, pnl, result_type, notes, sl_val, sl_val, tags_json))
            count += 1
            
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error importando fila {count+1}: {e}")
        return False

# --- GETTERS ---
# Modificados para traer los parciales y calcular analíticas correctamente

def get_open_trades(username):
    conn = get_connection()
    # Traemos initial_quantity para referencia y partial_realized_pnl
    query = "SELECT id, symbol, side, entry_price, quantity, initial_quantity, entry_date, notes, initial_stop_loss, current_stop_loss, tags, partial_realized_pnl FROM trades WHERE username = %s AND (exit_price IS NULL OR exit_price = 0)"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df

def get_closed_trades(username):
    conn = get_connection()
    # En cerrados, quantity ya fue restaurado a initial_quantity en close_trade
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
