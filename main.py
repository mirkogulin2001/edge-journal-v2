import streamlit as st
import pandas as pd
import database as db
import auth
import time
import plotly.express as px 
import yfinance as yf
from datetime import date

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Edge Journal", page_icon="📓", layout="wide")

db.init_db()

# --- SESIÓN ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = None
if 'user_name' not in st.session_state: st.session_state['user_name'] = None

# --- LOGIN ---
def login_page():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("Edge Journal 🔐")
        st.caption("Tu bitácora de trading profesional en la nube.")
        
        tab1, tab2 = st.tabs(["Ingresar", "Registrarse"])
        
        with tab1:
            username = st.text_input("Usuario", key="login_user")
            password = st.text_input("Contraseña", type="password", key="login_pass")
            if st.button("Entrar", type="primary"):
                user_data = db.get_user(username)
                if user_data:
                    stored_hash = user_data[1]
                    real_name = user_data[2]
                    if auth.check_password(password, stored_hash):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['user_name'] = real_name
                        st.success(f"¡Bienvenido, {real_name}!")
                        time.sleep(1)
                        st.rerun()
                    else: st.error("Contraseña incorrecta.")
                else: st.error("Usuario no encontrado.")

        with tab2:
            new_user = st.text_input("Nuevo Usuario", key="reg_user")
            new_name = st.text_input("Nombre Real", key="reg_name")
            new_pass = st.text_input("Crear Contraseña", type="password", key="reg_pass")
            
            if st.button("Crear Cuenta"):
                if new_user and new_pass and new_name:
                    hashed_pw = auth.hash_password(new_pass)
                    if db.create_user(new_user, hashed_pw, new_name):
                        st.success("¡Cuenta creada! Inicia sesión.")
                    else: st.error("Error al crear usuario.")
                else: st.warning("Completa todos los campos.")

# --- DASHBOARD ---
def dashboard_page():
    with st.sidebar:
        st.header(f"Hola, {st.session_state['user_name']}")
        st.write(f"ID: `{st.session_state['username']}`")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.divider()
        st.caption("Edge Journal Cloud v2.1")

    st.title("Gestión de Cartera 🏦")

    tab_active, tab_history, tab_stats = st.tabs(["⚡ Posiciones Abiertas", "📚 Bitácora Cerrada", "📊 Analytics Pro"])

    # ------------------------------------------------------------------
    # TAB 1: GESTIÓN ACTIVA (CON LIVE DATA) 📡
    # ------------------------------------------------------------------
    with tab_active:
        col_left, col_right = st.columns([1, 2]) # Hacemos la derecha más ancha para la tabla
        
        # ABRIR TRADE (Izquierda)
        with col_left:
            st.subheader("➕ Nueva Posición")
            with st.form("new_trade_form"):
                c1, c2 = st.columns(2)
                symbol = c1.text_input("Ticker (Ej: SPY)").upper()
                side = c2.selectbox("Dirección", ["LONG", "SHORT"])
                price = st.number_input("Precio Entrada", min_value=0.0, format="%.2f")
                qty = st.number_input("Cantidad", min_value=1, step=1)
                date_in = st.date_input("Fecha", value=date.today())
                notes = st.text_area("Notas")
                
                if st.form_submit_button("🚀 Ejecutar", type="primary"):
                    if symbol and price > 0:
                        db.open_new_trade(st.session_state['username'], symbol, side, price, qty, date_in, notes)
                        st.success(f"Orden: {side} {symbol}")
                        time.sleep(0.5)
                        st.rerun()

        # MONITOR EN VIVO (Derecha)
        with col_right:
            st.subheader("📡 Monitor de Mercado")
            df_open = db.get_open_trades(st.session_state['username'])
            
            if not df_open.empty:
                # --- MAGIA DE LIVE DATA ---
                # Creamos listas para guardar los datos nuevos
                current_prices = []
                unrealized_pnls = []
                
                # Barra de progreso para que se vea pro mientras carga
                prog_bar = st.progress(0)
                total_trades = len(df_open)
                
                for i, row in df_open.iterrows():
                    try:
                        # Buscamos precio en Yahoo Finance
                        ticker = yf.Ticker(row['symbol'])
                        # fast_info suele ser más rápido que history
                        cur_price = ticker.fast_info['last_price']
                        
                        # Si falla fast_info, intentamos history (plan B)
                        if cur_price is None: 
                            cur_price = ticker.history(period="1d")['Close'].iloc[-1]
                            
                    except:
                        cur_price = row['entry_price'] # Si falla internet, usamos precio entrada
                    
                    # Calcular PnL Latente
                    if row['side'] == 'LONG':
                        u_pnl = (cur_price - row['entry_price']) * row['quantity']
                    else: # SHORT
                        u_pnl = (row['entry_price'] - cur_price) * row['quantity']
                        
                    current_prices.append(cur_price)
                    unrealized_pnls.append(u_pnl)
                    prog_bar.progress((i + 1) / total_trades)
                
                prog_bar.empty() # Borrar barra al terminar
                
                # Agregamos las columnas al DataFrame visual
                df_open['Price'] = current_prices
                df_open['U. PnL'] = unrealized_pnls
                
                # MOSTRAR TABLA CON COLORES
                # Usamos st.dataframe con column_config para colorear el PnL
                st.dataframe(
                    df_open.drop(columns=['id', 'notes']), # Ocultamos notas e ID para espacio
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "entry_price": st.column_config.NumberColumn("Entrada", format="$ %.2f"),
                        "Price": st.column_config.NumberColumn("Actual", format="$ %.2f"),
                        "entry_date": st.column_config.DateColumn("Fecha"),
                        "U. PnL": st.column_config.NumberColumn(
                            "PnL Latente",
                            format="$ %.2f",
                        )
                    }
                )
                
                # MÉTRICA DE PNL TOTAL LATENTE
                total_unrealized = sum(unrealized_pnls)
                color_metric = "normal"
                if total_unrealized > 0: color_metric = "off" # Verde (truco de streamlit)
                st.metric(label="PnL Flotante Total", value=f"$ {total_unrealized:,.2f}", delta=total_unrealized)

                st.divider()
                
                # --- SECCIÓN DE CIERRE (Mantenemos tu lógica) ---
                st.write("👇 **Gestionar Posición**")
                df_open['display_label'] = df_open.apply(lambda x: f"#{x['id']} | {x['symbol']} | PnL: ${x['U. PnL']:.2f}", axis=1)
                
                c_sel, c_act = st.columns([2, 1])
                with c_sel:
                    trade_selection = st.selectbox("Seleccionar:", df_open['display_label'], label_visibility="collapsed")
                
                selected_id = int(trade_selection.split("|")[0].replace("#", "").strip())
                trade_data = df_open[df_open['id'] == selected_id].iloc[0]

                t_close, t_delete = st.tabs(["🔒 CERRAR", "🗑️ BORRAR"])
                
                with t_close:
                    with st.form("close_trade_form"):
                        c_ex1, c_ex2 = st.columns(2)
                        # Sugerimos el precio actual automáticamente ;)
                        exit_price = c_ex1.number_input("Precio Salida", value=float(trade_data['Price']), min_value=0.0, format="%.2f")
                        exit_date = c_ex2.date_input("Fecha", value=date.today())
                        
                        if st.form_submit_button("Confirmar Cierre"):
                            raw_pnl = (exit_price - trade_data['entry_price']) * trade_data['quantity'] if trade_data['side'] == "LONG" else (trade_data['entry_price'] - exit_price) * trade_data['quantity']
                            pnl = float(raw_pnl)
                            db.close_trade(selected_id, exit_price, exit_date, pnl)
                            st.success(f"Cerrado! PnL: ${pnl:.2f}")
                            time.sleep(1)
                            st.rerun()
                
                with t_delete:
                    if st.button("Eliminar Registro"):
                        db.delete_trade(selected_id)
                        st.rerun()

            else:
                st.info("Sin posiciones abiertas.")

    # --- TAB 2: CERRADAS ---
    with tab_history:
        st.subheader("📜 Bitácora de Operaciones Cerradas")
        df_closed = db.get_closed_trades(st.session_state['username'])
        
        if not df_closed.empty:
            st.dataframe(
                df_closed.drop(columns=['id']),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pnl": st.column_config.NumberColumn("PnL", format="$ %.2f"),
                    "entry_price": st.column_config.NumberColumn("In", format="$ %.2f"),
                    "exit_price": st.column_config.NumberColumn("Out", format="$ %.2f"),
                    "entry_date": st.column_config.DateColumn("Fecha"),
                }
            )
            
            # SECCIÓN BORRAR HISTORIAL
            with st.expander("🛠️ Gestionar / Eliminar Historial"):
                st.write("Si cerraste un trade por error o con datos incorrectos, puedes borrarlo aquí y cargarlo de nuevo.")
                df_closed['display_label'] = df_closed.apply(lambda x: f"#{x['id']} | {x['symbol']} | PnL: ${x['pnl']:.2f}", axis=1)
                hist_selection = st.selectbox("Selecciona trade cerrado:", df_closed['display_label'], key="sel_closed")
                hist_id = int(hist_selection.split("|")[0].replace("#", "").strip())
                
                if st.button("🗑️ Eliminar Registro del Historial"):
                    db.delete_trade(hist_id)
                    st.success("Registro eliminado correctamente.")
                    time.sleep(1)
                    st.rerun()
        else:
            st.write("Aún no has cerrado operaciones.")

    # --- TAB 3: ANALYTICS ---
    with tab_stats:
        st.subheader("Tablero de Comando")
        df_all = db.get_all_trades_for_analytics(st.session_state['username'])
        
        if not df_all.empty:
            df_c = df_all[df_all['exit_price'] > 0].copy()
            df_o = df_all[(df_all['exit_price'].isna()) | (df_all['exit_price'] == 0)].copy()
            
            total_pnl = df_c['pnl'].sum() if not df_c.empty else 0
            open_exposure = (df_o['entry_price'] * df_o['quantity']).sum() if not df_o.empty else 0
            
            wins = len(df_c[df_c['pnl'] > 0])
            total_closed = len(df_c)
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("PnL Total", f"$ {total_pnl:,.2f}")
            k2.metric("Exposición", f"$ {open_exposure:,.2f}")
            k3.metric("Win Rate", f"{win_rate:.1f}%")
            k4.metric("Cerrados", total_closed)
            
            st.divider()
            
            g1, g2 = st.columns([2, 1])
            with g1:
                st.subheader("📈 Equity Curve")
                if not df_c.empty:
                    df_c = df_c.sort_values('entry_date')
                    df_c['cumulative_pnl'] = df_c['pnl'].cumsum()
                    fig_line = px.line(df_c, x='entry_date', y='cumulative_pnl', markers=True)
                    fig_line.update_traces(line_color='#00BBA2', line_width=3)
                    st.plotly_chart(fig_line, use_container_width=True)
                else: st.info("Cierra trades para ver la curva.")
            
            with g2:
                st.subheader("🍰 Cartera Actual")
                if not df_o.empty:
                    df_o['position_size'] = df_o['entry_price'] * df_o['quantity']
                    pie_data = df_o.groupby('symbol')['position_size'].sum().reset_index()
                    fig_pie = px.pie(pie_data, values='position_size', names='symbol', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else: st.info("Sin posiciones abiertas.")
        else:
            st.warning("No hay datos para analizar.")

def main():
    if st.session_state['logged_in']: dashboard_page()
    else: login_page()

if __name__ == '__main__': main()


