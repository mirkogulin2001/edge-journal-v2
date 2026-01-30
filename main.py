import streamlit as st
import pandas as pd
import database as db
import auth
import time
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date

# --- CONFIGURACIÓN DE PÁGINA ---
# Esto SIEMPRE debe ir primero
st.set_page_config(page_title="Edge Journal", page_icon="📓", layout="wide")

# Inicializar DB
db.init_db()

# --- GESTIÓN DE SESIÓN ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = None
if 'user_name' not in st.session_state: st.session_state['user_name'] = None

# --- VISTA: LOGIN ---
def login_page():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.title("Edge Journal 🔐")
        st.caption("Tu bitácora de trading profesional en la nube.")
        
        tab1, tab2 = st.tabs(["Ingresar", "Registrarse"])
        
        # LOGIN
        with tab1:
            username = st.text_input("Usuario", key="login_user")
            password = st.text_input("Contraseña", type="password", key="login_pass")
            if st.button("Entrar", type="primary"):
                user_data = db.get_user(username)
                if user_data:
                    # user_data = (username, password, name, created_at, initial_balance)
                    stored_hash = user_data[1]
                    real_name = user_data[2]
                    if auth.check_password(password, stored_hash):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['user_name'] = real_name
                        st.success(f"¡Bienvenido, {real_name}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Contraseña incorrecta.")
                else:
                    st.error("Usuario no encontrado.")

        # REGISTRO
        with tab2:
            new_user = st.text_input("Nuevo Usuario", key="reg_user")
            new_name = st.text_input("Nombre Real", key="reg_name")
            new_pass = st.text_input("Crear Contraseña", type="password", key="reg_pass")
            
            if st.button("Crear Cuenta"):
                if new_user and new_pass and new_name:
                    hashed_pw = auth.hash_password(new_pass)
                    # Al crear, se usa el default de 10000 que pusimos en database.py
                    if db.create_user(new_user, hashed_pw, new_name):
                        st.success("¡Cuenta creada! Ahora inicia sesión.")
                    else:
                        st.error("Error: El usuario ya existe o error de conexión.")
                else:
                    st.warning("Completa todos los campos.")

# --- VISTA: DASHBOARD PRINCIPAL ---
def dashboard_page():
    # --- SIDEBAR (CAPITAL INICIAL) ---
    with st.sidebar:
        st.header(f"Hola, {st.session_state['user_name']}")
        
        # Recuperar balance inicial
        user_info = db.get_user(st.session_state['username'])
        # Protección por si la columna nueva aún no se lee bien
        try:
            current_balance = float(user_info[4]) if user_info and len(user_info) > 4 and user_info[4] is not None else 10000.0
        except:
            current_balance = 10000.0
            
        new_bal = st.number_input("Capital Inicial ($)", value=current_balance, step=1000.0)
        
        # Si cambia el input, actualizamos BD
        if new_bal != current_balance:
            db.update_initial_balance(st.session_state['username'], new_bal)
            st.rerun()

        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.divider()
        st.caption("Edge Journal v3.5 Pro")

    st.title("Gestión de Cartera 🏦")
    tab_active, tab_history, tab_stats = st.tabs(["⚡ Posiciones & Mercado", "📚 Bitácora & R:R", "📊 Analytics Pro"])

    # ------------------------------------------------------------------
    # TAB 1: OPERATIVA
    # ------------------------------------------------------------------
    with tab_active:
        col_left, col_right = st.columns([1, 2])
        
        # ABRIR TRADE
        with col_left:
            st.subheader("➕ Nueva Orden")
            with st.form("new_trade"):
                c1, c2 = st.columns(2)
                symbol = c1.text_input("Ticker").upper()
                side = c2.selectbox("Side", ["LONG", "SHORT"])
                
                c3, c4 = st.columns(2)
                price = c3.number_input("Precio Entrada", min_value=0.0, format="%.2f")
                qty = c4.number_input("Cantidad", min_value=1, step=1)
                
                sl_val = st.number_input("Stop Loss Inicial ($)", min_value=0.0, format="%.2f", help="Tu nivel de invalidación")
                date_in = st.date_input("Fecha", value=date.today())
                notes = st.text_area("Tesis")
                
                if st.form_submit_button("🚀 Ejecutar", type="primary"):
                    if symbol and price > 0:
                        db.open_new_trade(st.session_state['username'], symbol, side, price, qty, date_in, notes, sl_val, sl_val)
                        st.success(f"Orden {symbol} enviada.")
                        time.sleep(0.5)
                        st.rerun()

        # MONITOR LIVE
        with col_right:
            st.subheader("📡 Gestión Activa")
            df_open = db.get_open_trades(st.session_state['username'])
            
            if not df_open.empty:
                current_prices, unrealized_pnls = [], []
                prog = st.progress(0)
                
                for i, row in df_open.iterrows():
                    try:
                        t = yf.Ticker(row['symbol'])
                        # Intentamos obtener precio rápido
                        cp = t.fast_info['last_price']
                        if cp is None: cp = t.history(period='1d')['Close'].iloc[-1]
                    except: 
                        cp = row['entry_price'] # Fallback si falla internet
                    
                    pnl = (cp - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - cp) * row['quantity']
                    current_prices.append(cp)
                    unrealized_pnls.append(pnl)
                    prog.progress((i+1)/len(df_open))
                prog.empty()
                
                df_open['Price'] = current_prices
                df_open['Floating PnL'] = unrealized_pnls
                
                st.dataframe(
                    df_open.drop(columns=['id', 'notes', 'initial_stop_loss']), 
                    use_container_width=True, hide_index=True,
                    column_config={
                        "entry_price": st.column_config.NumberColumn("In", format="$%.2f"),
                        "Price": st.column_config.NumberColumn("Now", format="$%.2f"),
                        "current_stop_loss": st.column_config.NumberColumn("Stop Loss", format="$%.2f"),
                        "Floating PnL": st.column_config.NumberColumn("PnL", format="$%.2f")
                    }
                )

                total_float = sum(unrealized_pnls)
                st.metric("PnL Latente", f"${total_float:,.2f}", delta=total_float)
                
                st.divider()
                
                # GESTOR
                df_open['label'] = df_open.apply(lambda x: f"#{x['id']} {x['symbol']} | PnL: ${x['Floating PnL']:.0f}", axis=1)
                sel = st.selectbox("Seleccionar Operación:", df_open['label'])
                sel_id = int(sel.split("#")[1].split(" ")[0])
                row = df_open[df_open['id'] == sel_id].iloc[0]
                
                t_close, t_sl, t_del = st.tabs(["🔒 Cerrar", "🛡️ Ajustar Stop Loss", "🗑️ Borrar"])
                
                with t_close:
                    with st.form("close_f"):
                        c_ex1, c_ex2 = st.columns(2)
                        ep = c_ex1.number_input("Salida", value=float(row['Price']), format="%.2f")
                        ed = c_ex2.date_input("Fecha", value=date.today())
                        if st.form_submit_button("Confirmar Cierre"):
                            raw_pnl = (ep - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - ep) * row['quantity']
                            db.close_trade(sel_id, ep, ed, float(raw_pnl))
                            st.success("Cerrado!")
                            time.sleep(1); st.rerun()

                with t_sl:
                    st.write(f"SL Inicial: **${row['initial_stop_loss']}**")
                    with st.form("update_sl"):
                        new_sl = st.number_input("Nuevo Stop Loss (Trailing)", value=float(row['current_stop_loss']), format="%.2f")
                        if st.form_submit_button("Actualizar Stop Loss"):
                            db.update_stop_loss(sel_id, new_sl)
                            st.success("Actualizado."); time.sleep(1); st.rerun()

                with t_del:
                    if st.button("Eliminar"):
                        db.delete_trade(sel_id); st.rerun()
            else: st.info("Sin posiciones.")

    # ------------------------------------------------------------------
    # TAB 2: HISTORIAL
    # ------------------------------------------------------------------
    with tab_history:
        st.subheader("📚 Análisis de Risk : Reward")
        df_c = db.get_closed_trades(st.session_state['username'])
        if not df_c.empty:
            rr_list = []
            for i, r in df_c.iterrows():
                try:
                    risk = abs(r['entry_price'] - r['initial_stop_loss'])
                    # Si no hay SL o es igual a entrada, riesgo es minúsculo para no dividir por 0
                    if risk == 0: risk = 0.01 
                    
                    # R Units = PnL / (Risk * Qty)
                    r_units = r['pnl'] / (risk * r['quantity'])
                    rr_list.append(r_units)
                except: rr_list.append(0)
            
            df_c['R Units'] = rr_list
            
            st.dataframe(
                df_c.drop(columns=['id']),
                use_container_width=True, hide_index=True,
                column_config={
                    "pnl": st.column_config.NumberColumn("PnL", format="$%.2f"),
                    "R Units": st.column_config.NumberColumn("R Multiplier", format="%.2f R"),
                    "initial_stop_loss": st.column_config.NumberColumn("SL Init", format="$%.2f")
                }
            )
            
            # Opción de borrar historial
            with st.expander("🛠️ Eliminar registro histórico"):
                 df_c['del_lbl'] = df_c.apply(lambda x: f"#{x['id']} {x['symbol']} (${x['pnl']:.2f})", axis=1)
                 to_del = st.selectbox("Elegir:", df_c['del_lbl'])
                 id_del = int(to_del.split("#")[1].split(" ")[0])
                 if st.button("Borrar del historial"):
                     db.delete_trade(id_del); st.rerun()

        else: st.write("Sin datos.")

    # ------------------------------------------------------------------
    # TAB 3: ANALYTICS AVANZADO
    # ------------------------------------------------------------------
    with tab_stats:
        st.subheader("🧪 Análisis Cuantitativo")
        df_all = db.get_all_trades_for_analytics(st.session_state['username'])
        
        if not df_all.empty:
            df_closed = df_all[df_all['exit_price'] > 0].copy()
            df_open = df_all[(df_all['exit_price'].isna()) | (df_all['exit_price'] == 0)].copy()
            
            # PnL Latente
            unrealized_pnl = 0.0
            if not df_open.empty:
                for _, r in df_open.iterrows():
                    try:
                        t = yf.Ticker(r['symbol'])
                        cp = t.fast_info['last_price'] or r['entry_price']
                        val = (cp - r['entry_price']) * r['quantity'] if r['side'] == 'LONG' else (r['entry_price'] - cp) * r['quantity']
                        unrealized_pnl += val
                    except: pass

            if not df_closed.empty:
                # Datos básicos
                total_ops = len(df_closed)
                pnl_acum = df_closed['pnl'].sum()
                wins = df_closed[df_closed['pnl'] > 0]
                losses = df_closed[df_closed['pnl'] <= 0]
                
                n_wins = len(wins); n_losses = len(losses)
                win_rate = n_wins / total_ops
                loss_rate = n_losses / total_ops
                
                avg_win_usd = wins['pnl'].mean() if n_wins > 0 else 0
                avg_loss_usd = losses['pnl'].mean() if n_losses > 0 else 0
                
                # R Multiples
                df_closed['risk_amount'] = abs(df_closed['entry_price'] - df_closed['initial_stop_loss']) * df_closed['quantity']
                df_closed['r_multiple'] = df_closed.apply(lambda x: x['pnl'] / x['risk_amount'] if x['risk_amount'] > 0 else 0, axis=1)
                
                avg_win_r = df_closed[df_closed['pnl'] > 0]['r_multiple'].mean() if n_wins > 0 else 0
                avg_loss_r = df_closed[df_closed['pnl'] <= 0]['r_multiple'].mean() if n_losses > 0 else 0
                
                payoff_ratio = abs(avg_win_usd / avg_loss_usd) if avg_loss_usd != 0 else 0
                math_expectancy = (win_rate * payoff_ratio) - loss_rate
                roi_pct = (pnl_acum / current_balance) * 100
                
                # Drawdown
                df_closed = df_closed.sort_values('entry_date')
                df_closed['cumulative_pnl'] = df_closed['pnl'].cumsum()
                df_closed['equity_curve'] = current_balance + df_closed['cumulative_pnl']
                df_closed['peak'] = df_closed['equity_curve'].cummax()
                df_closed['dd_usd'] = df_closed['equity_curve'] - df_closed['peak']
                df_closed['dd_pct'] = (df_closed['dd_usd'] / df_closed['peak']) * 100
                
                max_dd_usd = df_closed['dd_usd'].min()
                max_dd_pct = df_closed['dd_pct'].min()

                # Metrics Grid
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Ops", total_ops)
                m2.metric("Win Rate", f"{win_rate*100:.1f}%")
                m3.metric("Loss Rate", f"{loss_rate*100:.1f}%")
                m4.metric("PnL Total", f"${pnl_acum:,.2f}", delta=pnl_acum)

                m5, m6, m7, m8 = st.columns(4)
                m5.metric("Avg Win $", f"${avg_win_usd:,.0f}")
                m6.metric("Avg Loss $", f"${avg_loss_usd:,.0f}")
                m7.metric("Avg Win R", f"{avg_win_r:.2f}R")
                m8.metric("Avg Loss R", f"{avg_loss_r:.2f}R")
                
                m9, m10, m11, m12 = st.columns(4)
                m9.metric("Payoff (B/R)", f"{payoff_ratio:.2f}")
                m10.metric("Esperanza", f"{math_expectancy:.2f}")
                m11.metric("Unrealized", f"${unrealized_pnl:,.0f}", delta=unrealized_pnl)
                m12.metric("ROI Total", f"{roi_pct:.2f}%")

                m13, m14 = st.columns(2)
                m13.metric("Max DD ($)", f"${max_dd_usd:,.2f}", delta=max_dd_usd)
                m14.metric("Max DD (%)", f"{max_dd_pct:.2f}%", delta=max_dd_pct)
                
                st.divider()

                # Charts
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig_dd = px.area(df_closed, x='entry_date', y='dd_pct', title="Drawdown (%)")
                    fig_dd.update_traces(line_color='red', fillcolor='rgba(255,0,0,0.2)')
                    st.plotly_chart(fig_dd, use_container_width=True)

                with col_g2:
                    fig_eq = px.line(df_closed, x='entry_date', y='equity_curve', title="Equity Curve")
                    fig_eq.update_traces(line_color='#00FFAA', line_width=3)           
                    if unrealized_pnl != 0:
                        last_d = df_closed['entry_date'].iloc[-1]
                        last_e = df_closed['equity_curve'].iloc[-1]
                        fig_eq.add_trace(go.Scatter(x=[last_d, date.today()], y=[last_e, last_e+unrealized_pnl], 
                                                    mode='lines+markers', name='Proy.', line=dict(color='yellow', dash='dot')))
                    st.plotly_chart(fig_eq, use_container_width=True)

            else: st.info("Cierra operaciones para ver métricas.")
        else: st.warning("Sin datos.")

# --- CONTROLADOR PRINCIPAL ---
def main():
    if st.session_state['logged_in']:
        dashboard_page()
    else:
        login_page()

if __name__ == '__main__':
    main()

