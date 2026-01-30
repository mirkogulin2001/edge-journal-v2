import streamlit as st
import pandas as pd
import database as db
import auth
import time
import plotly.express as px
import plotly.graph_objects as go # Necesario para gráficos avanzados
import yfinance as yf
from datetime import date

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
        tab1, tab2 = st.tabs(["Ingresar", "Registrarse"])
        with tab1:
            username = st.text_input("Usuario", key="login_user")
            password = st.text_input("Contraseña", type="password", key="login_pass")
            if st.button("Entrar", type="primary"):
                user_data = db.get_user(username)
                if user_data and auth.check_password(password, user_data[1]):
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.session_state['user_name'] = user_data[2]
                    st.rerun()
                else: st.error("Error de credenciales")
        with tab2:
            nu = st.text_input("Nuevo Usuario")
            nn = st.text_input("Nombre Real")
            np = st.text_input("Contraseña", type="password")
            if st.button("Crear Cuenta"):
                if db.create_user(nu, auth.hash_password(np), nn):
                    st.success("Creado!")
                else: st.error("Error al crear")

# --- DASHBOARD ---
def dashboard_page():
    with st.sidebar:
        st.header(f"Hola, {st.session_state['user_name']}")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.divider()
        st.caption("Edge Journal v3.0 Pro")

    st.title("Gestión de Cartera 🏦")
    tab_active, tab_history, tab_stats = st.tabs(["⚡ Posiciones & Mercado", "📚 Bitácora & R:R", "📊 Analytics Avanzado"])

    # --- TAB 1: OPERATIVA ---
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
                
                # --- NUEVO: STOP LOSS ---
                sl_val = st.number_input("Stop Loss Inicial ($)", min_value=0.0, format="%.2f", help="Tu nivel de invalidación")
                
                date_in = st.date_input("Fecha", value=date.today())
                notes = st.text_area("Tesis")
                
                if st.form_submit_button("🚀 Ejecutar", type="primary"):
                    if symbol and price > 0:
                        # Al inicio, el SL Actual es igual al Inicial
                        db.open_new_trade(st.session_state['username'], symbol, side, price, qty, date_in, notes, sl_val, sl_val)
                        st.success(f"Orden {symbol} enviada.")
                        time.sleep(0.5)
                        st.rerun()

        # MONITOR LIVE
        with col_right:
            st.subheader("📡 Gestión Activa")
            df_open = db.get_open_trades(st.session_state['username'])
            
            if not df_open.empty:
                # Live Data Logic
                current_prices, unrealized_pnls = [], []
                prog = st.progress(0)
                
                for i, row in df_open.iterrows():
                    try:
                        t = yf.Ticker(row['symbol'])
                        cp = t.fast_info['last_price'] or t.history(period='1d')['Close'].iloc[-1]
                    except: cp = row['entry_price']
                    
                    pnl = (cp - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - cp) * row['quantity']
                    current_prices.append(cp)
                    unrealized_pnls.append(pnl)
                    prog.progress((i+1)/len(df_open))
                prog.empty()
                
                df_open['Price'] = current_prices
                df_open['Floating PnL'] = unrealized_pnls
                
                # Tabla Principal
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

                # Total Latente
                total_float = sum(unrealized_pnls)
                st.metric("PnL Latente (Proyección)", f"${total_float:,.2f}", delta=total_float)
                
                st.divider()
                
                # --- GESTOR DE POSICIÓN ---
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
                            st.success("Stop Loss Actualizado.")
                            time.sleep(1); st.rerun()

                with t_del:
                    if st.button("Eliminar"):
                        db.delete_trade(sel_id)
                        st.rerun()
            else: st.info("Mercado tranquilo. Sin posiciones.")

    # --- TAB 2: HISTORIAL & R:R ---
    with tab_history:
        st.subheader("📚 Análisis de Risk : Reward")
        df_c = db.get_closed_trades(st.session_state['username'])
        if not df_c.empty:
            # Calculamos R:R
            # Riesgo = |Entrada - SL Inicial|
            # Recompensa = |Salida - Entrada|
            # R:R = Recompensa / Riesgo
            
            rr_list = []
            for i, r in df_c.iterrows():
                try:
                    risk = abs(r['entry_price'] - r['initial_stop_loss'])
                    reward = abs(r['exit_price'] - r['entry_price'])
                    if risk > 0:
                        rr = reward / risk
                        # Si perdió dinero, el R:R se muestra negativo para indicar pérdida de unidades de riesgo
                        if r['pnl'] < 0: rr = -1.0 # Simbólico o calculado como pérdida
                        # Mejor cálculo: PnL / (Riesgo * Qty) = R units ganadas/perdidas
                        r_units = r['pnl'] / (risk * r['quantity'])
                        rr_list.append(r_units)
                    else:
                        rr_list.append(0)
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
        else: st.write("Sin datos.")

    # --- TAB 3: EQUITY CURVE AREA PRO ---
    with tab_stats:
        st.subheader("📈 Crecimiento de Cuenta (Proyección)")
        df_all = db.get_all_trades_for_analytics(st.session_state['username'])
        
        if not df_all.empty:
            # 1. Preparar datos REALIZADOS
            df_closed = df_all[df_all['exit_price'] > 0].copy()
            df_closed = df_closed.sort_values('entry_date')
            
            if not df_closed.empty:
                # EJE X = Número de Trade (1, 2, 3...)
                df_closed['trade_number'] = range(1, len(df_closed) + 1)
                df_closed['equity'] = df_closed['pnl'].cumsum()
                
                last_trade_num = df_closed['trade_number'].iloc[-1]
                current_equity = df_closed['equity'].iloc[-1]
                
                # 2. Calcular PROYECCIÓN (Open PnL)
                df_open = df_all[(df_all['exit_price'].isna()) | (df_all['exit_price'] == 0)].copy()
                floating_pnl = 0
                if not df_open.empty:
                    # Necesitamos calcular el floating PnL rápido aquí también
                    # (Podríamos optimizar pasando el dato desde la Tab 1, pero esto es más seguro)
                    for _, r in df_open.iterrows():
                        try:
                            t = yf.Ticker(r['symbol'])
                            cp = t.fast_info['last_price'] or r['entry_price']
                            val = (cp - r['entry_price']) * r['quantity'] if r['side'] == 'LONG' else (r['entry_price'] - cp) * r['quantity']
                            floating_pnl += val
                        except: pass
                
                projected_equity = current_equity + floating_pnl
                
                # 3. CONSTRUIR GRÁFICO COMBINADO
                fig = go.Figure()
                
                # A) Área Sólida (Realizado)
                fig.add_trace(go.Scatter(
                    x=df_closed['trade_number'], 
                    y=df_closed['equity'],
                    fill='tozeroy', # Relleno de área
                    mode='lines+markers',
                    name='Equity Realizada',
                    line=dict(color='#00FFAA', width=3), # Verde Cripto
                    fillcolor='rgba(0, 255, 170, 0.1)' # Verde transparente
                ))
                
                # B) Línea Proyectada (Dotted)
                # Conectamos el último punto real con el punto proyectado (Trade N + 1)
                fig.add_trace(go.Scatter(
                    x=[last_trade_num, last_trade_num + 1],
                    y=[current_equity, projected_equity],
                    mode='lines+markers',
                    name='Proyección (Open PnL)',
                    line=dict(color='yellow', width=3, dash='dot'),
                    marker=dict(size=10, symbol='star')
                ))

                fig.update_layout(
                    title="Curva de Capital + Proyección Latente",
                    xaxis_title="Número de Trade",
                    yaxis_title="Capital Acumulado ($)",
                    template="plotly_dark",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Stats textuales
                c1, c2, c3 = st.columns(3)
                c1.metric("Equity Realizada", f"${current_equity:,.2f}")
                c2.metric("Floating PnL", f"${floating_pnl:,.2f}", delta=floating_pnl)
                c3.metric("Equity Proyectada", f"${projected_equity:,.2f}")
                
            else: st.info("Cierra trades para empezar a dibujar la curva.")
        else: st.warning("Sin datos.")

def main():
    if st.session_state['logged_in']: dashboard_page()
    else: login_page()

if __name__ == '__main__': main()
