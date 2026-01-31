import streamlit as st
import pandas as pd
import database as db
import auth
import time
import json
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Edge Journal", page_icon="📓", layout="wide")

# CSS: Achicar KPIs y Ajustar Editor
st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 18px !important; }
div[data-testid="stMetricLabel"] { font-size: 12px !important; }
div[data-testid="stDataEditor"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

db.init_db()

# --- SESIÓN ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = None
if 'user_name' not in st.session_state: st.session_state['user_name'] = None
if 'strategy_config' not in st.session_state: st.session_state['strategy_config'] = {}

# Variable nueva para mantener la tabla en memoria mientras editas
if 'config_df_draft' not in st.session_state: st.session_state['config_df_draft'] = None

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
                    if auth.check_password(password, user_data[1]):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['user_name'] = user_data[2]
                        # Cargar config
                        try:
                            config = user_data[5]
                            if isinstance(config, str): config = json.loads(config)
                            st.session_state['strategy_config'] = config
                        except: st.session_state['strategy_config'] = {}
                        st.rerun()
                    else: st.error("Pass incorrecta")
                else: st.error("Usuario no encontrado")

        with tab2:
            new_user = st.text_input("Nuevo Usuario")
            new_name = st.text_input("Nombre Real")
            new_pass = st.text_input("Contraseña", type="password")
            if st.button("Crear Cuenta"):
                if db.create_user(new_user, auth.hash_password(new_pass), new_name):
                    st.success("Creado!"); st.rerun()
                else: st.error("Error al crear")

# --- DASHBOARD ---
def dashboard_page():
    with st.sidebar:
        st.header(f"Hola, {st.session_state['user_name']}")
        
        user_info = db.get_user(st.session_state['username'])
        try: 
            current_balance = float(user_info[4]) if user_info and len(user_info) > 4 and user_info[4] else 10000.0
            # IMPORTANTE: Solo actualizamos la config DESDE LA DB si no estamos editando
            # Pero para el uso general, refrescamos la variable local
            raw_config = user_info[5] if len(user_info) > 5 else {}
            st.session_state['strategy_config'] = raw_config if isinstance(raw_config, dict) else (json.loads(raw_config) if raw_config else {})
        except: current_balance = 10000.0
        
        new_bal = st.number_input("Capital Inicial ($)", value=current_balance, step=1000.0)
        if new_bal != current_balance:
            db.update_initial_balance(st.session_state['username'], new_bal); st.rerun()

        st.divider()
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.session_state['config_df_draft'] = None # Limpiar borrador al salir
            st.rerun()
        st.caption("Edge Journal v7.0 Sticky")

    st.title("Gestión de Cartera 🏦")
    tab_active, tab_history, tab_stats, tab_config = st.tabs(["⚡ Posiciones", "📚 Historial", "📊 Analytics", "⚙️ Estrategia"])

    # --- TAB 1: OPERATIVA ---
    with tab_active:
        col_left, col_right = st.columns([1, 2])
        with col_left:
            st.subheader("➕ Nueva Orden")
            with st.form("new_trade"):
                c1, c2 = st.columns(2)
                symbol = c1.text_input("Ticker").upper()
                side = c2.selectbox("Side", ["LONG", "SHORT"])
                
                c3, c4 = st.columns(2)
                price = c3.number_input("Precio Entrada", min_value=0.0, format="%.2f")
                qty = c4.number_input("Cantidad", min_value=1, step=1)
                
                # --- MENÚS DINÁMICOS ---
                st.markdown("---")
                st.caption("Estrategia")
                selected_tags = {}
                config = st.session_state.get('strategy_config', {})
                
                if config:
                    dc1, dc2 = st.columns(2)
                    for i, (category, options) in enumerate(config.items()):
                        col = dc1 if i % 2 == 0 else dc2
                        with col:
                            valid_options = [o for o in options if o and str(o).strip() != ""]
                            if valid_options:
                                val = st.selectbox(category, valid_options)
                                selected_tags[category] = val
                else: st.info("Ve a la pestaña '⚙️ Estrategia'.")

                st.markdown("---")
                sl_val = st.number_input("Stop Loss Inicial ($)", min_value=0.0, format="%.2f")
                date_in = st.date_input("Fecha", value=date.today())
                notes = st.text_area("Tesis")
                
                if st.form_submit_button("🚀 Ejecutar", type="primary"):
                    if symbol and price > 0:
                        db.open_new_trade(st.session_state['username'], symbol, side, price, qty, date_in, notes, sl_val, sl_val, selected_tags)
                        st.success("Orden enviada."); time.sleep(0.5); st.rerun()

        with col_right:
            st.subheader("📡 Gestión Activa")
            df_open = db.get_open_trades(st.session_state['username'])
            if not df_open.empty:
                prices, pnls = [], []
                prog = st.progress(0)
                for i, row in df_open.iterrows():
                    try:
                        t = yf.Ticker(row['symbol'])
                        cp = t.fast_info['last_price'] or t.history(period='1d')['Close'].iloc[-1]
                    except: cp = row['entry_price']
                    pnl = (cp - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - cp) * row['quantity']
                    prices.append(cp); pnls.append(pnl)
                    prog.progress((i+1)/len(df_open))
                prog.empty()
                df_open['Price'] = prices; df_open['Floating PnL'] = pnls
                
                df_open['Estrategia'] = df_open['tags'].apply(lambda x: " | ".join([f"{k}:{v}" for k,v in (json.loads(x) if isinstance(x, str) else (x if x else {})).items()]))
                
                st.dataframe(df_open.drop(columns=['id','notes','initial_stop_loss','tags']), use_container_width=True, hide_index=True,
                             column_config={"entry_price":st.column_config.NumberColumn("In",format="$%.2f"),
                                            "Price":st.column_config.NumberColumn("Now",format="$%.2f"),
                                            "Floating PnL":st.column_config.NumberColumn("PnL",format="$%.2f")})
                
                st.metric("PnL Latente", f"${sum(pnls):,.2f}", delta=sum(pnls))
                st.divider()
                
                df_open['label'] = df_open.apply(lambda x: f"#{x['id']} {x['symbol']} | PnL: ${x['Floating PnL']:.0f}", axis=1)
                sel = st.selectbox("Seleccionar:", df_open['label'])
                sel_id = int(sel.split("#")[1].split(" ")[0])
                row = df_open[df_open['id'] == sel_id].iloc[0]
                
                t1, t2, t3 = st.tabs(["Cerrar", "Ajustar SL", "Borrar"])
                with t1:
                    with st.form("close"):
                        ex_p = st.number_input("Salida", value=float(row['Price']), format="%.2f")
                        ex_d = st.date_input("Fecha", value=date.today())
                        if st.form_submit_button("Confirmar"):
                            r_pnl = (ex_p - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - ex_p) * row['quantity']
                            db.close_trade(sel_id, ex_p, ex_d, float(r_pnl)); st.success("Cerrado!"); time.sleep(1); st.rerun()
                with t2:
                    with st.form("sl_upd"):
                        n_sl = st.number_input("Nuevo SL", value=float(row['current_stop_loss']), format="%.2f")
                        if st.form_submit_button("Actualizar"):
                            db.update_stop_loss(sel_id, n_sl); st.success("Listo"); time.sleep(1); st.rerun()
                with t3:
                    if st.button("Eliminar"): db.delete_trade(sel_id); st.rerun()
            else: st.info("Sin posiciones.")

    # --- TAB 2: HISTORIAL ---
    with tab_history:
        st.subheader("📚 Bitácora")
        df_c = db.get_closed_trades(st.session_state['username'])
        if not df_c.empty:
            r_list = []
            for i, r in df_c.iterrows():
                try:
                    risk = abs(r['entry_price'] - r['initial_stop_loss'])
                    if risk == 0: risk = 0.01
                    r_list.append(r['pnl'] / (risk * r['quantity']))
                except: r_list.append(0)
            df_c['R'] = r_list
            df_c['Estrategia'] = df_c['tags'].apply(lambda x: " ".join([f"[{v}]" for k,v in (json.loads(x) if isinstance(x, str) else (x if x else {})).items()]))
            st.dataframe(df_c.drop(columns=['id', 'tags']), use_container_width=True, hide_index=True,
                         column_config={"pnl": st.column_config.NumberColumn("PnL", format="$%.2f"), "R": st.column_config.NumberColumn("R", format="%.2fR")})
            
            with st.expander("Eliminar"):
                del_sel = st.selectbox("Elegir:", df_c.apply(lambda x: f"#{x['id']} {x['symbol']} (${x['pnl']:.0f})", axis=1))
                if st.button("Borrar"): db.delete_trade(int(del_sel.split("#")[1].split(" ")[0])); st.rerun()
        else: st.write("Sin datos.")

    # --- TAB 3: ANALYTICS ---
    with tab_stats:
        st.subheader("🧪 Análisis Cuantitativo")
        df_all = db.get_all_trades_for_analytics(st.session_state['username'])
        if not df_all.empty:
            df_closed = df_all[df_all['exit_price'] > 0].copy()
            df_open = df_all[(df_all['exit_price'].isna()) | (df_all['exit_price'] == 0)].copy()
            unrealized_pnl = 0.0; worst_case_pnl = 0.0; num_open_trades = 0; pie_data = []; total_invested_cash = 0.0
            if not df_open.empty:
                num_open_trades = len(df_open)
                for _, r in df_open.iterrows():
                    try: 
                        t = yf.Ticker(r['symbol'])
                        cp = t.fast_info['last_price'] or r['entry_price']
                        market_val = cp * r['quantity']
                        unrealized_pnl += (market_val - (r['entry_price'] * r['quantity'])) if r['side'] == 'LONG' else ((r['entry_price'] - cp) * r['quantity'])
                    except: cp = r['entry_price']; market_val = cp * r['quantity']
                    sl = r['current_stop_loss'] if r['current_stop_loss'] > 0 else r['entry_price']
                    wc_val = (sl - r['entry_price']) * r['quantity'] if r['side'] == 'LONG' else (r['entry_price'] - sl) * r['quantity']
                    worst_case_pnl += wc_val
                    total_invested_cash += (r['entry_price'] * r['quantity'])
                    pie_data.append({'Asset': r['symbol'], 'Value': market_val})

            if not df_closed.empty:
                df_closed = df_closed.sort_values('entry_date')
                df_closed['trade_num'] = range(1, len(df_closed) + 1)
                tot = len(df_closed); pnl_tot = df_closed['pnl'].sum()
                wins = df_closed[df_closed['pnl'] > 0]; losses = df_closed[df_closed['pnl'] <= 0]
                wr = len(wins)/tot; lr = len(losses)/tot
                avg_w = wins['pnl'].mean() if len(wins)>0 else 0; avg_l = losses['pnl'].mean() if len(losses)>0 else 0
                df_closed['cum_pnl'] = df_closed['pnl'].cumsum()
                df_closed['equity'] = current_balance + df_closed['cum_pnl']
                df_closed['peak'] = df_closed['equity'].cummax()
                df_closed['dd_pct'] = ((df_closed['equity'] - df_closed['peak']) / df_closed['peak']) * 100
                max_dd = df_closed['dd_pct'].min()
                seed_row = pd.DataFrame([{'trade_num': 0, 'equity': current_balance, 'dd_pct': 0}])
                df_chart = pd.concat([seed_row, df_closed[['trade_num', 'equity', 'dd_pct']]], ignore_index=True)

                kpis, charts = st.columns([1.3, 2])
                with kpis:
                    st.markdown("#### 🎯 KPIs Matrix")
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Ops", tot); k2.metric("Win%", f"{wr*100:.0f}%")
                    k3.metric("Loss%", f"{lr*100:.0f}%"); k4.metric("PnL", f"${pnl_tot:,.0f}")
                    k5, k6, k7, k8 = st.columns(4)
                    k5.metric("ROI", f"{(pnl_tot/current_balance)*100:.1f}%")
                    k6.metric("Open", f"${unrealized_pnl:,.0f}")
                    k7.metric("Win$", f"${avg_w:,.0f}"); k8.metric("Loss$", f"${avg_l:,.0f}")
                    k9, k10, k11, k12 = st.columns(4)
                    payoff = abs(avg_w/avg_l) if avg_l!=0 else 0; e_math = (wr * payoff) - lr
                    k9.metric("E(Math)", f"{e_math:.2f}"); k10.metric("Payoff", f"{payoff:.1f}")
                    k11.metric("Risk(SL)", f"${worst_case_pnl:,.0f}", delta=worst_case_pnl, delta_color="inverse")
                    k12.metric("MaxDD", f"{max_dd:.1f}%")

                with charts:
                    y_min = current_balance
                    min_realized = df_chart['equity'].min()
                    if min_realized < y_min: y_min = min_realized
                    if not df_open.empty:
                        last_e = df_chart['equity'].iloc[-1]
                        if (last_e + unrealized_pnl) < y_min: y_min = last_e + unrealized_pnl
                        if (last_e + worst_case_pnl) < y_min: y_min = last_e + worst_case_pnl
                    fig = px.area(df_chart, x='trade_num', y='equity', title="🚀 Equity Curve", labels={'trade_num':'#', 'equity':'$'})
                    fig.update_traces(line_color='#00FFFF', line_width=2, fillcolor='rgba(0, 255, 255, 0.15)')
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis_range=[y_min * 0.995, None])
                    if not df_open.empty:
                        last_n = df_chart['trade_num'].iloc[-1]; last_e = df_chart['equity'].iloc[-1]; target_n = last_n + num_open_trades
                        proj_equity = last_e + unrealized_pnl; risk_equity = last_e + worst_case_pnl
                        fig.add_trace(go.Scatter(x=[last_n, target_n], y=[last_e, proj_equity], mode='lines+markers', name='Equity Unrealized', line=dict(color='#008B8B', dash='dot', width=1)))
                        fig.add_trace(go.Scatter(x=[last_n, target_n], y=[last_e, risk_equity], mode='lines+markers', name='Riesgo (SL)', line=dict(color='#FF4B4B', dash='dot', width=1)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c_pie, c_dist = st.columns(2)
                    with c_pie:
                        pie_data.append({'Asset': 'CASH', 'Value': max(0, (current_balance + pnl_tot) - total_invested_cash)})
                        fig_pie = px.pie(pd.DataFrame(pie_data), values='Value', names='Asset', title="🍰 Asignación", hole=0.4)
                        fig_pie.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with c_dist:
                        fig_hist = px.histogram(df_closed, x="pnl", nbins=20, title="🔔 Distribución")
                        fig_hist.update_traces(marker_color='#9B59B6', opacity=0.8)
                        fig_hist.add_vline(x=df_closed['pnl'].mean(), line_dash="dash", line_color="yellow")
                        fig_hist.add_vline(x=0, line_dash="solid", line_color="white", opacity=0.5)
                        fig_hist.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                        st.plotly_chart(fig_hist, use_container_width=True)

                    fig_dd = px.area(df_chart, x='trade_num', y='dd_pct', title="📉 Drawdown")
                    fig_dd.update_traces(line_color='#FF4B4B', line_width=2, fillcolor='rgba(255, 75, 75, 0.2)', name='Drawdown', showlegend=True)
                    fig_dd.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0), showlegend=True)
                    st.plotly_chart(fig_dd, use_container_width=True)
            else: st.info("Cierra operaciones para ver métricas.")
        else: st.warning("Sin datos.")

    # ------------------------------------------------------------------
    # TAB 4: CONFIGURACIÓN (CON MEMORIA TEMPORAL) ⚙️
    # ------------------------------------------------------------------
    with tab_config:
        st.subheader("⚙️ Editor de Estrategia")
        st.markdown("Define tus Setups. **Guarda los cambios** al finalizar.")
        
        # 1. INICIALIZAR BORRADOR (Solo si está vacío)
        if st.session_state['config_df_draft'] is None:
            # Traer de DB
            current_config = st.session_state.get('strategy_config', {})
            # Convertir a DataFrame rellenando huecos
            if current_config:
                max_len = max([len(v) for v in current_config.values()]) if current_config else 0
                padded_data = {k: v + [""] * (max_len - len(v)) for k, v in current_config.items()}
                st.session_state['config_df_draft'] = pd.DataFrame(padded_data)
            else:
                st.session_state['config_df_draft'] = pd.DataFrame({
                    "Setup": ["Principal (SUP)", "Secundario (SS)", ""],
                    "Grado": ["Mayor", "Menor", ""]
                })

        # 2. SECCIÓN AGREGAR COLUMNA (Modifica el Borrador)
        with st.expander("➕ Agregar Nueva Columna"):
            c1, c2 = st.columns([3, 1])
            new_col = c1.text_input("Nombre Columna")
            if c2.button("Agregar"):
                if new_col and new_col not in st.session_state['config_df_draft'].columns:
                    st.session_state['config_df_draft'][new_col] = "" # Agregar al borrador
                    st.rerun() # Recargar para mostrar en la tabla de abajo

        # 3. EDITOR (Muestra y Edita el Borrador)
        # Importante: Asignamos el resultado de vuelta al state
        edited_df = st.data_editor(st.session_state['config_df_draft'], num_rows="dynamic", use_container_width=True)
        # Actualizamos el state con lo que el usuario escribe para que no se pierda al recargar
        st.session_state['config_df_draft'] = edited_df

        # 4. GUARDAR (Persistir a DB)
        if st.button("💾 Guardar Cambios", type="primary"):
            new_config_dict = {}
            for col in st.session_state['config_df_draft'].columns:
                raw_val = st.session_state['config_df_draft'][col].dropna().astype(str).tolist()
                clean_val = [v.strip() for v in raw_val if v.strip() != ""]
                if clean_val: new_config_dict[col] = clean_val
            
            if db.update_strategy_config(st.session_state['username'], new_config_dict):
                st.session_state['strategy_config'] = new_config_dict # Actualizar global
                st.success("Guardado exitoso!")
                time.sleep(1); st.rerun()
            else: st.error("Error al guardar.")

        # Botón para resetear si metiste la pata
        if st.button("🔄 Descartar Cambios (Recargar de DB)"):
            st.session_state['config_df_draft'] = None
            st.rerun()

def main():
    if st.session_state['logged_in']: dashboard_page()
    else: login_page()

if __name__ == '__main__': main()
