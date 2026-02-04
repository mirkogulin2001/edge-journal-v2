import streamlit as st
import pandas as pd
import database as db
import auth
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import yfinance as yf
from datetime import date, datetime, timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Edge Journal", page_icon="📓", layout="wide")

st.markdown("""
<style>
div[data-testid="stMetricValue"] { font-size: 18px !important; }
div[data-testid="stMetricLabel"] { font-size: 12px !important; }
.stButton button { width: 100%; }
</style>
""", unsafe_allow_html=True)

CUSTOM_TEAL_PALETTE = [
    "#00897B", "#00ACC1", "#26A69A", "#4DD0E1", 
    "#80DEEA", "#00695C", "#00838F", "#004D40", 
    "#006064", "#1DE9B6", "#00BFA5", "#A7FFEB"
]

GRID_STYLE = dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.15)')

db.init_db()

# --- SESIÓN ---
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False
if 'username' not in st.session_state: st.session_state['username'] = None
if 'user_name' not in st.session_state: st.session_state['user_name'] = None
if 'strategy_config' not in st.session_state: st.session_state['strategy_config'] = {}

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
                if user_data:
                    if auth.check_password(password, user_data[1]):
                        st.session_state['logged_in'] = True
                        st.session_state['username'] = username
                        st.session_state['user_name'] = user_data[2]
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
            if not st.session_state.get('editing_config', False):
                raw_config = user_info[5] if len(user_info) > 5 else {}
                st.session_state['strategy_config'] = raw_config if isinstance(raw_config, dict) else (json.loads(raw_config) if raw_config else {})
        except: current_balance = 10000.0
        
        new_bal = st.number_input("Capital Inicial ($)", value=current_balance, step=1000.0)
        if new_bal != current_balance:
            db.update_initial_balance(st.session_state['username'], new_bal); st.rerun()

        st.divider()
        with st.expander("📥 Importar Historial (Excel)", expanded=True):
            st.caption("Sube tu archivo .xlsx o .csv")
            uploaded_file = st.file_uploader("Arrastra aquí", type=['xlsx', 'csv'])
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'): df_import = pd.read_csv(uploaded_file)
                    else: df_import = pd.read_excel(uploaded_file)
                    st.success(f"Archivo leído: {len(df_import)} filas.")
                    if st.button("Procesar e Importar Ahora"):
                        if db.import_batch_trades(st.session_state['username'], df_import):
                            st.balloons(); st.success(f"¡Listo! {len(df_import)} operaciones importadas."); time.sleep(2); st.rerun()
                except Exception as e: st.error(f"Error: {e}")
        st.divider()
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False; st.rerun()
        st.caption("Edge Journal v16.4 Labels")

    st.title("Gestión de Cartera 🏦")
    tab_active, tab_history, tab_stats, tab_performance, tab_config = st.tabs(["⚡ Posiciones", "📚 Historial", "📊 Analytics", "📈 Performance", "⚙️ Estrategia"])

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
                st.markdown("---")
                st.caption("Estrategia")
                selected_tags = {}
                config = st.session_state.get('strategy_config', {})
                if config:
                    dc1, dc2 = st.columns(2)
                    for i, (category, options) in enumerate(config.items()):
                        col = dc1 if i % 2 == 0 else dc2
                        with col:
                            if isinstance(options, str): options = [x.strip() for x in options.split(',')]
                            valid_options = [str(o) for o in options if str(o).strip() != ""]
                            if valid_options: val = st.selectbox(category, valid_options); selected_tags[category] = val
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
                    pnl_latente = (cp - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - cp) * row['quantity']
                    prices.append(cp); pnls.append(pnl_latente)
                    prog.progress((i+1)/len(df_open))
                prog.empty()
                df_open['Price'] = prices; df_open['Floating PnL'] = pnls
                df_open['Estrategia'] = df_open['tags'].apply(lambda x: " | ".join([f"{k}:{v}" for k,v in (json.loads(x) if isinstance(x, str) else (x if x else {})).items()]))
                
                st.dataframe(df_open.drop(columns=['id','notes','initial_stop_loss','tags','initial_quantity', 'partial_realized_pnl']), use_container_width=True, hide_index=True,
                             column_config={"entry_price":st.column_config.NumberColumn("In",format="$%.2f"),
                                            "Price":st.column_config.NumberColumn("Now",format="$%.2f"),
                                            "Floating PnL":st.column_config.NumberColumn("PnL Lat.",format="$%.2f"),
                                            "quantity":st.column_config.NumberColumn("Qty", format="%d")})
                
                total_floating = sum(pnls)
                total_partial_banked = df_open['partial_realized_pnl'].fillna(0).sum()
                k1, k2 = st.columns(2)
                k1.metric("PnL Latente (Abierto)", f"${total_floating:,.2f}", delta=total_floating)
                k2.metric("PnL Realizado (Parciales)", f"${total_partial_banked:,.2f}", delta=total_partial_banked, help="Dinero ya cobrado de operaciones que siguen abiertas.")
                st.divider()
                
                df_open['label'] = df_open.apply(lambda x: f"#{x['id']} {x['symbol']} (Q: {x['quantity']})", axis=1)
                sel = st.selectbox("Seleccionar Operación:", df_open['label'])
                sel_id = int(sel.split("#")[1].split(" ")[0])
                row = df_open[df_open['id'] == sel_id].iloc[0]
                
                t1, t2, t3, t4 = st.tabs(["Cerrar TOTAL", "✂️ Cierre PARCIAL", "Ajustar SL", "Borrar"])
                with t1:
                    with st.form("close"):
                        c_ex1, c_ex2 = st.columns(2)
                        ex_p = c_ex1.number_input("Precio Salida Final", value=float(row['Price']), format="%.2f")
                        ex_d = c_ex2.date_input("Fecha", value=date.today())
                        tentative_pnl = (ex_p - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - ex_p) * row['quantity']
                        total_pnl_preview = tentative_pnl + (row['partial_realized_pnl'] or 0)
                        default_idx = 0 if total_pnl_preview > 0 else 1 
                        res_type = st.radio("Clasificación Global", ["WIN", "LOSS", "BE"], index=default_idx, horizontal=True)
                        if st.form_submit_button("Confirmar Cierre Total"):
                            db.close_trade(sel_id, ex_p, ex_d, float(tentative_pnl), res_type)
                            st.success("Operación finalizada y consolidada."); time.sleep(1); st.rerun()
                with t2:
                    with st.form("partial"):
                        max_qty = int(row['quantity'])
                        st.write(f"Tienes **{max_qty}** acciones.")
                        c_p1, c_p2 = st.columns(2)
                        qty_partial = c_p1.number_input("Cantidad a vender", min_value=1, max_value=max_qty, value=min(1, max_qty))
                        price_partial = c_p2.number_input("Precio de venta", value=float(row['Price']), format="%.2f")
                        pnl_chunk = (price_partial - row['entry_price']) * qty_partial if row['side'] == 'LONG' else (row['entry_price'] - price_partial) * qty_partial
                        st.caption(f"💰 PnL de este parcial: **${pnl_chunk:,.2f}**")
                        if st.form_submit_button("Ejecutar Parcial"):
                            if qty_partial == max_qty: st.warning("⚠️ Para cerrar todo, usa la pestaña 'Cerrar TOTAL'.")
                            else:
                                if db.execute_partial_close(sel_id, qty_partial, price_partial, float(pnl_chunk)):
                                    st.success(f"Vendidas {qty_partial} acciones."); time.sleep(1); st.rerun()
                with t3:
                    with st.form("sl_upd"):
                        n_sl = st.number_input("Nuevo SL", value=float(row['current_stop_loss']), format="%.2f")
                        if st.form_submit_button("Actualizar"):
                            db.update_stop_loss(sel_id, n_sl); st.success("Listo"); time.sleep(1); st.rerun()
                with t4:
                    if st.button("Eliminar"): db.delete_trade(sel_id); st.rerun()
            else: st.info("Sin posiciones.")

    # --- TAB 2: HISTORIAL ---
    with tab_history:
        st.subheader("📚 Bitácora de Operaciones")
        df_c = db.get_closed_trades(st.session_state['username'])
        
        with st.expander("🛠️ Gestionar Registros (Borrar)", expanded=False):
            col_single, col_nuke = st.columns([2, 1])
            with col_single:
                st.markdown("##### 🗑️ Borrar un Trade")
                if not df_c.empty:
                    del_sel = st.selectbox("Seleccionar:", df_c.apply(lambda x: f"#{x['id']} {x['symbol']} (${x['pnl']:.0f})", axis=1))
                    if st.button("Borrar Seleccionado"): 
                        trade_id_to_del = int(del_sel.split("#")[1].split(" ")[0])
                        db.delete_trade(trade_id_to_del); st.toast("Trade eliminado."); time.sleep(1); st.rerun()
                else: st.caption("No hay trades para borrar.")
            with col_nuke:
                st.markdown("##### ☢️ Zona Nuclear")
                confirm_nuke = st.checkbox("Confirmar borrado total")
                if st.button("BORRAR TODO", type="primary", disabled=not confirm_nuke):
                    if db.delete_all_trades(st.session_state['username']):
                        st.toast("🔥 Historial eliminado."); time.sleep(1); st.rerun()

        if not df_c.empty:
            df_c['tags_dict'] = df_c['tags'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
            with st.expander("🔍 Filtros Avanzados", expanded=False):
                f1, f2, f3 = st.columns(3)
                filter_ticker = f1.text_input("Ticker").upper()
                filter_side = f2.multiselect("Dirección", ["LONG", "SHORT"])
                filter_result = f3.multiselect("Resultado", ["WIN", "LOSS", "BE"])
                st.markdown("---")
                st.caption("Filtros de Estrategia")
                config = st.session_state.get('strategy_config', {})
                dynamic_filters = {}
                if config:
                    cols = st.columns(3)
                    for i, (key, options) in enumerate(config.items()):
                        with cols[i % 3]:
                            if isinstance(options, str): options = [x.strip() for x in options.split(',')]
                            selection = st.multiselect(f"{key}", options)
                            if selection: dynamic_filters[key] = selection

            if filter_ticker: df_c = df_c[df_c['symbol'].str.contains(filter_ticker)]
            if filter_side: df_c = df_c[df_c['side'].isin(filter_side)]
            if filter_result:
                if 'result_type' not in df_c.columns: df_c['result_type'] = df_c['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')
                else: df_c['result_type'] = df_c['result_type'].fillna('WIN')
                df_c = df_c[df_c['result_type'].isin(filter_result)]
            if dynamic_filters:
                for key, values in dynamic_filters.items():
                    df_c = df_c[df_c['tags_dict'].apply(lambda tags: tags.get(key) in values)]

            if not df_c.empty:
                filtered_pnl = df_c['pnl'].sum()
                filtered_count = len(df_c)
                filtered_wr = len(df_c[df_c['pnl']>0]) / filtered_count * 100
                st.info(f"🔎 **{filtered_count} trades** | PnL: **${filtered_pnl:,.2f}** | WR: **{filtered_wr:.1f}%**")
                r_list = []
                for i, r in df_c.iterrows():
                    try:
                        risk = abs(r['entry_price'] - r['initial_stop_loss'])
                        if risk == 0: risk = 0.01
                        r_list.append(r['pnl'] / (risk * r['quantity']))
                    except: r_list.append(0)
                df_c['R'] = r_list
                df_c['Estrategia'] = df_c['tags_dict'].apply(lambda x: " ".join([f"[{v}]" for k,v in x.items()]))
                st.dataframe(df_c.drop(columns=['id', 'tags', 'tags_dict']), use_container_width=True, hide_index=True,
                             column_config={"pnl": st.column_config.NumberColumn("PnL", format="$%.2f"), "R": st.column_config.NumberColumn("R", format="%.2fR"), "result_type": st.column_config.TextColumn("Res", width="small")})
            else: st.warning("Sin resultados.")
        else: st.write("Sin datos.")

    # --- TAB 3: ANALYTICS (VISUAL PRO + LABELS + MATH) ---
    with tab_stats:
        st.subheader("🧪 Análisis Cuantitativo")
        df_all = db.get_all_trades_for_analytics(st.session_state['username'])
        if not df_all.empty:
            df_closed = df_all[df_all['exit_price'] > 0].copy()
            df_open = df_all[(df_all['exit_price'].isna()) | (df_all['exit_price'] == 0)].copy()
            
            unrealized_pnl = 0.0
            total_partial_pnl_open = 0.0
            total_invested_cash = 0.0
            pie_data = []
            
            if not df_open.empty:
                for _, r in df_open.iterrows():
                    try: 
                        t = yf.Ticker(r['symbol'])
                        cp = t.fast_info['last_price'] or r['entry_price']
                    except: cp = r['entry_price']
                    floating = (cp - r['entry_price']) * r['quantity'] if r['side'] == 'LONG' else (r['entry_price'] - cp) * r['quantity']
                    unrealized_pnl += floating
                    if 'partial_realized_pnl' in r: total_partial_pnl_open += (r['partial_realized_pnl'] or 0.0)
                    market_val = cp * r['quantity']
                    total_invested_cash += (r['entry_price'] * r['quantity'])
                    pie_data.append({'Asset': r['symbol'], 'Value': market_val})

            if not df_closed.empty:
                df_closed = df_closed.sort_values('entry_date')
                if 'result_type' in df_closed.columns:
                    df_closed.loc[df_closed['result_type'].isna() & (df_closed['pnl'] > 0), 'result_type'] = 'WIN'
                    df_closed.loc[df_closed['result_type'].isna() & (df_closed['pnl'] <= 0), 'result_type'] = 'LOSS'
                else: df_closed['result_type'] = df_closed['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

                tot = len(df_closed); pnl_closed = df_closed['pnl'].sum()
                wins_df = df_closed[df_closed['result_type'] == 'WIN']
                losses_df = df_closed[df_closed['result_type'] == 'LOSS']
                be_df = df_closed[df_closed['result_type'] == 'BE']
                n_wins = len(wins_df); n_losses = len(losses_df); n_be = len(be_df)
                wr = n_wins / tot; lr = n_losses / tot; be_rate = n_be / tot
                avg_w = wins_df['pnl'].mean() if n_wins > 0 else 0
                avg_l = abs(losses_df['pnl'].mean()) if n_losses > 0 else 0
                
                df_closed['cum_pnl'] = df_closed['pnl'].cumsum()
                df_closed['equity'] = current_balance + df_closed['cum_pnl']
                df_closed['peak'] = df_closed['equity'].cummax()
                df_closed['dd_pct'] = ((df_closed['equity'] - df_closed['peak']) / df_closed['peak']) * 100
                max_dd = df_closed['dd_pct'].min()
                current_dd = df_closed['dd_pct'].iloc[-1] if not df_closed.empty else 0

                st.markdown("#### 🎯 KPIs Matrix")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Ops", tot); k2.metric("Win%", f"{wr*100:.0f}%"); k3.metric("Loss%", f"{lr*100:.0f}%"); k4.metric("BE%", f"{be_rate*100:.0f}%")
                
                k5, k6, k7, k8 = st.columns(4)
                total_banked = pnl_closed + total_partial_pnl_open
                k5.metric("PnL Realizado", f"${total_banked:,.0f}"); k6.metric("Avg Win", f"${avg_w:,.0f}"); k7.metric("Avg Loss", f"${avg_l:,.0f}"); k8.metric("ROI", f"{(total_banked/current_balance)*100:.1f}%")
                
                k9, k10, k11, k12 = st.columns(4)
                payoff = (avg_w / avg_l) if avg_l > 0 else 0
                
                # --- NUEVO CÁLCULO E(MATH) NORMALIZADO (ABSOLUTO) ---
                # E = (Win% * Payoff) - Loss%
                e_math_abs = (wr * payoff) - lr
                
                k9.metric("E(Math)", f"{e_math_abs:.2f}") # Sin signo $
                k10.metric("Payoff Ratio", f"{payoff:.2f}")
                k11.metric("Max Drawdown", f"{max_dd:.2f}%", delta=max_dd, delta_color="inverse")
                k12.metric("Current DD", f"{current_dd:.2f}%")

                st.markdown("---")
                
                c_main, c_side = st.columns([2, 1])
                
                with c_main:
                    seed_row = pd.DataFrame([{'trade_num': 0, 'equity': current_balance, 'dd_pct': 0}])
                    df_chart = pd.concat([seed_row, df_closed[['equity', 'dd_pct']]], ignore_index=True)
                    df_chart['trade_num'] = range(len(df_chart))
                    
                    fig = px.area(df_chart, x='trade_num', y='equity', title="🚀 Equity Curve")
                    fig.update_traces(line_color='#00FFFF', line_width=2, fillcolor='rgba(0, 255, 255, 0.15)')
                    fig.update_xaxes(**GRID_STYLE)
                    
                    min_y = df_chart['equity'].min() * 0.99
                    max_y = df_chart['equity'].max() * 1.01
                    if max_y - min_y < 100: min_y -= 50; max_y += 50
                    fig.update_yaxes(range=[min_y, max_y], **GRID_STYLE)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    fig_dd = px.area(df_chart, x='trade_num', y='dd_pct', title="📉 Drawdown Under Water")
                    fig_dd.update_traces(line_color='#FF4B4B', line_width=1, fillcolor='rgba(255, 75, 75, 0.2)')
                    fig_dd.update_xaxes(**GRID_STYLE); fig_dd.update_yaxes(**GRID_STYLE)
                    st.plotly_chart(fig_dd, use_container_width=True)

                with c_side:
                    # HISTOGRAMA (BINS DINÁMICOS)
                    fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
                    pnl_data = df_closed['pnl'].dropna()
                    
                    if len(pnl_data) > 1:
                        try:
                            kde = stats.gaussian_kde(pnl_data)
                            x_grid = np.linspace(pnl_data.min(), pnl_data.max(), 200)
                            y_kde = kde(x_grid)
                            fig_hist.add_trace(go.Scatter(x=x_grid, y=y_kde, mode='lines', line=dict(color='rgba(0, 150, 255, 0.4)', width=2), fill='tozeroy', fillcolor='rgba(0, 150, 255, 0.1)', name='Teórica'), secondary_y=True)
                        except: pass

                    # CALCULO DE BINS (SQRT RULE)
                    optimal_bins = int(np.sqrt(len(df_closed))) if not df_closed.empty else 10
                    # Forzamos un mínimo de bins para que no se vea feo si hay pocos datos
                    if optimal_bins < 5: optimal_bins = 5

                    # Definimos el rango total para calcular el ancho del bin
                    data_range = pnl_data.max() - pnl_data.min()
                    bin_size = data_range / optimal_bins if data_range > 0 else 10

                    be_data = pnl_data[(pnl_data >= -1) & (pnl_data <= 1)]
                    loss_data = pnl_data[pnl_data < -1]
                    win_data = pnl_data[pnl_data > 1]

                    fig_hist.add_trace(go.Histogram(x=win_data, marker_color='#00FFAA', marker_line_color='black', marker_line_width=1, opacity=0.85, name='WIN', xbins=dict(start=1, size=bin_size)), secondary_y=False)
                    fig_hist.add_trace(go.Histogram(x=loss_data, marker_color='#FF4B4B', marker_line_color='black', marker_line_width=1, opacity=0.85, name='LOSS', xbins=dict(end=-1, size=bin_size)), secondary_y=False)
                    fig_hist.add_trace(go.Histogram(x=be_data, marker_color='#AAAAAA', marker_line_color='black', marker_line_width=1, opacity=0.85, name='BE', xbins=dict(start=-1, end=1, size=2)), secondary_y=False)

                    fig_hist.update_layout(title="🔔 Distribución PnL", barmode='overlay', height=350, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
                    fig_hist.update_xaxes(**GRID_STYLE); fig_hist.update_yaxes(secondary_y=False, **GRID_STYLE); fig_hist.update_yaxes(secondary_y=True, showgrid=False, showticklabels=True)
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # PIE CHART (LABELS ACTIVOS)
                    current_cash = (current_balance + total_banked) - total_invested_cash
                    if current_cash < 0: current_cash = 0
                    pie_data.append({'Asset': 'CASH', 'Value': current_cash})
                    
                    fig_pie = px.pie(pd.DataFrame(pie_data), values='Value', names='Asset', title="🍰 Asignación Actual", hole=0.4, color_discrete_sequence=CUSTOM_TEAL_PALETTE)
                    # AQUÍ ESTÁ EL CAMBIO: textposition='inside', textinfo='percent+label'
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    fig_pie.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)

            else: st.info("Cierra operaciones para ver métricas.")
        else: st.warning("Sin datos.")

    # --- TAB 4: PERFORMANCE ---
    with tab_performance:
        st.subheader("📈 Rendimiento Temporal (%)")
        time_filters = ["Todo", "YTD (Este Año)", "Año Anterior"]
        selected_filter = st.radio("Rango:", time_filters, index=0, horizontal=True)
        df_all = db.get_closed_trades(st.session_state['username'])
        if not df_all.empty:
            df_perf = df_all.copy()
            df_perf['exit_date'] = pd.to_datetime(df_perf['exit_date'])
            daily_pnl = df_perf.groupby('exit_date')['pnl'].sum()
            overall_start = daily_pnl.index.min()
            today = pd.Timestamp(date.today())
            full_history_range = pd.date_range(start=overall_start, end=today)
            daily_pnl_reindexed = daily_pnl.reindex(full_history_range).fillna(0)
            equity_curve_series = current_balance + daily_pnl_reindexed.cumsum()
            
            start_date_filter = overall_start; end_date_filter = today
            if selected_filter == "YTD (Este Año)": start_date_filter = pd.Timestamp(date(today.year, 1, 1))
            elif selected_filter == "Año Anterior": start_date_filter = pd.Timestamp(date(today.year - 1, 1, 1)); end_date_filter = pd.Timestamp(date(today.year - 1, 12, 31))
            if start_date_filter < overall_start: start_date_filter = overall_start

            sliced_equity = equity_curve_series[(equity_curve_series.index >= start_date_filter) & (equity_curve_series.index <= end_date_filter)]
            if not sliced_equity.empty:
                base_value = sliced_equity.iloc[0] 
                pct_change_series = ((sliced_equity - base_value) / base_value) * 100
                df_chart_time = pd.DataFrame({'Date': pct_change_series.index, 'Return': pct_change_series.values})
                fig_ts = px.line(df_chart_time, x='Date', y='Return', title=f"Retorno {selected_filter}")
                fig_ts.update_traces(line_color='#00BFA5', line_width=3, hovertemplate='%{x}<br>Retorno: %{y:.2f}%')
                fig_ts.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
                fig_ts.update_layout(yaxis_title="Retorno (%)", yaxis_tickformat=".2f%", xaxis_title="Fecha", height=450, showlegend=False)
                fig_ts.update_xaxes(**GRID_STYLE); fig_ts.update_yaxes(**GRID_STYLE)
                st.plotly_chart(fig_ts, use_container_width=True)
                total_return = pct_change_series.iloc[-1]
                abs_pnl = sliced_equity.iloc[-1] - sliced_equity.iloc[0]
                st.info(f"📅 **Periodo:** {selected_filter} | **Retorno:** {total_return:.2f}% | **PnL:** ${abs_pnl:,.2f}")
            else: st.warning(f"No hay datos para el periodo {selected_filter}")
        else: st.info("Necesitas cerrar operaciones para ver la evolución temporal.")

    # --- TAB 5: CONFIGURACIÓN SIMPLE ---
    with tab_config:
        st.subheader("⚙️ Configuración de Estrategia")
        current_config = st.session_state.get('strategy_config', {})
        if not current_config: current_config = {"Setup": ["SUP", "SS"], "Grado": ["Mayor", "Menor"]}
        data_list = []
        for k, v in current_config.items():
            opts_str = ", ".join(v) if isinstance(v, list) else str(v)
            data_list.append({"Parámetro": k, "Opciones (separadas por coma)": opts_str})
        df_config = pd.DataFrame(data_list)
        edited_df = st.data_editor(df_config, num_rows="dynamic", use_container_width=True, key="master_config_editor")
        if st.button("💾 Guardar Toda la Configuración", type="primary"):
            new_config_dict = {}
            for index, row in edited_df.iterrows():
                param_name = str(row.get("Parámetro", "")).strip()
                opts_raw = str(row.get("Opciones (separadas por coma)", ""))
                if param_name:
                    opts_list = [x.strip() for x in opts_raw.split(',') if x.strip()]
                    new_config_dict[param_name] = opts_list
            if db.update_strategy_config(st.session_state['username'], new_config_dict):
                st.session_state['strategy_config'] = new_config_dict
                st.success("¡Configuración actualizada correctamente!"); time.sleep(1); st.rerun()
            else: st.error("Hubo un error al guardar.")

def main():
    if st.session_state['logged_in']: dashboard_page()
    else: login_page()

if __name__ == '__main__': main()
