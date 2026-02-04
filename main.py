import streamlit as st
import pandas as pd
import database as db
import auth
import time
import json
import plotly.express as px
import plotly.graph_objects as go
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

# Paleta Personalizada
CUSTOM_TEAL_PALETTE = [
    "#00897B", "#00ACC1", "#26A69A", "#4DD0E1", 
    "#80DEEA", "#00695C", "#00838F", "#004D40", 
    "#006064", "#1DE9B6", "#00BFA5", "#A7FFEB"
]

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
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.caption("Edge Journal v12.1 Schwab Style")

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
                        c_ex1, c_ex2 = st.columns(2)
                        ex_p = c_ex1.number_input("Salida", value=float(row['Price']), format="%.2f")
                        ex_d = c_ex2.date_input("Fecha", value=date.today())
                        
                        tentative_pnl = (ex_p - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - ex_p) * row['quantity']
                        default_idx = 0 if tentative_pnl > 0 else 1 
                        res_type = st.radio("Clasificación", ["WIN", "LOSS", "BE"], index=default_idx, horizontal=True)
                        
                        if st.form_submit_button("Confirmar Cierre"):
                            r_pnl = (ex_p - row['entry_price']) * row['quantity'] if row['side'] == 'LONG' else (row['entry_price'] - ex_p) * row['quantity']
                            db.close_trade(sel_id, ex_p, ex_d, float(r_pnl), res_type)
                            st.success("Cerrado!"); time.sleep(1); st.rerun()
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
        st.subheader("📚 Bitácora de Operaciones")
        df_c = db.get_closed_trades(st.session_state['username'])
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
                with st.expander("🛠️ Acciones"):
                    del_sel = st.selectbox("Eliminar:", df_c.apply(lambda x: f"#{x['id']} {x['symbol']} (${x['pnl']:.0f})", axis=1))
                    if st.button("Borrar Seleccionado"): db.delete_trade(int(del_sel.split("#")[1].split(" ")[0])); st.rerun()
            else: st.warning("Sin resultados.")
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
                
                if 'result_type' in df_closed.columns:
                    df_closed.loc[df_closed['result_type'].isna() & (df_closed['pnl'] > 0), 'result_type'] = 'WIN'
                    df_closed.loc[df_closed['result_type'].isna() & (df_closed['pnl'] <= 0), 'result_type'] = 'LOSS'
                else: df_closed['result_type'] = df_closed['pnl'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')

                tot = len(df_closed); pnl_tot = df_closed['pnl'].sum()
                wins_df = df_closed[df_closed['result_type'] == 'WIN']; losses_df = df_closed[df_closed['result_type'] == 'LOSS']; be_df = df_closed[df_closed['result_type'] == 'BE']
                n_wins = len(wins_df); n_losses = len(losses_df); n_be = len(be_df)
                wr = n_wins / tot; lr = n_losses / tot; be_rate = n_be / tot
                avg_w = wins_df['pnl'].mean() if n_wins > 0 else 0; avg_l = losses_df['pnl'].mean() if n_losses > 0 else 0
                
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
                    k1.metric("Ops", tot); k2.metric("Win%", f"{wr*100:.0f}%"); k3.metric("Loss%", f"{lr*100:.0f}%"); k4.metric("BE%", f"{be_rate*100:.0f}%")
                    k5, k6, k7, k8 = st.columns(4)
                    k5.metric("PnL", f"${pnl_tot:,.0f}"); k6.metric("Win$", f"${avg_w:,.0f}"); k7.metric("Loss$", f"${avg_l:,.0f}"); k8.metric("ROI", f"{(pnl_tot/current_balance)*100:.1f}%")
                    k9, k10, k11, k12 = st.columns(4)
                    payoff = abs(avg_w/avg_l) if avg_l!=0 else 0; e_math = (wr * payoff) - lr
                    k9.metric("E(Math)", f"{e_math:.2f}"); k10.metric("Payoff", f"{payoff:.1f}"); k11.metric("Risk(SL)", f"${worst_case_pnl:,.0f}", delta=worst_case_pnl, delta_color="inverse"); k12.metric("MaxDD", f"{max_dd:.1f}%")

                with charts:
                    all_y_values = df_chart['equity'].tolist()
                    if not df_open.empty:
                        last_e = df_chart['equity'].iloc[-1]
                        all_y_values.append(last_e + unrealized_pnl)
                        all_y_values.append(last_e + worst_case_pnl)
                    y_min_dynamic = min(all_y_values); y_max_dynamic = max(all_y_values)
                    range_diff = y_max_dynamic - y_min_dynamic
                    margin = range_diff * 0.05 if range_diff != 0 else y_max_dynamic * 0.01
                    final_min = y_min_dynamic - margin; final_max = y_max_dynamic + margin

                    fig = px.area(df_chart, x='trade_num', y='equity', title="🚀 Equity Curve (x Trade)", labels={'trade_num':'#', 'equity':'$'})
                    fig.update_traces(line_color='#00FFFF', line_width=2, fillcolor='rgba(0, 255, 255, 0.15)')
                    fig.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), yaxis_range=[final_min, final_max])
                    if not df_open.empty:
                        last_n = df_chart['trade_num'].iloc[-1]; last_e = df_chart['equity'].iloc[-1]; target_n = last_n + num_open_trades
                        proj_equity = last_e + unrealized_pnl; risk_equity = last_e + worst_case_pnl
                        fig.add_trace(go.Scatter(x=[last_n, target_n], y=[last_e, proj_equity], mode='lines+markers', name='Equity Unrealized', line=dict(color='#008B8B', dash='dot', width=1)))
                        fig.add_trace(go.Scatter(x=[last_n, target_n], y=[last_e, risk_equity], mode='lines+markers', name='Riesgo (SL)', line=dict(color='#FF4B4B', dash='dot', width=1)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    c_pie, c_dist = st.columns(2)
                    with c_pie:
                        pie_data.append({'Asset': 'CASH', 'Value': max(0, (current_balance + pnl_tot) - total_invested_cash)})
                        fig_pie = px.pie(pd.DataFrame(pie_data), values='Value', names='Asset', title="🍰 Asignación", hole=0.4, color_discrete_sequence=CUSTOM_TEAL_PALETTE)
                        fig_pie.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0))
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with c_dist:
                        color_map = {'WIN': '#00FFAA', 'LOSS': '#FF4B4B', 'BE': '#AAAAAA'}
                        fig_hist = px.histogram(df_closed, x="pnl", nbins=20, title="🔔 Distribución PnL", color="result_type", color_discrete_map=color_map)
                        fig_hist.update_layout(height=280, margin=dict(l=0,r=0,t=30,b=0), showlegend=True)
                        st.plotly_chart(fig_hist, use_container_width=True)

                    fig_dd = px.area(df_chart, x='trade_num', y='dd_pct', title="📉 Drawdown (x Trade)")
                    fig_dd.update_traces(line_color='#FF4B4B', line_width=2, fillcolor='rgba(255, 75, 75, 0.2)', name='Drawdown', showlegend=True)
                    fig_dd.update_layout(height=200, margin=dict(l=0,r=0,t=30,b=0), showlegend=True)
                    st.plotly_chart(fig_dd, use_container_width=True)
            else: st.info("Cierra operaciones para ver métricas.")
        else: st.warning("Sin datos.")

    # ------------------------------------------------------------------
    # TAB 4: PERFORMANCE (SCHWAB STYLE) 📈 - ACTUALIZADO V12.1
    # ------------------------------------------------------------------
    with tab_performance:
        st.subheader("📈 Rendimiento Temporal (%)")
        
        time_filters = ["Todo", "YTD (Este Año)", "Año Anterior"]
        selected_filter = st.radio("Rango:", time_filters, index=0, horizontal=True)

        df_all = db.get_closed_trades(st.session_state['username'])
        
        if not df_all.empty:
            df_perf = df_all.copy()
            df_perf['exit_date'] = pd.to_datetime(df_perf['exit_date'])
            
            # --- PREPARACIÓN DE DATOS (HISTORIA COMPLETA) ---
            daily_pnl = df_perf.groupby('exit_date')['pnl'].sum()
            
            # Crear línea de tiempo desde el primer trade de la historia hasta hoy
            overall_start = daily_pnl.index.min()
            today = pd.Timestamp(date.today())
            full_history_range = pd.date_range(start=overall_start, end=today)
            
            # Rellenar con 0 los días sin trades
            daily_pnl_reindexed = daily_pnl.reindex(full_history_range).fillna(0)
            
            # Calcular Equity ACUMULADO día a día (Capital Inicial + Ganancias Históricas)
            equity_curve_series = current_balance + daily_pnl_reindexed.cumsum()
            
            # --- FILTRADO INTELIGENTE (ESTILO SCHWAB) ---
            # Para que YTD empiece en 0%, necesitamos normalizar respecto al valor de inicio del periodo.
            
            start_date_filter = overall_start
            end_date_filter = today

            if selected_filter == "YTD (Este Año)":
                start_date_filter = pd.Timestamp(date(today.year, 1, 1))
            elif selected_filter == "Año Anterior":
                start_date_filter = pd.Timestamp(date(today.year - 1, 1, 1))
                end_date_filter = pd.Timestamp(date(today.year - 1, 12, 31))
            
            # Asegurar que start_date no sea anterior al primer trade (para no graficar la nada)
            if start_date_filter < overall_start:
                start_date_filter = overall_start

            # Cortar la serie de Equity
            sliced_equity = equity_curve_series[
                (equity_curve_series.index >= start_date_filter) & 
                (equity_curve_series.index <= end_date_filter)
            ]
            
            if not sliced_equity.empty:
                # --- NORMALIZACIÓN (El truco para que empiece en 0%) ---
                # Tomamos el primer valor del periodo seleccionado como "Base 100"
                base_value = sliced_equity.iloc[0] 
                
                # Calculamos el % de retorno relativo a esa base
                pct_change_series = ((sliced_equity - base_value) / base_value) * 100
                
                df_chart_time = pd.DataFrame({
                    'Date': pct_change_series.index,
                    'Return': pct_change_series.values
                })
                
                # --- GRÁFICO TIPO SCHWAB ---
                fig_ts = px.line(df_chart_time, x='Date', y='Return', title=f"Retorno {selected_filter}")
                
                # Estilo Limpio (Sin Relleno, Línea Cyan)
                fig_ts.update_traces(
                    line_color='#00BFA5', # Teal brillante
                    line_width=3,
                    hovertemplate='%{x}<br>Retorno: %{y:.2f}%'
                )
                
                # Línea Cero Punteada
                fig_ts.add_hline(y=0, line_dash="dot", line_color="white", opacity=0.5)
                
                fig_ts.update_layout(
                    yaxis_title="Retorno (%)",
                    yaxis_tickformat=".2f%",
                    xaxis_title="Fecha",
                    height=450,
                    showlegend=False
                )
                
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # Info
                total_return = pct_change_series.iloc[-1]
                abs_pnl = sliced_equity.iloc[-1] - sliced_equity.iloc[0]
                st.info(f"📅 **Periodo:** {selected_filter} | **Retorno:** {total_return:.2f}% | **PnL:** ${abs_pnl:,.2f}")
                
            else:
                st.warning(f"No hay datos para el periodo {selected_filter}")
        else:
            st.info("Necesitas cerrar operaciones para ver la evolución temporal.")

    # --- TAB 5: CONFIGURACIÓN ---
    with tab_config:
        st.subheader("⚙️ Editor de Estrategia")
        st.info("Escribe las opciones separadas por coma. Los cambios se mantienen hasta que guardes.")
        current_config = st.session_state.get('strategy_config', {})
        keys_to_delete = []
        for category, options in current_config.items():
            with st.container():
                c_name, c_vals, c_del = st.columns([2, 5, 1])
                c_name.markdown(f"**{category}**")
                val_str = ", ".join(options) if isinstance(options, list) else str(options)
                new_val = c_vals.text_input(f"Opciones ({category})", value=val_str, label_visibility="collapsed", key=f"input_{category}")
                if c_del.button("🗑️", key=f"del_{category}"): keys_to_delete.append(category)
                st.divider()
        if keys_to_delete:
            for k in keys_to_delete: del st.session_state['strategy_config'][k]
            st.rerun()
        with st.expander("➕ Agregar Nuevo Parámetro"):
            c_n, c_v, c_b = st.columns([2, 5, 1])
            new_k = c_n.text_input("Nombre"); new_v = c_v.text_input("Opciones (separadas por coma)")
            if c_b.button("Agregar"):
                if new_k:
                    st.session_state['strategy_config'][new_k] = [x.strip() for x in new_v.split(',') if x.strip()]
                    st.rerun()
        if st.button("💾 Guardar Cambios en Nube", type="primary"):
            final_config = {}
            for key in st.session_state['strategy_config'].keys():
                widget_key = f"input_{key}"
                if widget_key in st.session_state:
                    raw_text = st.session_state[widget_key]
                    clean_list = [x.strip() for x in raw_text.split(',') if x.strip() != ""]
                    final_config[key] = clean_list
                else: final_config[key] = st.session_state['strategy_config'][key]
            if db.update_strategy_config(st.session_state['username'], final_config):
                st.session_state['strategy_config'] = final_config
                st.success("Configuración guardada correctamente."); time.sleep(1); st.rerun()
            else: st.error("Error al guardar.")

def main():
    if st.session_state['logged_in']: dashboard_page()
    else: login_page()

if __name__ == '__main__': main()
