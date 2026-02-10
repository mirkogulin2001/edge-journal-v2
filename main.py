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
import matplotlib.pyplot as plt

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
# Cambio: Grilla blanca con transparencia media (0.2)
GRID_STYLE = dict(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.2)')

db.init_db()

# --- FUNCIONES FINANCIERAS ---
def get_risk_metrics(returns_series):
    if len(returns_series) < 2: return 0, 0
    rf_daily = 0.04 / 252 
    excess_ret = returns_series - rf_daily
    std = returns_series.std()
    sharpe = (excess_ret.mean() / std * np.sqrt(252)) if std != 0 else 0
    neg_ret = returns_series[returns_series < 0]
    downside = neg_ret.std()
    sortino = (excess_ret.mean() / downside * np.sqrt(252)) if downside != 0 else 0
    return sharpe, sortino

def calculate_alpha_beta(port_returns, bench_returns):
    # Validaciones básicas
    if len(port_returns) < 2 or len(bench_returns) < 2: return 0, 0
    
    # Alinear datos (Inner Join)
    df_join = pd.concat([port_returns, bench_returns], axis=1, join='inner').dropna()
    if df_join.empty: return 0, 0
    
    p_ret = df_join.iloc[:, 0]
    b_ret = df_join.iloc[:, 1]
    
    # 1. BETA (Volatilidad diaria)
    cov = np.cov(p_ret, b_ret)[0][1]
    var = np.var(b_ret)
    beta = cov / var if var != 0 else 0
    
    # 2. RETORNO TOTAL ACUMULADO
    rp_total = (1 + p_ret).prod() - 1
    rm_total = (1 + b_ret).prod() - 1
    
    # 3. CÁLCULO DE TIEMPO ROBUSTO (Por Fechas)
    # En lugar de contar filas, medimos la distancia en años entre la primera y última fecha.
    start_date = df_join.index[0]
    end_date = df_join.index[-1]
    time_years = (end_date - start_date).days / 365.25
    
    # Si por error de datos time_years es 0, asumimos un mínimo
    if time_years <= 0: time_years = len(p_ret) / 252

    # Tasa Libre de Riesgo ajustada al tiempo real
    risk_free_annual = 0.04
    risk_free_period = risk_free_annual * time_years
    
    # 4. JENSEN'S ALPHA
    # Fórmula: Tu Retorno - (Tasa Segura + Beta * (Mercado - Tasa Segura))
    alpha = rp_total - (risk_free_period + beta * (rm_total - risk_free_period))
    
    return alpha, beta

# --- MOTOR MONTE CARLO ---
def run_monte_carlo_simulation(r_values, num_sims, max_dd_limit, confidence_level, start_capital):
    if not r_values or len(r_values) < 5:
        return None, "Necesitas al menos 5 trades cerrados."

    r_array = np.array(r_values)
    n_trades = 100 
    
    # --- 1. CÁLCULO DEL KELLY TEÓRICO (PUNTO DE PARTIDA) ---
    # Win Rate (p)
    wins = r_array[r_array > 0]
    p = len(wins) / len(r_array)
    
    # Payoff Ratio promedio (b)
    # Promedio de ganadores / Promedio de perdedores (en valor absoluto)
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    losses = np.abs(r_array[r_array <= 0])
    avg_loss = np.mean(losses) if len(losses) > 0 else 1 # Evitar div por 0
    
    if avg_loss == 0: avg_loss = 1
    b = avg_win / avg_loss
    
    # Fórmula Kelly Clásica: f = p - (q / b)
    # Donde q = 1 - p
    q = 1 - p
    kelly_fraction = p - (q / b)
    
    # Límites de seguridad (Kelly nunca mayor a 50% por cordura, ni menor a 0)
    if kelly_fraction < 0: kelly_fraction = 0.001 # Estrategia perdedora teóricamente
    if kelly_fraction > 0.5: kelly_fraction = 0.5
    
    # --- 2. GRID SEARCH (Desde 0.5% hasta Kelly) ---
    # En lugar de buscar hasta 25% a ciegas, buscamos solo hasta tu Kelly.
    # Si Kelly es 20%, probamos f desde 0.5% hasta 20%.
    
    # Creamos 50 pasos entre un riesgo mínimo y tu Kelly
    f_range = np.linspace(0.005, max(0.01, kelly_fraction), 50)
    
    best_f = 0.0
    best_median_metric = -np.inf
    
    # Simulación "Rápida" para encontrar f óptimo (Optimization Phase)
    opt_sims = 1000
    rand_indices = np.random.randint(0, len(r_array), size=(opt_sims, n_trades))
    shuffled_rs = r_array[rand_indices]

    for f in f_range:
        # Vectorización completa para velocidad
        growth_factors = np.maximum(1 + (f * shuffled_rs), 0)
        equity_curves = np.cumprod(growth_factors, axis=1)
        
        # Calcular Max Drawdown de cada curva
        peaks = np.maximum.accumulate(equity_curves, axis=1)
        dd_pcts = (equity_curves - peaks) / peaks
        max_dds = np.min(dd_pcts, axis=1)
        
        # EL FILTRO: ¿Cumple tu condición de confianza?
        # "Que en el 95% de los casos (confidence), el DD no sea peor que el límite"
        # Buscamos el percentil malo (ej: el 5% peor)
        dd_at_risk = np.percentile(max_dds, (1 - confidence_level) * 100)
        
        # Si el "peor caso probable" respeta tu límite:
        if dd_at_risk >= -max_dd_limit:
            # Vemos cuánto dinero gana (Mediana)
            median_end = np.median(equity_curves[:, -1])
            
            # Si gana más que el anterior mejor f, lo guardamos.
            # Como f más alto suele dar más ganancia (hasta Kelly), esto tiende a elegir el f más alto posible que sea seguro.
            if median_end > best_median_metric:
                best_median_metric = median_end
                best_f = f
    
    # --- 3. SIMULACIÓN FINAL (Con el f ganador) ---
    # Ahora sí corremos las 3000 o 5000 simulaciones completas para los gráficos
    final_rand_indices = np.random.randint(0, len(r_array), size=(num_sims, n_trades))
    final_shuffled_rs = r_array[final_rand_indices]
    
    growth_factors = np.maximum(1 + (best_f * final_shuffled_rs), 0)
    equity_curves = start_capital * np.cumprod(growth_factors, axis=1)
    
    final_balances = equity_curves[:, -1]
    peaks = np.maximum.accumulate(equity_curves, axis=1)
    dds = (equity_curves - peaks) / peaks
    max_dds = np.min(dds, axis=1)
    
    median_balance = np.median(final_balances)
    median_dd_metric = np.median(max_dds)
    
    return {
        'optimal_f': best_f,
        'kelly_theoretical': kelly_fraction, # Devolvemos esto para mostrarlo si quieres
        'median_balance': median_balance,
        'median_dd': median_dd_metric,
        'equity_curves': equity_curves,
        'final_balances': final_balances,
        'max_dds': max_dds
    }, None

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
        st.caption("Edge Journal v20.2 Symmetrical")

    st.title("Gestión de Cartera 🏦")
    tab_active, tab_history, tab_stats, tab_performance, tab_montecarlo, tab_config, tab_edge = st.tabs(["⚡ Posiciones", "📚 Historial", "📊 Analytics", "📈 Performance", "🎲 Monte Carlo", "⚙️ Estrategia","🧬 Edge Evolution"])

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
                k1.metric("PnL Latente (Abierto)", f"${total_floating:,.0f}", delta=f"{total_floating:,.0f}")
                k2.metric("PnL Realizado (Parciales)", f"${total_partial_banked:,.0f}", delta=None)
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
                # st.markdown("##### ☢️ Zona Nuclear") # <-- LÍNEA ELIMINADA
                # Agregamos un pequeño espacio superior para alinear con el título de la izquierda
                st.write("") 
                st.write("") 
                confirm_nuke = st.checkbox("Confirmar borrado total")
                # El resto sigue igual...
                if st.button("BORRAR TODO", type="primary", disabled=not confirm_nuke):
                    if db.delete_all_trades(st.session_state['username']):
                        st.toast("🔥 Historial eliminado."); time.sleep(1); st.rerun()

        if not df_c.empty:
            df_c['tags_dict'] = df_c['tags'].apply(lambda x: json.loads(x) if isinstance(x, str) and x else {})
            r_vals = []
            for i, r in df_c.iterrows():
                try:
                    risk = abs(r['entry_price'] - r['initial_stop_loss'])
                    if risk == 0: risk = 0.01
                    r_vals.append(r['pnl'] / (risk * r['quantity']))
                except: r_vals.append(0)
            df_c['R'] = r_vals

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
                
                df_c['Estrategia'] = df_c['tags_dict'].apply(lambda x: " ".join([f"[{v}]" for k,v in x.items()]))
                st.dataframe(df_c.drop(columns=['id', 'tags', 'tags_dict', 'R']), use_container_width=True, hide_index=True,
                             column_config={"pnl": st.column_config.NumberColumn("PnL", format="$%.2f"), "result_type": st.column_config.TextColumn("Res", width="small")})
            else: st.warning("Sin resultados.")
        else: st.write("Sin datos.")

    # --- TAB 3: ANALYTICS ---
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
                e_math_abs = (wr * payoff) - lr
                k9.metric("E(Math)", f"{e_math_abs:.2f}"); k10.metric("Payoff Ratio", f"{payoff:.2f}"); k11.metric("Max Drawdown", f"{max_dd:.2f}%", delta=None); k12.metric("Current DD", f"{current_dd:.2f}%")

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
                    fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
                    pnl_data = df_closed['pnl'].dropna()
                    if len(pnl_data) > 1:
                        try:
                            kde = stats.gaussian_kde(pnl_data)
                            x_grid = np.linspace(pnl_data.min(), pnl_data.max(), 200)
                            y_kde = kde(x_grid)
                            fig_hist.add_trace(go.Scatter(x=x_grid, y=y_kde, mode='lines', line=dict(color='rgba(0, 150, 255, 0.4)', width=2), fill='tozeroy', fillcolor='rgba(0, 150, 255, 0.1)', name='Teórica'), secondary_y=True)
                        except: pass

                    optimal_bins = int(np.sqrt(len(df_closed))) if not df_closed.empty else 10
                    if optimal_bins < 5: optimal_bins = 5
                    data_range = pnl_data.max() - pnl_data.min()
                    bin_size = data_range / optimal_bins if data_range > 0 else 10

                    be_data = pnl_data[(pnl_data >= -1) & (pnl_data <= 1)]
                    loss_data = pnl_data[pnl_data < -1]
                    win_data = pnl_data[pnl_data > 1]

                    # Cambio: marker_color ahora es un celeste traslucido (rgba(135, 206, 235, 0.8))
                    fig_hist.add_trace(go.Histogram(x=win_data, marker_color='#00FFFF', marker_line_color='black', marker_line_width=1, opacity=1, name='WIN', xbins=dict(start=1, size=bin_size)), secondary_y=False)
                    # Nota: cambié opacity a 1 porque ya la estamos manejando dentro del rgba del color.
                    fig_hist.add_trace(go.Histogram(x=loss_data, marker_color='#FF4B4B', marker_line_color='black', marker_line_width=1, opacity=0.85, name='LOSS', xbins=dict(end=-1, size=bin_size)), secondary_y=False)
                    fig_hist.add_trace(go.Histogram(x=be_data, marker_color='#AAAAAA', marker_line_color='black', marker_line_width=1, opacity=0.85, name='BE', xbins=dict(start=-1, end=1, size=2)), secondary_y=False)

                    fig_hist.update_layout(title="🔔 Distribución PnL", barmode='overlay', height=350, margin=dict(l=0,r=0,t=40,b=0), showlegend=False)
                    fig_hist.update_xaxes(**GRID_STYLE); fig_hist.update_yaxes(secondary_y=False, **GRID_STYLE); fig_hist.update_yaxes(secondary_y=True, showgrid=False, showticklabels=True)
                    st.plotly_chart(fig_hist, use_container_width=True)

                    current_cash = (current_balance + total_banked) - total_invested_cash
                    if current_cash < 0: current_cash = 0
                    pie_data.append({'Asset': 'CASH', 'Value': current_cash})
                    
                    # Cambio: Nuevo título sin emoji
                    fig_pie = px.pie(pd.DataFrame(pie_data), values='Value', names='Asset', title="Composición de Portfolio", hole=0.4, color_discrete_sequence=CUSTOM_TEAL_PALETTE)
                    fig_pie.update_traces(textposition='outside', textinfo='label+percent')
                    fig_pie.update_layout(height=500, margin=dict(l=0,r=0,t=30,b=0), showlegend=False)
                    st.plotly_chart(fig_pie, use_container_width=True)

            else: st.info("Cierra operaciones para ver métricas.")
        else: st.warning("Sin datos.")

# --- TAB 4: PERFORMANCE ---
    with tab_performance:
        st.subheader("📈 Rendimiento vs Benchmark")
        
        # Selector de Rango
        time_filters = ["Todo", "YTD (Este Año)", "Año Anterior"]
        selected_filter = st.radio("Rango:", time_filters, index=0, horizontal=True)
        
        # Obtener datos
        df_all = db.get_closed_trades(st.session_state['username'])
        
        if not df_all.empty:
            df_perf = df_all.copy()
            df_perf['exit_date'] = pd.to_datetime(df_perf['exit_date'])
            
            today = pd.Timestamp(date.today())
            start_date_filter = df_perf['exit_date'].min()
            end_date_filter = today

            # Lógica de Fechas
            if selected_filter == "YTD (Este Año)": 
                start_date_filter = pd.Timestamp(date(today.year, 1, 1))
            elif selected_filter == "Año Anterior": 
                start_date_filter = pd.Timestamp(date(today.year - 1, 1, 1))
                end_date_filter = pd.Timestamp(date(today.year - 1, 12, 31))
            
            # Filtrar DataFrame
            df_perf = df_perf[(df_perf['exit_date'] >= start_date_filter) & (df_perf['exit_date'] <= end_date_filter)]
            
            if not df_perf.empty:
                # Construcción de la Curva de Equity Diaria
                daily_pnl = df_perf.groupby('exit_date')['pnl'].sum()
                date_range = pd.date_range(start=start_date_filter, end=end_date_filter)
                daily_pnl = daily_pnl.reindex(date_range).fillna(0)
                cumulative_pnl = daily_pnl.cumsum()
                
                # Equity y Retornos del Portfolio
                portfolio_equity = current_balance + cumulative_pnl
                portfolio_returns = portfolio_equity.pct_change().fillna(0)
                portfolio_cum_ret = ((portfolio_equity - current_balance) / current_balance) * 100

                # Descarga de datos del Benchmark (SPY)
                with st.spinner("Descargando datos de mercado (SPY)..."):
                    try:
                        spy_ticker = yf.Ticker("SPY")
                        # Pedimos un buffer de 2 días extra para asegurar datos
                        spy_data = spy_ticker.history(start=start_date_filter, end=end_date_filter + timedelta(days=2))
                        
                        if not spy_data.empty:
                            spy_data = spy_data['Close']
                            spy_data.index = spy_data.index.tz_localize(None) # Quitar zona horaria para compatibilidad
                            spy_data = spy_data.reindex(date_range).ffill().fillna(method='bfill')
                            
                            spy_returns = spy_data.pct_change().fillna(0)
                            spy_cum_ret = ((spy_data - spy_data.iloc[0]) / spy_data.iloc[0]) * 100
                            
                            # Cálculos de Riesgo
                            port_sharpe, port_sortino = get_risk_metrics(portfolio_returns)
                            spy_sharpe, spy_sortino = get_risk_metrics(spy_returns)
                            alpha, beta = calculate_alpha_beta(portfolio_returns, spy_returns)
                            
                            # --- MÉTRICAS (Ahora son 5 columnas) ---
                            period_return = portfolio_cum_ret.iloc[-1] if not portfolio_cum_ret.empty else 0.0

                            m0, m1, m2, m3, m4 = st.columns(5)
                            
                            m0.metric("Retorno (Periodo)", f"{period_return:.2f}%")
                            m1.metric("Beta (vs SPY)", f"{beta:.2f}", help="< 1: Menos volátil que el mercado.")
                            m2.metric("Sharpe (Portfolio / SPY)", f"{port_sharpe:.2f} / {spy_sharpe:.2f}")
                            m3.metric("Sortino (Portfolio / SPY)", f"{port_sortino:.2f} / {spy_sortino:.2f}")
                            m4.metric("Jensen's Alpha", f"{alpha:.2%}", help="Retorno extra sobre el mercado.")
                            
                            # --- GRÁFICO COMPARATIVO ---
                            fig_perf = go.Figure()
                            
                            # Traza Portfolio
                            fig_perf.add_trace(go.Scatter(
                                x=portfolio_cum_ret.index, 
                                y=portfolio_cum_ret.values, 
                                mode='lines', 
                                name='Tu Portfolio', 
                                line=dict(color='#00FFFF', width=3)
                            ))
                            
                            # Traza Benchmark
                            fig_perf.add_trace(go.Scatter(
                                x=spy_cum_ret.index, 
                                y=spy_cum_ret.values, 
                                mode='lines', 
                                name='S&P 500 (SPY)', 
                                line=dict(color='#E0E0E0', width=2)
                            ))
                            
                            # Línea base 0
                            fig_perf.add_hline(y=0, line_dash="dash", line_color="gray")
                            
                            # Configuración del Layout (Con height=600)
                            fig_perf.update_layout(
                                title="Rendimiento Acumulado vs Benchmark", 
                                yaxis_title="Retorno (%)", 
                                yaxis_tickformat=".2f%", 
                                hovermode="x unified", 
                                legend=dict(y=1.1, orientation="h"),
                                height=600 # Altura aumentada
                            )
                            
                            fig_perf.update_xaxes(**GRID_STYLE)
                            fig_perf.update_yaxes(**GRID_STYLE)
                            
                            st.plotly_chart(fig_perf, use_container_width=True)
                        else: 
                            st.warning("No se pudieron obtener datos del SPY para este periodo.")
                    except Exception as e: 
                        st.error(f"Error conectando con Yahoo Finance: {e}")
            else: 
                st.info("No hay operaciones en el rango seleccionado.")
        else: 
            st.info("Cierra operaciones para ver tu rendimiento.")

# --- TAB 5: MONTE CARLO (ESTÉTICA FINAL) ---
    with tab_montecarlo:
        st.subheader("🚀 Simulador Monte Carlo (Basado en Kelly Teórico)")
        st.caption("Simula el futuro de tu cuenta aplicando la Fórmula de Kelly estricta ajustada por tu factor de preferencia.")
        
        # 1. INPUTS
        c1, c2 = st.columns([1, 2])
        n_sims = c1.number_input("Simulaciones", 1000, 10000, 3000, step=500)
        kelly_factor = c2.slider("Factor de Ajuste (1.0 = Full Kelly, 0.5 = Half Kelly)", 0.1, 1.5, 1.0, 0.1)
        
        # 2. CARGA Y CÁLCULO DE DATOS
        df_c = db.get_closed_trades(st.session_state['username'])
        
        if not df_c.empty:
            if 'R' not in df_c.columns:
                r_vals = []
                for i, r in df_c.iterrows():
                    try:
                        risk = abs(r['entry_price'] - r['initial_stop_loss'])
                        if risk == 0: risk = 0.01
                        r_vals.append(r['pnl'] / (risk * r['quantity']))
                    except: r_vals.append(0)
                df_c['R'] = r_vals
            
            r_list = df_c['R'].tolist()
            r_array = np.array(r_list)
            
            # --- CÁLCULO ESTADÍSTICAS ---
            wins = r_array[r_array > 0]
            losses = r_array[r_array < 0]
            total_trades = len(r_array)
            
            if total_trades > 0 and len(losses) > 0:
                win_rate = len(wins) / total_trades
                loss_rate = len(losses) / total_trades
                avg_win_r = np.mean(wins) if len(wins) > 0 else 0
                avg_loss_r = np.mean(np.abs(losses)) 
                payoff_ratio = avg_win_r / avg_loss_r if avg_loss_r != 0 else 0
                kelly_algebraic = win_rate - (loss_rate / payoff_ratio)
            else:
                win_rate = 0; loss_rate = 0; payoff_ratio = 0; kelly_algebraic = 0
            
            # --- VISUALIZACIÓN INPUTS ---
            st.markdown("### 📊 Estadísticas de tu Edge (Base de Cálculo)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Win Rate", f"{win_rate*100:.1f}%")
            k2.metric("Loss Rate", f"{loss_rate*100:.1f}%")
            k3.metric("R/R Real (Payoff)", f"{payoff_ratio:.2f}")
            k_delta_color = "normal" if kelly_algebraic > 0 else "inverse"
            k4.metric("Kelly Teórico", f"{kelly_algebraic*100:.2f}%", delta_color=k_delta_color)
            
            st.markdown("---")

            # 3. EJECUCIÓN
            if st.button("🚀 Ejecutar Análisis", type="primary"):
                if len(r_list) < 5:
                    st.error("Necesitas al menos 5 trades cerrados.")
                elif kelly_algebraic <= 0:
                    st.error(f"Tu Kelly es {kelly_algebraic*100:.2f}%. No tienes ventaja matemática positiva.")
                else:
                    base_f = kelly_algebraic
                    applied_f = base_f * kelly_factor
                    
                    st.success(f"Base (Kelly Teórico): {base_f*100:.2f}% ➡️ **Aplicado en Simulación ({kelly_factor}x): {applied_f*100:.2f}%**")
                    
                    # --- DASHBOARD VISUAL ---
                    g_m1, g_m2, g_m3 = st.columns(3)
                    metric_bal = g_m1.empty()
                    metric_dd = g_m2.empty()
                    metric_var = g_m3.empty()
                    
                    st.markdown("---")
                    
                    # Títulos limpios (sin emojis)
                    g1, g2, g3 = st.columns(3)
                    with g1: st.markdown("##### Simulación (Equity)"); chart_curves = st.empty()
                    with g2: st.markdown("##### Distribución de Retornos"); chart_hist_ret = st.empty()
                    with g3: st.markdown("##### Distribución de Max Drawdown %"); chart_hist_dd = st.empty()
                    
                    progress_bar = st.progress(0)
                    
                    # --- BUCLE DE SIMULACIÓN ---
                    batch_size = int(n_sims / 15)
                    if batch_size < 50: batch_size = 50
                    
                    all_final_balances = []
                    all_max_dds = []
                    display_curves = [] 
                    n_trades = 100 
                    
                    for i in range(0, n_sims, batch_size):
                        current_batch_size = min(batch_size, n_sims - i)
                        rand_indices = np.random.randint(0, len(r_array), size=(current_batch_size, n_trades))
                        batch_rs = r_array[rand_indices]
                        
                        growth_factors = np.maximum(1 + (applied_f * batch_rs), 0)
                        batch_curves = current_balance * np.cumprod(growth_factors, axis=1)
                        
                        batch_finals = batch_curves[:, -1]
                        peaks = np.maximum.accumulate(batch_curves, axis=1)
                        dds = (batch_curves - peaks) / peaks
                        batch_mdds = np.min(dds, axis=1)
                        
                        all_final_balances.extend(batch_finals)
                        all_max_dds.extend(batch_mdds)
                        
                        if len(display_curves) < 800:
                            display_curves.extend(batch_curves[:20])
                        
                        # Estadísticas
                        curr_median_bal = np.median(all_final_balances)
                        curr_median_dd = np.median(all_max_dds)
                        dd_95_limit = np.percentile(all_max_dds, 5) 
                        delta_pct = ((curr_median_bal - current_balance) / current_balance) * 100
                        
                        # Actualizar Métricas
                        metric_bal.metric("Proyección Mediana", f"${curr_median_bal:,.0f}", delta=f"{delta_pct:.1f}%")
                        metric_dd.metric("Mediana Max Drawdown", f"{curr_median_dd*100:.2f}%")
                        metric_var.metric("Límite 95% Confianza", f"{dd_95_limit*100:.2f}%")
                        
                        # --- GRÁFICOS ---
                        
                        # 1. Curvas (Matplotlib)
                        fig_eq, ax = plt.subplots(figsize=(5, 4)) 
                        fig_eq.patch.set_facecolor('#0E1117')
                        ax.set_facecolor('#0E1117')
                        for spine in ax.spines.values(): spine.set_color('white')
                        ax.tick_params(colors='white', labelsize=8); ax.grid(color='#333333', linestyle='--')
                        
                        c_arr = np.array(display_curves)
                        if len(c_arr) > 0:
                            ax.plot(c_arr.T, color='white', alpha=0.04, linewidth=0.5)
                            ax.plot(np.median(c_arr, axis=0), color='#00FFAA', linewidth=2, label='Mediana')
                            ax.axhline(y=current_balance, color='gray', linestyle=':', alpha=0.5)
                            legend = ax.legend(loc='upper left', frameon=False, fontsize=8)
                            plt.setp(legend.get_texts(), color='white')
                        
                        ax.set_title(f"Proyección ({n_trades} trades)", color='white', fontsize=10)
                        chart_curves.pyplot(fig_eq)
                        plt.close(fig_eq)
                        
                        # 2. Histograma Retornos (Plotly)
                        curr_rets = (np.array(all_final_balances) / current_balance) - 1
                        mean_ret = np.mean(curr_rets)
                        median_ret = np.median(curr_rets)
                        upper_limit = np.percentile(curr_rets, 95)
                        if upper_limit < 0.5: upper_limit = 0.5
                        
                        fig_h1 = go.Figure(go.Histogram(x=curr_rets, nbinsx=80, marker_color='#00FFFF', opacity=0.7, name='Distribución'))
                        
                        # Líneas Verticales (Sin texto anotado, solo líneas)
                        fig_h1.add_vline(x=0, line_width=1, line_color="gray") 
                        fig_h1.add_vline(x=mean_ret, line_dash="dash", line_color="#FFA500") # Naranja
                        fig_h1.add_vline(x=median_ret, line_dash="dot", line_color="#00D000") # Verde Matrix (se ve mejor en dark)
                        
                        # Truco para Leyenda: Trazas vacías con el mismo estilo
                        fig_h1.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Media', line=dict(color='#FFA500', dash='dash')))
                        fig_h1.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Mediana', line=dict(color='#00D000', dash='dot')))
                        
                        fig_h1.update_layout(
                            height=300, margin=dict(l=0,r=0,t=20,b=0), 
                            xaxis_tickformat='.0%', 
                            showlegend=True, # Leyenda activada
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), # Leyenda arriba
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', range=[min(curr_rets), upper_limit * 1.5]), 
                            yaxis=dict(showgrid=False)
                        )
                        chart_hist_ret.plotly_chart(fig_h1, use_container_width=True, key=f"hr_{i}")
                        
                        # 3. Histograma Drawdowns (Plotly)
                        mean_dd = np.mean(all_max_dds)
                        median_dd = np.median(all_max_dds)
                        
                        fig_h2 = go.Figure(go.Histogram(x=all_max_dds, nbinsx=60, marker_color='#FF4B4B', opacity=0.7, name='Distribución'))
                        
                        # Líneas Verticales
                        fig_h2.add_vline(x=mean_dd, line_dash="dash", line_color="#FFA500") 
                        fig_h2.add_vline(x=median_dd, line_dash="dot", line_color="#00D000")
                        fig_h2.add_vline(x=dd_95_limit, line_dash="dash", line_color="yellow")
                        
                        # Leyenda (Trazas Dummy)
                        fig_h2.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Media', line=dict(color='#FFA500', dash='dash')))
                        fig_h2.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Mediana', line=dict(color='#00D000', dash='dot')))
                        fig_h2.add_trace(go.Scatter(x=[None], y=[None], mode='lines', name='Límite 95%', line=dict(color='yellow', dash='dash')))

                        fig_h2.update_layout(
                            height=300, margin=dict(l=0,r=0,t=20,b=0), 
                            xaxis_tickformat='.1%', 
                            showlegend=True,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'), 
                            yaxis=dict(showgrid=False)
                        )
                        chart_hist_dd.plotly_chart(fig_h2, use_container_width=True, key=f"hdd_{i}")
                        
                        progress_bar.progress(min(1.0, (i + batch_size) / n_sims))
                    
                    st.success("✅ Simulación Completada.")

        else:
            st.info("Cierra operaciones para ver tus estadísticas.")
    # --- TAB 6: CONFIGURACIÓN SIMPLE ---
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
# --- TAB 7: EDGE EVOLUTION (NUEVO) ---
    with tab_edge:
        st.subheader("🧬 Evolución de tu Edge")
        st.caption("Visualiza cómo maduran tus estadísticas a medida que acumulas experiencia (trades).")
        
        # Obtener datos
        df_ev = db.get_closed_trades(st.session_state['username'])
        
        if not df_ev.empty and len(df_ev) > 5:
            # 1. Preparar Datos (Calcular R)
            if 'R' not in df_ev.columns:
                r_vals = []
                for i, r in df_ev.iterrows():
                    try:
                        risk = abs(r['entry_price'] - r['initial_stop_loss'])
                        if risk == 0: risk = 0.01
                        r_vals.append(r['pnl'] / (risk * r['quantity']))
                    except: r_vals.append(0)
                df_ev['R'] = r_vals
            
            # Ordenar por fecha de salida (Fundamental para la evolución temporal)
            df_ev['exit_date'] = pd.to_datetime(df_ev['exit_date'])
            df_ev = df_ev.sort_values('exit_date')
            
            # 2. Bucle de Cálculo Evolutivo (Rolling Metrics)
            history_dates = []
            evo_wr = []
            evo_lr = []
            evo_be = []
            evo_rr = []
            evo_expectancy = []
            
            # Acumuladores
            count_win = 0
            count_loss = 0
            count_be = 0
            sum_win_r = 0
            sum_loss_r = 0 # En absoluto
            
            # Recorremos trade por trade
            for idx, row in df_ev.iterrows():
                r = row['R']
                
                # Clasificar
                if r > 0.05: # Umbral para considerar Win (ajustable)
                    count_win += 1
                    sum_win_r += r
                elif r < -0.05: # Umbral para Loss
                    count_loss += 1
                    sum_loss_r += abs(r)
                else:
                    count_be += 1
                
                total_trades = count_win + count_loss + count_be
                
                # Calcular Métricas del momento
                curr_wr = count_win / total_trades
                curr_lr = count_loss / total_trades
                curr_be = count_be / total_trades
                
                # Payoff (RR Ratio)
                avg_win = sum_win_r / count_win if count_win > 0 else 0
                avg_loss = sum_loss_r / count_loss if count_loss > 0 else 1 # Evitar div/0
                if avg_loss == 0: avg_loss = 1
                curr_rr = avg_win / avg_loss
                
                # Esperanza Matemática E(X) = (WR * RR) - LR
                # Nota: Usamos la fórmula simplificada que pediste.
                # Para ser purista R-Multiple sería: (WR * AvgWinR) - (LR * AvgLossR)
                # Tu fórmula: (WR * RR) - LR
                curr_ex = (curr_wr * curr_rr) - curr_lr
                
                # Guardar
                evo_wr.append(curr_wr)
                evo_lr.append(curr_lr)
                evo_be.append(curr_be)
                evo_rr.append(curr_rr)
                evo_expectancy.append(curr_ex)
                history_dates.append(row['exit_date'])

            # Creación de eje X (Número de trade)
            x_axis = list(range(1, len(history_dates) + 1))
            
            # 3. GRAFICAR (LAYOUT 3 COLUMNAS)
            c1, c2, c3 = st.columns(3)
            
            # --- GRÁFICO 1: EVOLUCIÓN DE TASAS (WR, LR, BE) ---
            with c1:
                st.markdown("##### 🎯 Tasas (Win/Loss/BE)")
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=x_axis, y=evo_wr, mode='lines', name='Win Rate', line=dict(color='#00FF00', width=2)))
                fig1.add_trace(go.Scatter(x=x_axis, y=evo_lr, mode='lines', name='Loss Rate', line=dict(color='#FF4B4B', width=2)))
                fig1.add_trace(go.Scatter(x=x_axis, y=evo_be, mode='lines', name='BE Rate', line=dict(color='gray', width=1, dash='dot')))
                
                fig1.update_layout(
                    height=300, margin=dict(l=0,r=0,t=30,b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, title="Trades"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='.0%'),
                    legend=dict(orientation="h", y=1.1, x=0),
                    hovermode="x unified"
                )
                st.plotly_chart(fig1, use_container_width=True)

            # --- GRÁFICO 2: EVOLUCIÓN DEL PAYOFF (R/R) ---
            with c2:
                st.markdown("##### ⚖️ Ratio R/B Real")
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=x_axis, y=evo_rr, mode='lines', name='Payoff', line=dict(color='#FFA500', width=2)))
                
                # Línea de referencia 1:1 o 2:1 según prefieras
                fig2.add_hline(y=1.5, line_dash="dot", line_color="rgba(255,255,255,0.3)", annotation_text="Obj 1.5")
                
                fig2.update_layout(
                    height=300, margin=dict(l=0,r=0,t=30,b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, title="Trades"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    showlegend=False,
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # --- GRÁFICO 3: EVOLUCIÓN DE ESPERANZA MATEMÁTICA ---
            with c3:
                st.markdown("##### 🔮 Esperanza E(X)")
                fig3 = go.Figure()
                
                # Color dinámico (Verde si es positiva, Roja si es negativa al final)
                final_color = '#00BFFF' if evo_expectancy[-1] > 0 else '#FF4B4B'
                
                fig3.add_trace(go.Scatter(x=x_axis, y=evo_expectancy, mode='lines', name='E(X)', line=dict(color=final_color, width=2)))
                fig3.add_hline(y=0, line_color="white", line_width=1) # Línea cero
                
                fig3.update_layout(
                    height=300, margin=dict(l=0,r=0,t=30,b=0),
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, title="Trades"),
                    yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                    showlegend=False,
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True)
                
            # Métricas Resumen al final
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Win Rate Actual", f"{evo_wr[-1]*100:.1f}%", delta=f"{(evo_wr[-1] - evo_wr[0])*100:.1f}% vs Inicio")
            m2.metric("R/B Promedio Actual", f"{evo_rr[-1]:.2f}", delta=f"{evo_rr[-1] - evo_rr[0]:.2f} vs Inicio")
            m3.metric("Esperanza Matemática", f"{evo_expectancy[-1]:.2f} R", help="Promedio de R ganados por trade neto.")

        else:
            st.info("Necesitas registrar al menos 5 operaciones cerradas para ver la evolución de tu edge.")






























