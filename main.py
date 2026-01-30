import streamlit as st
import pandas as pd
import database as db
import auth
import time
import plotly.express as px 
from datetime import date

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Edge Journal", page_icon="📓", layout="wide")

# Inicializar conexión a BD
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
                    # user_data = (username, password_hash, name, created_at)
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
                    if db.create_user(new_user, hashed_pw, new_name):
                        st.success("¡Cuenta creada! Ahora inicia sesión en la otra pestaña.")
                    else:
                        st.error("Error: El usuario ya existe o hubo un problema de conexión.")
                else:
                    st.warning("Completa todos los campos.")

# --- VISTA: DASHBOARD PRINCIPAL ---
def dashboard_page():
    # BARRA LATERAL
    with st.sidebar:
        st.header(f"Hola, {st.session_state['user_name']}")
        st.write(f"ID: `{st.session_state['username']}`")
        if st.button("Cerrar Sesión"):
            st.session_state['logged_in'] = False
            st.session_state['username'] = None
            st.rerun()
        st.divider()
        st.caption("Edge Journal Cloud v2.0")

    st.title("Gestión de Cartera 🏦")

    # TRES PESTAÑAS PRINCIPALES
    tab_active, tab_history, tab_stats = st.tabs(["⚡ Posiciones Abiertas", "📚 Bitácora Cerrada", "📊 Analytics Pro"])

    # ------------------------------------------------------------------
    # TAB 1: GESTIÓN ACTIVA (Abrir y Cerrar Trades)
    # ------------------------------------------------------------------
    with tab_active:
        col_left, col_right = st.columns([1, 1])
        
        # IZQUIERDA: ABRIR NUEVA POSICIÓN
        with col_left:
            st.subheader("➕ Abrir Nueva Posición")
            with st.form("new_trade_form"):
                c1, c2 = st.columns(2)
                symbol = c1.text_input("Ticker (Ej: SPY)").upper()
                side = c2.selectbox("Dirección", ["LONG", "SHORT"])
                
                c3, c4 = st.columns(2)
                price = c3.number_input("Precio Entrada", min_value=0.0, format="%.2f")
                qty = c4.number_input("Cantidad", min_value=1, step=1)
                
                date_in = st.date_input("Fecha Entrada", value=date.today())
                notes = st.text_area("Tesis de Inversión / Notas")
                
                if st.form_submit_button("🚀 Ejecutar Orden", type="primary"):
                    if symbol and price > 0:
                        db.open_new_trade(st.session_state['username'], symbol, side, price, qty, date_in, notes)
                        st.success(f"Orden ejecutada: {side} {symbol}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("Faltan datos obligatorios.")

        # DERECHA: VER Y CERRAR POSICIONES
        with col_right:
            st.subheader("🔓 Posiciones Abiertas")
            df_open = db.get_open_trades(st.session_state['username'])
            
            if not df_open.empty:
                # 1. Mostrar Tabla Simple
                st.dataframe(
                    df_open,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": None, # Ocultar ID
                        "entry_price": st.column_config.NumberColumn("Entrada", format="$ %.2f"),
                        "entry_date": st.column_config.DateColumn("Fecha"),
                    }
                )
                
                st.divider()
                st.write("👇 **Cerrar Operación**")
                
                # 2. Selector Inteligente
                # Creamos una etiqueta legible para el dropdown
                df_open['display_label'] = df_open.apply(
                    lambda x: f"#{x['id']} | {x['symbol']} ({x['side']}) | Qty: {x['quantity']}", axis=1
                )
                
                trade_selection = st.selectbox("Selecciona la posición a cerrar:", df_open['display_label'])
                
                # Obtener el ID real de la selección
                selected_id = int(trade_selection.split("|")[0].replace("#", "").strip())
                # Obtener los datos de esa fila para calcular PnL
                trade_data = df_open[df_open['id'] == selected_id].iloc[0]
                
                # Formulario de Cierre
                with st.form("close_trade_form"):
                    st.caption(f"Cerrando {trade_data['symbol']} (Entrada: ${trade_data['entry_price']})")
                    c_exit1, c_exit2 = st.columns(2)
                    exit_price = c_exit1.number_input("Precio Salida", min_value=0.0, format="%.2f")
                    exit_date = c_exit2.date_input("Fecha Salida", value=date.today())
                    
                    if st.form_submit_button("🔒 Confirmar Cierre"):
                        if exit_price > 0:
                            # Calcular PnL Matemático
                            pnl = 0.0
                            if trade_data['side'] == "LONG":
                                pnl = (exit_price - trade_data['entry_price']) * trade_data['quantity']
                            else: # SHORT
                                pnl = (trade_data['entry_price'] - exit_price) * trade_data['quantity']
                            
                            # Guardar en BD
                            db.close_trade(selected_id, exit_price, exit_date, pnl)
                            
                            st.balloons()
                            st.success(f"Trade cerrado. PnL: ${pnl:.2f}")
                            time.sleep(1.5)
                            st.rerun()
            else:
                st.info("No tienes posiciones abiertas actualmente.")

    # ------------------------------------------------------------------
    # TAB 2: HISTORIAL (Cerradas)
    # ------------------------------------------------------------------
    with tab_history:
        st.subheader("📜 Bitácora de Operaciones Cerradas")
        df_closed = db.get_closed_trades(st.session_state['username'])
        
        if not df_closed.empty:
            st.dataframe(
                df_closed,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pnl": st.column_config.NumberColumn("PnL Realizado", format="$ %.2f"),
                    "entry_price": st.column_config.NumberColumn("Entrada", format="$ %.2f"),
                    "exit_price": st.column_config.NumberColumn("Salida", format="$ %.2f"),
                    "entry_date": st.column_config.DateColumn("Fecha In"),
                }
            )
        else:
            st.write("Aún no has cerrado operaciones.")

    # ------------------------------------------------------------------
    # TAB 3: ANALYTICS (Gráficos)
    # ------------------------------------------------------------------
    with tab_stats:
        st.subheader("Tablero de Comando")
        # Traemos TODO para calcular métricas globales
        df_all = db.get_all_trades_for_analytics(st.session_state['username'])
        
        if not df_all.empty:
            # Separar Abiertos vs Cerrados para métricas distintas
            df_c = df_all[df_all['exit_price'] > 0].copy() # Cerrados
            df_o = df_all[(df_all['exit_price'].isna()) | (df_all['exit_price'] == 0)].copy() # Abiertos
            
            # --- KPIs SUPERIORES ---
            total_pnl = df_c['pnl'].sum() if not df_c.empty else 0
            open_exposure = (df_o['entry_price'] * df_o['quantity']).sum() if not df_o.empty else 0
            
            # Win Rate
            wins = len(df_c[df_c['pnl'] > 0])
            total_closed = len(df_c)
            win_rate = (wins / total_closed * 100) if total_closed > 0 else 0
            
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("PnL Total Realizado", f"$ {total_pnl:,.2f}")
            k2.metric("Exposición Activa", f"$ {open_exposure:,.2f}")
            k3.metric("Win Rate", f"{win_rate:.1f}%")
            k4.metric("Trades Cerrados", total_closed)
            
            st.divider()
            
            # --- GRÁFICOS ---
            g1, g2 = st.columns([2, 1])
            
            with g1:
                st.subheader("📈 Curva de Capital (Equity Curve)")
                if not df_c.empty:
                    df_c = df_c.sort_values('entry_date')
                    df_c['cumulative_pnl'] = df_c['pnl'].cumsum()
                    
                    fig_line = px.line(df_c, x='entry_date', y='cumulative_pnl', markers=True)
                    fig_line.update_traces(line_color='#00BBA2', line_width=3)
                    st.plotly_chart(fig_line, use_container_width=True)
                else:
                    st.info("Necesitas cerrar trades para generar la curva.")
            
            with g2:
                st.subheader("🍰 Cartera Actual")
                if not df_o.empty:
                    # Calculamos tamaño de posición: Precio * Cantidad
                    df_o['position_size'] = df_o['entry_price'] * df_o['quantity']
                    # Agrupar por Ticker (por si tienes varias entradas en el mismo activo)
                    pie_data = df_o.groupby('symbol')['position_size'].sum().reset_index()
                    
                    fig_pie = px.pie(pie_data, values='position_size', names='symbol', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No hay posiciones abiertas.")
        else:
            st.warning("No hay datos para analizar. ¡Carga tu primer trade!")

# --- CONTROLADOR PRINCIPAL ---
def main():
    if st.session_state['logged_in']:
        dashboard_page()
    else:
        login_page()

if __name__ == '__main__':
    main()