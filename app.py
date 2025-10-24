import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("📈 Herramienta de Trading y Backtester Automático")

# --- 1. CONFIGURACIÓN EN LA BARRA LATERAL ---
st.sidebar.header("Parámetros de Configuración")
ACTIVO = st.sidebar.text_input("Símbolo del Activo", value="AAPL")
TIMEFRAME = st.sidebar.selectbox("Timeframe", ['1d', '1h', '5m'], index=0)
PERIODO = st.sidebar.selectbox("Período de Datos", ['1y', '2y', '5y'], index=1)

# Parámetros del Backtester
st.sidebar.header("Parámetros del Backtester")
STOP_LOSS_PCT = st.sidebar.slider("Stop-Loss (%)", 1, 20, 5) / 100
TAKE_PROFIT_PCT = st.sidebar.slider("Take-Profit (%)", 1, 30, 10) / 100

# --- 2. OBTENER DATOS Y CALCULAR INDICADORES ---
@st.cache_data
def cargar_datos(activo, periodo, intervalo):
    datos = yf.download(activo, period=periodo, interval=intervalo)
    if datos.empty:
        st.error(f"No se pudieron descargar los datos para {activo}. Revisa el símbolo.")
        return None
    
    # Verificar si tenemos un MultiIndex en las columnas y aplanarlo si es necesario
    if isinstance(datos.columns, pd.MultiIndex):
        st.warning("Los datos tienen un MultiIndex. Aplanando las columnas...")
        # Aplanar el MultiIndex
        datos.columns = ['_'.join(col).strip() for col in datos.columns.values]
        # Renombrar columnas para que coincidan con lo que espera pandas_ta
        datos = datos.rename(columns={
            'Open_' + activo: 'Open',
            'High_' + activo: 'High',
            'Low_' + activo: 'Low',
            'Close_' + activo: 'Close',
            'Adj Close_' + activo: 'Adj Close',
            'Volume_' + activo: 'Volume'
        })
    
    return datos

datos_historicos = cargar_datos(ACTIVO, PERIODO, TIMEFRAME)

if datos_historicos is not None:
    # --- 3. APLICAR ALGORITMOS Y SEÑALES ---
    try:
        datos_historicos.ta.ema(length=20, append=True)
        datos_historicos.ta.ema(length=50, append=True)
        datos_historicos.ta.rsi(length=14, append=True)
        datos_historicos['tendencia_alcista'] = (datos_historicos['Close'] > datos_historicos['EMA_20']) & (datos_historicos['Close'] > datos_historicos['EMA_50'])
        datos_historicos['senal_compra_filtrada'] = (datos_historicos['Close'] > datos_historicos['EMA_20']) & (datos_historicos['Close'].shift(1) <= datos_historicos['EMA_20']) & datos_historicos['tendencia_alcista'] & (datos_historicos['RSI_14'] < 70)
    except Exception as e:
        st.error(f"Error al calcular indicadores técnicos: {str(e)}")
        st.stop()

    # --- 4. BACKTESTER ---
    resultados_operaciones = []
    en_posicion = False
    precio_entrada = 0

    for i in range(len(datos_historicos)):
        fila_actual = datos_historicos.iloc[i]
        if not en_posicion and fila_actual['senal_compra_filtrada']:
            en_posicion = True
            precio_entrada = fila_actual['Close']
        elif en_posicion:
            precio_salida = 0
            if fila_actual['High'] >= precio_entrada * (1 + TAKE_PROFIT_PCT):
                precio_salida = precio_entrada * (1 + TAKE_PROFIT_PCT)
            elif fila_actual['Low'] <= precio_entrada * (1 - STOP_LOSS_PCT):
                precio_salida = precio_entrada * (1 - STOP_LOSS_PCT)
            elif fila_actual['senal_compra_filtrada']: # Salida por nueva señal
                precio_salida = fila_actual['Close']
            
            if precio_salida > 0:
                rentabilidad = (precio_salida - precio_entrada) / precio_entrada
                resultados_operaciones.append(rentabilidad)
                en_posicion = False

    # --- 5. MOSTRAR RESULTADOS ---
    st.header("Resultados del Backtester")
    if not resultados_operaciones:
        st.warning("No se generaron operaciones en el período seleccionado con los parámetros actuales.")
    else:
        total_ops = len(resultados_operaciones)
        ops_ganadoras = sum(1 for r in resultados_operaciones if r > 0)
        rentabilidad_total = sum(resultados_operaciones)
        porcentaje_aciertos = (ops_ganadoras / total_ops) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Operaciones", total_ops)
        col2.metric("Ops. Ganadoras", ops_ganadoras)
        col3.metric("% Aciertos", f"{porcentaje_aciertos:.2f}%")
        col4.metric("Rentabilidad Total", f"{rentabilidad_total * 100:.2f}%", delta=f"{rentabilidad_total * 100:.2f}%")

    # --- 6. VISUALIZACIÓN ---
    st.header(f"Gráfico de {ACTIVO}")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=('Precio', 'RSI (14)'), row_width=[0.2, 0.7])
    fig.add_trace(go.Candlestick(x=datos_historicos.index, open=datos_historicos['Open'], high=datos_historicos['High'], low=datos_historicos['Low'], close=datos_historicos['Close'], name='Precio'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['EMA_20'], line=dict(color='orange', width=1), name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['EMA_50'], line=dict(color='blue', width=1), name='EMA 50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos[datos_historicos['senal_compra_filtrada']].index, y=datos_historicos[datos_historicos['senal_compra_filtrada']]['Close'], mode='markers', marker_symbol='triangle-up', marker_size=12, marker_color='lime', name='Señal Compra'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, y=datos_historicos['RSI_14'], line=dict(color='purple'), name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=800)
    st.plotly_chart(fig, use_container_width=True)
