import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

st.set_page_config(layout="wide", initial_sidebar_state="expanded")

st.title("📈 Herramienta de Trading y Backtester Mejorada")

# --- 1. CONFIGURACIÓN EN LA BARRA LATERAL ---
st.sidebar.header("Parámetros de Configuración")
ACTIVO = st.sidebar.text_input("Símbolo del Activo", value="AAPL")
TIMEFRAME = st.sidebar.selectbox("Timeframe", ['1d', '1h', '5m'], index=0)
PERIODO = st.sidebar.selectbox("Período de Datos", ['1y', '2y', '5y'], index=1)

# Parámetros del Backtester
st.sidebar.header("Parámetros del Backtester")
STOP_LOSS_PCT = st.sidebar.slider("Stop-Loss (%)", 1, 20, 5) / 100
TAKE_PROFIT_PCT = st.sidebar.slider("Take-Profit (%)", 1, 30, 10) / 100
USE_TRAILING_STOP = st.sidebar.checkbox("Usar Trailing Stop", value=True)
TRAILING_STOP_PCT = st.sidebar.slider("Trailing Stop (%)", 1, 20, 3) / 100

# Estrategia a utilizar
st.sidebar.header("Estrategia de Trading")
ESTRATEGIA = st.sidebar.selectbox("Selecciona Estrategia", 
                                  ['Momentum', 'Mean Reversion', 'Combinada'], 
                                  index=2)

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
    # Calcular indicadores básicos
    datos_historicos.ta.ema(length=20, append=True)
    datos_historicos.ta.ema(length=50, append=True)
    datos_historicos.ta.rsi(length=14, append=True)
    datos_historicos.ta.bbands(length=20, std=2, append=True)
    datos_historicos.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Calcular media móvil de volumen
    datos_historicos['Volume_SMA'] = datos_historicos['Volume'].rolling(window=20).mean()
    
    # ATR para ajustes dinámicos
    datos_historicos.ta.atr(length=14, append=True)
    
    # Estrategia 1: Momentum (mejorada)
    datos_historicos['tendencia_alcista'] = (datos_historicos['Close'] > datos_historicos['EMA_20']) & (datos_historicos['Close'] > datos_historicos['EMA_50'])
    datos_historicos['volumen_alto'] = datos_historicos['Volume'] > datos_historicos['Volume_SMA'] * 1.2
    datos_historicos['senal_momentum'] = (
        (datos_historicos['Close'] > datos_historicos['EMA_20']) & 
        (datos_historicos['Close'].shift(1) <= datos_historicos['EMA_20']) & 
        datos_historicos['tendencia_alcista'] & 
        (datos_historicos['RSI_14'] < 70) &
        datos_historicos['volumen_alto']
    )
    
    # Estrategia 2: Mean Reversion
    datos_historicos['senal_mean_reversion'] = (
        (datos_historicos['Close'] < datos_historicos['BBL_20_2.0']) & 
        (datos_historicos['RSI_14'] < 30) &
        datos_historicos['volumen_alto']
    )
    
    # Estrategia 3: Combinada
    datos_historicos['senal_combinada'] = datos_historicos['senal_momentum'] | datos_historicos['senal_mean_reversion']
    
    # Seleccionar la estrategia
    if ESTRATEGIA == 'Momentum':
        datos_historicos['senal_compra'] = datos_historicos['senal_momentum']
    elif ESTRATEGIA == 'Mean Reversion':
        datos_historicos['senal_compra'] = datos_historicos['senal_mean_reversion']
    else:  # Combinada
        datos_historicos['senal_compra'] = datos_historicos['senal_combinada']
    
    # --- 4. BACKTESTER MEJORADO ---
    resultados_operaciones = []
    en_posicion = False
    precio_entrada = 0
    stop_loss_actual = 0
    take_profit_actual = 0
    
    for i in range(len(datos_historicos)):
        fila_actual = datos_historicos.iloc[i]
        
        # Entrada en posición
        if not en_posicion and fila_actual['senal_compra']:
            en_posicion = True
            precio_entrada = fila_actual['Close']
            
            # Calcular stop loss y take profit dinámicos basados en ATR
            atr_actual = fila_actual['ATRr_14']
            stop_loss_actual = precio_entrada * (1 - max(STOP_LOSS_PCT, atr_actual * 2))
            take_profit_actual = precio_entrada * (1 + TAKE_PROFIT_PCT)
            
        # Gestión de posición existente
        elif en_posicion:
            precio_salida = 0
            
            # Actualizar trailing stop si está activado
            if USE_TRAILING_STOP:
                nuevo_trailing_stop = fila_actual['Close'] * (1 - TRAILING_STOP_PCT)
                if nuevo_trailing_stop > stop_loss_actual:
                    stop_loss_actual = nuevo_trailing_stop
            
            # Condiciones de salida
            if fila_actual['High'] >= take_profit_actual:
                precio_salida = take_profit_actual
            elif fila_actual['Low'] <= stop_loss_actual:
                precio_salida = stop_loss_actual
            elif ESTRATEGIA == 'Mean Reversion' and fila_actual['Close'] >= datos_historicos.iloc[i-1]['BBM_20_2.0']:
                precio_salida = fila_actual['Close']  # Salir al volver a la media
            
            # Registrar operación
            if precio_salida > 0:
                rentabilidad = (precio_salida - precio_entrada) / precio_entrada
                resultados_operaciones.append({
                    'entrada': precio_entrada,
                    'salida': precio_salida,
                    'rentabilidad': rentabilidad,
                    'fecha_entrada': datos_historicos.index[i-1],
                    'fecha_salida': datos_historicos.index[i]
                })
                en_posicion = False
    
    # --- 5. MOSTRAR RESULTADOS MEJORADOS ---
    st.header("Resultados del Backtester")
    if not resultados_operaciones:
        st.warning("No se generaron operaciones en el período seleccionado con los parámetros actuales.")
    else:
        df_operaciones = pd.DataFrame(resultados_operaciones)
        total_ops = len(df_operaciones)
        ops_ganadoras = sum(1 for r in df_operaciones['rentabilidad'] if r > 0)
        rentabilidad_total = df_operaciones['rentabilidad'].sum()
        porcentaje_aciertos = (ops_ganadoras / total_ops) * 100
        
        # Calcular métricas adicionales
        rentabilidad_media = df_operaciones['rentabilidad'].mean()
        max_ganancia = df_operaciones['rentabilidad'].max()
        max_perdida = df_operaciones['rentabilidad'].min()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Operaciones", total_ops)
        col2.metric("Ops. Ganadoras", ops_ganadoras)
        col3.metric("% Aciertos", f"{porcentaje_aciertos:.2f}%")
        col4.metric("Rentabilidad Total", f"{rentabilidad_total * 100:.2f}%", delta=f"{rentabilidad_total * 100:.2f}%")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Rentabilidad Media", f"{rentabilidad_media * 100:.2f}%")
        col6.metric("Máx. Ganancia", f"{max_ganancia * 100:.2f}%")
        col7.metric("Máx. Pérdida", f"{max_perdida * 100:.2f}%")
        col8.metric("Factor Beneficio", f"{abs(df_operaciones[df_operaciones['rentabilidad'] > 0]['rentabilidad'].sum() / df_operaciones[df_operaciones['rentabilidad'] < 0]['rentabilidad'].sum()):.2f}")
        
        # Tabla de operaciones
        with st.expander("Ver Detalles de Operaciones"):
            st.dataframe(df_operaciones)
    
    # --- 6. VISUALIZACIÓN MEJORADA ---
    st.header(f"Gráfico de {ACTIVO}")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, 
                        subplot_titles=('Precio', 'RSI (14)', 'Volumen'), 
                        row_width=[0.2, 0.2, 0.7])
    
    # Gráfico de precios
    fig.add_trace(go.Candlestick(x=datos_historicos.index, 
                                open=datos_historicos['Open'], 
                                high=datos_historicos['High'], 
                                low=datos_historicos['Low'], 
                                close=datos_historicos['Close'], 
                                name='Precio'), row=1, col=1)
    
    # Medias móviles
    fig.add_trace(go.Scatter(x=datos_historicos.index, 
                            y=datos_historicos['EMA_20'], 
                            line=dict(color='orange', width=1), 
                            name='EMA 20'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, 
                            y=datos_historicos['EMA_50'], 
                            line=dict(color='blue', width=1), 
                            name='EMA 50'), row=1, col=1)
    
    # Bandas de Bollinger
    fig.add_trace(go.Scatter(x=datos_historicos.index, 
                            y=datos_historicos['BBU_20_2.0'], 
                            line=dict(color='lightgray', width=1), 
                            name='Banda Superior'), row=1, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, 
                            y=datos_historicos['BBL_20_2.0'], 
                            line=dict(color='lightgray', width=1), 
                            fill='tonexty', 
                            name='Banda Inferior'), row=1, col=1)
    
    # Señales de compra
    fig.add_trace(go.Scatter(x=datos_historicos[datos_historicos['senal_compra']].index, 
                            y=datos_historicos[datos_historicos['senal_compra']]['Close'], 
                            mode='markers', 
                            marker_symbol='triangle-up', 
                            marker_size=12, 
                            marker_color='lime', 
                            name='Señal Compra'), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=datos_historicos.index, 
                            y=datos_historicos['RSI_14'], 
                            line=dict(color='purple'), 
                            name='RSI 14'), row=2, col=1)
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volumen
    fig.add_trace(go.Bar(x=datos_historicos.index, 
                        y=datos_historicos['Volume'], 
                        name='Volumen'), row=3, col=1)
    fig.add_trace(go.Scatter(x=datos_historicos.index, 
                            y=datos_historicos['Volume_SMA'], 
                            line=dict(color='orange', width=1), 
                            name='Media Volumen'), row=3, col=1)
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=1200)
    st.plotly_chart(fig, use_container_width=True)
