La Lógica Detrás de la Señal de Compra
Primero, recordemos qué significa ese triángulo verde (🔺) que aparece en el gráfico. Tu herramienta solo muestra una señal de compra cuando se cumplen TRES condiciones al mismo tiempo:

Cruce Alcista de Corto Plazo: El precio de cierre cruza hacia arriba la Media Móvil Exponencial (EMA) de 20 períodos. Esto indica un posible cambio de momentum a corto plazo.
Confirmación de Tendencia Alcista: El precio de cierre está por encima de ambas medias móviles (la de 20 y la de 50). Esto asegura que no estamos comprando en una tendencia general a la baja.
No Sobrecomprado: El Índice de Fuerza Relativa (RSI) es menor a 70. Esto nos ayuda a evitar comprar en un pico de precio, cuando el activo podría estar "sobrecalentado" y a punto de corregir a la baja.
En resumen: la señal de compra busca un momento de entrada con momentum a corto plazo, dentro de una tendencia alcista confirmada y a un precio razonable.

Guía Práctica: Cómo Interpretar la Herramienta
Sigue estos pasos cada vez que quieras analizar un activo:

Paso 1: Configura tu Escenario
Elige el Activo: En la barra lateral, prueba con diferentes símbolos (AAPL, TSLA, MSFT, BTC-USD, etc.).
Ajusta el Período: Usa 2y (2 años) para tener una buena perspectiva de cómo funciona la estrategia en diferentes ciclos de mercado.
Configura tu Gestión del Riesgo: Los sliders de Stop-Loss (5%) y Take-Profit (10%) son cruciales. Definen cuánto estás dispuesto a perder por operación y cuál es tu objetivo de ganancia.
Paso 2: Analiza el Gráfico Principal
Mira la parte superior del gráfico:

Busca los Triángulos Verdes (🔺): Esos son tus puntos de entrada ideales según la estrategia. Pasa el ratón sobre ellos para ver la fecha y el precio exacto.
Observa las Medias Móviles (EMA 20 naranja, EMA 50 azul):
¿El precio se mantiene consistentemente por encima de ellas? ¡Buena señal!
¿El precio las usa como soporte (rebota en ellas)? ¡Excelente!
Si el precio cae por debajo de la EMA 50, la estrategia no dará señales hasta que vuelva a subir.
Visualiza la Tendencia: ¿El gráfico general va de abajo a arriba (izquierda a derecha)? Eso es una tendencia alcista.
Paso 3: Observa el Indicador RSI
Mira la parte inferior del gráfico:

Zona de Sobrecompra (>70): Si el RSI está en esta zona, la estrategia no generará señales de compra. Es una señal de precaución.
Zona de Sobreventa (<30): Aunque tu estrategia no compra específicamente aquí, es un indicador de que el activo está "barato" y podría haber un rebote pronto.
Líneas Medias (30-70): La mayoría de tus señales de compra deberían ocurrir cuando el RSI está en esta zona, confirmando que no es un momento de euforia.
Paso 4: La Parte Más Importante - Los Resultados del Backtester
Esto es lo que valida (o invalida) la estrategia para ese activo. No te fíes solo del gráfico.

Total Operaciones: ¿Demasiadas pocas? La estrategia es muy selectiva. ¿Demasiadas? Quizás sea demasiado sensible.
Ops. Ganadoras y % Aciertos: Un % de aciertos alto (ej. >50%) es bueno, pero no lo es todo.
Rentabilidad Total: ¡Esta es la métrica clave! ¿La estrategia fue rentable en total durante los 2 años? Si el número es positivo y grande, significa que la estrategia funcionó bien para ese activo.
Ejemplo práctico de análisis:

"Para AAPL en 2 años, la estrategia generó 15 operaciones con un 60% de aciertos y una rentabilidad total del 25%. Parece sólida. Pero si pruebo con TSLA, la rentabilidad es del -5%. Esto significa que esta estrategia específica no funciona bien para TSLA, y debería buscar otra o ajustar los parámetros."

¿Cuándo Vender? La Lógica de Salida
Tu herramienta ya tiene definida una lógica de salida clara en el backtester. Así es como debes pensar en la venta:

Por Take-Profit (Ganancia): Cuando el precio alcanza tu objetivo de ganancia (ej. un 10% por encima de tu precio de compra). ¡Vende! Es la forma disciplinada de asegurar beneficios. La codicia es el enemigo del trader.
Por Stop-Loss (Pérdida): Cuando el precio cae a tu límite de pérdida (ej. un 5% por debajo de tu precio de compra). ¡Vende inmediatamente! Esto es lo más importante. Proteger tu capital es la prioridad número uno. Una pequeña pérdida es recuperable, una gran pérdida puede arruinarte.
Por Nueva Señal: Si aparece un nuevo triángulo verde mientras ya tienes una posición abierta, la estrategia considera que la oportunidad original ya no es válida y te obliga a salir para poder entrar en la nueva.
Advertencia Final y Próximos Pasos
Esto es un backtest: El rendimiento pasado no garantiza el rendimiento futuro. El mercado cambia.
No es en tiempo real: La herramienta analiza datos históricos. No sabe qué va a pasar mañana.
Experimenta: ¡Juega con los parámetros! ¿Qué pasa si pones un Stop-Loss del 3%? ¿Y si el Take-Profit es del 15%? Observa cómo cambian los resultados. Así encontrarás la configuración que mejor se adapta a tu tolerancia al riesgo.
Ahora tienes el conocimiento. Úsalo para analizar, no para adivinar. ¡Feliz trading y análisis
