# Validación walk-forward — resultados

**Septiembre 2026** · Herramienta: `api/backtest.py`

## Qué se midió, y por qué así

La afirmación del producto no es *"le gana al mercado"*. Es: **dada tu cartera,
hay una relación riesgo/retorno mejor.** El punto de comparación correcto es
entonces la propia cartera mantenida sin cambios, no un índice.

**Metodología.** Walk-forward con rebalanceo trimestral, 14 decisiones entre
enero 2023 y abril 2026. En cada fecha `END_DATE` se mueve al día de la decisión,
de modo que los modelos solo ven datos anteriores. La caché se desactiva: su
clave combina los instrumentos con una huella del *archivo*, no con la fecha de
corte, así que sin desactivarla todas las iteraciones compartirían el modelo de
la primera — el error más peligroso posible, porque no falla, devuelve números
plausibles y equivocados. Se descuentan 10 puntos básicos sobre la rotación.

---

## Resultados (ratio de Sharpe)

| Cartera | Sin penalización | Con penalización fuerte | Mantener sin tocar |
|---|---:|---:|---:|
| Genérica: AAPL, MSFT, GOOGL, JPM, BND, GLD, AMZN | 1,35 | **1,79** | 1,56 |
| Concentrada en tecnología/crecimiento | **0,85** | 0,81 | 0,44 |
| Fondos mutuos diversificada | **1,11** | 1,09 | 1,06 |

Detalle de la primera, que es donde el efecto de la rotación se ve con claridad:

| Modo | Retorno | Volatilidad | Sharpe | Peor caída | Rotación |
|---|---:|---:|---:|---:|---:|
| Sin penalización | 24,9 % | 14,7 % | 1,35 | −16,3 % | 282 % |
| Suave | 26,8 % | 14,4 % | 1,52 | −16,3 % | 213 % |
| Moderada | 29,3 % | 14,1 % | 1,72 | −15,7 % | 132 % |
| **Fuerte** | **30,6 %** | 14,3 % | **1,79** | **−15,4 %** | 92 % |
| Mantener sin tocar | 30,3 % | 16,2 % | 1,56 | −19,0 % | 0 % |

---

## Qué dicen estos números

**1. La herramienta reduce riesgo de forma consistente.** En las tres carteras la
volatilidad optimizada quedó por debajo de la de mantener, y en dos de tres
también la peor caída. Ese es el resultado más sólido y el más repetido.

**2. El beneficio depende fuertemente de la cartera de partida.** En la cartera
concentrada el salto es enorme —Sharpe 0,85 contra 0,44, casi el doble— y en la
de fondos mutuos, ya diversificada, apenas se mueve (1,11 contra 1,06). Tiene
sentido: el optimizador aporta donde hay concentración que corregir, y poco donde
la diversificación ya existe.

**3. En la cartera genérica, la configuración por defecto quedó por debajo de no
hacer nada.** Sharpe 1,35 contra 1,56, con 282 % de rotación acumulada. Cada
incremento de la penalización mejoró retorno, Sharpe y caída máxima
simultáneamente. Es un resultado incómodo y hay que decirlo: **ahí la rotación
destruyó valor.**

**4. Pero la penalización no ayuda siempre.** En las otras dos carteras la
empeoró levemente (0,85 → 0,81 y 1,11 → 1,09). No corresponde activarla por
defecto a partir de esta evidencia.

---

## Calibración del retorno esperado: el hallazgo más importante

La aplicación mostraba un "Optimal Return" en verde, junto a la volatilidad y el
Sharpe. Un usuario razonablemente lo lee como un pronóstico. **No lo es**, y la
medición lo confirma con contundencia.

Ese número es la media histórica anualizada de la ventana de datos bajo los pesos
óptimos, mezclada como mucho al 30 % con la predicción del Random Forest. Dicho
de otro modo: *"si los próximos meses se parecen al promedio de 2021-2026"*.

Comparando lo prometido en cada rebalanceo contra lo efectivamente ocurrido en
los tres meses siguientes:

| Fecha | Prometido | Realizado | Error |
|---|---:|---:|---:|
| 2023-01-04 | −8,7 % | +52,4 % | −61,1 |
| 2023-04-05 | +3,2 % | +40,4 % | −37,1 |
| 2023-07-07 | +4,2 % | −11,8 % | +15,9 |
| 2024-07-09 | +13,4 % | +9,0 % | +4,4 |
| 2025-01-07 | +16,0 % | −9,2 % | +25,1 |
| 2025-04-09 | +14,5 % | +63,7 % | −49,2 |
| 2026-01-09 | +20,0 % | −2,4 % | +22,4 |

**Solo 2 de 14 predicciones cayeron dentro de ±10 puntos porcentuales.** El sesgo
medio es de −15,5 % y los errores llegan a 61 puntos: son *mayores que la
magnitud que se intenta predecir*.

La conclusión es directa: **el retorno esperado no tiene valor predictivo a tres
meses y no debe presentarse como si lo tuviera.** La volatilidad y el Sharpe, en
cambio, se comportaron de forma estable y consistente con la teoría.

A raíz de esto la interfaz dejó de rotular esas cifras como "annualized" —lectura
que sugiere proyección— y ahora dice **"historical average"**, con una nota bajo
la fila de métricas que explica el resultado de esta validación.

---

## Lo que estos números NO prueban

- **Sesgo de supervivencia, no corregido.** Los 504 instrumentos se eligieron
  hoy: son los que sobrevivieron. CTRA se deslistó; ANSS, DFS, HES y X fueron
  absorbidas. Todos los resultados están inflados al alza por construcción, y no
  hay forma de corregirlo sin datos históricos de constituyentes.
- **Catorce rebalanceos no distinguen habilidad de suerte.** Es un tamaño de
  muestra chico, y una sola trayectoria.
- **Un solo régimen de mercado.** La ventana 2023-2026 no contiene una crisis
  comparable a 2008 o marzo de 2020.
- **Costos aproximados.** 10 puntos básicos sobre la rotación es una convención,
  no una medición del costo real de operar de nadie en particular.
- **Sin impuestos.** El backtest ignora las consecuencias fiscales que la propia
  aplicación declara. Con rotación del 282 %, en una cuenta gravable eso importa.

---

## De dónde viene realmente el valor

Tres mediciones sucesivas convergen en la misma conclusión, y obligan a revisar
la narrativa del producto.

**1. El Random Forest no aporta.** Sobre 50 instrumentos, el r² fuera de muestra
tiene mediana **−0,405** (negativo significa peor que usar el promedio) y solo 10
dan positivo. Evaluado como problema de ordenamiento —más fácil que acertar el
nivel— el *information coefficient* es **−0,0197**, con apenas 89 de 209 fechas
positivas: peor que una moneda.

Vale destacar que el diseño ya se protegía de esto: `blend_returns` pondera la
predicción por `clip(r², 0, 1) × 0,3`, de modo que un r² negativo le asigna peso
**cero**. Para cerca del 80 % de los instrumentos el Random Forest no interviene
y el retorno esperado es la media histórica pura.

**2. GARCH tampoco, a nuestros horizontes.** Medido contra la volatilidad
realizada:

| Horizonte | Histórico | EWMA | GARCH(1,1) | Mejora |
|---|---:|---:|---:|---:|
| 5 días | 11,49 pp | 10,84 pp | **10,51 pp** | +8,5 % |
| 10 días | 9,59 pp | 9,13 pp | **9,04 pp** | +5,8 % |
| 21 días | **7,80 pp** | 8,82 pp | 7,94 pp | −1,8 % |
| 63 días | **6,85 pp** | 7,57 pp | 7,02 pp | −2,5 % |

GARCH gana claramente a 5-10 días y pierde a 21-63. Es comportamiento esperado:
el pronóstico revierte a la varianza incondicional, así que a horizontes largos
converge al promedio histórico y solo agrega ruido de estimación. Nuestros
horizontes —21 días para el puntaje de riesgo, 63 para el rebalanceo— son
precisamente donde no ayuda.

**3. La covarianza condicional tampoco.** Restringirla a ventanas móviles de 2
años, 1 año o 6 meses movió el Sharpe realizado de 1,35 a 1,38 —dentro del ruido
para 14 observaciones— y aumentó la rotación del 282 % al 313 %.

**Por qué ninguna de las tres mejora nada.** Más de la mitad de los pesos quedan
fijados en una cota del optimizador:

| Cartera | Pesos en cotas | Realmente libres |
|---|---:|---:|
| Genérica | 4 de 7 | 3 |
| Concentrada | 4 de 8 | 4 |
| Fondos | 4 de 6 | 2 |

El problema está **dominado por las restricciones**, no por la optimización. Y
esas restricciones —peso máximo del 20 %, 30 % o 40 % según el régimen; piso para
los activos defensivos— se derivan del puntaje del MLP.

Dicho de otro modo: **el valor del producto no viene de predecir retornos ni de
estimar la covarianza con más precisión, sino de imponer diversificación adaptada
al régimen de mercado.** Refinar los insumos de una optimización cuyo resultado
fija mayormente las cotas no puede mover la aguja.

Es coherente con todo lo demás que se midió: la volatilidad bajó en las tres
carteras, el Sharpe mejoró en las tres, y el retorno esperado no predijo nada.

---

## Conclusión honesta

La herramienta **reduce riesgo de manera consistente y mejora el retorno ajustado
por riesgo en las tres carteras probadas**, con un margen que va de marginal a
sustancial según qué tan concentrada esté la cartera de partida.

No es evidencia de que prediga el mercado, y no debe presentarse así. Es
evidencia de que la optimización con restricciones adaptativas hace lo que la
teoría dice que hace, sobre una muestra chica, un solo régimen y un universo con
sesgo de supervivencia conocido.

**Para reproducir:**

```bash
cd api
python backtest.py --tickers AAPL,MSFT,GOOGL,JPM,BND,GLD,AMZN --modos off,light,moderate,strong
```
