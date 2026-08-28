# PortfolioAI — Whitepaper técnico

**Rebalanceo de carteras mediante aprendizaje automático y teoría moderna de
portafolio, con divulgación explícita de costos fiscales y de rotación.**

Versión 1.0 · Agosto 2026 · Beta privada
https://portfolioai-optimizer.vercel.app

> **Aviso.** Este documento describe un sistema de investigación con fines
> educativos. No constituye asesoramiento financiero ni fiscal. Las predicciones,
> puntajes de riesgo y recomendaciones provienen de modelos estadísticos
> entrenados sobre datos históricos y pueden ser incorrectos.

---

## 1. Problema

Un inversor minorista dispone hoy de abundante información —gráficos, ratios,
carteras modelo— y de ninguna respuesta a la única pregunta operativa que tiene:
**dado lo que poseo hoy, qué debo comprar y qué debo vender.**

El salto desde "su Sharpe es 0,77" hasta "venda USD 2.700 de Apple" queda del
lado del usuario. Ese salto exige estimar retornos esperados, construir una
matriz de covarianza, resolver una optimización con restricciones y traducir
pesos a montos. Es exactamente donde la mayoría se detiene.

Un segundo problema, menos visible, agrava al primero: las herramientas que sí
producen una cartera objetivo **omiten lo que cuesta llegar a ella**. Un
rebalanceo hacia el óptimo teórico puede implicar rotar la mitad de la cartera y
realizar ganancias imponibles considerables. Presentar el destino sin el precio
del viaje es, en la práctica, información incompleta.

---

## 2. Enfoque

PortfolioAI encadena cinco modelos, cada uno con una función acotada, y expone
tanto el resultado como su costo.

```
posiciones actuales
        │
        ├─► K-Means ──────────► perfil de comportamiento por activo
        ├─► Random Forest ────► retorno esperado por instrumento
        ├─► MLP ──────────────► puntaje de régimen de mercado (0–1)
        │                             │
        │                             ▼ (modula las cotas)
        └─► Markowitz (SLSQP) ─► pesos óptimos
                    │
                    ▼
            motor de rebalanceo ──► órdenes en dólares
                    │
                    ├─► divulgación fiscal (ganancia realizada, tipo de cuenta)
                    └─► divulgación de rotación (% de cartera movida)
```

Decisión de diseño central: **la salida es ejecutable**. No pesos objetivo, sino
montos contra las posiciones que el usuario efectivamente posee.

---

## 3. Modelos

### 3.1 K-Means — perfilado de activos

**No supervisado.** Agrupa los instrumentos de la cartera en tres perfiles según
tres características anualizadas: retorno, volatilidad y ratio de Sharpe.

- `n_clusters = min(3, n_activos)`, `random_state = 42`, `n_init = 10`
- Estandarización previa con `StandardScaler`
- Etiquetado posterior por reglas, no por índice de clúster: volatilidad media
  inferior a 0,10 ⇒ `Defensive`; en caso contrario, retorno medio por encima de
  la mediana ⇒ `Growth`; el resto ⇒ `Moderate`

El etiquetado por reglas evita que las etiquetas cambien de significado entre
corridas, un problema habitual al nombrar clústeres por su índice.

**Uso posterior:** el clúster `Defensive` fija un piso de asignación en el
optimizador (§3.4).

### 3.2 Random Forest — retorno esperado por instrumento

**Supervisado, un modelo independiente por instrumento.**

- `RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=5,
  random_state=42)`
- Ventana de observación: 60 sesiones. Horizonte objetivo: 63 sesiones
  (aproximadamente un trimestre)
- Variable objetivo: retorno compuesto del período siguiente
- Siete características derivadas de la ventana: retorno y volatilidad
  anualizados, retorno de las últimas 5 y 20 sesiones, volatilidad de las
  últimas 20, proporción de sesiones positivas, y desvío del último retorno
  respecto de la media de la ventana
- Recorte de valores atípicos en el objetivo a 2 desvíos estándar
- Evaluación mediante `r2_score` sobre partición de validación

**Mezcla con el histórico.** La predicción no se utiliza cruda. Se combina con el
retorno histórico ponderando por la calidad medida del modelo:

```
peso = clip(r², 0, 1) × 0,3
retorno_esperado = peso × predicción_RF + (1 − peso) × retorno_histórico
```

El factor 0,3 acota la influencia máxima del Random Forest al 30 %, incluso con
un r² perfecto. Es una decisión deliberadamente conservadora: en series
financieras un r² de 0,05 ya se considera informativo, y confiar plenamente en la
predicción sería injustificado.

### 3.3 MLP — puntaje de régimen de mercado

**Red neuronal densa.** Este es el componente más fácil de malinterpretar, de modo
que conviene ser preciso sobre qué predice.

- `MLPRegressor(hidden_layer_sizes=(64, 32, 16), activation='relu', solver='adam',
  learning_rate='adaptive', max_iter=500, early_stopping=True, random_state=42)`
- Entrada: cuatro estadísticos por instrumento (retorno anualizado, volatilidad
  anualizada, proporción de sesiones positivas, retorno de las últimas 5
  sesiones) **más la correlación media entre pares** de la cartera
- **Variable objetivo: la volatilidad realizada de la cartera en las 21 sesiones
  siguientes.** No es una probabilidad de pérdida ni una predicción direccional
- Normalización del objetivo y compresión sigmoidea: `expit(2,5 × (y − 0,5))`,
  que mantiene el puntaje dentro de (0, 1) sin fijarlo en los extremos

La inclusión de la correlación media es intencional: dos carteras con idéntica
volatilidad individual por activo pero distinta correlación entre ellos presentan
riesgos agregados muy diferentes.

**El puntaje no es decorativo.** Modula directamente las cotas del optimizador
(§3.4). Esa es la razón por la que existe.

### 3.4 Markowitz — optimización con restricciones

**Optimización con `scipy.optimize.minimize`, método SLSQP.**

- Matriz de covarianza con **contracción de Ledoit-Wolf**, anualizada (× 252).
  La contracción es necesaria porque la covarianza muestral es inestable cuando
  el número de instrumentos se acerca al de observaciones
- Tasa libre de riesgo: 5 %
- Restricción de igualdad: los pesos suman 1
- **Cotas dependientes del puntaje del MLP:**

| Puntaje de riesgo | Peso máximo por activo | Piso para activos defensivos |
|---|---|---|
| < 0,35 | 40 % | 2 % |
| 0,35 – 0,65 | 30 % | 2 % ó 10 % |
| > 0,65 | 20 % | 10 % |

Un régimen tranquilo habilita concentración; uno volátil la impide. Se resuelven
dos carteras: Máximo Sharpe y Mínima Varianza.

### 3.5 Motor de rebalanceo

Traduce pesos a instrucciones:

```
diferencia = peso_óptimo × valor_total − valor_actual
acción     = COMPRAR si diferencia > 50; VENDER si < −50; MANTENER en otro caso
```

El umbral de USD 50 evita recomendar operaciones cuyo costo de transacción
superaría el beneficio.

### 3.6 Capa de lenguaje natural

**Claude Haiku 4.5** cumple dos funciones: explicar el resultado en lenguaje
llano, y atender instrucciones conversacionales ("agregá TSLA") que reejecutan el
pipeline completo.

Consideración de seguridad relevante: la salida del modelo se inserta en el DOM
mediante `textContent` y `createTextNode`, **nunca** mediante `innerHTML`. Esto
neutraliza la inyección de HTML a través del prompt, que es el vector natural en
una aplicación con un modelo de lenguaje intermediando.

---

## 4. Divulgación como característica del producto

La diferencia más sustantiva de PortfolioAI frente a un optimizador convencional
no está en los modelos sino en lo que declara sobre sus propias recomendaciones.

### 4.1 Rotación

La cartera de ejemplo, optimizada sin restricciones adicionales, implica **una
rotación del 53,3 %**. Ese número se muestra sobre la tabla de operaciones, antes
que ninguna otra cosa. No requiere ningún dato fiscal, y es la cifra con mayor
probabilidad de hacer que alguien reconsidere antes de operar.

### 4.2 Modos opcionales del optimizador

Ambos **desactivados por defecto**. Con ambos inactivos la salida del optimizador
es idéntica a la de la versión previa a su incorporación, propiedad verificada
mediante pesos de referencia capturados antes del cambio.

**Minimizar operaciones.** Añade una penalización cuadrática por desviarse de la
cartera actual:

```
maximizar  Sharpe − λ · Σ (wᵢ − wᵢ_actual)²
```

Cuadrática y no L1 para preservar la suavidad del objetivo ante SLSQP. Se expone
en tres niveles (λ = 0,5 / 2,0 / 6,0). Resultados medidos:

| Modo | Rotación | Sharpe |
|---|---:|---:|
| Desactivado | 53,3 % | 0,969 |
| Suave | 48,8 % | 0,956 |
| Moderado | 26,7 % | 0,799 |
| Fuerte | 15,1 % | 0,694 |

**Ambas caras se muestran siempre.** Operar menos cuesta rendimiento teórico, y
ocultarlo convertiría una opción informada en una trampa.

El nombre importa: se llama "minimizar operaciones", no "optimización fiscal".
No requiere ningún dato tributario y reduce la carga impositiva únicamente como
efecto secundario. Nombrarlo de otro modo sería una afirmación que el mecanismo
no respalda.

**Selección fiscalmente informada.** Mismo mecanismo, con pesos por activo
derivados de la ganancia latente. Exige que el usuario haya ingresado sus propias
alícuotas, y permanece deshabilitado hasta entonces.

### 4.3 Divulgación fiscal

El principio rector separa **hechos** de **estimaciones**:

- *Hecho:* "esta venta realiza USD 1.410 de ganancia, USD 820 provenientes de
  lotes con antigüedad menor a un año". Se calcula a partir del costo que informa
  el intermediario.
- *Estimación:* "usted pagará USD 340". Requiere tramo impositivo, situación
  fiscal y jurisdicción. **Solo se produce con las alícuotas que el propio
  usuario ingresa**, y se rotula como ilustrativa.

**No existe ninguna alícuota por defecto en el código.** Una prueba automatizada
inspecciona el módulo en busca de valores que parezcan tasas y falla si aparecen.

Tres propiedades garantizadas por pruebas, no por convención:

1. Una cuenta cuyo tipo no se reconoce se reporta como *no identificada*, **nunca**
   como gravable. El silencio es preferible a una afirmación incorrecta sobre el
   tratamiento fiscal de un tercero.
2. Sin lotes fiscales se informa la ganancia total pero **no** su descomposición
   entre corto y largo plazo, en lugar de inventarla.
3. La jurisdicción determina el método. Canadá exige costo promedio ajustado;
   FIFO, LIFO y HIFO no son admisibles bajo sus reglas, de modo que la selección
   se fuerza y se explica.

**El método de selección de lotes se declara en pantalla.** En una venta parcial
el método domina el resultado: la misma operación de USD 5.000 realiza USD 500
bajo HIFO y USD 4.000 bajo FIFO. Cuando los métodos difieren se muestra el rango
completo, porque una cifra única sugeriría una precisión inexistente.

### 4.4 Carácter fiscal por tipo de activo

Las distribuciones tributan de manera distinta según el instrumento. Ninguna API
provee esta clasificación, de modo que se mantiene una **lista verificada
manualmente** (`api/tax_profiles.json`).

La alternativa —clasificar por coincidencia de patrones sobre el nombre— se
intentó y falló en ambas direcciones: omitió VNQ y XLRE (ambos inmobiliarios) y
LQD y EMB (ambos de renta fija), y clasificó erróneamente MA, QCOM, TXN y UNH por
contener la palabra "Incorporated". La lista abarca 38 instrumentos; los 250
restantes se tratan como renta variable con una etiqueta genérica explícita.

Incluye un caso frecuentemente omitido: los fideicomisos de metales preciosos
(GLD, SLV, IAU) tributan en Estados Unidos como objetos de colección, con una
alícuota de largo plazo de hasta 28 % en lugar de la habitual de 15/20 %. GLD
integra la cartera por defecto.

---

## 5. Integración con intermediarios

Conexión mediante **SnapTrade**, exclusivamente de lectura. El sistema no ejecuta
órdenes ni puede mover fondos, por decisión explícita: hacerlo modificaría
sustancialmente el encuadre regulatorio del producto.

**Modelo de identidad.** Un *principal* posee la conexión, con dos variantes tras
una misma interfaz: sesión efímera sin registro (actual) y cuenta persistente
(prevista). Todas las rutas resuelven un token portador contra un principal sin
distinguir la variante.

**Consideraciones de almacenamiento:**

- Las posiciones **nunca se persisten**: se solicitan, se reconcilian, se
  devuelven y se descartan
- Del token se almacena únicamente su hash SHA-256; el token existe solo en el
  navegador
- El `userSecret` del intermediario se cifra con Fernet mediante una clave
  independiente de la de la base de datos. Un volcado de la base sin esa clave no
  otorga acceso a ninguna cuenta

**Reconciliación.** Las posiciones se consolidan entre cuentas —la teoría de
portafolio opera sobre la exposición total— pero cada una conserva su desglose
por cuenta, de modo que la operación resultante indique dónde se ubica. Las
posiciones no modelables se enumeran con su motivo específico: fuera del universo
de instrumentos, tipo no admitido, o excedente del límite de 15. La truncación
silenciosa sería el peor resultado posible: optimizar contra una cartera que no
es la del usuario.

Las monedas no se mezclan. Se adopta como base la de la cuenta de mayor valor y
el resto se reporta por separado, sin conversión, porque no se dispone de tipos
de cambio y un total incorrecto es más difícil de detectar que una exclusión
declarada.

---

## 6. Arquitectura y datos

| Componente | Tecnología |
|---|---|
| API | FastAPI 0.115 sobre Railway |
| Modelos | scikit-learn 1.5.2, SciPy 1.14.1, NumPy 2.1.3 |
| Frontend | HTML/JS sin framework, Chart.js, sobre Vercel |
| Persistencia | Supabase (lista de espera, analítica, principales) |
| Lenguaje natural | API de Anthropic (Claude Haiku 4.5) |

**Universo:** 288 instrumentos — acciones, ETF sectoriales, de factores, de renta
fija y de materias primas, y vehículos de criptomonedas.

**Historia de precios:** 1.418 sesiones, del 2021-01-04 al 2026-08-26, cierres
ajustados, empaquetados en el repositorio. Actualización semanal automatizada
mediante GitHub Actions con validación previa a la escritura: se rechaza toda
actualización que reduzca el número de filas, elimine columnas, produzca precios
no positivos, o cuya última fila cubra menos del 90 % de los instrumentos.

**Costo computacional medido:**

| Instrumentos | Tiempo total | Random Forest | MLP |
|---:|---:|---:|---:|
| 7 | 21,7 s | 10,0 s | 11,6 s |
| 15 | 39,2 s | 30,0 s | 9,1 s |
| 30 | 71,8 s | 53,2 s | 18,5 s |
| 45 | 97,3 s | 60,1 s | 37,0 s |

El costo escala con el número de instrumentos porque se entrena un Random Forest
por cada uno. **El límite de 15 instrumentos se fundamenta en esta medición**, no
en una preferencia de diseño: 30 duplicaría el tiempo hasta un rango donde tanto
los proxies de red como la paciencia del usuario se agotan.

Los resultados se almacenan en caché con una clave que combina el conjunto de
instrumentos y una huella del archivo de precios, de modo que una actualización
de datos invalida automáticamente los resultados previos.

---

## 7. Verificación

**97 pruebas automatizadas**, ejecutadas en integración continua ante cada
modificación.

| Módulo | Pruebas | Foco |
|---|---:|---|
| `test_snaptrade.py` | 35 | Reconciliación, atribución por cuenta, control de acceso |
| `test_tax.py` | 34 | Períodos de tenencia, jurisdicción, degradación de fidelidad |
| `test_security.py` | 12 | Límites de tamaño y frecuencia |
| `test_optimizer_modes.py` | 11 | Equivalencia con la configuración por defecto |
| `test_api.py` | 5 | Extremo a extremo del pipeline |

Las pruebas de mayor valor son las **negativas**: que no se invente una
descomposición por período de tenencia que no se conoce; que una cuenta no
identificada no se declare gravable; que el optimizador con los modos
desactivados produzca exactamente el resultado previo a su existencia; y que un
cuerpo de petición de tamaño normal no sea rechazado por los límites de abuso.

### Una nota sobre metodología

La integración con intermediarios se escribió contra la documentación del SDK y
superó 74 pruebas. **La primera ejecución real falló antes de la primera llamada
de red:** el constructor había cambiado de firma. Por debajo había tres defectos
adicionales —la respuesta era un diccionario y no una lista, el identificador del
instrumento residía en otro campo, y el costo de compra tenía otro nombre—, cada
uno de los cuales impedía por completo la importación.

Ninguno era detectable mediante simulacros, porque los simulacros los había
escrito la misma persona que formuló los supuestos erróneos.

La corrección incorporó pruebas construidas sobre la respuesta real capturada. Es
la distinción operativa entre un prototipo y un producto: **verificar contra la
realidad, no contra la representación que uno tiene de ella.**

---

## 8. Postura de seguridad

Revisión completa realizada sobre el historial íntegro del repositorio y sobre el
comportamiento del servicio en producción.

**Verificado:** 68 confirmaciones sin secretos; autenticación comprobada
empíricamente —no por lectura de código— en los seis extremos que manejan datos;
tokens generados con `secrets.token_urlsafe(32)` y almacenados solo como hash;
validación de entrada por lista blanca de caracteres en los identificadores de
instrumentos; ausencia de secretos y de posiciones en los registros.

**Controles incorporados:** límite de 256 KB por petición; límites de frecuencia
por dirección IP diferenciados por extremo (20/hora para la optimización, 30/hora
para los extremos de lenguaje natural); y topes de tamaño sobre el contenido
enviado al modelo, dado que `max_tokens` restringe la salida pero no la entrada,
que es donde reside el costo que un tercero puede inducir.

**Limitación declarada:** el contador de frecuencia reside en la memoria del
proceso. Se reinicia con cada despliegue y no se comparte entre instancias.
Constituye una barrera frente al abuso casual, no un control frente a un
adversario determinado. La solución adecuada requiere un contador externo y se
encuentra pendiente.

---

## 9. Limitaciones

- **No predice el mercado.** Estima retornos esperados sobre datos históricos.
  Los modelos pueden equivocarse, y la aplicación lo declara en pantalla.
- **No constituye asesoramiento financiero ni fiscal.**
- **No ejecuta operaciones** ni se conecta a cuentas con capacidad transaccional.
- **Máximo 15 instrumentos por corrida**, por el costo computacional de §6.
- **La cobertura de datos fiscales depende del intermediario.** Sin lotes
  fiscales no se dispone de la descomposición por período de tenencia.
- **Jurisdicciones cubiertas: Estados Unidos y Canadá.** Fuera de ellas se
  reportan hechos sin marco tributario.
- **Fuera de alcance:** impuestos estatales y provinciales, recargos, y
  tratamientos particulares como el impuesto sobre la renta de inversiones o el
  mínimo alternativo.
- **La rentabilidad histórica no es indicativa de resultados futuros.**

---

## 10. Trabajo futuro

1. **Completar la validación con intermediarios reales.** Falta medir qué
   proporción provee lotes fiscales; la prueba en entorno de pruebas no lo
   determina.
2. **Cuentas de usuario persistentes**, sustituyendo las sesiones efímeras.
3. **Límites de frecuencia externalizados**, para superar la limitación de §8.
4. **Ampliación del universo** más allá de 288 instrumentos.
5. **Reducción del tiempo de respuesta en frío**, hoy el principal obstáculo de
   experiencia de uso.

---

## Referencias

- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance, 7(1).
- Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for
  large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2).
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1).
- Sharpe, W. F. (1966). *Mutual Fund Performance*. The Journal of Business, 39(1).

**Código:** https://github.com/axelpeluso/portfolioai-optimizer — licencia MIT
**Contacto:** hi@axelpeluso.com
