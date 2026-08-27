# PortfolioAI — resumen ejecutivo

**Agosto 2026 · Beta privada**
Landing: https://portfolioai-optimizer.vercel.app · Herramienta: `/app` (no listada)

---

## El problema

Quien invierte tiene una pregunta que ninguna herramienta le responde:
**dado lo que tengo hoy, que compro y que vendo?**

Las plataformas existentes muestran graficos, ratios y carteras modelo. El salto
de "tu Sharpe es 0.77" a "vende 2.700 dolares de Apple" queda del lado del
usuario, y ahi es donde la mayoria se detiene.

## Que hace PortfolioAI

Toma las posiciones reales de una persona y devuelve **instrucciones ejecutables
en dolares**: comprar X, vender Y, mantener Z — con el razonamiento de cada
modelo visible y el costo de la operacion declarado.

---

## Como funciona

Cinco modelos encadenados, cada uno con una funcion distinta:

| # | Modelo | Tipo | Funcion |
|---|---|---|---|
| 1 | K-Means | No supervisado | Perfila cada activo: Growth / Moderate / Defensive |
| 2 | Random Forest | Supervisado, uno por ticker | Proyecta el retorno del proximo trimestre |
| 3 | MLP | Red neuronal | Puntua el regimen de mercado de 0 a 1 |
| 4 | Markowitz / MPT | Optimizacion (SLSQP) | Resuelve Max-Sharpe y Min-Varianza |
| 5 | Motor de rebalanceo | Traduccion | Convierte pesos en ordenes en dolares |

El score del MLP **no es decorativo**: modifica los limites de posicion que el
optimizador puede usar. Mercado calmo permite concentrar; volatil obliga a
diversificar.

Encima, **Claude Haiku 4.5** explica el resultado en lenguaje llano y responde
ordenes conversacionales ("agrega TSLA") que re-ejecutan el pipeline completo.

---

## Que lo diferencia

**1. La salida es ejecutable.** No pesos objetivo — montos contra las posiciones
que la persona realmente tiene.

**2. Declara lo que la operacion cuesta.** La cartera de ejemplo rota el 53% en
un rebalanceo. Casi ninguna herramienta lo menciona. Nosotros lo mostramos arriba
de la tabla, y ofrecemos operar menos mostrando el precio de hacerlo:

| Modo | Rotacion | Sharpe |
|---|---:|---:|
| Sin penalizacion | 53.3% | 0.969 |
| Moderado | 26.7% | 0.799 |
| Fuerte | 15.1% | 0.694 |

Ambas caras siempre visibles. **Apagado por defecto**: la matematica del
optimizador no cambia salvo que el usuario lo pida explicitamente.

**3. Divulgacion fiscal que distingue hechos de estimaciones.**

- *Hecho* — "esta venta realiza 1.410 de ganancia, 820 de lotes con menos de un
  anio". Se calcula del costo que informa el broker.
- *Estimacion* — "vas a pagar 340". Requiere tramo, situacion fiscal y pais.
  **Solo aparece con las tasas que el propio usuario carga.** No hay ninguna tasa
  por defecto en el codigo.

Tres propiedades garantizadas por tests, no por convencion: una cuenta no
identificada nunca se declara gravable; sin lotes se informa la ganancia pero
**no** el reparto corto/largo en vez de inventarlo; y las cuentas canadienses
usan costo promedio, porque FIFO no esta permitido bajo sus reglas.

**4. Conexion con el broker.** Elimina la carga manual de posiciones. Solo
lectura: el producto no ejecuta ordenes ni puede mover dinero, por decision
explicita.

---

## Estado (medido, no estimado)

| | |
|---|---|
| Backend | ~3.300 lineas Python (FastAPI, en Railway) |
| Frontend | ~3.400 lineas (vanilla JS + Chart.js, en Vercel) |
| Tests | **80**, en CI sobre cada push |
| Universo | 288 instrumentos: acciones, ETFs sectoriales y de bonos, commodities, cripto |
| Historial | 2021-01-04 → 2026-08-26 (1.418 sesiones), refresco semanal automatizado |
| Tiempo de corrida | ~40 s en frio (15 tickers), instantaneo con cache |

Infraestructura: FastAPI + scikit-learn + SciPy en Railway, frontend en Vercel,
Supabase para lista de espera y analitica, Anthropic para la capa de lenguaje.

---

## Modelo de acceso

Beta privada. La herramienta **no esta enlazada publicamente**: se llega desde la
landing pidiendo una demo. Es deliberado — un score de riesgo de 0.88 o un
"vender 2.700" se malinterpretan facil sin contexto, y la conversacion previa
evita decisiones mal fundadas.

Las solicitudes de demo y el uso de funciones se registran en `/analytics`.

---

## Limitaciones que declaramos

- **No predice el mercado.** Proyecta retornos esperados sobre datos historicos.
  Los modelos pueden equivocarse y la aplicacion lo dice en pantalla.
- **No es asesoramiento financiero ni fiscal.** Herramienta de investigacion.
- **No ejecuta operaciones** ni se conecta a cuentas para operar.
- **Maximo 15 instrumentos** por corrida: cada uno agrega un Random Forest, y 30
  duplicaria el tiempo a ~72 s.
- **La conexion con brokers esta en validacion.** Se probo contra el entorno de
  pruebas esta semana; la cobertura de datos fiscales varia segun el broker.

---

## Proximos pasos

1. **Cerrar la validacion con brokers reales.** La prueba contra sandbox destapo
   cuatro bugs que ningun test con simulacros detectaba — la forma real del
   payload difiere de la documentada. Falta medir cuantos brokers reales aportan
   detalle de lotes fiscales.
2. **Cuentas de usuario.** Hoy las sesiones son efimeras; hacen falta cuentas
   persistentes para que la conexion sobreviva entre visitas.
3. **Ampliar el universo** mas alla de 288 instrumentos.
4. **Rendimiento**: la corrida en frio de 40 s es el mayor obstaculo de
   experiencia.

---

## Una nota sobre el metodo

La integracion con el broker se escribio contra la documentacion del SDK y
paso 74 tests. La primera ejecucion real fallo **antes de la primera llamada de
red**: el constructor habia cambiado de firma. Debajo habia tres bugs mas — la
respuesta era un diccionario y no una lista, el ticker venia en otro campo, y el
costo de compra tenia otro nombre.

Ninguno era detectable con simulacros, porque los simulacros los habia escrito la
misma persona que escribio los supuestos equivocados.

Se corrigieron y se agregaron tests con el payload real capturado. Es el tipo de
rigor que separa un prototipo de un producto: **probar contra la realidad, no
contra lo que uno cree que la realidad deberia ser.**
