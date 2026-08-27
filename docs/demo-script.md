# Guion de demo — PortfolioAI

Audiencia: producto / inversores. Duracion objetivo: **8-10 minutos**.

> **Regla de oro:** la primera optimizacion tarda ~40 segundos. No es un cuelgue,
> es el pipeline entrenando modelos. Ese silencio es el mejor momento de la demo
> para explicar que hay debajo — usalo, no lo disimules.

---

## Antes de empezar (15 minutos antes, no 2)

| # | Accion | Por que |
|---|---|---|
| 1 | Abrir https://atlantis-production-58a3.up.railway.app/health | Railway duerme sin trafico. El primer request puede tardar. Despertalo antes. |
| 2 | Abrir https://portfolioai-optimizer.vercel.app/ | Confirmar que la landing carga. |
| 3 | Abrir `/app` y **correr una optimizacion completa** | Calienta la cache de modelos: la corrida de la demo baja de ~40s a ~2s. |
| 4 | En `/app`, consola del navegador: `localStorage.removeItem('tour_state')` | Para que el tour guiado arranque en la demo. Si no, no aparece. |
| 5 | Cerrar pestañas de mas, silenciar notificaciones | |
| 6 | Tener a mano las capturas de respaldo (ver el final) | |

**Si vas a mostrar la conexion de brokerage**, ademas: verificar que
`/snaptrade/status` devuelva `enabled: true`. Si devuelve `false`, saltea esa
seccion — no improvises en vivo.

---

## El guion

### 1. El problema (1 min) — en la landing

Abrir https://portfolioai-optimizer.vercel.app/

> "Cualquiera que invierte tiene la misma pregunta y no la puede responder:
> *dado lo que tengo hoy, que deberia comprar o vender?* Las herramientas que
> existen te dan graficos y ratios. Ninguna te dice: vende 2.700 dolares de
> Apple, compra 3.400 de oro."

Bajar despacio por los cinco modelos. **No leerlos.** Senalar y decir:

> "Cinco modelos distintos, cada uno con un trabajo especifico. No es un chatbot
> con un grafico encima."

### 2. La herramienta (30 s)

Ir a `/app`.

> "Esta parte no es publica. Se accede despues de una demo, y ahora vamos a ver
> por que."

Si aparece el tour guiado, dejalo correr dos pasos y cerralo:

> "Todo el que entra recibe esto la primera vez."

### 3. La corrida (2-3 min) — el corazon

La cartera por defecto ya esta cargada. Senalar el panel izquierdo:

> "Instrumentos, valor total, y lo que tengo hoy de cada uno."

Clic en **Run Optimization**. **Mientras corre**, hablar sin apuro:

> "Ahora mismo: agrupa los activos por comportamiento, entrena un modelo por
> ticker para proyectar retornos, una red neuronal evalua el regimen de mercado,
> y con eso resuelve la frontera eficiente. Son cuarenta segundos porque esta
> entrenando de verdad, no consultando una tabla."

Cuando aparezcan los resultados, ir en orden:

1. **Fila de metricas** — "Sharpe antes y despues. Retorno por unidad de riesgo."
2. **Risk score** — "La red neuronal puntua el regimen actual de 0 a 1. Y esto
   no es decorativo: ese numero cambia los limites que el optimizador puede usar."
3. **Grafico antes/despues** — la reasignacion, visual.
4. **Tabla de rebalanceo** — **este es el momento clave**:

> "Aca esta la diferencia. No son pesos objetivo: es *vende 2.700 dolares de
> Apple*. Ejecutable."

### 4. La honestidad (1 min) — el diferenciador

Senalar el aviso de turnover arriba de la tabla:

> "Fijense en esto. Nos esta avisando que este rebalanceo rota el 53% de la
> cartera. La mayoria de las herramientas te muestran el resultado optimo y se
> callan lo que cuesta llegar."

Abrir **Optimizer → Minimize trading** en el panel izquierdo, marcarlo, correr
de nuevo (esta vez es rapido, cache caliente):

> "Y si eso te parece mucho, se penaliza operar. Turnover baja del 53% al 27%, y
> el Sharpe baja de 0.97 a 0.80. Mostramos las dos caras: operar menos cuesta
> rendimiento. El usuario decide, nosotros no escondemos el precio."

> "Por defecto esto esta apagado. La matematica del optimizador no cambia salvo
> que vos lo pidas."

### 5. Claude (1-2 min)

Clic en **Explain with Claude**. Mientras escribe:

> "Explica el resultado en lenguaje llano: que significa el score, por que las
> operaciones mas grandes tienen sentido."

Abrir el chat (boton abajo a la derecha) y escribir **`add TSLA`**:

> "Y responde ordenes. Edita la cartera, vuelve a correr el pipeline entero, y
> reporta que cambio."

### 6. Brokerage — solo si `enabled: true` (1-2 min)

> "Todo esto asume que cargaste tus posiciones a mano. El ultimo paso lo elimina."

Clic en **Connect brokerage** → SnapTrade Sandbox → conectar.

En el modal de revision, senalar lo que se excluye:

> "Trae las posiciones y dice exactamente que no puede modelar y por que. Bitcoin
> queda afuera porque no tenemos su historial de precios. Nada se descarta en
> silencio: optimizar contra una cartera que no es la tuya seria el peor
> resultado posible."

Importar y correr. En la tabla, senalar las lineas fiscales:

> "Y ahora cada venta dice cuanta ganancia realiza y en que cuenta esta. Si la
> cuenta es una IRA, no hay consecuencia fiscal y lo dice. Nunca inventamos una
> cifra: si el broker no manda el costo, decimos que no lo manda."

### 7. Cierre (30 s)

Volver a la landing, al formulario:

> "Es beta privada. Se entra despues de una demo, porque estos numeros se
> malinterpretan facil. Es una herramienta de investigacion, no asesoramiento."

---

## Preguntas probables

**"¿Predice el mercado?"**
> No, y desconfiaria de quien diga que si. Proyecta retornos esperados a partir
> de datos historicos y optimiza la relacion riesgo-retorno. Los modelos pueden
> equivocarse y lo decimos en pantalla.

**"¿Que lo diferencia de un robo-advisor?"**
> Un robo-advisor te administra la plata. Esto te dice que hacer con la tuya, te
> muestra el razonamiento de cada modelo, y te avisa lo que la operacion cuesta
> en impuestos y rotacion. Es transparencia, no delegacion.

**"¿Ejecuta operaciones?"**
> No, y es deliberado. Leemos posiciones, nunca operamos. Ejecutar cambiaria por
> completo el encuadre regulatorio del producto.

**"¿Cuantos usuarios tienen?"**
> Beta privada, acceso por demo. Lo que se mide hoy son solicitudes de demo y uso
> de las funciones — esta instrumentado en `/analytics`.

**"¿Por que tarda 40 segundos?"**
> Porque entrena un Random Forest por instrumento en cada corrida, contra cinco
> anios de datos. Se cachea: la segunda corrida de la misma cartera es
> instantanea. Es una decision consciente — preferimos entrenar de verdad antes
> que servir un resultado precalculado.

---

## Si algo falla

| Sintoma | Que hacer |
|---|---|
| La optimizacion no vuelve | Esperar 60s completos antes de tocar nada. Railway pudo haberse dormido. |
| Error de red | Recargar `/app`. Los resultados anteriores se pierden; correr de nuevo con la cache ya caliente. |
| Claude no responde | Seguir sin eso. La optimizacion no depende de Claude. |
| El brokerage falla | Cerrar el modal y seguir con entrada manual. **No debuguear en vivo.** |
| Todo cae | Pasar a las capturas y contar lo que se ve. |

**Capturas de respaldo a tener listas:** resultados completos de una corrida,
tabla de rebalanceo con las lineas fiscales, y el modal de importacion. Diez
minutos antes, sacalas de una corrida real.

---

## Lo que NO conviene decir

- No prometas rendimientos. Los numeros dependen de la cartera y la ventana.
- No digas "te dice que comprar" sin agregar que es investigacion, no
  asesoramiento. Esa frase esta en toda la aplicacion; que este tambien en la
  charla.
- No presentes la conexion de brokerage como terminada. Se valido contra sandbox
  esta semana y eso destapo cuatro bugs reales. Si preguntan, esa historia
  **suma**: demuestra que se prueba contra la realidad, no contra mocks.
