"""
backtest.py — evaluacion walk-forward del pipeline completo.

Que se mide
-----------
La afirmacion del producto NO es "le gana al mercado". Es: dada TU cartera, hay
una relacion riesgo/retorno mejor. Asi que el punto de comparacion es la propia
cartera mantenida sin cambios, no un indice.

Como se evita mirar el futuro
-----------------------------
1. END_DATE se mueve en cada iteracion, asi que los modelos solo ven datos
   anteriores a la fecha de decision.
2. La cache se desactiva. Su clave combina tickers con una huella del ARCHIVO,
   no con la fecha de corte, de modo que sin desactivarla todas las iteraciones
   compartirian el modelo de la primera. Ese seria el error mas peligroso de
   todos: no falla, devuelve numeros plausibles y equivocados.

Lo que este backtest NO puede corregir
--------------------------------------
El universo se eligio HOY: son los instrumentos que sobrevivieron. CTRA se
deslisto, ANSS/DFS/HES/X fueron absorbidas. Cualquier resultado esta sesgado al
alza por construccion, y no hay forma de arreglarlo sin datos historicos de
constituyentes. Leer los numeros con eso presente.

    python backtest.py --tickers AAPL,MSFT,GOOGL,JPM,BND,GLD,AMZN
"""

from __future__ import annotations

import argparse
import statistics
import sys
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import optimizer as opt

TRADING_DAYS = 252


def cargar_precios() -> pd.DataFrame:
    return pd.read_csv(opt.CSV_PATH, index_col=0, parse_dates=True)


def metricas(serie: pd.Series) -> dict:
    """Retorno anualizado, volatilidad y Sharpe de una curva de capital."""
    rets = serie.pct_change().dropna()
    if len(rets) < 2:
        return {"retorno": 0.0, "vol": 0.0, "sharpe": 0.0, "max_dd": 0.0}
    anios = len(rets) / TRADING_DAYS
    total = serie.iloc[-1] / serie.iloc[0]
    cagr = total ** (1 / anios) - 1 if anios > 0 else 0.0
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    dd = ((serie - serie.cummax()) / serie.cummax()).min()
    return {"retorno": cagr, "vol": vol,
            "sharpe": (cagr - opt.RISK_FREE) / vol if vol > 0 else 0.0,
            "max_dd": dd}


def curva(precios: pd.DataFrame, pesos_por_fecha: list, fechas: list,
          costo_bps: float = 0.0) -> tuple[pd.Series, float]:
    """Capital acumulado rebalanceando a los pesos dados en cada fecha.

    `costo_bps` descuenta el costo de operar en cada rebalanceo, proporcional a
    la rotacion. Sin esto, una estrategia que rota el 50% por trimestre se ve
    gratis, que es precisamente el sesgo que el modo de baja rotacion ataca.
    """
    valores, capital, rot_total = [], 1.0, 0.0
    previos = None
    for i, (w, ini) in enumerate(zip(pesos_por_fecha, fechas)):
        if costo_bps and previos is not None:
            rot = sum(abs(w.get(t, 0.0) - previos.get(t, 0.0))
                      for t in precios.columns) / 2
            rot_total += rot
            capital *= (1 - rot * costo_bps / 10_000)
        previos = w
        fin = fechas[i + 1] if i + 1 < len(fechas) else precios.index[-1]
        tramo = precios.loc[ini:fin]
        if len(tramo) < 2:
            continue
        pesos = np.array([w.get(t, 0.0) for t in precios.columns])
        rel = tramo / tramo.iloc[0]
        cartera = (rel * pesos).sum(axis=1) * capital
        valores.append(cartera.iloc[1:])
        capital = float(cartera.iloc[-1])
    serie = pd.concat(valores) if valores else pd.Series(dtype=float)
    return serie, rot_total


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="AAPL,MSFT,GOOGL,JPM,BND,GLD,AMZN")
    p.add_argument("--entrenamiento", type=int, default=504,
                   help="sesiones minimas de entrenamiento (default: 2 anios)")
    p.add_argument("--cada", type=int, default=63,
                   help="sesiones entre rebalanceos (default: trimestral)")
    p.add_argument("--modos", default="off,moderate",
                   help="modos del optimizador a comparar")
    p.add_argument("--cov-window", type=int, default=None,
                   help="sesiones para la covarianza (default: toda la ventana)")
    p.add_argument("--costo-bps", type=float, default=10.0,
                   help="costo por operar, en puntos basicos sobre la rotacion")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    precios = cargar_precios()[tickers].dropna()
    print(f"Instrumentos : {', '.join(tickers)}")
    print(f"Datos        : {precios.index[0].date()} -> {precios.index[-1].date()} "
          f"({len(precios)} sesiones)")

    puntos = list(range(args.entrenamiento, len(precios) - args.cada, args.cada))
    fechas = [precios.index[i] for i in puntos]
    if len(fechas) < 4:
        print("Historial insuficiente para un walk-forward util.")
        return 1
    print(f"Rebalanceos  : {len(fechas)} ({fechas[0].date()} -> {fechas[-1].date()})")
    print(f"Advertencia  : universo elegido hoy => sesgo de supervivencia al alza")
    print()

    # Sin cache: su clave no incluye la fecha de corte, asi que todas las
    # iteraciones compartirian el modelo de la primera.
    opt._load_cache = lambda key: None
    opt._save_cache = lambda *a, **k: None
    end_original = opt.END_DATE

    inicial = {t: 1.0 / len(tickers) for t in tickers}   # punto de partida comun
    resultados = {}

    for modo in [m.strip() for m in args.modos.split(",")]:
        penal = None if modo == "off" else modo
        print(f"--- modo: {modo} ---")
        pesos_hist, anterior = [], dict(inicial)
        calibracion = []
        for n, fecha in enumerate(fechas, 1):
            opt.END_DATE = fecha.strftime("%Y-%m-%d")   # el modelo no ve mas alla
            tenencias = {t: anterior.get(t, 0.0) * 100_000 for t in tickers}
            try:
                r = opt.run_full_analysis(tickers, tenencias, turnover_penalty=penal,
                                          cov_window=args.cov_window)
                w = r["optimization"]["optimal_weights"]
            except Exception as e:
                print(f"  {fecha.date()}: fallo ({type(e).__name__}), se mantiene")
                w = anterior
            pesos_hist.append(w)
            # Calibracion: que prometio el modelo vs que ocurrio despues.
            i_fecha = precios.index.get_loc(fecha)
            fin = min(i_fecha + args.cada, len(precios) - 1)
            if fin > i_fecha:
                tramo = precios.iloc[i_fecha:fin + 1]
                pesos_v = np.array([w.get(t, 0.0) for t in precios.columns])
                val = (tramo / tramo.iloc[0] * pesos_v).sum(axis=1)
                periodos = TRADING_DAYS / (fin - i_fecha)
                real = val.iloc[-1] ** periodos - 1
                calibracion.append((fecha, r["optimization"]["max_sharpe_metrics"]["return"], real))
            anterior = w
            rot = sum(abs(w.get(t, 0) - anterior.get(t, 0)) for t in tickers) / 2
            print(f"  [{n}/{len(fechas)}] {fecha.date()}  riesgo "
                  f"{r['risk_score']:.3f}  top: "
                  f"{max(w, key=w.get)} {max(w.values()):.0%}")
        resultados[modo] = curva(precios, pesos_hist, fechas, args.costo_bps)
        if calibracion:
            print()
            print(f"  calibracion del retorno esperado ({modo}):")
            print(f"    {'fecha':12} {'prometido':>11} {'realizado':>11} {'error':>10}")
            errs = []
            for f, pred, real in calibracion:
                errs.append(pred - real)
                print(f"    {str(f.date()):12} {pred:>10.1%} {real:>10.1%} {pred-real:>+9.1%}")
            sesgo = sum(errs) / len(errs)
            aciertos = sum(1 for e in errs if abs(e) < 0.10)
            print(f"    sesgo medio: {sesgo:+.1%}   "
                  f"dentro de +/-10 puntos: {aciertos}/{len(errs)}")

    opt.END_DATE = end_original

    # Referencia: la misma cartera inicial, mantenida sin tocar.
    resultados["mantener"] = curva(precios, [inicial], [fechas[0]], 0.0)

    print()
    print("=" * 74)
    print(f"{'estrategia':22} {'retorno':>8} {'vol':>7} {'sharpe':>7} "
          f"{'peor caida':>11} {'rotacion':>9}")
    print("=" * 74)
    for nombre, (serie, rot) in resultados.items():
        if serie.empty:
            continue
        m = metricas(serie)
        etiqueta = {"off": "optimizado", "moderate": "opt. + baja rotacion",
                    "mantener": "mantener sin tocar"}.get(nombre, nombre)
        print(f"{etiqueta:22} {m['retorno']:>7.1%} {m['vol']:>6.1%} "
              f"{m['sharpe']:>7.2f} {m['max_dd']:>10.1%} {rot:>8.0%}")
    print("=" * 74)
    print(f"Costo aplicado: {args.costo_bps:.0f} bps sobre la rotacion "
          f"(la referencia no opera, asi que no paga).")
    print("Sesgo de supervivencia no corregido: los resultados estan inflados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
