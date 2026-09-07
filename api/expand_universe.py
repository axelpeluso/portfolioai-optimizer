"""
expand_universe.py — amplia tickers.json validando cada candidato contra la fuente.

El punto clave del diseno: NO confia en la lista de candidatos. Cada simbolo se
prueba contra la fuente de precios y solo entra si devuelve historial suficiente
para que los modelos produzcan predicciones. Un candidato inventado o mal escrito
simplemente no pasa, en vez de contaminar el universo con un ticker muerto.

    python expand_universe.py --dry-run     # informa sin escribir
    python expand_universe.py               # actualiza tickers.json

Despues hay que correr refresh_prices.py para bajar el historial de los nuevos.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
TICKERS = BASE / "tickers.json"

# El pipeline necesita ventana 60 + horizonte 63 para el Random Forest. Por
# debajo de eso un instrumento aporta historial pero ninguna prediccion.
MIN_SESSIONS = 153

# Volatilidad anual por debajo de la cual el activo rompe el optimizador: la
# covarianza se vuelve casi singular y el Sharpe da valores absurdos. Es el caso
# de los money market, que economicamente son efectivo, no una posicion.
MIN_VOL_PCT = 0.5

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")


def probe(sym: str) -> dict:
    """Trae el historial de un simbolo y decide si es apto para el universo."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
           f"?period1=1609459200&period2=1788000000&interval=1d")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=25) as r:
            payload = json.load(r)
    except Exception as e:                       # noqa: BLE001
        return {"symbol": sym, "ok": False, "motivo": f"sin datos ({getattr(e,'code','?')})"}

    res = (payload.get("chart", {}).get("result") or [None])[0]
    if not res or not res.get("timestamp"):
        return {"symbol": sym, "ok": False, "motivo": "sin serie de precios"}

    ind = res.get("indicators", {})
    closes = ((ind.get("adjclose") or [{}])[0].get("adjclose")
              or (ind.get("quote") or [{}])[0].get("close") or [])
    vals = [c for c in closes if c]
    n = len(vals)
    if n < MIN_SESSIONS:
        return {"symbol": sym, "ok": False,
                "motivo": f"historial corto ({n} < {MIN_SESSIONS})"}

    rets = [vals[i] / vals[i - 1] - 1 for i in range(1, n)]
    vol = statistics.pstdev(rets) * (252 ** 0.5) * 100
    if vol < MIN_VOL_PCT:
        return {"symbol": sym, "ok": False,
                "motivo": f"volatilidad ~0 ({vol:.2f}%) — probable money market"}

    meta = res.get("meta") or {}
    return {
        "symbol": sym, "ok": True, "sesiones": n, "vol": round(vol, 1),
        "moneda": meta.get("currency"),
        "nombre": (meta.get("longName") or meta.get("shortName")
                   or meta.get("symbol") or sym),
        "desde": datetime.fromtimestamp(res["timestamp"][0], timezone.utc)
                         .date().isoformat(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", type=Path,
                   help="archivo con un simbolo por linea (por defecto: stdin)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--only-usd", action="store_true",
                   help="descartar lo que no cotice en USD (evita mezclar monedas)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    texto = (args.candidates.read_text(encoding="utf-8") if args.candidates
             else sys.stdin.read())
    cands = sorted({l.strip().upper() for l in texto.splitlines()
                    if l.strip() and not l.startswith("#")})

    actuales = json.loads(TICKERS.read_text(encoding="utf-8"))
    nuevos = [c for c in cands if c not in actuales]
    print(f"candidatos: {len(cands)}  |  ya en el universo: {len(cands)-len(nuevos)}"
          f"  |  a validar: {len(nuevos)}")
    if not nuevos:
        return 0

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(probe, nuevos))
    print(f"validados en {time.time()-t0:.0f}s\n")

    aceptados = [r for r in res if r["ok"]]
    if args.only_usd:
        otros = [r for r in aceptados if r.get("moneda") not in (None, "USD")]
        aceptados = [r for r in aceptados if r.get("moneda") in (None, "USD")]
        for r in otros:
            r["motivo"] = f"moneda {r.get('moneda')} — se mezclarian monedas"
        res = [r for r in res if r["ok"] is False] + otros
    rechazados = [r for r in res if not r.get("ok") or "motivo" in r and not r["ok"]]
    rechazados = [r for r in res if not r["ok"]] + \
                 [r for r in res if r["ok"] and "motivo" in r]

    print(f"ACEPTADOS ({len(aceptados)}):")
    for r in sorted(aceptados, key=lambda x: x["symbol"]):
        print(f"  {r['symbol']:10} {r['sesiones']:>5} ses  desde {r['desde']}  "
              f"vol {r['vol']:>5.1f}%  {str(r['nombre'])[:40]}")

    if rechazados:
        print(f"\nRECHAZADOS ({len(rechazados)}):")
        for r in sorted(rechazados, key=lambda x: x["symbol"]):
            print(f"  {r['symbol']:10} {r['motivo']}")

    if args.dry_run:
        print("\nDry run — tickers.json sin cambios.")
        return 0

    for r in aceptados:
        actuales[r["symbol"]] = str(r["nombre"])[:80]
    TICKERS.write_text(json.dumps(dict(sorted(actuales.items())),
                                  indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print(f"\ntickers.json: {len(actuales)-len(aceptados)} -> {len(actuales)}")
    print("Ahora corre:  python refresh_prices.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
