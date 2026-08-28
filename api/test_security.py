"""
Tests de los limites de abuso.

El que mas importa es el de regresion: que un cuerpo de tamano REAL no se
rechace. Romper el uso normal seria peor que el problema que estos limites
resuelven.

Correr: pytest -q   (desde api/)
"""
import json

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def _limpiar_contadores():
    """Sin esto los casos se contaminan entre si."""
    main._rate_state.clear()
    yield
    main._rate_state.clear()


# ── cap de tamano ─────────────────────────────────────────────
def test_cuerpo_desmedido_rechazado_con_413():
    grande = "x" * (main.MAX_BODY_BYTES + 1024)
    r = client.post("/optimize", content=json.dumps({"tickers": ["AAPL", "MSFT"],
                                                     "relleno": grande}),
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413
    assert "grande" in r.json()["detail"].lower()


def test_cuerpo_de_tamano_real_no_se_rechaza():
    """La regresion que importa: 15 posiciones con lotes fiscales pesan ~15 KB."""
    lots = [{"quantity": 10, "cost_basis": 1500.0, "purchased_price": 150.0,
             "original_purchase_date": "2024-03-01"} for _ in range(8)]
    ctx = {"positions": {f"T{i}": {"price": 180.5, "units": 25,
                                   "_account": "acc-uuid-0001",
                                   "average_purchase_price": 175.0,
                                   "tax_lots": lots} for i in range(15)},
           "accounts": {"a1": {"raw_type": "Individual", "currency": "USD"}}}
    body = {"tickers": ["AAPL", "MSFT"], "current_holdings": {"AAPL": 100},
            "tax_context": ctx}
    assert len(json.dumps(body)) > 10_000, "el caso deberia ser realmente grande"
    r = client.post("/optimize", json=body)
    assert r.status_code != 413, "un cuerpo normal nunca debe rechazarse"


# ── rate limiting ─────────────────────────────────────────────
def test_excederse_devuelve_429_con_retry_after():
    limite = main.RATE_LIMITS["optimize"]
    for _ in range(limite):
        main._rate_limit("optimize", "1.2.3.4", limite)
    assert main._rate_limit("optimize", "1.2.3.4", limite) is not None

    r = client.post("/optimize", json={"tickers": ["AAPL"]},
                    headers={"X-Forwarded-For": "1.2.3.4"})
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0


def test_los_buckets_son_independientes():
    """Agotar el cupo de LLM no debe bloquear /optimize."""
    for _ in range(main.RATE_LIMITS["llm"]):
        main._rate_limit("llm", "9.9.9.9", main.RATE_LIMITS["llm"])
    assert main._rate_limit("llm", "9.9.9.9", main.RATE_LIMITS["llm"]) is not None
    assert main._rate_limit("optimize", "9.9.9.9",
                            main.RATE_LIMITS["optimize"]) is None


def test_el_limite_es_por_ip():
    limite = main.RATE_LIMITS["optimize"]
    for _ in range(limite):
        main._rate_limit("optimize", "5.5.5.5", limite)
    assert main._rate_limit("optimize", "5.5.5.5", limite) is not None
    assert main._rate_limit("optimize", "6.6.6.6", limite) is None, "otra IP, otro cupo"


def test_usa_x_forwarded_for_detras_del_proxy():
    """Sin esto, en Railway todas las peticiones comparten la IP del proxy y el
    cupo se agota para todos a la vez."""
    class FakeReq:
        headers = {"x-forwarded-for": "203.0.113.7, 10.0.0.1"}
        client = type("c", (), {"host": "10.0.0.1"})()
    assert main._client_ip(FakeReq()) == "203.0.113.7"


# ── limites de entrada en los endpoints que cuestan dinero ────
def test_chat_rechaza_demasiados_mensajes():
    msgs = [{"role": "user", "content": "hola"}
            for _ in range(main.MAX_CHAT_MESSAGES + 1)]
    r = client.post("/chat", json={"messages": msgs})
    assert r.status_code == 400
    assert "mensajes" in r.json()["detail"].lower()


def test_chat_rechaza_conversacion_demasiado_larga():
    msgs = [{"role": "user", "content": "x" * 3000} for _ in range(10)]
    r = client.post("/chat", json={"messages": msgs})
    assert r.status_code == 400
    assert "larga" in r.json()["detail"].lower()


def test_chat_rechaza_portfolio_data_enorme():
    enorme = {"relleno": ["y" * 1000 for _ in range(200)]}
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hola"}],
                                   "portfolio_data": enorme})
    assert r.status_code == 400
    assert "portfolio_data" in r.json()["detail"]


def test_explain_rechaza_portfolio_data_enorme():
    enorme = {"relleno": ["y" * 1000 for _ in range(200)]}
    r = client.post("/explain", json={"portfolio_data": enorme})
    assert r.status_code == 400


def test_una_conversacion_normal_pasa_los_limites():
    """No debe fallar por los topes; puede fallar por falta de clave de API."""
    r = client.post("/chat", json={"messages": [{"role": "user", "content": "hola"}],
                                   "portfolio_data": {"tickers": ["AAPL"]}})
    assert r.status_code != 400, "un chat normal no debe rechazarse"


# ── fuga de informacion ───────────────────────────────────────
def test_el_error_de_sesion_no_expone_el_driver():
    """El detalle nombraba tablas y permisos de Supabase."""
    src = open(main.__file__, encoding="utf-8").read()
    assert "str(e)[:200]" not in src, "el mensaje del driver volvio a la respuesta"
