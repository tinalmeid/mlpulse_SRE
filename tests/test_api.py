# tests/test_api.py
"""Testes da API MLPulse."""
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_retorna_ok():
    """Verifica se o health check retorna status ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["version"] == "0.2.0"


def test_predict_sem_modelo_retorna_erro():
    """Verifica que predição sem modelo treinado retorna erro 400."""
    r = client.post("/predict", json={"x": [1.0, 2.0]})
    assert r.status_code == 400


def test_treino_e_predicao():
    """Treina o modelo e verifica se a predição está correta."""
    # Treinar
    r = client.post("/train", json={
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    assert r.status_code == 200
    assert r.json()["status"] == "treinado"
    assert abs(r.json()["coef"] - 2.0) < 0.01

    # Predizer
    r = client.post("/predict", json={"x": [6.0, 7.0]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2
    assert abs(preds[0] - 12.0) < 0.1
    assert abs(preds[1] - 14.0) < 0.1


def test_treino_com_tamanhos_diferentes_retorna_erro():
    """Verifica que treino com x e y de tamanhos diferentes retorna erro."""
    r = client.post("/train", json={
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0]
    })
    assert r.status_code == 422
