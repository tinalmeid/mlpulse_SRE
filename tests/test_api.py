# tests/test_api.py
# Autor: Tina de Almeida
# Março de 2026
"""Testes da API MLPulse.
São testes de integração que verificam o comportamento da API usando TestClient do FastAPI.
"""
from unittest.mock import patch, MagicMock
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


@patch("app.main.boto3.client")
def test_treino_e_predicao(mock_boto):
    """Treina o modelo e verifica se a predição está correta."""
    # Simula o cliente S3 sem precisar de credenciais reais
    mock_s3 = MagicMock()
    mock_boto.return_value = mock_s3

    # Treinar
    r = client.post("/train", json={
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [2.0, 4.0, 6.0, 8.0, 10.0]
    })
    assert r.status_code == 200
    assert r.json()["status"] == "treinado"
    assert abs(r.json()["coef"] - 2.0) < 0.01

    # Verificar que tentou fazer upload para o S3
    mock_s3.upload_file.assert_called_once()

    # Predizer
    r = client.post("/predict", json={"x": [6.0, 7.0]})
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert abs(preds[0] - 12.0) < 0.1
    assert abs(preds[1] - 14.0) < 0.1


@patch("app.main.boto3.client")
def test_treino_com_tamanhos_diferentes_retorna_erro(mock_boto):
    """Verifica que treino com x e y de tamanhos diferentes retorna erro."""
    r = client.post("/train", json={
        "x": [1.0, 2.0, 3.0],
        "y": [2.0, 4.0]
    })
    assert r.status_code == 422
