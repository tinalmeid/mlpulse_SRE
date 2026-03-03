# app/main.py
# Autor: Tina de Almeida
# Março de 2026
"""
MLPulse — API de Machine Learning
Serve predições via HTTP com suporte a treino de modelos sklearn.

Princípios aplicados:
    - SRP (Single Responsibility): cada função tem uma responsabilidade
    - Clean Code: nomes descritivos, funções pequenas e focadas
    - Docstrings: documentação em todas as funções públicas
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
import logging
import boto3
from botocore.exceptions import ClientError

# Bucket S3 para persistir o modelo
S3_BUCKET = "mlpulse-models-511197442274"
S3_MODEL_KEY = "models/model.pkl"

# ── Configuração de Logging ───────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
APP_TITLE = "MLPulse"
APP_VERSION = "0.2.0"

# ── Aplicação ─────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description="API de Machine Learning com boas práticas de engenharia"
)

# Estado global do modelo (carregado na memória)
_model: LinearRegression | None = None


# ── Schemas (contratos da API) ────────────────────────────────
class TrainRequest(BaseModel):
    """Dados de entrada para treinar o modelo."""

    x: list[float] = Field(
        ...,
        description="Variável independente (features)",
        example=[1.0, 2.0, 3.0, 4.0, 5.0]
    )
    y: list[float] = Field(
        ...,
        description="Variável dependente (target)",
        example=[2.0, 4.0, 6.0, 8.0, 10.0]
    )


class PredictRequest(BaseModel):
    """Dados de entrada para gerar predições."""

    x: list[float] = Field(
        ...,
        description="Valores para predizer",
        example=[6.0, 7.0, 8.0]
    )


class TrainResponse(BaseModel):
    """Resposta do endpoint de treino."""

    status: str
    coef: float
    intercept: float
    samples_used: int


class PredictResponse(BaseModel):
    """Resposta do endpoint de predição."""

    predictions: list[float]
    model_version: str


class HealthResponse(BaseModel):
    """Resposta do endpoint de health check."""

    status: str
    model_loaded: bool
    version: str


# ── Funções de Suporte (SRP) ──────────────────────────────────
def _save_model(model: LinearRegression) -> None:
    """
    Persiste o modelo treinado em disco e faz upload para o S3.

    Args:
        model: Instância treinada do LinearRegression.
    """
    # Salvar local
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info("Modelo salvo localmente em %s", MODEL_PATH)

    # Upload para S3
    try:
        s3 = boto3.client("s3")
        s3.upload_file(MODEL_PATH, S3_BUCKET, S3_MODEL_KEY)
        logger.info("Modelo enviado para s3://%s/%s", S3_BUCKET, S3_MODEL_KEY)
    except ClientError as error:
        logger.error("Erro ao enviar modelo para S3: %s", error)


def _train_model(x_values: list[float], y_values: list[float]) -> LinearRegression:
    """
    Treina um modelo de regressão linear.

    Args:
        x_values: Lista de valores da variável independente.
        y_values: Lista de valores da variável dependente.

    Returns:
        Modelo treinado.

    Raises:
        ValueError: Se x e y tiverem tamanhos diferentes.
    """
    if len(x_values) != len(y_values):
        raise ValueError(
            f"x e y devem ter o mesmo tamanho. "
            f"Recebido: x={len(x_values)}, y={len(y_values)}"
        )

    X = np.array(x_values).reshape(-1, 1)
    y = np.array(y_values)

    model = LinearRegression()
    model.fit(X, y)

    logger.info(
        "Modelo treinado — coef=%.4f, intercept=%.4f, amostras=%d",
        model.coef_[0], model.intercept_, len(x_values)
    )
    return model


def _get_model() -> LinearRegression:
    """
    Retorna o modelo carregado ou lança exceção se não estiver pronto.

    Returns:
        Modelo treinado em memória.

    Raises:
        HTTPException 400: Se o modelo ainda não foi treinado.
    """
    if _model is None:
        raise HTTPException(
            status_code=400,
            detail="Modelo não treinado. Envie dados para POST /train primeiro."
        )
    return _model


# ── Endpoints ────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
def health_check() -> HealthResponse:
    """
    Verifica se a API está respondendo e se o modelo está carregado.

    Returns:
        Status da API e do modelo.
    """
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        version=APP_VERSION
    )


@app.post("/train", response_model=TrainResponse, tags=["Modelo"])
def train(request: TrainRequest) -> TrainResponse:
    """
    Treina o modelo com os dados fornecidos e persiste em disco.

    Args:
        request: Dados de treino com listas x (features) e y (target).

    Returns:
        Coeficiente, intercepto e número de amostras usadas.

    Raises:
        HTTPException 422: Se x e y tiverem tamanhos diferentes.
    """
    global _model

    try:
        _model = _train_model(request.x, request.y)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    _save_model(_model)

    return TrainResponse(
        status="treinado",
        coef=round(float(_model.coef_[0]), 4),
        intercept=round(float(_model.intercept_), 4),
        samples_used=len(request.x)
    )


@app.post("/predict", response_model=PredictResponse, tags=["Modelo"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Gera predições usando o modelo treinado.

    Args:
        request: Lista de valores x para predizer.

    Returns:
        Lista de predições arredondadas em 4 casas decimais.

    Raises:
        HTTPException 400: Se o modelo não estiver treinado.
    """
    model = _get_model()

    X = np.array(request.x).reshape(-1, 1)
    predictions = model.predict(X).tolist()

    logger.info("Predição gerada para %d valores", len(request.x))

    return PredictResponse(
        predictions=[round(p, 4) for p in predictions],
        model_version=APP_VERSION
    )

# Fim do arquivo main.py
