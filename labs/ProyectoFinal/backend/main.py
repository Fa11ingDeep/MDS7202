from backend.generate_prediction import generate_prediction
from backend.models import PredictionRequest, PredictionResponse
from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="ChaucherApp Ticket Priority API",
    description="API para predecir el nivel de prioridad de tickets de soporte al cliente.",
    version="1.0.0",
)


@app.get("/")
def health_check() -> dict[str, str]:
    """
    Endpoint simple para verificar que la API está levantada.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """
    Endpoint de predicción.

    Recibe asunto y contenido del ticket, genera los embeddings usando
    generate_prediction y retorna la prioridad predicha.
    """
    try:
        prediccion = generate_prediction(
            asunto=payload.asunto,
            contenido=payload.contenido,
        )

        return PredictionResponse(prediccion=prediccion)

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Error al generar la predicción: {str(error)}",
        ) from error
