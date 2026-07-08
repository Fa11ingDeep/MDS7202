from pydantic import BaseModel, ConfigDict, Field, StrictStr


class PredictionRequest(BaseModel):
    """
    Request esperada por el endpoint /predict.

    Se usa StrictStr para evitar coerción automática de tipos.
    Por ejemplo, si asunto=123, FastAPI/Pydantic retornará error 422.
    """

    model_config = ConfigDict(extra="forbid")

    asunto: StrictStr = Field(
        ...,
        min_length=1,
        description="Asunto del ticket de soporte.",
        examples=["No puedo acceder a mi cuenta"],
    )

    contenido: StrictStr = Field(
        ...,
        min_length=1,
        description="Contenido o descripción del ticket de soporte.",
        examples=["Desde esta mañana no puedo ingresar a mi cuenta. Necesito hacer una transferencia urgente."],
    )


class PredictionResponse(BaseModel):
    """
    Response retornada por el endpoint /predict.
    """

    prediccion: StrictStr = Field(
        ...,
        description="Nivel de prioridad predicho por el modelo.",
        examples=["Alta"],
    )
