import pickle
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# Ruta donde se encuentra el mejor modelo entrenado
MODEL_PATH = Path("models/best_model.pkl")

# Crear aplicación FastAPI
app = FastAPI(
    title="API de Potabilidad del Agua",
    description="Predicción de potabilidad del agua usando XGBoost.",
    version="1.0.0",
)

# Cargar modelo al iniciar la aplicación
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Esquema de entrada para las mediciones de agua
class WaterMeasurement(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# Ruta principal con descripción de la API
@app.get("/")
def home():
    return {
        "modelo": "XGBoost optimizado con Optuna y registrado con MLflow",
        "problema": "Clasificar si una muestra de agua es potable o no potable.",
        "entrada": "9 mediciones químicas del agua.",
        "salida": {"potabilidad": "1 si es potable, 0 si no es potable"},
    }


# Ruta para predecir la potabilidad del agua
@app.post("/potabilidad/")
def predict_potability(measurement: WaterMeasurement):
    # Convertir entrada a DataFrame
    data = pd.DataFrame([measurement.model_dump()])

    # Obtener predicción
    prediction = model.predict(data)[0]

    # Retornar resultado
    return {"potabilidad": int(prediction)}


# Levantar servidor FastAPI
def main():
    uvicorn.run(
        "main:app",
        # host="127.0.0.1", descomenta esta línea y comenta la siguiente si se quiere testear
        # la parte 2 de la tarea
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


# Ejecutar aplicación
if __name__ == "__main__":
    main()
