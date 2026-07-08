import os
import pickle
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

N_EMBEDDING_DIMS = 1024
MODEL_PATH = Path(__file__).resolve().parents[1] / "modelo_final.pkl"


def construir_texto_embedding(asunto: str, contenido: str) -> str:
    """
    Construye el texto en el mismo formato usado para generar embeddings.
    """
    return f"Asunto_Ticket: {asunto}\nContenido_Ticket: {contenido}\n"


def cargar_modelo():
    """
    Carga el pipeline final entrenado.

    Este pipeline corresponde a MLP + Embeddings:
    - ColumnTransformer con StandardScaler sobre embedding_dim_1 ... embedding_dim_1024
    - MLPClassifier
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        modelo = pickle.load(f)

    return modelo


def generar_embedding(asunto: str, contenido: str) -> list[float]:
    """
    Genera el embedding del ticket usando el mismo modelo y dimensión
    usados durante el entrenamiento.
    """
    load_dotenv()

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if google_api_key is None:
        raise ValueError("No se encontró GOOGLE_API_KEY.")

    texto = construir_texto_embedding(asunto, contenido)

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=google_api_key,
        output_dimensionality=N_EMBEDDING_DIMS,
    )

    embedding = embedding_model.embed_query(texto)

    if len(embedding) != N_EMBEDDING_DIMS:
        raise ValueError(
            f"El embedding generado tiene dimensión {len(embedding)}, "
            f"pero el modelo espera {N_EMBEDDING_DIMS} dimensiones."
        )

    return embedding


def construir_input_pipeline(asunto: str, contenido: str) -> pd.DataFrame:
    """
    Construye el DataFrame esperado por el pipeline entrenado.

    Como el modelo final es MLP + Embeddings, solo se necesitan las columnas:
    embedding_dim_1, embedding_dim_2, ..., embedding_dim_1024.
    """
    embedding = generar_embedding(asunto, contenido)

    columnas_embedding = [f"embedding_dim_{i}" for i in range(1, N_EMBEDDING_DIMS + 1)]

    X = pd.DataFrame([embedding], columns=columnas_embedding)

    return X


def generate_prediction(asunto: str, contenido: str) -> str:
    """
    Predice el Nivel_Prioridad de un ticket nuevo.

    Parámetros mínimos:
    - asunto: asunto del ticket.
    - contenido: descripción o contenido del ticket.

    Retorna:
    - Una clase: Baja, Media, Alta o Critica.
    """
    modelo = cargar_modelo()
    X = construir_input_pipeline(asunto, contenido)

    prediccion = modelo.predict(X)[0]

    return str(prediccion)


if __name__ == "__main__":
    asunto_ejemplo = "No puedo acceder a mi cuenta"
    contenido_ejemplo = (
        "Desde esta mañana no puedo ingresar a mi cuenta. "
        "La aplicación muestra un error y necesito hacer una transferencia urgente."
    )

    prediccion = generate_prediction(
        asunto=asunto_ejemplo,
        contenido=contenido_ejemplo,
    )

    print("Asunto:", asunto_ejemplo)
    print("Contenido:", contenido_ejemplo)
    print("Predicción del ticket:", prediccion)
