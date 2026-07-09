import os
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


def get_backend_url() -> str:
    """
    Obtiene la URL del backend desde la variable de ambiente BACKEND_URL.
    Ejemplo en .env:
    BACKEND_URL="http://127.0.0.1:8000"
    """
    backend_url = os.getenv("BACKEND_URL")

    if backend_url is None:
        raise ValueError(
            "No se encontró BACKEND_URL. "
            "Debes definirla en el archivo .env, por ejemplo: "
            'BACKEND_URL="http://127.0.0.1:8000"'
        )

    return backend_url.rstrip("/")


def enviar_prediccion(
    asunto: str,
    contenido: str,
    canal_ticket: str,
    categoria_problema: str,
    tipo_cuenta: str,
    antiguedad_cuenta_dias: int,
) -> str:
    """
    Envía una solicitud al endpoint POST /predict del backend FastAPI.

    Nota:
    El modelo final desplegado es MLP + Embeddings, por lo que el backend
    usa solamente asunto y contenido para generar el embedding. Los otros
    atributos se mantienen en la interfaz para trazabilidad y para cumplir
    con la separación visual de atributos del ticket y del usuario.
    """
    backend_url = get_backend_url()
    endpoint = f"{backend_url}/predict"

    payload: dict[str, Any] = {
        "asunto": asunto,
        "contenido": contenido,
    }

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=120,
        )

        response.raise_for_status()

        data = response.json()
        prediccion = data["prediccion"]

        return f"Prioridad predicha: {prediccion}"

    except requests.exceptions.HTTPError:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        return f"Error HTTP al llamar al backend: {response.status_code}\n{detail}"

    except requests.exceptions.RequestException as error:
        return f"Error de conexión con el backend: {error}"

    except Exception as error:
        return f"Error inesperado: {error}"
