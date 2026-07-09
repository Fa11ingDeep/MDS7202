import os

import gradio as gr
from services import enviar_prediccion

CANAL_OPTIONS = [
    "App",
    "Web",
    "Correo",
    "WhatsApp",
]

CATEGORIA_OPTIONS = [
    "Fraude",
    "Transferencias",
    "Tarjeta digital",
    "Cuenta",
    "Préstamos",
    "Seguros",
    "Inversiones",
    "Pregunta general",
]

TIPO_CUENTA_OPTIONS = [
    "Free",
    "Premium",
    "Business",
]


CUSTOM_CSS = """
body {
    background: linear-gradient(135deg, #f4fbf8 0%, #eef7ff 100%);
}

#main-container {
    max-width: 1050px;
    margin: auto;
}

.chaucher-header {
    background: linear-gradient(135deg, #00A878 0%, #0077B6 100%);
    color: white;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
}

.chaucher-header h1 {
    margin: 0;
    font-size: 34px;
}

.chaucher-header p {
    margin-top: 10px;
    font-size: 16px;
}

.section-card {
    border-radius: 16px;
    padding: 18px;
    background: white;
    border: 1px solid #e7eef3;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
}

.prediction-box textarea {
    font-size: 20px !important;
    font-weight: 700 !important;
}
"""


with gr.Blocks(
    title="ChaucherApp - Priorización de Tickets",
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
) as demo:
    with gr.Column(elem_id="main-container"):
        gr.HTML(
            """
            <div class="chaucher-header">
                <h1>ChaucherApp</h1>
                <p>
                    Clasificador de prioridad de tickets de soporte al cliente.
                    La app llama al backend FastAPI y retorna la prioridad predicha por el modelo.
                </p>
            </div>
            """
        )

        gr.Markdown(
            """
            **Nota:** el modelo final corresponde a `MLP + Embeddings`, por lo que la predicción
            se realiza usando el texto del asunto y contenido del ticket. Los atributos adicionales
            se incluyen en la interfaz para ordenar el ingreso de información y mantener trazabilidad.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("## Atributos del ticket")

                    asunto = gr.Textbox(
                        label="Asunto del ticket",
                        placeholder="Ej: No puedo acceder a mi cuenta",
                        lines=1,
                    )

                    contenido = gr.Textbox(
                        label="Contenido del ticket",
                        placeholder=(
                            "Ej: Desde esta mañana no puedo ingresar a mi cuenta. "
                            "Necesito hacer una transferencia urgente."
                        ),
                        lines=7,
                    )

                    canal_ticket = gr.Dropdown(
                        label="Canal del ticket",
                        choices=CANAL_OPTIONS,
                        value="App",
                    )

                    categoria_problema = gr.Dropdown(
                        label="Categoría del problema",
                        choices=CATEGORIA_OPTIONS,
                        value="Cuenta",
                    )

            with gr.Column(scale=1):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("## Atributos del usuario")

                    tipo_cuenta = gr.Dropdown(
                        label="Tipo de cuenta",
                        choices=TIPO_CUENTA_OPTIONS,
                        value="Free",
                    )

                    antiguedad_cuenta_dias = gr.Number(
                        label="Antigüedad de la cuenta en días",
                        value=30,
                        precision=0,
                        minimum=0,
                    )

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("## Resultado")

                    resultado = gr.Textbox(
                        label="Predicción del modelo",
                        lines=3,
                        interactive=False,
                        elem_classes="prediction-box",
                    )

                    boton_predecir = gr.Button(
                        "Predecir prioridad",
                        variant="primary",
                    )

                    boton_limpiar = gr.ClearButton(
                        components=[
                            asunto,
                            contenido,
                            canal_ticket,
                            categoria_problema,
                            tipo_cuenta,
                            antiguedad_cuenta_dias,
                            resultado,
                        ],
                        value="Limpiar formulario",
                    )

        ejemplo_asunto = "Transacción desconocida y cuenta bloqueada"
        ejemplo_contenido = (
            "Tengo una transferencia no reconocida en mi cuenta y además no puedo "
            "acceder a la aplicación. Necesito ayuda urgente porque mi dinero está "
            "comprometido y temo que sea fraude."
        )

        gr.Examples(
            examples=[
                [
                    ejemplo_asunto,
                    ejemplo_contenido,
                    "App",
                    "Fraude",
                    "Premium",
                    420,
                ],
                [
                    "No puedo acceder a mi cuenta",
                    (
                        "Desde esta mañana no puedo ingresar a mi cuenta. "
                        "La aplicación muestra un error y necesito hacer una "
                        "transferencia urgente."
                    ),
                    "App",
                    "Cuenta",
                    "Free",
                    30,
                ],
                [
                    "Consulta sobre seguro contratado",
                    ("Quisiera saber cuándo comienza la cobertura del seguro que contraté desde la aplicación."),
                    "Correo",
                    "Seguros",
                    "Business",
                    900,
                ],
            ],
            inputs=[
                asunto,
                contenido,
                canal_ticket,
                categoria_problema,
                tipo_cuenta,
                antiguedad_cuenta_dias,
            ],
        )

        boton_predecir.click(
            fn=enviar_prediccion,
            inputs=[
                asunto,
                contenido,
                canal_ticket,
                categoria_problema,
                tipo_cuenta,
                antiguedad_cuenta_dias,
            ],
            outputs=resultado,
        )

if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
