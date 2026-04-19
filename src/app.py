import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import speech_recognition as sr
import numpy as np
import tempfile
import os
import sys
import scipy.io.wavfile as wav

sys.path.append(os.path.dirname(__file__))

from model import AgeEstimatorCNN
from speech_nlp import obtener_frase_aleatoria, verificar_lectura, generar_audio_guia
from gradcam import AgeGradCAM

# ---- Configuracion ----
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM = transforms.Compose([
    transforms.Resize(256),            # Re-escala un poco por encima
    transforms.CenterCrop(IMG_SIZE),   # Recorta el borde exacto (quita distracciones y marcos blancos)
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def cargar_modelo():
    model = AgeEstimatorCNN()
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

modelo = cargar_modelo()

def nueva_frase():
    return obtener_frase_aleatoria()

def bienvenida():
    """Genera mensaje de bienvenida al iniciar la app."""
    msg = "Bienvenido al sistema de verificación de edad. Por favor, lee en voz alta la frase que aparece en pantalla para comenzar."
    return generar_audio_guia(msg)

def transcribir_desde_gradio(audio):
    if audio is None:
        return None
    sample_rate, data = audio
    if data.ndim > 1:
        data = data[:, 0]
    if data.dtype != np.int16:
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = (data / max_val * 32767).astype(np.int16)
        else:
            data = data.astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, sample_rate, data)
        ruta_tmp = f.name
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(ruta_tmp) as source:
            audio_rec = recognizer.record(source)
        texto = recognizer.recognize_google(audio_rec, language="es-ES")
        return texto
    except (sr.UnknownValueError, sr.RequestError):
        return None
    finally:
        if os.path.exists(ruta_tmp):
            os.unlink(ruta_tmp)

def verificar_voz(audio, frase):
    """
    Verifica la voz y devuelve:
    - mensaje de resultado
    - visibilidad del paso 1
    - visibilidad del paso 2
    - audio de la guia
    """
    transcripcion = transcribir_desde_gradio(audio)
    resultado = verificar_lectura(frase, transcripcion, umbral=0.75)
    detectado = resultado.get("frase_detectada", "No detectado")
    similitud = resultado["similitud"]

    if resultado["verificado"]:
        msg = f"VERIFICADO ({similitud:.0%}) — Transcrito: \"{detectado}\"\nAvanzando a verificacion de edad..."
        audio_guia = generar_audio_guia("Identidad verificada con éxito. Ahora, por favor, sube una foto de tu rostro para verificar tu edad.")
        return msg, gr.update(visible=False), gr.update(visible=True), audio_guia
    else:
        msg = f"NO VERIFICADO ({similitud:.0%}) — Transcrito: \"{detectado}\"\nVuelve a intentarlo."
        audio_guia = generar_audio_guia("No he podido verificar tu voz. Por favor, inténtalo de nuevo leyendo la frase con claridad.")
        return msg, gr.update(visible=True), gr.update(visible=False), audio_guia

def predecir_edad(imagen):
    """Predice si es mayor de edad, la edad aproximada y genera el Grad-CAM."""
    if imagen is None:
        return "Sin imagen cargada.", None

    img_pil = Image.fromarray(imagen).convert("RGB")
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    # Iniciar el explicador usando la capa convolucional final de ResNet
    cam = AgeGradCAM(modelo, modelo.backbone.layer4)
    
    # Generar el heatmap y obtener predicciones 
    # (No usamos torch.no_grad() porque necesitamos los gradientes para la explicabilidad)
    heatmap_img, class_prob, age_pred = cam.generate_heatmap(tensor, imagen)
    
    prob_mayor = class_prob.item()
    edad_aprox = round(age_pred.item())

    # Determinar clasificación
    es_mayor = prob_mayor >= 0.5
    prob_display = prob_mayor if es_mayor else (1 - prob_mayor)

    if es_mayor:
        resultado = f"MAYOR DE EDAD ({prob_display:.0%} probabilidad)"
        voz_msg = f"El sistema estima que eres mayor de edad, con aproximadamente {edad_aprox} años."
    else:
        resultado = f"MENOR DE EDAD ({prob_display:.0%} probabilidad)"
        voz_msg = f"El sistema estima que eres menor de edad, con aproximadamente {edad_aprox} años."

    resultado += f"\n   Edad aproximada: {edad_aprox} años"
    audio_guia = generar_audio_guia(voz_msg)
    
    return resultado, heatmap_img, audio_guia


# ---- Estilos CSS ----
CSS = """
.hero-container {
    padding: 60px 20px;
    text-align: center;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    margin: 20px auto;
    max-width: 800px;
}
.hero-title {
    color: #1a365d;
    font-size: 2.8rem;
    font-weight: 800;
    margin-bottom: 1rem;
    letter-spacing: -0.025em;
}
.hero-subtitle {
    color: #4a5568;
    font-size: 1.2rem;
    max-width: 600px;
    margin: 0 auto 2.5rem auto;
    line-height: 1.6;
}
.center-btn {
    display: flex;
    justify-content: center;
    width: 100%;
}
"""

# ---- Interfaz Gradio ----

with gr.Blocks(title="Verificación de Edad", theme=gr.themes.Soft(), css=CSS) as demo:

    gr.Markdown("# Sistema de Verificación de Edad")

    # Altavoz para la guia por voz (Debe ser visible para asegurar la reproduccion en algunos navegadores)
    guia_audio = gr.Audio(interactive=False, label="Guía por Voz (IA)", autoplay=True)

    # ── INICIO: Desbloqueo de audio ──
    with gr.Group(visible=True) as inicio:
        gr.HTML("""
        <div class="hero-container">
            <h1 class="hero-title">Sistema de Verificación de Identidad</h1>
            <p class="hero-subtitle">
                Acceso seguro mediante biometría de voz y estimación de edad por visión artificial.
            </p>
        </div>
        """)
        with gr.Row(elem_classes="center-btn"):
            btn_iniciar = gr.Button("Iniciar Sistema de Verificación", variant="primary", size="lg")

    # ── PASO 1: Verificacion por voz ──
    with gr.Group(visible=False) as paso1:
        gr.Markdown("## Paso 1 — Verificacion por voz")
        gr.Markdown("Lee en voz alta la frase que aparece a continuacion y pulsa **Verificar voz**.")

        frase_estado = gr.Textbox(
            label="Frase a leer",
            value=obtener_frase_aleatoria(),
            interactive=False
        )
        btn_nueva_frase = gr.Button("Nueva frase", variant="secondary")

        entrada_audio = gr.Audio(
            label="Graba tu voz",
            sources=["microphone"],
            type="numpy"
        )

        btn_verificar_voz = gr.Button("Verificar voz", variant="primary", size="lg")
        salida_voz = gr.Textbox(label="Resultado verificacion", interactive=False)

    # ── PASO 2: Estimacion de edad ──
    with gr.Group(visible=False) as paso2:
        gr.Markdown("## Paso 2 — Verificacion de edad")
        gr.Markdown("Identidad verificada. Sube ahora una foto del rostro para verificar si es mayor de edad.")

        entrada_imagen = gr.Image(label="Foto del rostro", type="numpy")
        btn_predecir = gr.Button("Verificar edad", variant="primary", size="lg")
        salida_edad = gr.Textbox(label="Resultado", interactive=False)
        salida_heatmap = gr.Image(label="Lo que mira la IA (Grad-CAM)")

    # ── Eventos ──
    btn_nueva_frase.click(fn=nueva_frase, outputs=frase_estado)

    btn_verificar_voz.click(
        fn=verificar_voz,
        inputs=[entrada_audio, frase_estado],
        outputs=[salida_voz, paso1, paso2, guia_audio]
    )

    btn_predecir.click(
        fn=predecir_edad,
        inputs=[entrada_imagen],
        outputs=[salida_edad, salida_heatmap, guia_audio]
    )

    # El inicio activa la bienvenida y muestra el paso 1
    btn_iniciar.click(
        fn=lambda: (bienvenida(), gr.update(visible=False), gr.update(visible=True)),
        outputs=[guia_audio, inicio, paso1]
    )

if __name__ == "__main__":
    demo.launch()