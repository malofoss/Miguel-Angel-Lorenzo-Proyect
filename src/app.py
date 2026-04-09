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
from speech_nlp import obtener_frase_aleatoria, verificar_lectura

# ---- Configuracion ----
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
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
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None
    finally:
        if os.path.exists(ruta_tmp):
            os.unlink(ruta_tmp)

def procesar(imagen, audio, frase):
    if imagen is None:
        return "Sin imagen cargada.", "Sin imagen cargada."

    img_pil = Image.fromarray(imagen).convert("RGB")
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        edad = modelo(tensor).item()

    resultado_edad = f"Edad estimada: {round(edad)} anos"

    if audio is None:
        return resultado_edad, "No se recibio audio."

    transcripcion = transcribir_desde_gradio(audio)
    resultado_voz = verificar_lectura(frase, transcripcion, umbral=0.75)

    detectado = resultado_voz.get("frase_detectada", "No detectado")
    similitud = resultado_voz["similitud"]

    if resultado_voz["verificado"]:
        texto_voz = f'VERIFICADO ({similitud:.0%})\nTranscrito: "{detectado}"'
    else:
        texto_voz = f'NO VERIFICADO ({similitud:.0%})\nTranscrito: "{detectado}"'

    return resultado_edad, texto_voz


with gr.Blocks(title="Verificacion de Edad", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Sistema de Verificacion de Edad")
    gr.Markdown("Sube una foto de tu cara y lee la frase en voz alta para verificar tu identidad.")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Imagen")
            entrada_imagen = gr.Image(label="Foto del rostro", type="numpy")

        with gr.Column():
            gr.Markdown("### 2. Verificacion por voz")
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

    btn_verificar = gr.Button("Verificar", variant="primary", size="lg")

    with gr.Row():
        salida_edad = gr.Textbox(label="Resultado edad", interactive=False)
        salida_voz = gr.Textbox(label="Resultado verificacion", interactive=False)

    btn_nueva_frase.click(fn=nueva_frase, outputs=frase_estado)
    btn_verificar.click(
        fn=procesar,
        inputs=[entrada_imagen, entrada_audio, frase_estado],
        outputs=[salida_edad, salida_voz]
    )

if __name__ == "__main__":
    demo.launch()