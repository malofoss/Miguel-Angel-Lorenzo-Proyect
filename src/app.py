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
import cv2

sys.path.append(os.path.dirname(__file__))

from model import AgeEstimatorCNN
from speech_nlp import obtener_frase_aleatoria, verificar_lectura
from gradcam import AgeGradCAM

# ---- Configuracion ----
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    except (sr.UnknownValueError, sr.RequestError):
        return None
    finally:
        if os.path.exists(ruta_tmp):
            os.unlink(ruta_tmp)

def verificar_voz(audio, frase):
    """
    Verifica la voz y devuelve:
    - mensaje de resultado
    - visibilidad del paso 1 (se oculta si pasa)
    - visibilidad del paso 2 (se muestra si pasa)
    """
    transcripcion = transcribir_desde_gradio(audio)
    resultado = verificar_lectura(frase, transcripcion, umbral=0.75)
    detectado = resultado.get("frase_detectada", "No detectado")
    similitud = resultado["similitud"]

    if resultado["verificado"]:
        msg = f"VERIFICADO ({similitud:.0%}) — Transcrito: \"{detectado}\"\nAvanzando a verificacion de edad..."
        return msg, gr.update(visible=False), gr.update(visible=True)
    else:
        msg = f"NO VERIFICADO ({similitud:.0%}) — Transcrito: \"{detectado}\"\nVuelve a intentarlo."
        return msg, gr.update(visible=True), gr.update(visible=False)

def predecir_edad(imagen):
    """Predice si es mayor de edad, la edad aproximada y genera el Grad-CAM."""
    if imagen is None:
        return "Sin imagen cargada.", None

    # NOVEDAD: Detección y recorte de rostro
    gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        # Usar el rostro más grande detectado (por si hay gente atrás)
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Añadir margen del 20% para incluir frente y barbilla
        margin = int(w * 0.2)
        x1, y1 = max(0, x - margin), max(0, y - int(margin*1.5))
        x2, y2 = min(imagen.shape[1], x + w + margin), min(imagen.shape[0], y + h + margin)
        
        imagen_rostro = imagen[y1:y2, x1:x2]
    else:
        imagen_rostro = imagen # Fallback por si la IA clásica no ve la cara

    img_pil = Image.fromarray(imagen_rostro).convert("RGB")
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    # Iniciar el explicador usando la capa convolucional final de ResNet
    cam = AgeGradCAM(modelo, modelo.backbone.layer4)
    
    # Generar el heatmap y obtener predicciones 
    # (No usamos torch.no_grad() porque necesitamos los gradientes para la explicabilidad)
    heatmap_img, class_prob, age_pred = cam.generate_heatmap(tensor, imagen_rostro)
    
    prob_mayor = class_prob.item()
    edad_aprox = round(age_pred.item())

    # Determinar clasificación
    es_mayor = prob_mayor >= 0.5
    prob_display = prob_mayor if es_mayor else (1 - prob_mayor)

    if es_mayor:
        resultado = f"MAYOR DE EDAD ({prob_display:.0%} probabilidad)"
    else:
        resultado = f"MENOR DE EDAD ({prob_display:.0%} probabilidad)"

    resultado += f"\n   Edad aproximada: {edad_aprox} años"
    return resultado, heatmap_img


# ---- Interfaz Gradio ----

with gr.Blocks(title="Verificacion de Edad", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Sistema de Verificacion de Edad")

    # ── PASO 1: Verificacion por voz ──
    with gr.Group(visible=True) as paso1:
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
        outputs=[salida_voz, paso1, paso2]
    )

    btn_predecir.click(
        fn=predecir_edad,
        inputs=[entrada_imagen],
        outputs=[salida_edad, salida_heatmap]
    )

if __name__ == "__main__":
    demo.launch()