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

# ---- Configuration ----
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model.pth')
IMG_SIZE = 224
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRANSFORM = transforms.Compose([
    transforms.Resize(256),            # Re-scale slightly above target size
    transforms.CenterCrop(IMG_SIZE),   # Exact crop to remove distractions and white borders
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
    msg = "Welcome to the age verification system. Please read the phrase shown on the screen out loud to begin."
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
        texto = recognizer.recognize_google(audio_rec, language="en-US")
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
    detectado = resultado.get("frase_detectada", "Not detected")
    similitud = resultado["similitud"]

    if resultado["verificado"]:
        msg = f"VERIFIED ({similitud:.0%}) — Transcribed: \"{detectado}\"\nProceeding to age estimation..."
        audio_guia = generar_audio_guia("Identity successfully verified. Now, please upload a photo of your face to verify your age.")
        return msg, gr.update(visible=False), gr.update(visible=True), audio_guia
    else:
        msg = f"NOT VERIFIED ({similitud:.0%}) — Transcribed: \"{detectado}\"\nPlease try again."
        audio_guia = generar_audio_guia("I could not verify your voice. Please try again, making sure to read the phrase clearly.")
        return msg, gr.update(visible=True), gr.update(visible=False), audio_guia

def predecir_edad(imagen):
    """Predice si es mayor de edad, la edad aproximada y genera el Grad-CAM."""
    if imagen is None:
        return "No image uploaded.", None

    img_pil = Image.fromarray(imagen).convert("RGB")
    tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    # Iniciar el explicador usando la capa convolucional final de ResNet
    cam = AgeGradCAM(modelo, modelo.backbone.layer4)
    
    # Generar el heatmap y obtener predicciones 
    # (No usamos torch.no_grad() porque necesitamos los gradientes para la explicabilidad)
    heatmap_img, class_prob, age_pred = cam.generate_heatmap(tensor, imagen)
    
    prob_mayor = class_prob.item()
    edad_aprox = round(age_pred.item())

    # Determine classification based on estimated age (Consistency Fix)
    es_mayor = edad_aprox >= 18
    prob_display = prob_mayor if es_mayor else (1 - prob_mayor)

    if es_mayor:
        resultado = f"ADULT (Confidence: {prob_display:.0%})"
        voz_msg = f"The system estimates you are an adult, approximately {edad_aprox} years old."
    else:
        resultado = f"MINOR (Confidence: {prob_display:.0%})"
        voz_msg = f"The system estimates you are a minor, approximately {edad_aprox} years old."

    resultado += f"\n   Estimated Age: {edad_aprox} years old"
    audio_guia = generar_audio_guia(voz_msg)
    
    return resultado, heatmap_img, audio_guia


# ---- CSS Styles ----
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

with gr.Blocks(title="Age Verification System") as demo:

    gr.Markdown("# Age Verification System")

    # Speaker for the voice guide
    guia_audio = gr.Audio(interactive=False, label="Voice Guide (AI)", autoplay=True)

    # ── INICIO: Desbloqueo de audio ──
    with gr.Group(visible=True) as inicio:
        gr.HTML("""
        <div class="hero-container">
            <h1 class="hero-title">Identity Verification System</h1>
            <p class="hero-subtitle">
                Secure access through voice biometrics and computer vision-based age estimation.
            </p>
        </div>
        """)
        with gr.Row(elem_classes="center-btn"):
            btn_iniciar = gr.Button("Start Verification System", variant="primary", size="lg")

    # ── PASO 1: Verificacion por voz ──
    with gr.Group(visible=False) as paso1:
        gr.Markdown("## Step 1 — Voice Verification")
        gr.Markdown("Read the following phrase out loud and click **Verify Voice**.")

        frase_estado = gr.Textbox(
            label="Phrase to read",
            value=obtener_frase_aleatoria(),
            interactive=False
        )
        btn_nueva_frase = gr.Button("New phrase", variant="secondary")

        entrada_audio = gr.Audio(
            label="Record your voice",
            sources=["microphone"],
            type="numpy"
        )

        btn_verificar_voz = gr.Button("Verify Voice", variant="primary", size="lg")
        salida_voz = gr.Textbox(label="Verification Result", interactive=False)

    # ── PASO 2: Estimacion de edad ──
    with gr.Group(visible=False) as paso2:
        gr.Markdown("## Step 2 — Age Estimation")
        gr.Markdown("Identity verified. Now upload a face photo to check if you are an adult.")

        entrada_imagen = gr.Image(label="Face Photo", type="numpy")
        btn_predecir = gr.Button("Verify Age", variant="primary", size="lg")
        salida_edad = gr.Textbox(label="Result", interactive=False)
        salida_heatmap = gr.Image(label="IA Focus (Grad-CAM)")

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
    demo.launch(theme=gr.themes.Soft(), css=CSS)