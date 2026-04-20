import speech_recognition as sr
import random
import unicodedata
import re
from difflib import SequenceMatcher
from gtts import gTTS
import tempfile
import os

# === ANTI-FRAUD PHRASES ===
FRASES = [
    "Hello my name is John",
    "The big cat is sleeping",
    "I want to eat a green apple",
    "The weather is very nice",
    "I have a small house",
]

def normalizar(texto: str) -> str:
    """Elimina tildes, pasa a minusculas y quita puntuacion."""
    texto = texto.lower().strip()
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    texto = re.sub(r'[^a-z0-9\s]', '', texto)
    return texto

def similitud(a: str, b: str) -> float:
    """Devuelve similitud entre 0 y 1 usando SequenceMatcher."""
    return SequenceMatcher(None, normalizar(a), normalizar(b)).ratio()

def obtener_frase_aleatoria() -> str:
    """Selecciona una frase aleatoria del banco de frases."""
    return random.choice(FRASES)

def transcribir_audio(duracion: int = 5, idioma: str = "en-US") -> str | None:
    """
    Graba audio del microfono y lo transcribe.
    Devuelve el texto transcrito o None si falla.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("  [INFO] Adjusting background noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print(f"  [REC] Recording ({duracion}s)... Speak now!")
        try:
            audio = recognizer.listen(source, timeout=duracion + 2, phrase_time_limit=duracion)
        except sr.WaitTimeoutError:
            print("  [WARNING] No speech detected.")
            return None

    print("  [INFO] Transcribing...")
    try:
        texto = recognizer.recognize_google(audio, language=idioma)
        print(f"  [TEXT] Transcription: '{texto}'")
        return texto
    except sr.UnknownValueError:
        print("  [WARNING] Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"  [ERROR] Google API error: {e}")
        return None

def verificar_lectura(frase_esperada: str, texto_transcrito: str, umbral: float = 0.75) -> dict:
    """
    Compara la frase esperada con la transcripcion.
    Devuelve dict con resultado y puntuacion de similitud.
    """
    if texto_transcrito is None:
        return {"verificado": False, "similitud": 0.0, "motivo": "No transcription"}

    score = similitud(frase_esperada, texto_transcrito)
    verificado = score >= umbral

    return {
        "verificado": verificado,
        "similitud": round(score, 3),
        "frase_esperada": frase_esperada,
        "frase_detectada": texto_transcrito,
        "motivo": "OK" if verificado else f"Insufficient similarity ({score:.1%} < {umbral:.0%})"
    }

def pipeline_antifraude(duracion: int = 5, umbral: float = 0.75) -> dict:
    """
    Pipeline completo:
    1. Muestra frase al usuario
    2. Graba su voz
    3. Transcribe con STT
    4. Verifica similitud con NLP
    Devuelve el resultado final.
    """
    frase = obtener_frase_aleatoria()
    print(f"\n{'='*50}")
    print(f"  READ OUT LOUD:")
    print(f"  '{frase}'")
    print(f"{'='*50}\n")

    transcripcion = transcribir_audio(duracion=duracion)
    resultado = verificar_lectura(frase, transcripcion, umbral=umbral)

    estado = "VERIFIED" if resultado["verificado"] else "FRAUD DETECTED"
    print(f"\n  Result: {estado} (similarity: {resultado['similitud']:.1%})")
    return resultado

def generar_audio_guia(texto: str, idioma: str = "en") -> str:
    """
    Convierte texto a voz y lo guarda en un archivo temporal .mp3.
    Devuelve la ruta del archivo.
    """
    tts = gTTS(text=texto, lang=idioma)
    
    # Creamos un archivo temporal que no se borre inmediatamente
    # para que Gradio pueda leerlo antes de que desaparezca.
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_file.name)
    temp_file.close()
    
    return temp_file.name


if __name__ == "__main__":
    # Quick test of audio generation
    ruta = generar_audio_guia("Hello, this is a test of the voice guidance system.")
    print(f"Audio generated at: {ruta}")
    # Note: File stays in temp directory.