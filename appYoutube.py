from multiprocessing import Process, Queue
from googleapiclient.discovery import build
import re
import emoji
import json
import spacy
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import time
import os

# Preparación NLP
# nltk.download("stopwords")
stopwords_es = set(stopwords.words("spanish"))
nlp = spacy.load("es_core_news_sm")

# Listas de palabras clave
palabras_depresion = set([
    "depresion", "triste", "vacio", "ya no quiero", "no puedo mas", "autolesion",
    "inutil", "suicidio", "matarme", "quitarme la vida", "no quiero vivir", "muerte",
    "morir", "desaparecer", "me rindo", "me odio", "miedo", "terror", "suicidarme", "morirme", "murio"
])

palabras_ansiedad = set([
    "ansiedad", "ataque", "nervioso", "panico", "hiperventilo", "ansioso", "estresado",
    "estres", "preocupacion", "inquietud", "tension", "agitado", "preocupada", "preocupado",
    "nerviosa", "siento que me ahogo", "siento que me muero", "siento que me desmayo", "temblando",
    "taquicardia", "palpitaciones", "sudoracion", "siento que me vuelvo loco", "no puedo respirar",
    "temblor", "inquieto"
])


def clasificar(texto):
    texto = texto.lower()
    if any(p in texto for p in palabras_depresion):
        return "Depresión"
    elif any(p in texto for p in palabras_ansiedad):
        return "Ansiedad"
    else:
        return "Neutro"

def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = unidecode(texto)
    texto = emoji.replace_emoji(texto, '')
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"@\w+|#\w+", "", texto)
    texto = re.sub(r"[^a-zñáéíóúü\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def procesar_texto(texto):
    limpio = limpiar_texto(texto)
    doc = nlp(limpio)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords_es]
    return " ".join(tokens)

def scrape_youtube(frase, queue):
    API_KEY = "AIzaSyAUbqSax3QrAlwnxzoTATmY1KrFK8b2W5Y"
    youtube = build("youtube", "v3", developerKey=API_KEY)
    resultados = []

    try:
        search_response = youtube.search().list(
            q=frase,
            part="snippet",
            type="video",
            maxResults=9
        ).execute()

        for item in search_response.get("items", []):
            video_id = item["id"]["videoId"]
            titulo = item["snippet"]["title"]
            descripcion = item["snippet"].get("description", "")
            texto_completo = f"{titulo} {descripcion}"
            procesado = procesar_texto(texto_completo)
            etiqueta = clasificar(procesado)

            comentarios_limpios = []
            try:
                comments_response = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=5,
                    textFormat="plainText"
                ).execute()

                for comment in comments_response.get("items", []):
                    texto_com = comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                    limpio = procesar_texto(texto_com)
                    comentarios_limpios.append(limpio)

            except Exception as e:
                print(f"Error al obtener comentarios de {video_id}: {e}")

            if comentarios_limpios:
                resultados.append({
                    "titulo": titulo.strip(),
                    "comentarios": comentarios_limpios,
                    "frase": f"{etiqueta} feliz" if etiqueta != "Neutro" else "Neutro"
                })

        queue.put(resultados)
        print(f"Scraping YouTube para '{frase}' completado.")
    except Exception as e:
        print(f"Error en búsqueda de '{frase}': {e}")
        queue.put([])

def run_youtube_scraping():
    q1 = Queue()
    q2 = Queue()

    # Leer frases desde archivo (formato: ["frase1", "frase2"])
    if os.path.exists("frases_scraping.json"):
        with open("frases_scraping.json", "r", encoding="utf-8") as f:
            frases = json.load(f)
    else:
        frases = ["depresión", "ansiedad"]

    p1 = Process(target=scrape_youtube, args=(frases[0], q1))
    p2 = Process(target=scrape_youtube, args=(frases[1], q2))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(f"p1 terminado: {p1.exitcode}")
    print(f"p2 terminado: {p2.exitcode}")

    datos = []
    try:
        datos += q1.get(timeout=30)
        datos += q2.get(timeout=30)
    except Exception as e:
        print("Error al recoger datos:", e)

    with open("comentariosYouTubeMultiprocesoFinal.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    print("Datos guardados en 'comentariosYouTubeMultiprocesoFinal.json'")

if __name__ == "__main__":
    run_youtube_scraping()

