from multiprocessing import Process, Queue
from playwright.sync_api import sync_playwright
import time
import random
import json
import re
import emoji
import spacy
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import os

# Descargar recursos de NLTK si no están
nltk.download('stopwords', quiet=True)
stopwords_es = set(stopwords.words("spanish"))

# Carga modelo de spaCy
nlp = spacy.load("es_core_news_sm")

# Listas de palabras clave
palabras_depresion = {
    "depresion", "triste", "vacio", "ya no quiero", "no puedo mas", "autolesion",
    "inutil", "suicidio", "matarme", "quitarme la vida", "no quiero vivir", "muerte",
    "morir", "desaparecer", "me rindo", "me odio", "miedo", "terror", "suicidarme", "morirme", "murio"
}
palabras_ansiedad = {
    "ansiedad", "ataque", "nervioso", "panico", "hiperventilo", "ansioso", "estresado",
    "estres", "preocupacion", "inquietud", "tension", "agitado", "preocupada", "preocupado",
    "nerviosa", "siento que me ahogo", "siento que me muero", "siento que me desmayo", "temblando",
    "taquicardia", "palpitaciones", "sudoracion", "siento que me vuelvo loco", "no puedo respirar",
    "temblor", "inquieto"
}

def esperar_aleatorio(min_seg=1.5, max_seg=3.5):
    time.sleep(random.uniform(min_seg, max_seg))

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

def clasificar(texto):
    texto = texto.lower()
    if any(p in texto for p in palabras_depresion):
        return "Depresión"
    elif any(p in texto for p in palabras_ansiedad):
        return "Ansiedad"
    else:
        return "Neutro"

def scrape_tiktok(frase, queue, max_videos=6):
    resultados = []
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)
            page = browser.new_page()
            page.goto("https://www.tiktok.com", timeout=60000)
            esperar_aleatorio(2, 4)

            page.click('button[data-e2e="nav-search"]')
            esperar_aleatorio()

            input_locator = page.locator('input[data-e2e="search-user-input"]:visible')
            input_locator.fill("")
            for letra in frase:
                input_locator.type(letra)
                time.sleep(random.uniform(0.05, 0.2))
            esperar_aleatorio(1.5, 2)
            page.keyboard.press("Enter")
            esperar_aleatorio(2, 4)

            page.wait_for_selector('div[data-e2e="search_top-item"]', timeout=15000)
            page.click('div[data-e2e="search_top-item"]')
            esperar_aleatorio(2, 4)

            for i in range(max_videos):
                try:
                    page.wait_for_selector('p[data-e2e="comment-level-1"]', timeout=20000)
                    comentarios = page.locator('p[data-e2e="comment-level-1"]')

                    descripcion = page.locator('div[data-e2e="browse-video-desc"] span[data-e2e="new-desc-span"]')
                    desc_text = " ".join([
                        descripcion.nth(j).inner_text().strip()
                        for j in range(descripcion.count())
                        if descripcion.nth(j).inner_text().strip()
                    ])
                    desc_proc = procesar_texto(desc_text)
                    etiqueta = clasificar(desc_proc)

                    comentarios_limpios = []
                    for j in range(comentarios.count()):
                        txt = comentarios.nth(j).inner_text().strip()
                        if txt:
                            limpio = procesar_texto(txt)
                            comentarios_limpios.append(limpio)

                    if comentarios_limpios:
                        resultados.append({
                            "titulo": f"Video {i+1}",
                            "comentarios": comentarios_limpios,
                            "frase": f"{etiqueta} feliz" if etiqueta != "Neutro" else "Neutro"
                        })

                    if i < max_videos - 1:
                        page.click('button[data-e2e="arrow-right"]')
                        esperar_aleatorio(4, 8)
                except Exception as e:
                    print(f"⚠️ Error en el video {i+1}: {e}")
                    break

            browser.close()
    except Exception as e:
        print(f"❌ Error general en scrape_tiktok: {e}")
    queue.put(resultados)

def run_tiktok_scraping():
    q1 = Queue()
    q2 = Queue()

    # Leer frases desde archivo o usar predeterminadas
    if os.path.exists("frases_scraping.json"):
        with open("frases_scraping.json", "r", encoding="utf-8") as f:
            frases = json.load(f)
    else:
        frases = ["depresion", "ansiedad"]

    p1 = Process(target=scrape_tiktok, args=(frases[0], q1))
    p2 = Process(target=scrape_tiktok, args=(frases[1], q2))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print(f"Procesos finalizados: p1={p1.exitcode}, p2={p2.exitcode}")

    datos = []
    try:
        datos += q1.get(timeout=30)
        datos += q2.get(timeout=30)
    except Exception as e:
        print("⚠️ Error al recoger datos del queue:", e)

    with open("comentariosTikTokMultiprocesoFinal.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False)

    print("✅ Datos guardados en 'comentariosTikTokMultiprocesoFinal.json'")

if __name__ == "__main__":
    run_tiktok_scraping()
