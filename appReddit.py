import praw
import re
import emoji
import json
import spacy
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode

# Descargar recursos de NLTK si es necesario
#nltk.download("stopwords")
stopwords_es = set(stopwords.words("spanish"))

# Modelo SpaCy
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
    texto = re.sub(r"\.{2,}", " ", texto)
    texto = re.sub(r"[^a-zñáéíóúü\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def procesar_texto(texto):
    limpio = limpiar_texto(texto)
    doc = nlp(limpio)
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stopwords_es]
    return " ".join(tokens)

# Configurar Reddit
reddit = praw.Reddit(
    client_id="k8MV_WBoO6hwt3q-jZ6AgA",
    client_secret="VVr8HvR270KzyGNAouwCuFjoN2JuVA",
    user_agent="cp-scraper-by-jorge"
)

subreddits = ["depresion", "ansiedad", "tristeza", "psicologia"]
posts_finales = []

# Obtener publicaciones
for nombre_subreddit in subreddits:
    try:
        subreddit = reddit.subreddit(nombre_subreddit)
        for post in subreddit.new(limit=20):
            post.comments.replace_more(limit=0)

            titulo = post.title + " " + (post.selftext or "")
            titulo_procesado = procesar_texto(titulo)
            etiqueta = clasificar(titulo_procesado)

            comentarios_procesados = []
            for comentario in post.comments[:5]:
                cuerpo = comentario.body.strip()
                if cuerpo:
                    limpio = procesar_texto(cuerpo)
                    comentarios_procesados.append(limpio)

            if comentarios_procesados:
                posts_finales.append({
                    "titulo": titulo.strip(),
                    "comentarios": comentarios_procesados,
                    "frase": f"{etiqueta} feliz" if etiqueta != "Neutro" else "Neutro"
                })
    except Exception as e:
        print(f"⚠️ Error al procesar subreddit '{nombre_subreddit}': {e}")

# Guardar resultado
with open("comentariosRedditMultiprocesoFinal.json", "w", encoding="utf-8") as archivo_json:
    json.dump(posts_finales, archivo_json, indent=2, ensure_ascii=False)

print("✅ Archivo 'comentariosRedditMultiprocesoFinal.json' generado correctamente.")
