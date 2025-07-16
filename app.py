from flask import Flask, request, jsonify
import numpy as np

import openai
from openai import OpenAI
import os
from flask_cors import CORS
import subprocess
from playwright.sync_api import sync_playwright
import sys
import json


client = OpenAI( api_key='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')


# === Configuración ===
ID2LABEL = {0: "Otro", 1: "Depresión", 2: "Ansiedad"}
LABEL_DESC = {
    "Otro": "Texto sin señales directas de depresión o ansiedad.",
    "Depresión": "El texto presenta señales relacionadas con síntomas depresivos.",
    "Ansiedad": "El texto presenta señales asociadas a crisis o síntomas de ansiedad."
}

# Cargar modelo BERT
# model_path = "modelo_bert_depresion"
# model = BertForSequenceClassification.from_pretrained(model_path)
# tokenizer = BertTokenizer.from_pretrained(model_path)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()

# === Flask App ===
app = Flask(__name__)
CORS(app)  


def predecir(texto):
    # inputs = tokenizer(texto, truncation=True, padding=True, max_length=128, return_tensors="pt")
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     logits = outputs.logits
    #     probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    #     pred = np.argmax(probs)

    return jsonify({
        "etiqueta": ID2LABEL[pred],
        "confianza": round(float(probs[pred]), 4)
    })



@app.route("/palabrasClave", methods=["POST"])
def extraer_palabras_clave():
    try:
        data = request.get_json()
        print("JSON recibido en /palabrasClave:", data)

        respuestas = data.get("respuestas", [])

        if not respuestas or not isinstance(respuestas, list):
            return jsonify({"error": "Se esperaba una lista de respuestas"}), 400

        respuestas_concatenadas = " ".join(respuestas)
        if not respuestas_concatenadas.strip():
            return jsonify({"error": "Las respuestas están vacías"}), 400

        prompt = f"""
Analiza este conjunto de textos escritos por un usuario en contexto emocional.

Extrae exactamente 2 palabras clave **reales y frecuentes** que se repiten o reflejan el estado emocional del usuario. No agregues explicaciones. No inventes palabras. Solo devuelve las dos palabras más repetidas o emocionalmente fuertes, separadas por coma.

Texto del usuario:
{respuestas_concatenadas}

Formato de salida:
palabra1, palabra2
"""

        resultado = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages= [
                 {"role": "user", "content": prompt},
                # {
                #     "role": "user",
                #     "content": user_message
                # }
            ],
            
            temperature=1,
            max_completion_tokens=230,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        contenido = resultado.choices[0].message.content.strip()
        print("Respuesta GPT:", contenido)

        if "," not in contenido:
            return jsonify({"error": "La respuesta del modelo no está en el formato esperado"}), 500

        palabra1, palabra2 = map(str.strip, contenido.split(",", 1))

        return jsonify({
            "palabra1": palabra1,
            "palabra2": palabra2
        })

    except Exception as e:
        print("Error interno en /palabrasClave:", str(e))
        return jsonify({"error": f"No se pudieron extraer palabras clave: {str(e)}"}), 500






@app.route("/prediccionIndividual", methods=["POST"])
def prediccionIndividual():
    try:

        data = request.get_json()
        prompt = """
        
        Eres un psicólogo clínico con experiencia en evaluación de salud mental. 
        Analizas respuestas abiertas sobre el estado emocional de una persona para estimar un nivel aproximado de depresión y ansiedad, 
        expresado en porcentaje de severidad. Basas tu análisis en criterios de los instrumentos DASS-21, GAD-7 y BDI-II. Sé empático y objetivo en tu análisis.
        Las preguntas son estas ¿Cómo describirías tu estado de ánimo en los últimos días?

        * ¿Qué cosas te han hecho sentir preocupado, triste o estresado últimamente?

        * ¿Has tenido dificultades para dormir o para descansar bien? ¿Por qué crees que sucede eso?

        * ¿Hay algo que antes disfrutabas y que ahora ya no te interesa o no te provoca hacerlo?

        * ¿Cómo te sientes contigo mismo/a en este momento?

        * ¿Qué piensas o haces cuando te sientes muy mal emocionalmente?
        de las cuales, puede venir en cualquier orden con una respuesta, y por este caso lo primordial es la respuesta que debes devolver
        Te enviare cada pregunta y respuesta del usuario, una por una, por lo que debes responder con una etiqueta de depresión o ansiedad,
        y un porcentaje de confianza en tu respuesta, sin explicaciones adicionales asi: etiqueta || porcentaje, aqui va:"""

        prompt += f"\nPregunta: {data['pregunta']}\nRespuesta: {data['respuesta']}\n"

        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1000
        )

        resultado = completion.choices[0].message.content.strip()
        #print(f'etiqueta: {ID2LABEL[pred]}, confianza: {round(float(probs[pred]), 4)}')
        if "||" not in resultado:
            raise ValueError("Formato inesperado en la respuesta del modelo: " + resultado)

        etiqueta, porcentaje_str = [x.strip().lower() for x in resultado.split("||")]

        porcentaje_str = porcentaje_str.replace("%", "").strip()
        porcentaje = float(porcentaje_str)
        confianza = round(porcentaje / 100, 4)

        return jsonify({
            "etiqueta": etiqueta,
            "confianza": confianza
        })
        

    except Exception as e:
        print("Error en prediccionIndividual:", e)
        return jsonify({"error": str(e)}), 500
        

@app.route("/frasesParaScrapping", methods=["POST"])
def generar_frases_scraping_desde_json():
    try:
        data = request.get_json()

        respuestas_json = data.get("respuestas", [])

        print("Respuestas recibidas:", respuestas_json)
        # Filtrar respuestas relevantes
        respuestas_filtradas = [
            (item['respuesta'], item['etiqueta'])
            for item in respuestas_json
            if item['etiqueta'] in ['depresión', 'ansiedad']
        ]

        if not respuestas_filtradas:
            return jsonify({"error": "No hay respuestas con etiquetas válidas para generar frases."}), 400

        # Construir el prompt
        prompt = """
A continuación, te doy varias respuestas escritas por un usuario, cada una con su respectiva clasificación emocional obtenida mediante un modelo BERT (las posibles etiquetas son: Depresión, Ansiedad u Otro).

Tu tarea es:

1. Analizar el contenido de todas las respuestas y sus clasificaciones asociadas.
2. Identificar las palabras o frases más frecuentes, significativas o emocionalmente representativas por cada etiqueta (Depresión o Ansiedad).
3. Generar exactamente dos expresiones útiles para búsquedas en redes sociales, con el siguiente formato:
   ➤ [Etiqueta] [palabra_representativa]

Por ejemplo:  
Depresión vacío || Ansiedad miedo

(No uses expresiones con la etiqueta "Otro", ignora esas respuestas)

Devuelve solo una línea con 2 expresiones separadas por el símbolo `||`.
---
"""

        for i, (respuesta, etiqueta) in enumerate(respuestas_filtradas, start=1):
            prompt += f"{i}. Respuesta: {respuesta}\nClasificación: {etiqueta}\n"

        prompt += "\nDevuelve solo esto:\nEtiqueta palabra1 || Etiqueta palabra2"

        # Llamada a OpenAI
        completion = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1000
        )

        resultado = completion.choices[0].message.content.strip()
        return jsonify({"frases": resultado})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



def generar_explicacion_chatgpt(etiqueta, texto):
    prompt = f"""
Eres un experto en salud mental y lenguaje natural. Analiza el siguiente texto y explica por qué un modelo BERT clasificó este texto como "{etiqueta}".
Texto: "{texto}"

Da una explicación breve, clara y profesional para un usuario que podría estar buscando ayuda, no utilices palabras tecnicas complejas, nada de modelo BERT ni nada, algo que sea entendible para el usuario final
"""
    try:
        query = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages= [
                 {"role": "user", "content": prompt},
                # {
                #     "role": "user",
                #     "content": user_message
                # }
            ],
            
            temperature=1,
            max_completion_tokens=230,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        

        response = query.choices[0].message.content
        return response
    except Exception as e:
        return f"No se pudo generar una explicación: {str(e)}"

@app.route("/")
def home():
    return "API de clasificación de textos con BERT + explicación de ChatGPT lista."


@app.route("/frases", methods=["POST"])
def obtener_frases_scraping():
    try:
        data = request.get_json()
        if not data or "texto" not in data:
            return jsonify({"error": "Debes enviar un campo 'texto' en JSON."}), 400

        texto = data["texto"]
        #resultado = "Depresión"
        resultado = predecir(texto)
        frases = generar_frases_scraping_desde_json(texto, resultado["etiqueta"])

        return jsonify({
            "frases_scraping": frases,
            "etiqueta": resultado["etiqueta"],
            "descripcion": LABEL_DESC[resultado["etiqueta"]]
        })
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error: {str(e)}"}), 500






@app.route('/analizar-caso', methods=['POST'])
def analizar_caso_psicologico():
    try:
        data = request.get_json()
        #print("Datos recibidos en /analizar-caso:", data)
        red_social = data.get("redSocial")
        #print("Red social seleccionada:", red_social)
        frases_raw = data.get("palabrasScraping", "")
        print("Frases de scraping recibidas:", frases_raw)
        respuestas_usuario = data.get("respuestasUsuario", [])
        print("Respuestas del usuario recibidas:", respuestas_usuario)
        ciudad = data.get("ciudad", "Cuenca")  # Valor por defecto si no se envía
        print("Ciudad recibida:", ciudad)
        # Guardar ciudad en archivo para el scraper
        with open("ciudad_scraping.txt", "w", encoding="utf-8") as f:
            f.write(ciudad)



        if not red_social or not frases_raw or not respuestas_usuario:
            return jsonify({"error": "Faltan campos obligatorios en la solicitud"}), 400

        frases = [f.strip() for f in frases_raw.split("||")]

        # === Paso 1: Ejecutar el scraping solo si es Facebook ===
                # === Paso 1: Ejecutar el scraping según red social ===
        if red_social['redsocial'].lower() == "facebook":
            print(" Ejecutando scraping de Facebook...")

            # Guardar frases en archivo para el scraper
            with open("frases_scraping.json", "w", encoding="utf-8") as f:
                json.dump(frases, f)

            subprocess.run([sys.executable, "scraper_facebook.py"])

            if not os.path.exists("comentariosFacebookMultiprocesoFinal.json"):
                return jsonify({"error": "No se encontró el archivo de resultados del scraping"}), 500

            with open("comentariosFacebookMultiprocesoFinal.json", "r", encoding="utf-8") as f:
                contenido_scraping = json.load(f)

        elif red_social['redsocial'].lower() == "reddit":
            print(" Ejecutando scraping de Reddit")

            # Guardar frases en archivo para el scraper
            with open("frases_scraping.json", "w", encoding="utf-8") as f:
                json.dump(frases, f)

            subprocess.run([sys.executable, "appReddit.py"])

            if not os.path.exists("comentariosRedditMultiprocesoFinal.json"):
                return jsonify({"error": "No se encontró el archivo de resultados del scraping"}), 500

            with open("comentariosRedditMultiprocesoFinal.json", "r", encoding="utf-8") as f:
                contenido_scraping = json.load(f)

        elif red_social['redsocial'].lower() == "tiktok":
            return jsonify({
                "error": "Scraping para TikTok aún no está implementado. En construcción."
            }), 501

        elif red_social['redsocial'].lower() == "youtube":
            print(" Ejecutando api de Youtube")

            # Guardar frases en archivo para el scraper
            with open("frases_scraping.json", "w", encoding="utf-8") as f:
                json.dump(frases, f)

            subprocess.run([sys.executable, "appYoutube.py"])

            if not os.path.exists("comentariosYoutubeMultiprocesoFinal.json"):
                return jsonify({"error": "No se encontró el archivo de resultados del scraping"}), 500

            with open("comentariosYoutubeMultiprocesoFinal.json", "r", encoding="utf-8") as f:
                contenido_scraping = json.load(f)
        else:
            return jsonify({
                "error": f"Red social '{red_social}' no reconocida. Usa: Facebook, Reddit, TikTok o YouTube."
            }), 400


        # === Paso 2: Construir el prompt ===
        prompt = """
Actúa como un psicólogo profesional. A continuación se presentan las respuestas de un paciente a un formulario de evaluación emocional, donde cada respuesta ya ha sido clasificada por un modelo BERT como Depresión, Ansiedad u Otro.

También se han extraído comentarios reales de redes sociales (Facebook) relacionados con las emociones del paciente.

Tu tarea es:

1. Analizar en conjunto las respuestas del paciente, sus etiquetas y los comentarios de redes sociales.
2. Generar un análisis general del estado emocional del paciente en base a las respuestas del paciente.
3. Proporcionar una recomendación o reflexión positiva que ayude al usuario a comprender mejor su situación y sepa cómo manejarla emocionalmente.

--- RESPUESTAS DEL PACIENTE ---
"""

        for i, item in enumerate(respuestas_usuario, start=1):
            respuesta = item.get("respuesta", "")
            etiqueta = item.get("etiqueta", "")
            prompt += f"{i}. {respuesta}  (Clasificación: {etiqueta})\n"

        prompt += "\n--- COMENTARIOS EXTRAÍDOS DE FACEBOOK ---\n"
        for publicacion in contenido_scraping[:5]:  # solo los 5 primeros para simplificar el análisis
            prompt += f"- {publicacion['titulo']}\n"
            for comentario in publicacion.get("comentarios", [])[:3]:
                prompt += f"  • {comentario}\n"

        prompt += """
--- INSTRUCCIONES DE RESPUESTA ---

Actúa como un psicólogo profesional. Debes analizar las respuestas del usuario a un formulario emocional, cada una etiquetada como Depresión, Ansiedad u Otro. También se han extraído comentarios de redes sociales (Facebook) usando frases relacionadas con lo que siente el usuario. Estas frases no provienen del usuario, sino del análisis de patrones emocionales que otras personas expresan en línea.

Tu tarea es generar una respuesta estructurada con los siguientes tres elementos:

1. Un análisis emocional detallado basado en las respuestas del usuario.
2. Un análisis general de los comentarios obtenidos desde redes sociales, explicando qué emociones y temas son comunes en personas que usan frases similares.
3. Una recomendación o mensaje positivo redactado de forma profesional y empática, como lo haría un psicólogo.

🔴 Devuelve exclusivamente un objeto JSON válido, sin formato Markdown, sin etiquetas como ```json, sin comas sobrantes, sin explicaciones antes o después.

🔴 La estructura debe ser exactamente así (sin ningún carácter adicional):

{
  "analisis_emocional": "Texto extenso (mínimo 5 a 6 líneas).",
  "comentarios_sociales": "Texto extenso (mínimo 5 a 6 líneas).",
  "recomendacion_positiva": "Texto extenso (mínimo 5 a 6 líneas)."
}

Repite: devuelve **solo** el JSON, sin comillas externas ni código formateado.
"""

        # === Paso 3: Enviar a ChatGPT ===
        respuesta = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )

        ##analisis = respuesta.choices[0].message.content.strip()
        ##datos = json.loads(analisis)

        import re

        analisis = respuesta.choices[0].message.content.strip()

        # Intentar extraer el bloque JSON usando expresión regular
        match = re.search(r'\{.*\}', analisis, re.DOTALL)

        if match:
            try:
                datos = json.loads(match.group())

                ciudad_archivo = None
                if ciudad.lower() == 'cuenca':
                    ciudad_archivo = f"personasFacebookCiudadCuenca.json"
                elif ciudad.lower() == 'guayaquil':
                    ciudad_archivo = f"personasFacebookCiudadGuayaquil.json"
                elif ciudad.lower() == 'quito':
                    ciudad_archivo = f"personasFacebookCiudadQuito.json"

                psicologos = []

                if os.path.exists(ciudad_archivo):
                    with open(ciudad_archivo, "r", encoding="utf-8") as f:
                        psicologos = json.load(f)

                return jsonify({**datos, "psicologos": psicologos})
            except json.JSONDecodeError as e:
                return jsonify({"error": f"JSON inválido: {str(e)}", "respuesta_bruta": analisis}), 500
        else:
            return jsonify({"error": "No se pudo encontrar un bloque JSON en la respuesta", "respuesta_bruta": analisis}), 500


    except Exception as e:
        print("Error en /analizar-caso:", str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=True)
