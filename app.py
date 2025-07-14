from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

import openai
from openai import OpenAI
import os
from flask_cors import CORS


client = OpenAI( api_key='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')


# === Configuración ===
ID2LABEL = {0: "Otro", 1: "Depresión", 2: "Ansiedad"}
LABEL_DESC = {
    "Otro": "Texto sin señales directas de depresión o ansiedad.",
    "Depresión": "El texto presenta señales relacionadas con síntomas depresivos.",
    "Ansiedad": "El texto presenta señales asociadas a crisis o síntomas de ansiedad."
}

# Cargar modelo BERT
model_path = "modelo_bert_depresion"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# === Flask App ===
app = Flask(__name__)
CORS(app)  


def predecir(texto):
    inputs = tokenizer(texto, truncation=True, padding=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)

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

        if not data or "texto" not in data:
            return jsonify({"error": "No se proporcionó texto válido"}), 400

        texto = data["texto"]
        print(texto)

        if not texto.strip():
            return jsonify({"error": "Texto vacío"}), 400

        inputs = tokenizer(text=texto, truncation=True, padding=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            pred = np.argmax(probs)

        print(f'etiqueta: {ID2LABEL[pred]}, confianza: {round(float(probs[pred]), 4)}')
        return jsonify({
            "etiqueta": ID2LABEL[pred],
            "confianza": round(float(probs[pred]), 4)
        })
        

    except Exception as e:
        print("Error en prediccionIndividual:", e)
        return jsonify({"error": str(e)}), 500


@app.route("/frasesParaScrapping", methods=["POST"])
def generar_frases_scraping_desde_json():
    try:
        data = request.get_json()

        respuestas_json = data.get("respuestas", [])

        # Filtrar respuestas relevantes
        respuestas_filtradas = [
            (item['respuesta'], item['etiqueta'])
            for item in respuestas_json
            if item['etiqueta'] in ['Depresión', 'Ansiedad']
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
            max_tokens=100
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





if __name__ == "__main__":
    app.run(debug=True)
