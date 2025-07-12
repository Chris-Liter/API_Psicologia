from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

import openai
from openai import OpenAI
import os
from flask_cors import CORS


client = OpenAI( api_key='')


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

    return {
        "etiqueta": ID2LABEL[pred],
        "confianza": round(float(probs[pred]), 4)
    }

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

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"Datos recibidos: {data}")
        if not data or "texto" not in data:
            return jsonify({"error": "Debes enviar un campo 'texto' en JSON."}), 400

        texto = data["texto"]
        print(f"Texto recibido: {texto}")
        resultado = predecir(texto)
        explicacion = generar_explicacion_chatgpt(resultado["etiqueta"], texto)
        print(f"Resultado de la predicción: {resultado}")
        return jsonify({
            "resultado": resultado,
            "descripcion": LABEL_DESC[resultado["etiqueta"]],
            "explicacion_chatgpt": explicacion
        })
    except Exception as e:
        print(f"Error en la predicción: {str(e)}")
        return jsonify({"error": "Ocurrió un error al procesar la solicitud."}), 500

if __name__ == "__main__":
    app.run(debug=True)
