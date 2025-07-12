from flask import Flask, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import numpy as np

# Etiquetas
ID2LABEL = {0: "Otro", 1: "Depresión", 2: "Ansiedad"}

# Cargar modelo y tokenizer
model_path = "modelo_bert_depresion"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Preparar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Función de predicción
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

# App Flask
app = Flask(__name__)

@app.route("/")
def home():
    return "Modelo BERT de depresión y ansiedad está corriendo."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "texto" not in data:
        return jsonify({"error": "Se requiere el campo 'texto'."}), 400

    texto = data["texto"]
    resultado = predecir(texto)
    return jsonify(resultado)

if __name__ == "__main__":
    app.run(debug=True)
