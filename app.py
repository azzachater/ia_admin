import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify
from FishCNN import FishCNN
import io

# Initialisation de Flask
app = Flask(__name__)

# Chargement du modèle
checkpoint = torch.load("fish_model.pth", map_location=torch.device('cpu'))
class_names = checkpoint['class_names']
num_classes = len(class_names)

model = FishCNN(num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Transformation pour l'image entrante
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Pas d'image envoyée"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    try:
        image = Image.open(file).convert('RGB')
        image = transform(image).unsqueeze(0)  # Shape : (1, 3, 128, 128)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_name = class_names[predicted.item()]

        return jsonify({
            "prediction": class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return "API pour classification des poissons 🐟. POST /predict avec une image .png"

if __name__ == '__main__':
    app.run(debug=True)