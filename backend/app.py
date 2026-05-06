from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import logging
from utils.model_loader import loadModel
from utils.predictor import getTransform, predict

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

modelsDict = {
    "DenseNet-121 Baseline": "models/tumor_densenet121.pth",
    "ResNet-50 Baseline": "models/tumor_resnet50.pth",
    "DenseNet-121 Adv": "models/tumor_densenet121_adv.pth",
    "ResNet-50 Adv": "models/tumor_resnet50_adv.pth"
}

loaded_models = {}
for name, path in modelsDict.items():
    loaded_models[name] = loadModel(path, device)

@app.route("/api/compare", methods=["POST"])
def compare_models():
    try:
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        results = {}
        idxToClass = {0: "Normal", 1: "Tumor"}

        for modelName, model in loaded_models.items():
            transform = getTransform()
            tensor = transform(img).unsqueeze(0).to(device)
            pred, conf = predict(model, tensor)

            results[modelName] = {
                "class": idxToClass[pred],
                "confidence": float(conf)
            }

        bestModelName = max(results, key=lambda x: results[x]["confidence"])

        return jsonify({
            "results": results,
            "best_model": bestModelName
        })

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)