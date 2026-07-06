from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import io
import logging
import gc
from utils.model_loader import loadModel
from utils.predictor import getTransform, predict

app = Flask(__name__)
CORS(app)

device = torch.device("cpu")  # force CPU on Render free tier — no GPU available anyway

modelsDict = {
    "DenseNet-121 Baseline": "models/tumor_densenet121.pth",
    "ResNet-50 Baseline": "models/tumor_resnet50.pth",
    "DenseNet-121 Adv": "models/tumor_densenet121_adv.pth",
    "ResNet-50 Adv": "models/tumor_resnet50_adv.pth"
}

# NOTE: models are NOT loaded here anymore — loaded on-demand per request instead,
# one at a time, to avoid holding all 4 in memory simultaneously (which exceeded 512MB).

@app.route("/api/compare", methods=["POST"])
def compare_models():
    try:
        file = request.files["image"]
        img = Image.open(io.BytesIO(file.read())).convert("RGB")

        results = {}
        idxToClass = {0: "Normal", 1: "Tumor"}
        transform = getTransform()
        tensor = transform(img).unsqueeze(0).to(device)

        for modelName, path in modelsDict.items():
            model = loadModel(path, device)  # load just this one model
            pred, conf = predict(model, tensor)

            results[modelName] = {
                "class": idxToClass[pred],
                "confidence": float(conf)
            }

            # free this model from memory before loading the next one
            del model
            gc.collect()

        bestModelName = max(results, key=lambda x: results[x]["confidence"])

        return jsonify({
            "results": results,
            "best_model": bestModelName
        })

    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
