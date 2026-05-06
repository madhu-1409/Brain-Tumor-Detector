import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name="densenet121", num_classes=2, device="cpu"):
    
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        
    else:
        raise ValueError("Unsupported model! Choose 'resnet50' or 'densenet121'.")

    return model.to(device)

