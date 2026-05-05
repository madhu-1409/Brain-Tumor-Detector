import torch
import torch.nn as nn
from torchvision import models

def loadModel(path, device):
    ckpt = torch.load(path, map_location=device)
    
    # We know there are exactly 2 classes: Normal and Tumor
    num_classes = 2 
    
    if "resnet50" in path.lower():
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # For DenseNet-121
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    # Check if the state_dict is nested or direct
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model