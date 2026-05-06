import torch 
import torch.nn as nn 

def getClippingBounds(mean, std, device): 
    meanT = torch.tensor(mean).view(1,3,1,1).to(device) 
    stdT = torch.tensor(std).view(1,3,1,1).to(device) 

    min_clip = (0 - meanT) / stdT 
    max_clip = (1 - meanT) / stdT 
    return min_clip, max_clip 

def fgsm_attack(model, image, label, epsilon, minClip, maxClip): 
    imageAdv = image.clone().detach().requires_grad_(True) 

    model.zero_grad() 
    output = model(imageAdv) 
    loss = nn.CrossEntropyLoss()(output, label) 
    loss.backward() 

    grad = imageAdv.grad.sign() 
    attacked = imageAdv + epsilon * grad 

    attacked = torch.max(torch.min(attacked, maxClip), minClip) 
    return attacked.detach()