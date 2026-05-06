import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import copy
from data_loader import get_data_loaders
from model_dev import get_model

# --- 1. Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "../data" 
MODELS_TO_TRAIN = ["densenet121", "resnet50"] 
BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.001

def plot_training_history(history, model_name):
    """Generates verification graphs for Loss and Accuracy"""
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.title(f'{model_name} - Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Save verification graph
    graph_path = f"../models/{model_name}_metrics.png"
    plt.savefig(graph_path)
    print(f"Verification graph saved to: {graph_path}")
    plt.show()

def run_training():
    # Load data once for all models
    loaders, class_names, dataset_sizes = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    if not os.path.exists("../models"):
        os.makedirs("../models")

    for m_type in MODELS_TO_TRAIN:
        print(f"\n" + "="*30)
        print(f"STARTING TRAINING: {m_type}")
        print("="*30)
        
        # Initialize Architect
        model = get_model(model_name=m_type, num_classes=len(class_names), device=DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(EPOCHS):
            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in loaders[phase]:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]
                
                # Store history for graphing
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                    
                    # Track best weights
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                print(f'Epoch {epoch+1}/{EPOCHS} | {phase:10} | Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 1. Show Verification Graphs
        plot_training_history(history, m_type)

        # 2. Save the .pth file
        model.load_state_dict(best_model_wts)
        save_path = f"../models/tumor_{m_type}.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': {'normal': 0, 'tumor': 1},
            'model_name': m_type
        }, save_path)
        print(f"SUCCESS: Saved best {m_type} weights to {save_path}")

if __name__ == '__main__':
    run_training()