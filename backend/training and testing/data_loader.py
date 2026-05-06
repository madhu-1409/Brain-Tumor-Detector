import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir, batch_size=32, num_workers=0, pin_memory=True):
    """
    Complete Data Pipeline for Greyscale MRI Brain Tumor Scans.
    """
    
    # 1. Processing Recipe
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Folder Mapping
    data_transforms = {
        'train': train_transforms,
        'validation':   val_test_transforms,
        'test':  val_test_transforms
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'validation', 'test']
    }

    # 3. Optimized Loaders
    loaders = {
        'train': DataLoader(
            image_datasets['train'], 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, # Background helpers
            pin_memory=pin_memory    # Fast-track transfer
        ),
        'validation': DataLoader(
            image_datasets['validation'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        ),
        'test': DataLoader(
            image_datasets['test'], 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    }
    
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}

    return loaders, class_names, dataset_sizes

print("Data Loader finalized with Performance Tuning.")
