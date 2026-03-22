# Brain-Tumor-Detector

• Brain Tumor Detection using MRI images (binary classification: tumor vs non-tumor)

• Dataset with two classes: yes and no (imbalanced data)

• Image preprocessing: resizing (224×224), normalization

• Data augmentation: rotation, zoom, shift, horizontal flip

• Validation method: 80–20 train–validation split using ImageDataGenerator

• No separate test set used

• Built Custom CNN model (Conv2D, MaxPooling, Dropout, Dense)

• Applied Transfer Learning (EfficientNetB0) with fine-tuning

• CNN achieved ~74–78% validation accuracy

• EfficientNetB0 achieved ~62% validation accuracy

• CNN outperformed transfer learning on this dataset

• Evaluation using accuracy, loss, confusion matrix, classification report

• Issues faced: small dataset, class imbalance, overfitting

• Key insight: simple CNN performs better on small, domain-specific data

• Future improvements: class weighting, better preprocessing (CLAHE), more data, advanced models (ResNet/VGG)
