# Brain Tumor Detection Under Adversarial Threats
This project implements a deep learning-based system to detect brain tumors from MRI scans while evaluating and improving the model's robustness against FGSM (Fast Gradient Sign Method) adversarial attacks.

🚀 Project Overview
Medical AI systems are often vulnerable to "digital tricks" called adversarial attacks—subtle, invisible changes to an image that force an AI to give a wrong diagnosis. This project compares ResNet-50 and DenseNet-121 architectures to see which handles these threats better.

🧠 Core Features
    Dual Architecture Comparison: Evaluates ResNet-50 and DenseNet-121.
    Adversarial Simulation: Real-time generation of FGSM noise to test model limits.
    Robustness Training: Includes models trained specifically on adversarial examples to improve diagnostic reliability.
    Web Interface: A React-based dashboard for side-by-side comparison of "Clean" vs "Attacked" predictions.

📊 Results & Performance Analysis
•The core of this project involved a rigorous comparison between two popular architectures—DenseNet-121 and ResNet-50—across three different scenarios: Clean Data, Adversarial Attack (FGSM), and Adversarial Training.
•Baseline Superiority: In standard conditions using clean MRI images, the ResNet-50 model initially outperformed DenseNet-121 with a classification accuracy of 94%.
•Architecture Vulnerability: Upon applying the Fast Gradient Sign Method (FGSM) attack, a significant disparity in architectural robustness was observed. ResNet-50 suffered a catastrophic failure, with accuracy plummeting by 24% down to a 70% success rate.
•Inherent Robustness: In contrast, DenseNet-121 demonstrated superior stability. Its accuracy only decreased from 92% to 88% (a minor 4% drop), indicating that its dense connectivity pattern may provide better resistance to the perturbations introduced by FGSM noise.
•Effectiveness of Adversarial Training: The research successfully "hardened" the vulnerable ResNet-50 model. By implementing adversarial training, the model's performance under attack was restored from 70% to 85%, marking a substantial 15% gain in reliability.


# 🔬 Supplementary Study: Custom CNN vs. EfficientNetB0
In addition to the main system, an exploratory study was conducted to compare a Custom CNN against EfficientNetB0 to evaluate how model complexity affects performance on imbalanced, small-scale datasets.
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
