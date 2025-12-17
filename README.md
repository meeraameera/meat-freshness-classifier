# Meat Freshness Classifier

A Deep Learning solution for real-time classification of Fresh vs. Spoiled meat, deployed on Android.

---

## Problem Statement

Meat spoilage is a significant concern within food safety and supply chain management. The inability to accurately distinguish between fresh and spoiled meat poses potential health risks to consumers and contributes substantially to food waste. Fresh meat is typically characterized by a bright red color and minimal discolorations or dark spots, whereas spoiled meat may appear drier, slimy, and exhibit darker patches or a greenish/brownish tint.

## Project Goal

The primary goal was to build a robust Deep Learning model to accurately classify meat quality, thereby reducing health risks and food waste in the supply chain. This involved data pre-processing, training a custom Baseline CNN, and evaluating multiple pre-trained models before selecting and fine-tuning VGG16 for optimal performance.

## Dataset
Image dataset containing labeled Fresh and Spoiled meat samples, sourced from public dataset (https://www.kaggle.com/datasets/crowww/meat-quality-assessment-based-on-deep-learning/data).

## Technical Approach & Workflow

### 1. Baseline & Transfer Learning Exploration (`Solution Part 1.ipynb`)

* **Data Preparation:** The dataset was split into Training (70%), Validation (20%), and Testing (10%) sets. Images were normalized (`rescale=1./255`) and resized to $224 \times 224$ pixels.
* **Data Augmentation:** An `ImageDataGenerator` was applied to the training set, including rotation, shifting, zooming, and horizontal flipping, to improve model generalization.
* **Baseline Model Training:** A custom CNN was trained for 30 epochs but exhibited severe overfitting and poor generalization (low precision/recall $\approx 0.51$).
* **Transfer Learning Evaluation:** Multiple pre-trained models (InceptionV3, ResNet50, MobileNetV2, EfficientNetB0, VGG16) were loaded with frozen base layers and evaluated to identify the most promising architectures.

### 2. Fine-Tuning and Final Model Selection (`Solution Part 2.ipynb`)

* **Model Selection:** Based on initial performance, **InceptionV3, ResNet50, and VGG16** were selected for in-depth fine-tuning.
* **Fine-Tuning Implementation:** The base model's last **20 layers** were unfrozen to allow for task-specific feature learning. 
* **Training Parameters:** Training was conducted for 15 epochs using the Adam optimizer (with a reduced learning rate) and **class weights** to explicitly address class imbalance.
* **Final Result:** The **Fine-Tuned VGG16** model achieved the highest performance and demonstrated excellent generalization, indicated by a minimal gap between training and validation accuracy.

### 3. Android Studio Deployment

* The final optimized **Fine-Tuned VGG16 model** was prepared for mobile deployment.
* **Model Conversion:** The Keras model was converted into a **TensorFlow Lite (TFLite)** format for efficient, low-latency, on-device inference.
* **Platform:** Deployed using **Android Studio** with Gradle configuration to manage dependencies.
* **Goal:** Enable **real-time image classification** of meat (Fresh/Spoiled) using the device's camera. 

## Key Challenges Faced

* **Overfitting in Baseline Model:**
    * *Challenge:* The custom CNN showed severe overfitting and poor generalization, resulting in a high misclassification rate ($\approx 50\%$ precision/recall).
    * *Solution:* Transitioned to pre-trained **Transfer Learning** architectures and applied stronger regularization techniques (L2 Regularization and Dropout).
* **Model Bias/Poor Generalization in Transfer Learning:**
    * *Challenge:* Several lighter models exhibited strong bias towards the Fresh class, failing to identify most Spoiled instances.
    * *Solution:* Focused on higher-performing models (VGG16, ResNet50, InceptionV3) and utilized **class weights** during fine-tuning to enforce fair learning across both categories.

## Installation & Execution

### Model Training

- Clone the repository.
- Create and activate a virtual environment
- Install required dependencies
- Execute the Jupyter Notebooks (`Solution Part 1.ipynb` and `Solution Part 2.ipynb`) sequentially to train and fine-tune the final VGG16 model.

### Mobile Application

- Load the entire project structure into **Android Studio**.
- The mobile application can then be built and deployed onto a physical Android device or emulator for real-time testing of the classifier.
