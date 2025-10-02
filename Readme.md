# Iris Image Disorder Detection
## Objective

This project develops a simple AI prototype to detect eye disorders from iris images, classifying them as Healthy or Diseased (Unhealthy). The goal is to demonstrate understanding of image preprocessing, transfer learning, model training, evaluation, and explainability (Grad-CAM visualizations).

## Dataset

Source: Kaggle Iris Disease Dataset or similar public iris datasets.

Classes: Healthy and Unhealthy.

## Structure:

Augmented Dataset/
├── Healthy/
│   ├── Healthy1.jpg
│   ├── Healthy2.jpg
│   └── ...
└── Unhealthy/
    ├── Unhealthy1.jpg
    ├── Unhealthy2.jpg
    └── ...


## Splits: 60% Train, 20% Validation, 20% Test.

## Steps
### 1. Data Loading & Visualization

Loaded sample iris images using matplotlib.

Displayed random images from each class to verify data.

### 2. Preprocessing

Resized images to 224×224 pixels.

Normalized pixel values using preprocess_input (VGG16 preprocessing).

Used ImageDataGenerator for augmentation on training images:

Rotation, width/height shift, zoom, shear, horizontal flip.

Split data into train, validation, and test sets.

### 3. Model Building

Used VGG16 pre-trained on ImageNet (without top layer).

Frozen all convolutional layers for transfer learning.

Added custom layers:

Flatten

Dense(128, relu)

Dropout(0.5)

Dense(1, sigmoid) for binary classification.

Compiled with:

Optimizer: Adam (learning_rate=1e-4)

Loss: Binary Crossentropy

Metrics: Accuracy

### 4. Training

Trained for 20 epochs with EarlyStopping on validation accuracy.

Training used binary class mode for Healthy vs Unhealthy.

### 5. Evaluation

Evaluated on the test set:

Classification Report:

              precision    recall  f1-score   support

     Healthy       0.97      0.95      0.96        39
   Unhealthy       0.95      0.98      0.96        42

    accuracy                           0.96        81
   macro avg       0.96      0.96      0.96        81
weighted avg       0.96      0.96      0.96        81


Confusion Matrix:

[[37  2]
 [ 1 41]]


Achieved 96% test accuracy.

Plotted normalized confusion matrix and training vs validation loss/accuracy curves.

### 6. Explainability

Generated Grad-CAM heatmaps for test images.

Visualized areas of the iris that the model focused on for predictions.

Example:

Healthy and Unhealthy images overlaid with Grad-CAM heatmaps.

Usage
from tensorflow.keras.models import load_model
from gradcam import display_gradcam  # Custom Grad-CAM function

# Load trained model
vgg16 = load_model('vgg_model.keras')

# Display Grad-CAM for a sample image
display_gradcam('/path/to/Healthy117.jpg', vgg16)
display_gradcam('/path/to/Unhealthy1.jpg', vgg16)

Future Improvements

Expand dataset to improve model robustness.

Fine-tune last convolutional layers for better feature extraction.

Test other architectures like ResNet50 or EfficientNet.

Implement real-time iris disorder detection using webcam or mobile device.

References

VGG16 Paper

Kaggle Iris Disease Dataset

Grad-CAM implementation: tf.keras & OpenCV  

## Usage

### Generating Model Weights

The trained model weights (`model.weights.h5`) are not included due to size constraints (>25 MB).  
To generate them, follow these steps:

1. Open the notebook `iris_disorder_classification.ipynb` in Google Colab or Jupyter.
2. Ensure the dataset folder is present and correctly referenced in the notebook.
3. Run all cells sequentially. The model will be trained and weights saved automatically.

```python
# Example code snippet inside the notebook to save weights
from tensorflow.keras.models import load_model

# After defining and training your model (vgg16 or chosen CNN)
model.save_weights("model.weights.h5")
print("✅ Model weights saved as model.weights.h5")

