# Image_Classification

# 🧠 Image Classification using Convolutional Neural Networks (CNN)

This project implements an **image classification model** using a **Convolutional Neural Network (CNN)** built with **TensorFlow and Keras**, trained on the **CIFAR-10 dataset**.  
The model classifies images into one of **10 classes**, such as airplanes, cars, birds, cats, and more.

---

## 📸 Dataset
The project uses the **CIFAR-10 dataset**, which contains:
- **60,000 images** (32x32 pixels, RGB)
- **10 classes**:  
  `Plane`, `Car`, `Bird`, `Cat`, `Deer`, `Dog`, `Frog`, `Horse`, `Ship`, `Truck`

The dataset is automatically downloaded from `tensorflow.keras.datasets`.

---

## 🧩 Model Architecture

| Layer Type         | Parameters/Notes              |
|--------------------|------------------------------|
| Conv2D             | 32 filters, (3x3), ReLU       |
| MaxPooling2D       | (2x2)                        |
| Conv2D             | 64 filters, (3x3), ReLU       |
| MaxPooling2D       | (2x2)                        |
| Conv2D             | 64 filters, (3x3), ReLU       |
| Flatten            | —                            |
| Dense              | 64 neurons, ReLU activation   |
| Dense (Output)     | 10 neurons, Softmax activation|

**Optimizer:** Adam  
**Loss Function:** Sparse Categorical Crossentropy  
**Metric:** Accuracy  
**Epochs:** 10

---

## 🚀 How It Works

1. **Load and preprocess data**
   - Normalize pixel values (0–255 → 0–1)
   - Visualize sample images with their labels

2. **Build and train the CNN**
   - The model learns image features through convolution and pooling layers
   - Dense layers perform final classification

3. **Evaluate model performance**
   - Accuracy and loss are calculated on test data

4. **Save the trained model**
   - The trained model is saved as `image_classifier.h5`

5. **Test with custom images**
   - You can load your own image (e.g., `dog.jpg`) to test predictions.

---

## 🧠 Example Output

After training, you’ll get an output similar to:
