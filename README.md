# 🔢 Handwritten Digit Recognition — MNIST Neural Network

A feedforward neural network built with TensorFlow/Keras to classify handwritten digits (0–9) from the MNIST dataset. Trained and evaluated entirely in a Jupyter Notebook.

---

## 📌 Project Overview

This project builds and trains a simple Artificial Neural Network (ANN) on the classic MNIST dataset — 70,000 grayscale images of handwritten digits. The model learns to identify digits from pixel values alone.

| Property | Details |
|---|---|
| Dataset | MNIST (via `keras.datasets`) |
| Training samples | 60,000 images |
| Test samples | 10,000 images |
| Image size | 28 × 28 pixels |
| Model type | Feedforward ANN (Dense layer) |
| Test accuracy | ~80% |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

---

## 🧠 Model Architecture

```
Input Layer   →  784 nodes  (28×28 pixels flattened)
Dense Layer   →  10 nodes   (sigmoid activation)
Output        →  Digit class (0–9)
```

- **Loss function:** Sparse Categorical Crossentropy  
- **Optimizer:** Adam  
- **Epochs:** 5  

---

## 📈 Training Results

| Epoch | Accuracy | Loss |
|---|---|---|
| 1 | 55.3% | 2.1879 |
| 2 | 71.2% | 1.9769 |
| 3 | 73.9% | 1.7901 |
| 4 | 76.1% | 1.6263 |
| 5 | 77.6% | 1.4835 |

**Final test accuracy: 80.22%**

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/niswanth2007-debug/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Install dependencies
```bash
pip install tensorflow numpy matplotlib scikit-learn jupyter
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook
```

Open `digit_recognition.ipynb` and run all cells.

---

## 📂 Project Structure

```
handwritten-digit-recognition/
│
├── digit_recognition.ipynb   # Main notebook — data loading, training, evaluation
└── README.md
```

---

## 🔍 How It Works

1. **Load data** — MNIST dataset is loaded directly via Keras
2. **Explore data** — Shape inspection and digit visualisation using Matplotlib
3. **Preprocess** — Images flattened from 28×28 to 784-dimensional vectors; pixel values normalised to [0, 1]
4. **Train** — Single Dense layer trained for 5 epochs
5. **Evaluate** — Model tested on 10,000 unseen images
6. **Predict** — `np.argmax()` used to extract predicted digit class

---

## 📌 What I Learned

- Loading and preprocessing image datasets with Keras
- Building and training a basic ANN with TensorFlow
- Normalising pixel values for better model convergence
- Evaluating model accuracy on unseen test data
- Visualising MNIST digit images with Matplotlib

---

## 🔮 Future Improvements

- [ ] Add hidden layers to improve accuracy
- [ ] Switch to a CNN (Convolutional Neural Network) for better performance
- [ ] Add a confusion matrix to analyse misclassifications
- [ ] Build a simple UI to draw and predict digits in real time

---

## 👤 Author

**Niswanth B**  
ECE Undergrad @ SASTRA Deemed University  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-niswanth--b-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://linkedin.com/in/niswanth-b-2304063a0)
[![GitHub](https://img.shields.io/badge/GitHub-niswanth2007--debug-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/niswanth2007-debug)
