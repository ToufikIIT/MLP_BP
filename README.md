# XOR Neural Network (MLP + Backpropagation)

A simple implementation of a **Multi-Layer Perceptron (MLP)** using **Python and NumPy** to learn the XOR function.  
This project demonstrates how neural networks learn nonlinear relationships using **forward propagation** and **backpropagation**.

---

## 🧠 Problem

| A | B | Y = A ⊕ B |
|---|---|------------|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

---

## ⚙️ Network Structure

- Input layer: 2 neurons (A, B)  
- Hidden layer: 2 neurons (sigmoid activation)  
- Output layer: 1 neuron (sigmoid activation)  
- Training: Stochastic Gradient Descent (SGD)

---

## 📄 File

- **`XOR_MLP_BP.py`** → Main script containing the implementation.

---

## ▶️ How to Run

```bash
pip install numpy
python XOR_MLP_BP.py
