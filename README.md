# 🧠 AML - Gender Classification with Neural Networks

This is the **Advanced Machine Learning project** from 3rd year at **ENSAE Paris**.

> Project by Antoine Rougier, Grégoire Brugère, and Marin Petibon.

---

## 📚 Project Summary

We aim to **predict a person's gender based on voice data** using neural networks.

The key theoretical concept explored is **Batch Normalization**. We compare:

- A simple feedforward neural network
- A neural network with BatchNorm layers

To observe the effects of BatchNorm, we also experiment with adding **Gaussian noise** and examine its impact on **Internal Covariate Shift**.

---

## 📁 Dataset

We use the public dataset from Kaggle:  
https://www.kaggle.com/datasets/primaryobjects/voicegender

The data is automatically downloaded and extracted when the project is run.

---

## 🧪 Project Structure

AML/
├── data/ # voice.csv will be placed here
├── src/
│ ├── main.py # main script to run full pipeline
│ ├── pre_processing/ # data preprocessing scripts
│ │ └── pre_processing.py
│ ├── models/ # model definitions
│ │ ├── model_simple.py
│ │ └── model_BN.py
│ ├── train/ # training loop
│ │ └── entrainement.py