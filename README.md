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

## ⚙️ Setup & Run

Here’s how to get started in one command-line session:

```bash
# Clone the repository
git clone https://github.com/antoinerougier/AML.git
cd AML

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Access aux données 
python src/pre_processing/data_download.py

# Run the full pipeline
python main.py
