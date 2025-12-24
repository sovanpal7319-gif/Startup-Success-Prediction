# 🚀 Startup Success Prediction using Machine Learning

A machine learning project to predict whether a startup will succeed or fail using real-world heterogeneous data.

---

## 📌 Project Overview

Startup success is influenced by multiple factors such as funding, market category, team size, and operational metrics.  
This project applies several machine learning algorithms and compares their performance to identify the **most reliable model for startup success prediction**.

All data preprocessing, training, evaluation, and visualization are implemented in a **single Jupyter Notebook** for clarity and reproducibility.

---

## 🎯 Objective

- Predict **Startup Success vs Failure**
- Compare classical ML models with ensemble and boosting methods
- Handle **categorical features** and **class imbalance**
- Evaluate models using **F1-score** instead of accuracy

---

## 📂 Project Structure
```
Startup-Success-Prediction/
│
├── Startup_Success_Prediction.ipynb # Complete project (EDA + ML + Evaluation)
├── data/
│ └── startup_data.csv # Dataset
├── results/
│ └── confusion_matrix.png # Model evaluation visuals 
└── README.md

```
---

## 🧠 Models Implemented

The following models were trained and evaluated inside the notebook:

- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest
- logistic Regression
- xgboost
- **CatBoost (Best Performing Model)**

---

## 🏆 Best Model: CatBoost

- **Evaluation Metric:** F1-score  
- **Best F1-score:** `0.861`

### Why CatBoost?
- Native handling of **categorical features**
- No need for explicit feature scaling
- Ordered boosting reduces overfitting
- Performs well on heterogeneous and imbalanced data

---

## 📊 Evaluation Strategy

- 5-Fold Cross Validation
- Confusion Matrix Analysis
- Precision, Recall, and F1-score comparison
- Focus on **minority class performance**

📌 Accuracy was not used as the primary metric due to class imbalance.

---

## ⚙️ Technologies Used

- **Language:** Python  
- **Environment:** Jupyter Notebook  
- **Libraries:**  
  - NumPy  
  - Pandas  
  - Scikit-learn  
  - CatBoost  
  - Matplotlib  
  - Seaborn  
> Libraries are imported directly inside the notebook.
---

## ▶️ How to Run the Notebook

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/Startup-Success-Prediction.git
cd Startup-Success-Prediction
```
2️⃣  Open the Notebook
jupyter notebook Startup_Success_Prediction.ipynb

3️⃣ Run all cells sequentially to reproduce the results.


📈 Key Results

Baseline models showed limited performance due to categorical dominance

Random Forest improved results but showed mild overfitting

CatBoost achieved the highest and most stable F1-score

Confusion matrix confirmed better minority-class prediction

🔮 Future Work

Model explainability using SHAP

Hyperparameter tuning using Optuna

Model deployment using FastAPI

Real-time startup success prediction system
