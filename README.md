# 🏦 Banking AI: End-to-End Loan Repayment Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)

An interactive Machine Learning application that predicts the likelihood of loan repayment based on historical borrower data. This project covers the entire pipeline from **SQL Data Extraction** to **Cloud Deployment**.

## 🚀 Live Demo
[🔗 Click here to access the Live Web App]([YOUR_STREAMLIT_LINK_HERE](https://abankloanpredictionapp-7pqhzysbn5zt7pmshqycdv.streamlit.app/))

---

## 📌 Project Overview
The goal of this project is to automate the credit risk assessment process. Using a **Random Forest Classifier**, the system analyzes borrower attributes to determine if a loan should be approved or rejected. 

### Key Features:
* **Real-time Prediction:** Instant risk assessment via a web interface.
* **Safety Rails:** Integrated business logic (e.g., minimum credit score cutoffs).
* **Imbalance Handling:** Model optimized for skewed datasets where defaults are rare but costly.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Data Source:** MySQL (retrieved via SQLAlchemy)
* **Modeling:** Scikit-Learn (Random Forest, StandardScaler)
* **Frontend:** Streamlit
* **Deployment:** GitHub & Streamlit Cloud

---

## 📊 Data & Methodology
### 1. Data Engineering
* **SQL Extraction:** Data was queried from a banking database.
* **Preprocessing:** Handled missing values, performed **One-Hot Encoding** on categorical features, and used **StandardScaler** to normalize numeric data.
* **Feature Alignment:** Ensured the web app inputs match the model's expected 20+ feature dimensions.

### 2. Modeling
We utilized a **Random Forest Ensemble** to capture non-linear relationships.
* **Class Weighting:** Used `class_weight='balanced'` to account for the minority of "Default" cases.
* **Optimization:** Compressed the model using `joblib` to maintain high performance under GitHub's 25MB file limit.



---

## 🖥️ How to Use the App
1.  **Enter Personal Info:** Provide Age and Marital Status.
2.  **Input Financials:** Enter Credit Score and proposed Interest Rate.
3.  **Analyze:** Click "Run Risk Analysis."
4.  **Decision:** The app will return an "Approved" or "Rejected" status along with a **Repayment Confidence Score**.

---

## 📁 Repository Structure
* `app.py`: The Streamlit web application code.
* `loan_model.pkl`: The trained Random Forest model.
* `scaler.pkl`: Pre-trained scaler for data normalization.
* `features.pkl`: List of feature names to ensure data alignment.
* `requirements.txt`: Python dependencies for the cloud environment.

---

## 💡 Key Challenges & Solutions
* **Model Bias:** The model was initially too optimistic due to data imbalance. **Solution:** Introduced a custom 70% probability threshold for approvals.
* **Size Constraints:** The model was 27MB. **Solution:** Applied Zlib compression via `joblib` to bring it under the 25MB GitHub limit.

## ⚖️ License
This project is for educational purposes as part of a Capstone Project.
