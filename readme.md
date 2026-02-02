# AutoML Insurance Cost Prediction 🚑💰

An end-to-end Machine Learning project that predicts medical insurance charges using AutoML (PyCaret), evaluates model performance, deploys predictions via a Streamlit web application, and visualizes business insights using Power BI.

---

## 📌 Project Overview

Medical insurance cost prediction helps insurance companies with:
- Risk assessment
- Premium optimization
- Customer segmentation

This project demonstrates a complete ML lifecycle:
- Data preparation
- Automated model selection and tuning
- Model evaluation and interpretation
- Deployment and business analytics

---

## 🧠 Tech Stack

- Python
- PyCaret (AutoML)
- Pandas, NumPy
- Scikit-learn
- Streamlit (Deployment)
- Power BI (Visualization)
- GitHub

---

## 📊 Dataset

The dataset contains demographic and health-related attributes:

- Age  
- Sex  
- BMI  
- Number of children  
- Smoking status  
- Region  
- Insurance charges (target)

**File used:**
insurance.csv
---

## ⚙️ AutoML Pipeline

The AutoML workflow includes:
- Data preprocessing
- Categorical encoding
- Model comparison and tuning
- Final model selection (Gradient Boosting Regressor)

![AutoML Pipeline](screenshot/pipeline.png)

---

## 📈 Model Evaluation

Model performance was evaluated using:
- MAE
- RMSE
- R² Score
- Residual analysis

![Model Evaluation](screenshot/automl_result.png)  
![Residual Plot](screenshot/residuals.png)

---

## 🔍 Feature Importance

Feature importance analysis shows which factors influence insurance cost the most.

![Feature Importance](screenshot/feature_importance.png)

**Key insights:**
- Smoking status has the highest impact
- BMI and age are strong contributing factors
- Region and number of children have moderate influence

---

## 🔮 Prediction Results

Predictions were generated and saved for further analysis.


**Output file:**
insutrance_model_prediction.csv
---

## 📊 Power BI Business Dashboard

Predicted values were integrated into Power BI to generate business insights such as:
- Actual vs Predicted Charges
- Average Charges by Smoking Status
- BMI vs Predicted Charges
- Age vs Predicted Charges

![Power BI Dashboard](screenshot/evaluation.png)

---

## 🚀 Deployment (Streamlit App)

A Streamlit web application allows users to input customer details and instantly receive insurance cost predictions.

Run locally:
```bash
python -m streamlit run app.py
