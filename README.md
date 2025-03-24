# Healthcare Claims & Revenue Analysis

## 📌 Overview
This project explores **healthcare claims processing** and **hospital revenue forecasting** using **machine learning and time series models**. It provides insights into **claim denials**, **reimbursement patterns**, and **revenue trends** to help hospitals optimize financial outcomes.

## 🚀 Features
- **Claims Denial Prediction**: Models to classify which insurance claims are likely to be denied.
- **Feature Importance Analysis**: Identify key factors that contribute to claim rejections.
- **Revenue Forecasting**: Compare **ARIMA** and **Prophet** models to predict hospital revenue trends.
- **Data Visualization**: Interactive and static plots for claim status, revenue distribution, and forecasts.

## 📊 Key Insights
- **Claim Denials:** The best predictor for claim denial is **AR.Status** (Accounts Receivable Status), followed by **Billed.Amount** and **Insurance Type**.
- **Model Performance:** XGBoost showed **better balance** in predicting denied claims compared to Logistic Regression and Decision Trees.
- **Revenue Forecasting:** Prophet captured seasonal trends better than ARIMA, but further tuning is required for accuracy improvements.

## 🛠️ Tech Stack
- **R**: Data processing, modeling, and visualization.
- **tidyverse**: Data wrangling and visualization.
- **forecast & prophet**: Time series forecasting.
- **caret & xgboost**: Machine learning models.

## 📂 Project Structure
```
├── data/                   # Raw and cleaned datasets
├── scripts/                # R scripts for analysis
├── results/                # Model outputs and visualizations
├── hsp_claims_and_revenue.R # Main analysis script
├── README.md               # Project documentation
```

## 🔥 Next Steps
This project provides a solid foundation, but there’s room for deeper analysis:
- **Hyperparameter tuning** to improve prediction accuracy.
- **Additional features** (e.g., patient demographics, claim processing time).
- **Integration with real-time claim processing systems**.

## 🤝 Contributing
Pull requests and suggestions are welcome! Let’s collaborate to enhance healthcare financial analytics.

## 📜 License
Free to use and modify.

---
📧 **Contact:** [pnjujuna@gmail.com]

