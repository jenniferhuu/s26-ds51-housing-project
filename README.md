# California Housing Price Prediction Project

This project predicts the closing price of California single-family residential properties using Trestle/CoreLogic MLS data. The project follows a weekly data science workflow: exploratory data analysis, preprocessing, baseline modeling, feature engineering, advanced modeling, expanded evaluation, and a simple Streamlit prediction app.

## Project Goal

The goal is to estimate `ClosePrice` for California single-family residences using structured property features such as living area, bedrooms, bathrooms, lot size, property age, and location-based school district information.

The final model is intended as a machine learning estimate, not a formal appraisal. It is most reliable for typical residential sale prices and less reliable for unusual low-priced transactions or high-end luxury properties.

## Dataset

The dataset consists of monthly sold property CSV files from Trestle/CoreLogic MLS data.

The analysis focuses on:

- `PropertyType = Residential`
- `PropertySubType = SingleFamilyResidence`

The target variable is:

- `ClosePrice`

Core modeling features include:

- `LivingArea`
- `Bedrooms`
- `Bathrooms`
- `LotSize`
- `PropertyAge`
- missing-value indicators
- engineered features
- school district information

## Repository Structure

```text
01_exploration.ipynb
02_preprocessing.ipynb
03_baseline_model.ipynb
04_model_comparison.ipynb
05_advanced_models.ipynb
06_evaluation.ipynb
geocoding_update.ipynb
app.py
requirements.txt
metrics_summary.csv
price_band_bias_summary.csv
week6_old_vs_new_feature_comparison.csv
week7_xgboost_test_predictions.csv
xgboost_price_model.joblib