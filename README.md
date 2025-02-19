<h1 align="center">Taiwan Real Estate Analysis & Prediction</h1>

<p align="center">
  <strong>Exploratory Data Analysis & Machine Learning Model</strong><br>
  A data-driven approach to understanding and predicting home prices in Taiwan.
</p>

## 📌 Project Overview

This project focuses on analyzing and predicting real estate home prices in Taiwan using machine learning techniques. It consists of two Jupyter notebooks:

- **real_estate_eda.ipynb** - Exploratory Data Analysis (EDA), data preprocessing, and storage in MongoDB.
- **real_estate_model.ipynb** - Machine Learning model development for fire prediction.

## 🚀 Technologies Used

- **Python** 🐍
- **Pandas, NumPy** 📊
- **Matplotlib, Seaborn** 📈
- **MongoDB** 🗄️
- **Scikit-learn** 🤖
- **Jupyter Notebook** 📓

## 📊 Data & Analysis

### 🔍 Exploratory Data Analysis (EDA)
- Loaded and cleaned the dataset, handling missing values.
- Conducted feature engineering and transformations.
- Visualized key relationships using **Seaborn** and **Matplotlib**.

![taiwan_heatmap](https://github.com/user-attachments/assets/2031a1ce-7835-4ded-9c90-75786a1951f3)

We see from the visual that house unit price feature is correlated postively with the number of convenience stores nearby and is negatively correlated with the distance from the nearest MRT station. 


### 🤖 Machine Learning Model
- Encoded categorical variables and split the data into training and testing sets.
- Used **RandomizedSearchCV** and **GridSearchCV** for hyperparameter tuning.
- Built and evaluated a predictive model for fire occurrence based on meteorological factors.

![real_estate_features](https://github.com/user-attachments/assets/c12facf7-e659-48b1-aa02-ebb31d3f51a4)

The following visual further confirms our findings that distance to nearest MRT stations, number of convenience stores, and age are the most important features when making predictions on the house prices


## 📢 Results & Insights
- Identified key factors influencing home prices.
- Developed a machine learning model to predict home prices.
- Improved model accuracy through feature selection and hyperparameter tuning.
- Created an Artifical DataFrame with two houses and made predictions on the DataFrame

