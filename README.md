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
The following table shows basic statistical analysis of each our features



<table>
  <thead>
    <tr>
      <th>feature</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>X2 house age</td>
      <td>17.712560</td>
      <td>11.392485</td>
      <td>0.00000</td>
      <td>9.0250</td>
      <td>16.1000</td>
      <td>28.150</td>
      <td>43.800</td>
    </tr>
    <tr>
      <td>X3 distance to the nearest MRT station</td>
      <td>1083.885689</td>
      <td>1262.109595</td>
      <td>23.38284</td>
      <td>289.3248</td>
      <td>492.2313</td>
      <td>1454.279</td>
      <td>6488.021</td>
    </tr>
    <tr>
      <td>X4 number of convenience stores</td>
      <td>4.094203</td>
      <td>2.945562</td>
      <td>0.00000</td>
      <td>1.0000</td>
      <td>4.0000</td>
      <td>6.000</td>
      <td>10.000</td>
    </tr>
    <tr>
      <td>Y house price of unit area</td>
      <td>37.980193</td>
      <td>13.606488</td>
      <td>7.60000</td>
      <td>27.7000</td>
      <td>38.4500</td>
      <td>46.600</td>
      <td>117.500</td>
    </tr>
  </tbody>
</table>

![taiwan_heatmap](https://github.com/user-attachments/assets/2031a1ce-7835-4ded-9c90-75786a1951f3)

We see from the visual that house unit price feature is correlated postively with the number of convenience stores nearby and is negatively correlated with the distance from the nearest MRT station. 


### 🤖 Machine Learning Model
- Encoded categorical variables and split the data into training and testing sets.
- Used **RandomizedSearchCV** and **GridSearchCV** for hyperparameter tuning.
- Built and evaluated a predictive model for fire occurrence based on meteorological factors.

#### Tuned Random Forest

``` python
param_grid = [{'bootstrap': [True, False], 
               'max_depth':[10,20,30,40,50,60,70,80,90, 100, 110, 120], 
               'min_samples_leaf':[1,3,4], 
               'min_samples_split':[2,6,10], 'n_estimators':[5,20,50,100]}]
f_reg = RandomForestRegressor()
Random_rf = RandomizedSearchCV(f_reg, param_grid, cv=10, verbose=2, n_jobs=-1)
Random_rf.fit(X_train_scaled, y_train)
best_random_grid = Random_rf.best_estimator_
bestrf_pred = best_random_grid.predict(X_test_scaled)
mae = MAE(y_test, bestrf_pred)
r2 = r2_score(y_test, bestrf_pred)
```
<table>
  <thead>
    <tr>
      <th>Metric</th>
      <th>Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>R2 Score</td>
      <td>0.7728</td>
    </tr>
    <tr>
      <td>MAE</td>
      <td>4.390851</td>
    </tr>
  </tbody>
</table>


![real_estate_features](https://github.com/user-attachments/assets/c12facf7-e659-48b1-aa02-ebb31d3f51a4)

The following visual further confirms our findings that distance to nearest MRT stations, number of convenience stores, and age are the most important features when making predictions on the house prices


## 📢 Results & Insights
- Identified key factors influencing home prices.
- Developed a machine learning model to predict home prices.
- Improved model accuracy through feature selection and hyperparameter tuning.
- Created an Artifical DataFrame with two houses and made predictions on the DataFrame

