# Hause-Price-Prediction

# 🏠 House Prices: Advanced Regression Techniques

Welcome to my solution for the Kaggle competition [**House Prices: Advanced Regression Techniques**](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques). The objective is to predict the final sale price of homes in Ames, Iowa, using rich tabular data with 79 explanatory variables.

---

## 📌 Project Goals

- Accurately predict house sale prices using regression models
- Apply both traditional machine learning and deep learning techniques
- Optimize performance using advanced preprocessing, feature engineering, and model tuning
- Deploy clean and reproducible pipelines

---

## 🧠 Approaches Used

### 🔹 1. Deep Neural Network (DNN) - TensorFlow / Keras
- Custom neural network architecture with:
  - Dense layers with ReLU activations
  - Batch Normalization
  - Dropout for regularization
  - L2 weight regularization
- Used `Huber` loss function for robustness against outliers
- Log-transformed target (`log1p(y)`) to normalize skewed distribution
- Feature scaling and one-hot encoding via `ColumnTransformer`
- Callbacks:
  - EarlyStopping
  - ReduceLROnPlateau
  - ModelCheckpoint
- TensorBoard integration for training visualization
- Final performance:
  - ✅ **Test RMSE**: ~28,450  
  - ✅ **Test MAE**: ~19,595  
  - ✅ **R² Score**: ~0.85

### 🔹 2. (Optional) XGBoost Regressor (Not fully shown in this repo)
- Feature engineering and selection
- Hyperparameter tuning via `RandomizedSearchCV`
- Recursive Feature Elimination with Cross-Validation (RFECV)

---

## ⚙️ Preprocessing Pipeline

Implemented using `sklearn`'s `ColumnTransformer`:
- Numerical features:
  - Imputed missing values (mean)
  - Scaled using `StandardScaler`
- Categorical features:
  - Imputed (mode)
  - Encoded using `OneHotEncoder` (handle unknowns)
  
All transformations were applied consistently to both training and test sets.

---

## 📈 Evaluation Metrics

The model was evaluated using:

- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Coefficient of Determination (R² Score)

These metrics were computed on the test set **after reversing the log transformation** with `np.expm1()`.

---

## 📤 Submission Pipeline for Kaggle

- Transformed Kaggle test set using the trained `preprocessor`
- Predicted log-transformed values using the trained neural network
- Reverted predictions to original price scale using `np.expm1()`
- Saved predictions in the required submission format (`submission.csv`)

```python
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": y_kaggle_preds
})
submission.to_csv("submission.csv", index=False)
