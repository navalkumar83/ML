# ML
ML Model
# Predicting Employee Compensation

This project focuses on predicting employee compensation (CTC) based on experience, city, and role using various regression models. The aim is to create a reliable predictive model that HR departments can use to make informed salary offers.

## Table of Contents

- [Problem Statement](#problem-statement)
- [Approach](#approach)
- [Available ML Models](#available-ml-models)
- [Best Performing Model](#best-performing-model)
- [Steps to Improve Model Performance](#steps-to-improve-model-performance)
- [Files Included](#files-included)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Problem Statement

The project aims to predict the Compensation (CTC) of employees using their experience (in months), city, and role as input features. By leveraging machine learning models, the goal is to develop a system that can accurately estimate an employee’s compensation, assisting HR departments in making data-driven decisions.

## Approach

### Data Exploration and Preprocessing
- **Loading the Dataset:** The dataset was loaded and initially explored using `pandas` to understand the structure and key statistics.
- **Feature Engineering:** 
  - Dropped irrelevant columns like `College`.
  - Applied one-hot encoding to categorical variables such as `City` and `Role`.
- **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets.
  
### Model Selection
Five regression models were selected to explore different ways of predicting CTC:
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Random Forest Regressor**
5. **Gradient Boosting Regressor**

### Model Training and Evaluation
- Each model was trained on the training set and evaluated using the test set.
- Key evaluation metrics included Mean Squared Error (MSE) and R² score.
- The intercept of the Linear Regression model was also extracted and analyzed.

### Hyperparameter Tuning
- For models like Ridge, Lasso, Random Forest, and Gradient Boosting, hyperparameter tuning was performed using GridSearchCV to find the best model parameters.

## Available ML Models

- **Linear Regression:** A baseline model assuming a linear relationship between the features and CTC.
- **Ridge Regression:** A regularized version of Linear Regression using L2 penalty to prevent overfitting.
- **Lasso Regression:** Another regularized model using L1 penalty, which also helps with feature selection.
- **Random Forest Regressor:** An ensemble model that constructs multiple decision trees and averages their outputs to improve prediction accuracy.
- **Gradient Boosting Regressor:** An advanced ensemble technique that builds trees sequentially, each one correcting errors made by the previous trees.

## Best Performing Model

After evaluating all models, the **Random Forest Regressor** was found to have the best performance, achieving the lowest MSE and highest R² score. This model’s ability to capture complex, non-linear relationships between features contributed to its superior performance.

## Steps to Improve Model Performance

To further enhance the model’s accuracy:
1. **Advanced Hyperparameter Tuning:** Use Random Search or Bayesian Optimization for finer tuning of model parameters.
2. **Feature Engineering:** Investigate the creation of new features or transformations to provide more insights to the model.
3. **Ensemble Methods:** Explore stacking or blending multiple models to leverage their strengths.
4. **Cross-Validation:** Apply k-fold cross-validation to ensure the model's performance is consistent across different subsets of data.
5. **Data Augmentation:** Consider gathering additional data or applying techniques to enrich the existing dataset.

## Files Included

- **`ML_case_Study.csv`:** The dataset used for training and testing the models.
- **`final project ML.ipynb`:** The Python script containing the complete code for data preprocessing, model training, evaluation, and hyperparameter tuning.
- **`README.md`:** This README file, providing an overview and instructions for the project.

## Acknowledgements

This project utilizes powerful libraries like `pandas`, `numpy`, `scikit-learn`, `seaborn`, and `matplotlib` for data manipulation, visualization, and model building. Thanks to the open-source community for making these tools available.


