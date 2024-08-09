import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import joblib

# Load data
cell_df = pd.read_csv('sdp1.xlsx - Sheet1.csv')

# Data preprocessing
# CONVERTING TOTAL EXPLOSIVES WEIGHT OBJECT TO INT

cell_df['Total explosives weight'] = cell_df['Total explosives weight'].str.replace(',', '')
cell_df['Total explosives weight'] = cell_df['Total explosives weight'].astype(int)


# CONVERTING BURDEN x SPACING OBJECT TO INT

cell_df['Burden × Spacing [m]'] = cell_df['Burden × Spacing [m]'].fillna(0).astype(int)

# prompt: i want to replace the nan values with 0 in deck

cell_df['Deck    [m]'] = cell_df['Deck    [m]'].fillna(0)

# REMOVING NAN VALUES IN AOP
cell_df['AOP'] = cell_df['AOP'].fillna(0)

# CONVERTING AVERAGE EXPLOSIVE OBJECT TO INT

cell_df['Average explosive'] = cell_df['Average explosive'].str.replace(',', '')
cell_df['Average explosive'] = cell_df['Average explosive'].astype(float)



# Define features and target variable
feature_cols = ['Hole. dia. [mm]', 'No. Of Holes', 'Hole depth  [m]', 'Burden × Spacing [m]', 'Deck    [m]', 'Top stemming[m]', 'Average explosive', 'Total explosives weight']
X = cell_df[feature_cols]
y = cell_df['AOP']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Train SVR model
svm_model = SVR(kernel='linear')
svm_model.fit(X_train, y_train)

# Predictions on the training set using SVR
y_train_pred_svr = svm_model.predict(X_train)

# Predictions on the test set using SVR
y_test_pred_svr = svm_model.predict(X_test)

# Adding SVR predictions as a new feature to the original feature set
X_train_augmented = np.column_stack((X_train, y_train_pred_svr))
X_test_augmented = np.column_stack((X_test, y_test_pred_svr))

# Training Gradient Boosting Regressor on the augmented feature set
params = {
    "n_estimators": 100,
    "min_samples_split": 5,
    "max_depth": 6,
    "learning_rate": 0.01,
    "loss": "squared_error",
}

reg = GradientBoostingRegressor(**params)
reg.fit(X_train_augmented, y_train)

# Save the trained models
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(reg, 'gb_model.pkl')