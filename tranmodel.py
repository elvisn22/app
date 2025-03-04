import pandas as pd
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split

# Example synthetic dataset (replace with your actual dataset)
data = {
    'air_temp': [300, 310, 320, 315, 305],
    'process_temp': [310, 315, 320, 310, 325],
    'rotational_speed': [1500, 1600, 1700, 1550, 1650],
    'torque': [30, 32, 33, 31, 29],
    'tool_wear': [50, 55, 60, 58, 52],
    'type_l': [1, 0, 1, 1, 0],
    'type_m': [0, 1, 1, 0, 1],
    'failure_type': [0, 1, 0, 1, 0]  # Example target variable
}

# Create DataFrame from data
df = pd.DataFrame(data)

# Features (X) and target (y)
X = df.drop(columns=['failure_type'])
y = df['failure_type']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format for XGBoost (optimized data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Specify model parameters
params = {
    'objective': 'binary:logistic',  # For binary classification, modify if multiclass
    'eval_metric': 'logloss',
    'max_depth': 3,  # Tree depth (tune based on your data)
    'eta': 0.1,      # Learning rate (tune based on your data)
    'subsample': 0.8  # Proportion of data used for training (tune based on your data)
}

# Train the model
model = xgb.train(params, dtrain, num_boost_round=100)  # num_boost_round is the number of boosting rounds (iterations)

# Save the trained model using joblib
joblib.dump(model, 'xgb_model_fold_4.joblib')
print("Model saved as xgb_model_fold_4.joblib")

# Optionally, save the model in XGBoost's native format (JSON)
model.save_model('xgb_model_fold_4.json')
print("Model saved as xgb_model_fold_4.json")

# Test loading the model with joblib
loaded_model_joblib = joblib.load('xgb_model_fold_4.joblib')
print("Model loaded successfully with joblib")

# Test loading the model with XGBoost's native format
loaded_model_json = xgb.Booster()
loaded_model_json.load_model('xgb_model_fold_4.json')
print("Model loaded successfully from JSON format")
