import xgboost as xgb
import pandas as pd
import joblib

# Train a new model (use your actual data here)
data = pd.read_csv('your_dataset.csv')  # Replace with your dataset
X = data.drop('target', axis=1)
y = data['target']

dtrain = xgb.DMatrix(X, label=y)
params = {'objective': 'multi:softmax', 'num_class': 6}
model = xgb.train(params, dtrain, num_boost_round=10)

# Save the model again
joblib.dump(model, 'xgb_model_fold_4.joblib')
