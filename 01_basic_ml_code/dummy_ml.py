"""
Dummy code to create an ML model. Lots of code snippets, could be run from a notebook or python terminal.
"""
import os
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, log_loss


# load data and split train and test sets
x_data = pd.read_csv('../data/x_data.csv')
y_data = pd.read_csv('../data/y_data.csv')
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.33, random_state=42
)

# train model
param_grid = [{"penalty": ["l1"], "C": [1.0]}]
model = GridSearchCV(
        LogisticRegression(solver="liblinear"), param_grid, cv=5, scoring="roc_auc",
    )
model.fit(x_train, y_train)
print(model.best_estimator_)

# model performance
y_pred_prob = model.predict_proba(x_test)[:, 1]
test_roc_auc = roc_auc_score(y_test, y_pred_prob)
test_log_loss = log_loss(y_test, y_pred_prob)
print(f"cv_roc_auc: {model.best_score_:.10f};")
print(f"test_roc_auc: {test_roc_auc:.10f};")
print(f"test_log_loss: {test_log_loss:.10f};")

# save model artifacts (model, metrics, graphs, etc)
model_dir = "../models"
dump(model, os.path.join(model_dir, "model.pkl"))
