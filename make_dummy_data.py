from sklearn.datasets import make_classification
import pandas as pd
import os


# Create dummy classification data
x_data, y_data = make_classification(n_samples=10000, n_features=20, n_classes=2)

x_data = pd.DataFrame(x_data)
y_data = pd.DataFrame(y_data)

# make data dir
data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

x_data.to_csv(os.path.join(data_dir, "x_data.csv"), index=False)
y_data.to_csv(os.path.join(data_dir, "y_data.csv"), index=False)
