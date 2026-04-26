import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib

data = pd.read_csv("asl_data.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

joblib.dump(model, "asl_model.pkl")

print("Model trained and saved!")