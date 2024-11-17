from joblib import load
import numpy as np
from sklearn.datasets import load_iris

model = load("./ML_Projects/models/RandomForestClassifier_iris_model.pkvml")
print("Model loaded successfully")

# Sepal length: 5.1, Sepal width: 3.5, Petal length: 1.4, Petal width: 0.2
new_data = np.array([[5.1, 3.5, 1.4, 0.2]])
iris=load_iris()

predicted_class = model.predict(new_data)
print(f"Predicted Class: {predicted_class[0]}")  # Output will be 0, 1, or 2 (corresponding to species)
print(f"Predicted Species: {iris.target_names[predicted_class[0]]}")
