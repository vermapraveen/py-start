import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

iris=load_iris()
X,y= iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

#make prediction
prediction=model.predict(X_test)

#evaluate model
accuracy=accuracy_score(y_test, prediction)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

dump(model, "./ML_Projects/models/RandomForestClassifier_iris_model.pkvml")
print("Model saved as RandomForestClassifier_iris_model.pkvml")

