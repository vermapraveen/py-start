from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the dataset
iris = load_iris()
X, y = iris.data, iris.target

# Create and train the Decision Tree
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree_model.fit(X, y)

# Print the tree structure
tree_rules = export_text(tree_model, feature_names=iris.feature_names)
print(tree_rules)

# Predict for a new data point
new_data = [[5.1, 3.5, 1.4, 0.2]]
prediction = tree_model.predict(new_data)
print(f"Predicted Class: {iris.target_names[prediction[0]]}")
