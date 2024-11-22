from sklearn.tree import plot_tree, DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target
tree_model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
tree_model.fit(X, y)

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()
