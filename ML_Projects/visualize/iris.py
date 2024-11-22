from sklearn.datasets import load_iris
import pandas as pd

# Load the dataset
iris = load_iris()

# Convert to a DataFrame for better readability
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target  # Add the target column
iris_df['target_name'] = iris_df['target'].map({i: name for i, name in enumerate(iris.target_names)})

# Display the first few rows
print(iris_df.head())
