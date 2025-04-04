import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# Load dataset (Modify the file path accordingly)
data = pd.read_csv("dataset.csv")  # Ensure dataset has features and target columns

# Splitting into features and target variable
X = data.drop(columns=["target"])  # Modify 'target' based on your dataset
y = data["target"]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Model Accuracy: {accuracy:.4f}")

# Print Decision Tree structure
print(export_text(model, feature_names=list(X.columns)))