# train_model.py
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load sample dataset (you can replace this with your own dataset)
X, y = load_iris(return_X_y=True)

# Train a simple model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('app/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved successfully as model.pkl")
