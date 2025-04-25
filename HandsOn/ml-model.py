import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv('Iris.csv')

# Features and target selection
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalWidthCm']]
y = df['PetalLengthCm']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

er 

# Load and predict with the saved model
loaded_model = joblib.load('iris_linear_model.pkl')
y_pred = loaded_model.predict(X_test)

# Evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", round(mse, 4))
print("R² Score (Accuracy):", round(r2, 4))

# Plot: Actual vs Predicted PetalLengthCm
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label='Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual PetalLengthCm')
plt.ylabel('Predicted PetalLengthCm')
plt.title('Actual vs Predicted PetalLengthCm')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300)
plt.show()