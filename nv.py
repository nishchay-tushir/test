import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB

# Load the dataset
df = pd.read_csv("./tennis_dataset.csv")

# Separate features and target variable
X = df.drop("PlayTennis", axis=1)
y = df["PlayTennis"]

# Initialize the Categorical Naive Bayes model
model = CategoricalNB()

# Train the model on the entire dataset
model.fit(X, y)

# Generate random test cases
random_test_cases = pd.DataFrame({
    "Weather": np.random.randint(0, 3, size=10),
    "Temperature": np.random.randint(0, 3, size=10)
})

# Make predictions on the random test cases
predictions = model.predict(random_test_cases)

# Print the random test cases and corresponding predictions
for i in range(len(random_test_cases)):
    print("Weather:", random_test_cases["Weather"].iloc[i], "  Temperature:", random_test_cases["Temperature"].iloc[i], "  Prediction:", predictions[i])