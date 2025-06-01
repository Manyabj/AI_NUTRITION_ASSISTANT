import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
import sys
sys.path.append("C:/Users/manya/OneDrive/Desktop/AI-ASSITANT")  # Adjust path to your project root
from preprocessing.data_preparation import train_test_split 

import numpy as np
import tensorflow as tf
import joblib
import pandas as pd


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X_train = pd.read_csv('server/model/X_train.csv')
X_test = pd.read_csv('server/model/X_test.csv')
y_train = pd.read_csv('server/model/y_train.csv')
y_test = pd.read_csv('server/model/y_test.csv')

# Print shape of training data for verification
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Define the neural network model with multiple outputs
model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(3)  # 3 outputs: carbs, protein, fat
])


# Compile the model using mean squared error for regression tasks
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=10, validation_data=(X_test, y_test))

# Evaluate the model on the test set to see its performance
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, TEST MAE: {mae}")

# Make predictions on test data
predictions = model.predict(X_test)

# Extract predictions from the first test sample
predicted_calories = predictions[0][0]
predicted_carbs = predictions[0][1]
predicted_calories = predictions[0][0]
predicted_carbs = predictions[0][1]
predicted_protein = predictions[0][2]
predicted_fat = predictions[0][2]

print(f"Predicted Calories: {predicted_calories:.2f}")
print(f"Predicted Carbs: {predicted_carbs:.2f}")
print(f"Predicted Protein: {predicted_protein:.2f}")
print(f"Predicted Fat: {predicted_fat:.2f}")

# Save the trained model
model.save('server/model/nutrition_model.h5')

print("Model training completed, saved, and food suggestions generated.")
