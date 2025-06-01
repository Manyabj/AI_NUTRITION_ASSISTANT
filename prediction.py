import numpy as np
import joblib
import tensorflow as tf

# Load preprocessing objects
label_encoders = joblib.load('server/model/label_encoders.pkl')
scaler = joblib.load('server/model/scaler.pkl')

# Example of input data for prediction (replace with actual user data)
input_data = {
    'age': 30,
    'gender': 'male',
    'activity_level': 'moderate',
    'goal': 'muscle gain',
    'weight': 75,  # in kg
    'height': 180,  # in cm
    'current_calories': 2500
}

# Encode categorical data (keep the order of features consistent)
input_data['gender'] = label_encoders['gender'].transform([input_data['gender']])[0]
input_data['activity_level'] = label_encoders['activity_level'].transform([input_data['activity_level']])[0]
input_data['goal'] = label_encoders['goal'].transform([input_data['goal']])[0]

scaled_data = scaler.transform([[input_data['weight'],input_data['height'],input_data['current_calories']]])

# Combine scaled numerical features with encoded categorical features
final_input = np.hstack((
    scaled_data,
    [input_data['gender'], input_data['activity_level'], input_data['goal']]
)).reshape(1, -1)

# Debug print: confirm input shape
print(f"Final input shape: {final_input.shape}")

# Make prediction
try:
    predictions = model.predict(final_input)

    # Extract individual predictions
    predicted_calories = predictions[0][0]
    predicted_carbs = predictions[0][1]
    predicted_protein = predictions[0][2]
    predicted_fat = predictions[0][2]

    # Output the predictions
    print(f"Suggested Calories: {predicted_calories:.2f}")
    print(f"Suggested Carbs: {predicted_carbs:.2f}g")
    print(f"Suggested Protein: {predicted_protein:.2f}g")
    print(f"Suggested Fat: {predicted_fat:.2f}g")

except ValueError as e:
    print(f"Error during prediction: {e}")