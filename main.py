import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the dataset
df = pd.read_csv('Crop_recommendation.csv')

# Label encoding for the target variable
df['crop_map'] = df['label'].map({'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4, 'pigeonpeas': 5,
                                  'mothbeans': 6, 'mungbean': 7, 'blackgram': 8, 'lentil': 9, 'pomegranate': 10,
                                  'banana': 11, 'mango': 12, 'grapes': 13, 'watermelon': 14, 'muskmelon': 15,
                                  'apple': 17, 'orange': 18, 'papaya': 19, 'coconut': 20, 'cotton': 21,
                                  'jute': 22, 'coffee': 23})

# Drop the original label column
df.drop('label', axis=1, inplace=True)

# Split features and target
X = df.drop('crop_map', axis=1)
y = df['crop_map']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Train the model
rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Define the crop prediction function
def predict_crop(input_data):
    return rf.predict(input_data).astype('int')

def main():
    st.title('Crop Prediction')

    # Input fields
    N = st.text_input('Nitrogen')
    P = st.text_input('Phosphorus')
    K = st.text_input('Potassium')
    temperature = st.text_input('Temperature')
    humidity = st.text_input('Humidity')
    ph = st.text_input('pH Value')
    rainfall = st.text_input('Rainfall')

    data = ''
    # Button to trigger prediction
    if st.button('Predict Best Crop'):
        input_data = np.array([N, P, K, temperature, humidity, ph, rainfall], dtype='float64').reshape(1, -1)
        prediction = predict_crop(input_data)
        crop_map = {
            1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans', 5: 'pigeonpeas',
            6: 'mothbeans', 7: 'mungbean', 8: 'blackgram', 9: 'lentil', 10: 'pomegranate',
            11: 'banana', 12: 'mango', 13: 'grapes', 14: 'watermelon', 15: 'muskmelon',
            17: 'apple', 18: 'orange', 19: 'papaya', 20: 'coconut', 21: 'cotton',
            22: 'jute', 23: 'coffee'
        }
        data = crop_map.get(prediction[0], 'Unknown')

    # Display prediction result
    st.success('Recommended Crop: {}'.format(data))

if __name__ == '__main__':
    main()
