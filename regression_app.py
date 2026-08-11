import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model


model = load_model('regression_churn_model.h5')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geography.pkl', 'rb') as f:
    onehot_encoder_geography = pickle.load(f)

st.title("Customer Churn Prediction")


geography = st.selectbox("Geography", onehot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 100)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
exited = st.selectbox("Exited", [0, 1])
tenure = st.slider("Tenure", 0, 10)
num_of_products = st.slider("Number of Products", 1, 4)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
})

geography_encoded = onehot_encoder_geography.transform(
    input_data[['Geography']]
)
geography_encoded_df = pd.DataFrame(geography_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))


input_data = pd.concat([input_data.drop('Geography', axis = 1), geography_encoded_df], axis=1)

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

input_data_scaled = scaler.transform(input_data)

prediction = model.predict(input_data_scaled)


st.write("Predition :", prediction)