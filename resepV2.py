import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

st.write("""
# Aplikasi Prediksi Rating Resep
Berdasarkan Ingrediens

Oleh : Nezar Abdilah Prakasa

Data from : https://www.epicurious.com/

""")

st.sidebar.header('Ayo isi datamu!')


# Collects user input features into dataframe
def user_input_features():
    calories = st.sidebar.slider('Kalori', 0,10000,560)
    protein = st.sidebar.slider('Protein (mg)', 0,10000,73)
    fat = st.sidebar.slider('Lemak (mg)',0,10000,10)
    sodium = st.sidebar.slider('Sodium (mg)',0,10000,3698)
    data = {'Kalori': calories,
            'Protein (mg)': protein,
            'Lemak (mg)': fat,
            'Sodium (mg)':sodium}
    features = pd.DataFrame(data, index=[0])
    return features
df = user_input_features()


# Displays the user input features
st.subheader('Silahkan isi dengan slider yang ada di sebelah kiri')
st.write(df)

# Aplikasi Prediksi Rating Resep
("""
Method : Random Forest
""")

# Reads in saved classification model
load_clf = pickle.load(open('g_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)



st.subheader('Prediksi')
st.write(prediction)


st.write("""
#
Notes:
Rating dengan skala 5/5
""")
