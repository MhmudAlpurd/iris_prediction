import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict(data):
    clf = joblib.load("rf_model.sav")
    return clf.predict(data)



st.title('Classify Iris Flowers')
st.markdown('You can predict type of the iris,  \
    using randorm forest model. Change the roles!')

st.header("Features")

col1, col2 = st.columns(2)

with col1:
    st.text('Sepal values')
    sepal_l = st.slider('Sepal length (cm)', 0.0, 10.0, 0.5)
    sepal_w = st.slider('Sepal Width (cm)', 0.0, 10.0, 0.5)

with col2:
    st.text('Petal values')
    petal_l = st.slider('Petal length (cm)', 0.0, 10.0, 0.5)
    petal_w = st.slider('Petal Width (cm)', 0.0, 10.0, 0.5)


if st.button("Predict!"):
    result = predict(
        np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    
    st.text(result[0])