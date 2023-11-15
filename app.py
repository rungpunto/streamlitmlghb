#import lib
import streamlit as st
from joblib import load

#st.title('Hello World')

#load the model from disk

model = load('titanic_model.joblib')

#Create Streamlit Web App

st.title('Titanic Survival Prediction')

#sidebar with menu

st.sidebar.title('Menu')

#Menu

menu = ['Home','Prediction']

#Sidebar selection

st.sidebar.selectbox('',menu)

#input with slider

age = st.slider('Age',0.42,80.0,30.0)

sibSp = st.slider('SibSp',0,8,0)

parch = st.slider('Parch',0,6,0)

fare = st.slider('Fare',0.0,512.30,32.20)

#Add Prediction button

predict_button = st.button('Predict')

if predict_button:
    #รับค่ามาเก็บในตัวแปรแบบ list
    input_data = [[age,sibSp,parch,fare]]

#หาค่าความน่าจะเป็น
    predict_proba = model.predict_proba(input_data)

    #ทำนายผล

    prediction = model.predict(input_data)

    #แสดงผลลัพธ์
    st.subheader('Prediction')

    if prediction[0] ==1:
        st.write('Survived')
    else:
        st.write('Not Survived')


#แสดงความน่าจะเป็น

    st.subheader('Prediction Probability')
    st.write(f'Survived: {predict_proba[0][1]:.2f}')
    st.write(f'Did not survived: {predict_proba[0][0]:.2f}')
