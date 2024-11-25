import streamlit as st
import pandas as pd
import pickle as pk
from PIL import Image
#loading the model

model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Header for app
#st.header("Loan Predictor App")
im = Image.open('logo.png')
st.set_page_config(page_title="Loan Predictor App",page_icon=im)

html_temp = """
    <div style="background-color:#f63350 ;padding:10px">
    <h1 style="color:white;text-align:center;">
    Loan Predictor App </h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.image('background.png')


st.markdown("""
    Welcome to the **Loan Approval Predictor**!  
    This app helps predict whether you are likely to be approved for a loan based on your details.
    Fill in the details below to get started!
""")
# Custom styling for the app
st.markdown(
    """
    <style>
    .main {background-color: #f5f5f5; padding: 20px;}
    .header {color: #3e5c5b;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stSlider>div>div>input {background-color: #f0f0f0;}
    </style>
    """, unsafe_allow_html=True)

# User input for interaction

no_of_dep = st.number_input('Number of Dependents', min_value =0, max_value = 5)
education_level = st.radio('Education Level',['Graduate','Not Graduate'])
employment = st.radio('Are you self Employed?',['Yes','No'])
annual_income = st.number_input('Annual Income', min_value = 0, max_value = 100000000, step = 100000)
loan_amount = st.number_input('Loan Amount', min_value = 0, max_value =100000000, step = 100000 )
loan_dur = st.number_input('Loan Duration (in years)', min_value = 0, max_value = 30, step = 1 )
cibil_score = st.slider('CIBIL Score', 300, 900)
assets = st.number_input('Total Assets Value', min_value = 0, max_value = 100000000, step =100000 )

# Converting categorical to numerical
education_num = 0 if education_level == 'Graduate' else 1
self_emp_num = 1 if employment == 'Yes' else 0

# Predict loan approval status

if st.button('Predict'):
    # Display loading spinner while making the prediction
    with st.spinner('Making the prediction...'):
        import time  # To simulate processing time
        time.sleep(2) 
        pred_data = pd.DataFrame([[no_of_dep, education_num, self_emp_num, annual_income, loan_amount, loan_dur, cibil_score, assets]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'assets'])
        pred_data = scaler.transform(pred_data)
        prediction = model.predict(pred_data)
    
        if prediction[0] == 0:
            st.success("Congratulation!, Your loan is approved")
        else:
            st.error('Sorry!, Your loan is rejected')
        
st.markdown("""
    ---
    **Disclaimer**: This is a machine learning model for loan prediction. It is not the final decision-making tool for actual loan approval. Consult with a bank or financial institution for official approval.
""")        