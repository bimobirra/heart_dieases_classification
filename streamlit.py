import streamlit as st
import pandas as pd
import joblib
import time

# Load Model, Encoder and Scaler
rf_model = joblib.load('model/rf_model.pkl')
encoder = joblib.load('model/encoder.pkl')
scaler = joblib.load('model/scaler.pkl')

# Header
st.header('Heart Dieases Prediction')
st.write('---')

# Tab initialization
tab1, tab2 = st.tabs(['Problem', 'Predict'])

with tab1:
    st.subheader("Problem")
    st.write("Heart disease, also known as cardiovascular disease, is one of the leading causes of death globally, with an estimated 17.9 million deaths from this disease in 2019. It is very important to detect the disease as early as possible so that it can be prevented, for example through counseling and the administration of medication.")
    st.write('---')

    st.subheader("Solution")
    st.write("By building a predictive model, we can identify at an early stage whether a patient has a presence of heart disease. This allows healthcare professionals to design faster and more personalized interventions, thereby improving the chances of successful treatment and the patient's quality of life.")
    st.write('---')

    st.subheader("Data")
    st.write("Data source: https://www.kaggle.com/datasets/bharath011/heart-disease-classification-dataset")
    st.write("Data used:")
    st.write("age: Patient's Age in Years")
    st.write("gender: Patient's Gender")
    st.write("impulse: Patient's Pulse Rate (beats per minute)")
    st.write("pressurehigh: Systolic BP (High)")
    st.write("pressurelow: Diastolic BP (Low)")
    st.write("glucose: Glucose Level")
    st.write("kcm: KCM Value")
    st.write("troponin: Troponin Level")
    st.write('---')

    st.subheader("Models")
    st.write("The models used in this project, Random Forest Classifier and K-Nearest Neighbors (KNN), were chosen as they are both effective and straightforward for classification tasks. Random Forest is well-regarded for its high accuracy and ability to prevent overfitting, while KNN is known for its simplicity. Upon evaluation, the RandomForestClassifier was selected due to its superior performance, achieving 97% accuracy compared to 75% for the K-Nearest Neighbors model.")
    st.write('---')

with tab2:
    st.subheader("Prediction Results")
    st.write("Please Input the data on the side panel")
    st.write("---")
    # Get Input Data

    age = st.sidebar.number_input(label="Patient's Age", min_value=0, max_value=100, value=0)

    gender = ['Female', 'Male']

    selected_gender = st.sidebar.selectbox(label="Gender", options=gender)

    impulse = st.sidebar.number_input(label="Pulse Rate (beats per minute)", min_value=0, max_value=1000, value=0)

    preassure_high = st.sidebar.number_input(label="Systolic BP (High)", min_value=0, max_value=250, value=0)

    preassure_low = st.sidebar.number_input(label="Diastolic BP (Low)", min_value=0, max_value=200, value=0)

    glucose = st.sidebar.number_input(label="Glucose Level", min_value=0, max_value=550, value=0)

    kcm = st.sidebar.number_input(label="KCM Value", min_value=0., max_value=300., value=0., step=1., format="%.2f")

    troponin = st.sidebar.number_input(label="Troponin Level", min_value=0., max_value=10., value=0.,step=1., format="%.3f")

    predict_button = st.sidebar.button('Run Prediction')

    # If button is clicked
    if predict_button:
        data = {
            'age': age,
            'gender': selected_gender,
            'impulse': impulse,
            'preassurehigh': preassure_high,
            'preassurelow': preassure_low,
            'glucose': glucose,
            'kcm': kcm,
            'troponin': troponin,
        }

        data_df = pd.DataFrame(data, index=[0])

        st.subheader("Patient's data")
        st.dataframe(data_df)

        # Changes value of gender to 0 if female in 1 if male
        data_df['gender'] = 0 if selected_gender == 'Female' else 1

        # Scaling
        numeric = data_df.select_dtypes(include='number').columns

        data_df[numeric] = scaler.transform(data_df[numeric])

        # Predict the data based on the scaled input
        prediction = rf_model.predict(data_df)

        with st.spinner("Analyzing..."):
            time.sleep(3)

        st.write('---')
        st.subheader("Prediction result")

        # Print prediction result
        if prediction == 0:
            st.success("Based on the provided data, the model predicts a low probability of heart disease presence.")
        else:
            st.error("Based on the provided data, the model predicts a high probability of heart disease presence. Disclaimer: This is not a medical diagnosis. Please consult a healthcare professional.")
