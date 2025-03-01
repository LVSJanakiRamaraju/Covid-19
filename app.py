import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_squared_error, r2_score


# Load your data
covid_data = pd.read_csv("Covid Dataset.csv")

# Data Preprocessing (assuming this is the same as your original code)
e = LabelEncoder()
for col in ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Running Nose', 'Asthma',
            'Chronic Lung Disease', 'Headache', 'Heart Disease', 'Diabetes', 'Hyper Tension',
            'Abroad travel', 'Contact with COVID Patient', 'Attended Large Gathering',
            'Visited Public Exposed Places', 'Family working in Public Exposed Places',
            'Wearing Masks', 'Sanitization from Market', 'COVID-19', 'Gastrointestinal ', 'Fatigue ']:
    covid_data[col] = e.fit_transform(covid_data[col])

x = covid_data.drop('COVID-19', axis=1)
y = covid_data['COVID-19']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)


# KNN Model Training (assuming this is the same as your original code)
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 50)}
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(x_train, y_train)


# Streamlit app
st.title("COVID-19 Prediction App")

# Input fields
Breathing_Problem = st.number_input("Breathing Problem (1 for Yes, 0 for No)", min_value=0, max_value=1)
Fever = st.number_input("Fever (1 for Yes, 0 for No)", min_value=0, max_value=1)
Dry_Cough = st.number_input("Dry Cough (1 for Yes, 0 for No)", min_value=0, max_value=1)
Sore_throat = st.number_input("Sore Throat (1 for Yes, 0 for No)", min_value=0, max_value=1)
Running_Nose = st.number_input("Running Nose (1 for Yes, 0 for No)", min_value=0, max_value=1)
Asthma = st.number_input("Asthma (1 for Yes, 0 for No)", min_value=0, max_value=1)
Chronic_Lung_Disease = st.number_input("Chronic Lung Disease (1 for Yes, 0 for No)", min_value=0, max_value=1)
Headache = st.number_input("Headache (1 for Yes, 0 for No)", min_value=0, max_value=1)
Heart_Disease = st.number_input("Heart Disease (1 for Yes, 0 for No)", min_value=0, max_value=1)
Diabetes = st.number_input("Diabetes (1 for Yes, 0 for No)", min_value=0, max_value=1)
Hyper_Tension = st.number_input("Hyper Tension (1 for Yes, 0 for No)", min_value=0, max_value=1)
Fatigue = st.number_input("Fatigue (1 for Yes, 0 for No)", min_value=0, max_value=1)
Gastrointestinal = st.number_input("Gastrointestinal (1 for Yes, 0 for No)", min_value=0, max_value=1)
Abroad_travel = st.number_input("Abroad Travel (1 for Yes, 0 for No)", min_value=0, max_value=1)
Contact_with_COVID_Patient = st.number_input("Contact with COVID Patient (1 for Yes, 0 for No)", min_value=0, max_value=1)
Attended_Large_Gathering = st.number_input("Attended Large Gathering (1 for Yes, 0 for No)", min_value=0, max_value=1)
Visited_Public_Exposed_Places = st.number_input("Visited Public Exposed Places (1 for Yes, 0 for No)", min_value=0, max_value=1)
Family_working_in_Public_Exposed_Places = st.number_input("Family Working in Public Exposed Places (1 for Yes, 0 for No)", min_value=0, max_value=1)



# Dummy values for the missing features 'Wearing Masks' and 'Sanitization from Market'
Wearing_Masks = 1  # or 0 - you should decide on a reasonable default
Sanitization_from_Market = 0 # or 0 - you should decide on a reasonable default


if st.button("Predict"):
    patient = [[Breathing_Problem, Fever, Dry_Cough, Sore_throat, Running_Nose, Asthma, Chronic_Lung_Disease, Headache, Heart_Disease, Diabetes, Hyper_Tension, Fatigue, Gastrointestinal, Abroad_travel, Contact_with_COVID_Patient, Attended_Large_Gathering, Visited_Public_Exposed_Places, Family_working_in_Public_Exposed_Places, Wearing_Masks, Sanitization_from_Market]]
    result = knn_cv.predict(patient)

    st.write("\nResults : ", result)
    if result == 1:
        st.write('You may be affected with COVID-19 virus! Please get RTPCR test ASAP and stay in Quarantine for 14 days!')
    else:
        st.write('You do not have any symptoms of COVID-19. Stay home! Stay safe!')