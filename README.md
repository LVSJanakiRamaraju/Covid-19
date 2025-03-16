# COVID-19 Prediction App

## Overview
This project is a **COVID-19 Prediction App** built using **Streamlit** and **K-Nearest Neighbors (KNN)** algorithm. The application takes user inputs based on various symptoms and risk factors and predicts whether the user might be affected by COVID-19.

## Features
- User-friendly web-based interface using **Streamlit**
- Predicts COVID-19 infection risk based on symptoms and risk factors
- Uses **K-Nearest Neighbors (KNN)** for classification
- Data preprocessing with **Label Encoding**
- Hyperparameter tuning with **GridSearchCV**
- Displays prediction results and recommendations

## Technologies Used
  - **Python**
  - **Streamlit** (for web application)
  - **Scikit-learn** (for machine learning model)
  
  - **Pandas** (for data preprocessing)
  
  - **NumPy** (for numerical computations)
  
  - **Matplotlib** (for visualization, if needed)

## Dataset
The dataset used in this project is **Covid Dataset.csv**, which contains various attributes related to COVID-19 symptoms and risk factors. Such as
- Fever

- Breathing Problem

- Dry Cough

- Diabetes

- Contact with COVID-19 Patient

- Travel History

And many more...

The dataset is preprocessed using Label Encoding, and a train-test split (80%-20%) is performed to train the model.

## Installation & Setup
### 1. Clone the Repository
```bash
git clone https://github.com/LVSJanakiRamaraju/Covid19.git
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

## Usage
- Run the app using `streamlit run app.py`.
- Enter symptoms and risk factors in the input fields.
- Click on **Predict** to get the results.
- The app will display whether the user is likely affected or not.

## Dependencies
- **Python 3.x**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

## Model Details
- The dataset is preprocessed using **Label Encoding**.
- A **K-Nearest Neighbors (KNN)** classifier is trained.
- Hyperparameter tuning is done using **GridSearchCV**.
- The trained model is used for real-time predictions in the Streamlit app.

## Example Output
```
Results: [1]
You may be affected with COVID-19 virus! Please get RTPCR test ASAP and stay in Quarantine for 14 days!
```
OR
```
Results: [0]
You do not have any symptoms of COVID-19. Stay home! Stay safe!
```

## Contact
For any queries or collaborations, contact us via **rajakanumuri2005@gmail.com**

## License
This project is **open-source** and free to use.
