import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="wide")


@st.cache_data
def load_data():
    try:
        df_local = pd.read_csv("heart.csv")
        return df_local
    except FileNotFoundError:
        st.error("Error: 'heart.csv' not found in the app directory. Please add the dataset and re-run.")
        st.stop()

df = load_data()


categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

df_ml = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df_ml.drop('HeartDisease', axis=1)
y = df_ml['HeartDisease']

# -------------------------
# Train / Test split and scaling
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------
# Model training (cached)
# -------------------------
@st.cache_resource
def build_model():
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)
    return model

model = build_model()

# -------------------------
# App UI
# -------------------------
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "This app predicts the likelihood of heart disease from simple clinical attributes. "
    "**For educational purposes only! not a medical diagnosis.**"
)

st.sidebar.header("Patient Information")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 54)
    sex = st.sidebar.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.sidebar.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 130)
    cholesterol = st.sidebar.slider('Cholesterol (mg/dl)', 80, 610, 240)
    fasting_bs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl', (0, 1))
    resting_ecg = st.sidebar.selectbox('Resting ECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.sidebar.slider('Max Heart Rate Achieved', 60, 202, 150)
    exercise_angina = st.sidebar.selectbox('Exercise-Induced Angina', ('N', 'Y'))
    oldpeak = st.sidebar.slider('Oldpeak (ST depression)', 0.0, 6.2, 1.0)
    st_slope = st.sidebar.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

    data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Combine with original data to ensure same encoding columns
combined = pd.concat([input_df, df.drop('HeartDisease', axis=1)], axis=0)
combined_encoded = pd.get_dummies(combined, columns=categorical_cols, drop_first=True)
user_encoded = combined_encoded.iloc[0:1]

# Make sure the user's input has all the columns the model expects
user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)

# Scale the user input using the same scaler
user_scaled = scaler.transform(user_encoded)

# Display input summary
st.subheader("Input summary")
st.write(input_df.style.set_properties(**{'background-color': '#FFF8DC', 'color': 'black'}))

# Prediction
if st.sidebar.button("Predict"):
    pred = model.predict(user_scaled)[0]
    proba = model.predict_proba(user_scaled)[0][1] * 100  # probability of positive class

    st.subheader("Prediction")
    if pred == 1:
        st.error(f"High risk of heart disease — Probability: {proba:.2f}%")
        st.markdown("Recommendation: Please consult a medical professional for a full evaluation.")
    else:
        st.success(f"Low risk of heart disease — Probability: {proba:.2f}%")
        st.markdown("Recommendation: Continue healthy lifestyle practices and regular checkups.")

st.sidebar.markdown("---")
st.sidebar.info("This app is a demo using Logistic Regression. Not a substitute for professional medical advice.")
