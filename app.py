import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Lead Prediction Dashboard", layout="wide")

st.markdown("""
    <h1 style='text-align: center;'>🎯 Lead Conversion Prediction</h1>
    <p style='text-align: center; color: gray;'>
        Enter lead details and predict conversion probability
    </p>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("leads_basic_details.csv")

df = load_data()

# --------------------------------------------------
# SELECT TARGET
# --------------------------------------------------
target_col = st.selectbox("Select Target Column", df.columns)

# --------------------------------------------------
# PREPROCESS + TRAIN MODEL (HIDDEN)
# --------------------------------------------------
df_model = df.copy()

# Drop ID columns
for col in df_model.columns:
    if "id" in col.lower():
        df_model.drop(col, axis=1, inplace=True)

# Encode categorical
le_dict = {}
for col in df_model.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    le_dict[col] = le

X = df_model.drop(target_col, axis=1)
y = df_model[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)

# --------------------------------------------------
# UI LAYOUT
# --------------------------------------------------
col1, col2 = st.columns([1, 1])

# ---------------- LEFT: INPUT ----------------
with col1:
    st.markdown("### 🧾 Enter Lead Details")

    user_input = {}

    for col in X.columns:
        if df[col].dtype == "object":
            user_input[col] = st.selectbox(col, df[col].unique())
        else:
            user_input[col] = st.number_input(
                col, value=float(df[col].mean())
            )

# ---------------- RIGHT: RESULT ----------------
with col2:
    st.markdown("### 📊 Prediction Result")

    if st.button("🚀 Predict Now"):

        input_df = pd.DataFrame([user_input])

        # Encode
        for col in input_df.columns:
            if col in le_dict:
                input_df[col] = le_dict[col].transform(input_df[col])

        # Scale
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)
        prob = model.predict_proba(input_scaled)

        confidence = np.max(prob) * 100

        # ---------------- RESULT CARD ----------------
        if prediction[0] == 1:
            st.markdown(f"""
                <div style='padding:20px; border-radius:10px; background-color:#d4edda'>
                    <h2 style='color:green;'>✅ Converted</h2>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='padding:20px; border-radius:10px; background-color:#f8d7da'>
                    <h2 style='color:red;'>❌ Not Converted</h2>
                    <h4>Confidence: {confidence:.2f}%</h4>
                </div>
            """, unsafe_allow_html=True)