import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write("This app uses a **Logistic Regression** model trained on the "
             "Titanic dataset to predict whether a passenger would have survived.")
    st.write("Built with Python, scikit-learn and Streamlit.")
    st.caption("Made by ZAKIR AHMED")

# ---------- Main page ----------
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger details, then click **Predict**.")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("🎫 Ticket class", [1, 2, 3],
                          format_func=lambda x: {1: "1st class", 2: "2nd class", 3: "3rd class"}[x])
    sex = st.radio("🧑 Sex", ["Male", "Female"], horizontal=True)

with col2:
    age = st.slider("🎂 Age", 1, 80, 30)
    fare = st.slider("💰 Ticket fare (£)", 0, 512, 32)

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    X = pd.DataFrame(
        [[pclass, age, fare, 1 if sex == "Male" else 0]],
        columns=["pclass", "age", "fare", "sex_male"],
    )

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    if pred == 1:
        st.success("✅ This passenger would likely SURVIVE")
        st.balloons()
    else:
        st.error("❌ This passenger would likely NOT survive")

    st.metric("Survival probability", f"{prob * 100:.1f}%")
    st.progress(int(prob * 100))