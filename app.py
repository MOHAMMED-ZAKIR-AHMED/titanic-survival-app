import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

FEATURES = ["pclass", "age", "fare", "sex_male"]
NICE_NAMES = {
    "pclass": "Ticket class",
    "age": "Age",
    "fare": "Fare",
    "sex_male": "Sex (male)",
}

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.write("This app uses a **Logistic Regression** model trained on the "
             "Titanic dataset to predict whether a passenger would have survived.")
    st.write("Built with Python, scikit-learn and Streamlit.")
    st.caption("Built by ZAKIR AHMED")

# ---------- Title and inputs ----------
st.title("🚢 Titanic Survival Predictor")
st.write("Enter the passenger details, then explore the prediction and how the model works.")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("🎫 Ticket class", [1, 2, 3],
                          format_func=lambda x: {1: "1st class", 2: "2nd class", 3: "3rd class"}[x])
    sex = st.radio("🧑 Sex", ["Male", "Female"], horizontal=True)
with col2:
    age = st.slider("🎂 Age", 1, 80, 30)
    fare = st.slider("💰 Ticket fare (£)", 0, 512, 32)

X = pd.DataFrame(
    [[pclass, age, fare, 1 if sex == "Male" else 0]],
    columns=FEATURES,
)

st.divider()

tab1, tab2 = st.tabs(["🔮 Prediction", "🧠 How it works"])

# ---------- Tab 1: Prediction ----------
with tab1:
    if st.button("Predict", type="primary", use_container_width=True):
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        if pred == 1:
            st.success("✅ This passenger would likely SURVIVE")
            st.balloons()
        else:
            st.error("❌ This passenger would likely NOT survive")

        st.metric("Survival probability", f"{prob * 100:.1f}%")
        st.progress(int(prob * 100))
    else:
        st.info("Click **Predict** to see the result. Then open the **How it works** tab.")

# ---------- Tab 2: How it works ----------
with tab2:
    st.subheader("The big picture")
    st.markdown(
        """
        1. **Input:** you enter 4 details (class, age, fare, sex).
        2. **Score:** the model multiplies each input by a *weight* it learned
           from real Titanic passengers, adds them up, and adds a starting value.
        3. **Probability:** the score is squeezed into a number between 0 and 1
           using the *sigmoid* function.
        4. **Decision:** if the probability is 50% or more, the model says **Survived**,
           otherwise **Not survived**.
        """
    )

    st.latex(r"z = b + w_1 \cdot class + w_2 \cdot age + w_3 \cdot fare + w_4 \cdot sex")
    st.latex(r"P(\text{survive}) = \frac{1}{1 + e^{-z}}")

    # Learned weights from the saved model
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])

    st.subheader("1. What the model learned")
    st.write("These weights were learned during training in Colab. "
             "A **positive** weight raises survival chances, a **negative** weight lowers them.")

    weights_df = pd.DataFrame({
        "Feature": [NICE_NAMES[f] for f in FEATURES],
        "Weight": np.round(coefs, 4),
        "Effect": ["⬆️ increases survival" if c > 0 else "⬇️ decreases survival" for c in coefs],
    })
    st.dataframe(weights_df, hide_index=True, use_container_width=True)
    st.caption(f"Starting value (intercept) b = {intercept:.4f}")

    st.subheader("2. Calculation for the passenger you selected")
    values = X.iloc[0].values
    contrib = coefs * values

    calc_df = pd.DataFrame({
        "Feature": [NICE_NAMES[f] for f in FEATURES],
        "Your input": values,
        "Weight": np.round(coefs, 4),
        "Contribution (input × weight)": np.round(contrib, 3),
    })
    st.dataframe(calc_df, hide_index=True, use_container_width=True)

    z = intercept + contrib.sum()
    prob = 1 / (1 + np.exp(-z))

    c1, c2 = st.columns(2)
    c1.metric("Total score (z)", f"{z:.3f}")
    c2.metric("Probability after sigmoid", f"{prob * 100:.1f}%")
    st.caption("This matches the probability shown in the Prediction tab.")

    st.subheader("3. What pushed the prediction up or down")
    chart_df = pd.DataFrame({
        "Contribution": np.append(contrib, intercept),
    }, index=[NICE_NAMES[f] for f in FEATURES] + ["Starting value (b)"])
    st.bar_chart(chart_df)
    st.caption("Bars above zero push towards survival, bars below zero push towards not surviving.")

    st.subheader("About the model")
    st.markdown(
        """
        - **Algorithm:** Logistic Regression (scikit-learn)
        - **Dataset:** Titanic (from seaborn)
        - **Data preparation:** missing ages filled with the median, `sex` converted to
          `sex_male` (1 = male, 0 = female)
        - **Limits:** only 4 inputs are used, so it is a simple model for learning
          purposes, not a perfect predictor.
        """
    )
st.divider()
st.caption("Built by ZAKIR AHMED | [GitHub](https://github.com/MOHAMMED-ZAKIR-AHMED/) | [LinkedIn](https://www.linkedin.com/in/zakir00)")
