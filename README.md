# 🚢 Titanic Survival Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/user/mohammed-zakir-ahmed)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression-orange)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)

A web app that predicts whether a Titanic passenger would have survived, and shows the survival probability.

👉 **[Try the live app](https://titanic-survival-logistic-reg-app.streamlit.app/)**


## 🎯 What it does
Enter a passenger's details, click **Predict**, and get:
- ✅ / ❌ Survived or not survived
- 📊 Survival probability
- 🧠 A **How it works** tab that shows the model's weights and the step-by-step calculation

## 🧾 Inputs
| Input | Description |
|---|---|
| `pclass` | Ticket class (1, 2 or 3) |
| `age` | Passenger age |
| `fare` | Ticket fare |
| `sex_male` | 1 = male, 0 = female |

## 🧪 Try these
| Passenger | Expected result |
|---|---|
| Female, 1st class, age 25, fare 100 | High survival chance |
| Male, 3rd class, age 40, fare 8 | Low survival chance |

<details>
<summary>🔧 Data preparation (click to open)</summary>

- Dataset: Titanic (loaded from seaborn)
- Missing `age` values filled with the median age
- `sex` converted to `sex_male` using one-hot encoding
- Features used: pclass, age, fare, sex_male
- Model: Logistic Regression (scikit-learn)
- Accuracy: 81%

</details>

<details>
<summary>📁 Project files (click to open)</summary>

- `app.py`: Streamlit interface
- `model.pkl`: trained model
- `requirements.txt`: libraries needed
- `Titanic Logistic Regression.ipynb`: training notebook

</details>

## 🛠️ Tech stack
Python · pandas · scikit-learn · Streamlit

## 👤 Author
**ZAKIR AHMED**
· [LinkedIn](https://www.linkedin.com/in/zakir00)
