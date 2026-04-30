# ------------------ INSTALL ------------------
# pip install streamlit pandas numpy xgboost scikit-learn

# ------------------ IMPORTS ------------------
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AFL Predictor", layout="wide")

st.title("🏉 AFL Match Outcome Predictor (Machine Learning)")
st.markdown("""
This model predicts AFL match outcomes using historical match data,  
team performance trends, venue, and weather conditions.
""")

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/mizzony/AFL/main/afl_data.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df['Attendance'] = df['Attendance'].str.replace(',', '').astype(float)
    df['Rainfall'] = df['Rainfall'].fillna(df['Rainfall'].median())

    df = df[(df['HomeTeamScore'] >= 0) & (df['AwayTeamScore'] >= 0)]

    # Feature engineering
    recent = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=180)]
    home_avg = recent.groupby('HomeTeam')['HomeTeamScore'].mean()
    away_avg = recent.groupby('AwayTeam')['AwayTeamScore'].mean()

    df['HomeTeam_PastAvgPoints'] = df['HomeTeam'].map(home_avg).fillna(0)
    df['AwayTeam_PastAvgPoints'] = df['AwayTeam'].map(away_avg).fillna(0)

    return df, home_avg, away_avg

data, home_avg, away_avg = load_data()

# ------------------ MODEL ------------------
X = data[['HomeTeam','Year','Rainfall','Venue',
          'HomeTeam_PastAvgPoints','AwayTeam','AwayTeam_PastAvgPoints']]
y = data['Win']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

label_encoders = {}
for col in X_train.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])
    label_encoders[col] = le

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    seed=42,
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# ------------------ MODEL PERFORMANCE ------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

#st.sidebar.metric("Model Accuracy", f"{acc*100:.2f}%")

# ------------------ UI ------------------
st.header("🔮 Predict Match Outcome")

col1, col2, col3 = st.columns(3)

with col1:
    home_team = st.selectbox("Home Team", sorted(data['HomeTeam'].unique()))

with col2:
    away_team = st.selectbox("Away Team", sorted(data['AwayTeam'].unique()), index=1)

with col3:
    venue = st.selectbox("Venue", sorted(data['Venue'].unique()))

rain = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
year = st.number_input("Year", 2020, 2026, 2024)

# ------------------ PREDICTION ------------------
if st.button("Predict"):
    try:
        input_df = pd.DataFrame({
            'HomeTeam':[label_encoders['HomeTeam'].transform([home_team])[0]],
            'Year':[year],
            'Rainfall':[rain],
            'Venue':[label_encoders['Venue'].transform([venue])[0]],
            'HomeTeam_PastAvgPoints':[home_avg.get(home_team, 0)],
            'AwayTeam':[label_encoders['AwayTeam'].transform([away_team])[0]],
            'AwayTeam_PastAvgPoints':[away_avg.get(away_team, 0)]
        })

        proba = model.predict_proba(input_df)[0][1]

        if proba > 0.5:
            st.success(f"🏡 {home_team} likely to WIN ({proba*100:.1f}%)")
        else:
            st.success(f"🚶 {away_team} likely to WIN ({(1-proba)*100:.1f}%)")

    except Exception as e:
        st.error(f"Prediction error: {e}")
