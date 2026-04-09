import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.title("Student Performance ML Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a CSV file")
    st.stop()

st.write(df.head())

st.header("EDA")
st.write(df.describe())
st.write(df.isnull().sum())

fig, ax = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

df = df.fillna(df.mean(numeric_only=True))

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

st.write(df.head())

target = st.selectbox("Select Target Variable", df.columns)

X = df.drop(columns=[target])
y = df[target]

test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model_choice = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest"]
)

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor()

if st.button("Train Model"):
    model.fit(X_train, y_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    st.write(cv_scores)
    st.write(np.mean(cv_scores))

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(mse)
    st.write(r2)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred)
    st.pyplot(fig2)