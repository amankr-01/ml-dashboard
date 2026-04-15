import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 Student Performance ML Dashboard")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Upload a dataset to proceed")
    st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Summary")
    st.write(df.describe())

with col2:
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

numeric_df = df.select_dtypes(include=np.number)

st.subheader("Correlation Heatmap")
fig, ax = plt.subplots()
ax.imshow(numeric_df.corr())
ax.set_xticks(range(len(numeric_df.columns)))
ax.set_yticks(range(len(numeric_df.columns)))
ax.set_xticklabels(numeric_df.columns, rotation=90)
ax.set_yticklabels(numeric_df.columns)
st.pyplot(fig)

df = df.fillna(df.mean(numeric_only=True))

le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

st.sidebar.header("Model Settings")

target = st.sidebar.selectbox("Target Variable", df.columns)

test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.2)

model_choice = st.sidebar.selectbox(
    "Model",
    ["Linear Regression", "Random Forest"]
)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

if model_choice == "Linear Regression":
    model = LinearRegression()
else:
    model = RandomForestRegressor()

if st.sidebar.button("Train Model"):

    model.fit(X_train, y_train)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    st.subheader("K-Fold Validation")
    st.write(cv_scores)
    st.write("Average Score:", np.mean(cv_scores))

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Performance Metrics")

    m1, m2 = st.columns(2)
    m1.metric("MSE", round(mse, 4))
    m2.metric("R² Score", round(r2, 4))

    st.subheader("Actual vs Predicted")

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)
