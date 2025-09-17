import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import os

st.set_page_config(page_title="Netflix Stock Price Prediction", layout="wide")

st.title("Netflix (NFLX) Stock Price Prediction App")

# Sidebar
st.sidebar.header("Upload Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload Netflix stock data (CSV)", type=["csv"])
model_path = st.sidebar.text_input("Keras Model Path", value="keras_model.h5")
time_step = st.sidebar.number_input("Time step (lookback days)", min_value=1, max_value=365, value=60)

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    # Convert Date column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    st.subheader("Netflix Stock Data")
    st.write(df.head())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    # Plot closing price
    st.subheader("Closing Price vs Time")
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(df['Close'], label='Closing Price')
    ax.legend()
    st.pyplot(fig)

    # Preprocessing
    scaler = MinMaxScaler(feature_range=(0,1))
    close_data = df['Close'].values.reshape(-1,1)
    scaled_close = scaler.fit_transform(close_data)

    # Train/test split
    split_index = int(len(scaled_close)*0.7)
    train = scaled_close[:split_index]
    test = scaled_close[split_index - time_step:]

    def create_sequences(data, time_step):
        X, y = [], []
        for i in range(time_step, len(data)):
            X.append(data[i-time_step:i, 0])
            y.append(data[i, 0])
        return np.array(X).reshape(-1, time_step, 1), np.array(y)

    X_test, y_test = create_sequences(test, time_step)

    # Load model and predict
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success(f"Model loaded: {model_path}")

        y_pred = model.predict(X_test)
        y_pred_rescaled = scaler.inverse_transform(y_pred)
        y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

        # Plot predictions
        st.subheader("Predicted vs Actual Prices (Test Set)")
        fig2, ax2 = plt.subplots(figsize=(12,5))
        ax2.plot(y_test_rescaled, label='Actual Price')
        ax2.plot(y_pred_rescaled, label='Predicted Price')
        ax2.legend()
        st.pyplot(fig2)

        # Errors
        mae = np.mean(np.abs(y_pred_rescaled - y_test_rescaled))
        st.write(f"**Mean Absolute Error (MAE):** {mae:.4f}")

        mse= np.mean((y_pred_rescaled - y_test_rescaled)**2)
        st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")

        rmse = np.sqrt(mse)
        st.write(f"**Root Mean Squared Error (RMSE):** {rmse:.4f}")

        r_squared = 1 - (np.sum((y_test_rescaled - y_pred_rescaled)**2) / np.sum((y_test_rescaled - np.mean(y_test_rescaled))**2))
        st.write(f"**R-squared:** {r_squared:.4f}")

        # Future predictions
        st.subheader("Predict Future Prices")
        predict_days = st.number_input("Days to predict into the future", min_value=1, max_value=365, value=30)

        if st.button("Predict Future"):
            last_seq = scaled_close[-time_step:].reshape(1, time_step, 1)
            future_preds = []
            seq = last_seq.copy()
            for _ in range(predict_days):
                next_pred = model.predict(seq)[0,0]
                future_preds.append(next_pred)
                seq = np.append(seq[:,1:,:], [[[next_pred]]], axis=1)
            future_preds_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

            st.write(pd.DataFrame(future_preds_rescaled, columns=["Predicted Price"]))

            fig3, ax3 = plt.subplots(figsize=(12,5))
            ax3.plot(future_preds_rescaled, marker='o')
            ax3.set_title("Future Predicted Prices")
            st.pyplot(fig3)
    else:
        st.warning("Please provide a valid model file (keras_model.h5)")
else:
    st.warning("Please upload Netflix stock data (NFLX.csv)")

