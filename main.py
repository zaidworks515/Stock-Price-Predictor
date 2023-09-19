import numpy as np
from flask import Flask, render_template, request, send_file
import pandas as pd
from io import BytesIO
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from future import future_predictions  # Import the function from future.py
import webbrowser

app = Flask(__name__, static_folder='static')
app.config['STATIC_FOLDER'] = 'static'

X = None
y=None
model = None
uploaded_data=None
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1)) # for inverse scaling

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/instructions.html')
def examples():
    return render_template('instructions.html')

@app.route('/model_behavior', methods=['POST'])
def model_behavior():
    global X,y, model, uploaded_data, scaler, scaler_y

    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        uploaded_data = pd.read_csv(file)

        standardized_data = pd.DataFrame()
        standardized_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(
            uploaded_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])
        standardized_data['Date'] = pd.to_datetime(uploaded_data['Date'])  # Ensure 'Date' column is in datetime format
        standardized_data.set_index('Date', inplace=True)  # Set 'Date' as the index

        data_html = standardized_data.to_html()

        X = standardized_data[['Open', 'High', 'Low', 'Adj Close', 'Volume']]
        y = standardized_data['Close']

        model = Sequential()
        model.add(LSTM(128, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='rmsprop', loss='mae', metrics=['mse'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, shuffle=False)

        epochs = 0
        if len(standardized_data) < 2000:
            epochs = 16
        elif len(standardized_data) < 3000:
            epochs = 22
        elif len(standardized_data) < 3500:
            epochs = 30
        else:
            epochs = 40

        history = model.fit(X_train, y_train, epochs=epochs, validation_split=0.3, use_multiprocessing=True)
        y_pred = model.predict(X_test)

        scaler_y.fit(uploaded_data[['Close']])  # Fit scaler on original close prices
        y_test_original = scaler_y.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        y_pred_original = scaler_y.inverse_transform(y_pred).flatten()

        plt.figure(figsize=(12, 6))
        plt.plot(uploaded_data['Date'].iloc[-len(y_test):], y_test_original, label='Actual', color='blue')
        plt.plot(uploaded_data['Date'].iloc[-len(y_test):], y_pred_original, label='Predicted', color='green')
        plt.xlabel('Date')
        plt.ylabel('Close Prices')
        plt.title('Actual vs. Predicted Close Prices')
        plt.legend()
        plot_buffer = BytesIO()
        plt.savefig(plot_buffer, format='png')
        plot_buffer.seek(0)

        dates_list = standardized_data.index.strftime('%Y-%m-%d').tolist()

        actual_values = y_test_original.tolist()  #JSON serializable format
        predicted_values = y_pred_original.tolist()

        return render_template('results.html', data=data_html, actual_values=actual_values, predicted_values=predicted_values, dates=dates_list, plot_url=send_file(plot_buffer, mimetype='image/png'))

@app.route('/future.html')
def show_future():
    global X, model, scaler, uploaded_data, scaler_y
    future_dates, future_preds_scaled = future_predictions(X, model)
    future_preds_scaled = np.array(future_preds_scaled)
    future_preds_scaled = future_preds_scaled.reshape(-1, 1)
    future_preds_original = scaler_y.inverse_transform(future_preds_scaled).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_preds_original, label='Predicted', color='green')
    plt.xlabel('Date')
    plt.ylabel('Close Prices')
    plt.title('Future Predicted Close Prices')
    plt.legend()
    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format='png')
    plot_buffer.seek(0)

    return render_template('future.html', dates=future_dates, predicted_values=future_preds_original.tolist(), plot_url=send_file(plot_buffer, mimetype='image/png'))


if __name__ == "__main__":
    port = 5000
    url = f"http://localhost:{port}"
    webbrowser.open(url)    #To automatically open the browser with localhost
    app.run(debug=True, port=port, use_reloader=False)
