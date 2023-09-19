from datetime import datetime, timedelta
import numpy as np

def future_predictions(X, model):
    today = datetime.now()
    future_dates = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 11)]

    X_future = X.tail(1).values
    future_predictions = []  # Store the predicted values

    for _ in range(10):
        y_pred_future = model.predict(X_future.reshape(1, -1, 1))
        future_predictions.append(float(y_pred_future[0, 0]))

        X_future = X_future[0, 1:]
        X_future = np.append(X_future, y_pred_future[0, 0])
        X_future = np.expand_dims(X_future, axis=0)

    return future_dates, future_predictions


