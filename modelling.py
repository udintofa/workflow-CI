import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import joblib

if __name__ == "__main__":
    # Load dataset hasil preprocessing
    train = pd.read_csv("preprocessed_data/train_processed.csv")
    test = pd.read_csv("preprocessed_data/test_processed.csv")

    # Ganti dengan nama kolom targetmu
    target = "Cholesterol Total (mg/dL)"

    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    # Buat model regresi
    model = LinearRegression()

    # Mulai experiment run
    with mlflow.start_run(run_name="baseline_linear_regression"):
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Hitung metrik
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred) * 100

        # Log manual juga (meskipun autolog sudah simpan)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

    print("Training selesai! Jalankan 'mlflow ui' lalu cek di browser http://127.0.0.1:5000")
