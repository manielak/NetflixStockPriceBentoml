from pathlib import Path
import sys
import pandas as pd
import bentoml
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.metrics import mean_squared_error
from dvclive import Live

if __name__ == "__main__":
    train_data_path = sys.argv[1]
    test_data_path = sys.argv[2]
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    train_data = train_data.rename(columns={"Date": "ds", "Close": "y"})

    model = Prophet()
    model.fit(train_data)

    future = test_data[["Date"]].copy().rename(columns={"Date": "ds"})
    forecast = model.predict(future)

    actual_data = test_data["Close"].values
    predicted_data = forecast["yhat"].values

    with Live() as live:
        mse = mean_squared_error(actual_data, predicted_data)
        print(f"Mean Squared Error: {mse}")
        live.log_metric("mse", mse)

    model_path = Path("model")
    model_path.mkdir(parents=True, exist_ok=True)

    with open(model_path / "model.json", "w") as f:
        json_model = model_to_json(model)
        f.write(json_model)

    bentoml.picklable_model.save_model(
        "NetflixModel",
        model,
        signatures={"predict": {"batchable": False}})