from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data.repo_dataset import get_public_dataset

train_data, _, test_data = get_public_dataset("nyc_taxi")

for data in [train_data, test_data]:
  data.roll(lookback=100, horizon=1)

# training
forecaster = TCNForecaster.from_tsdataset(train_data)
forecaster.fit(train_data)

# prediction
prediction = forecaster.predict(test_data)

# evaluation
mse, smape = forecaster.evaluate(test_data, metrics=["mse", "smape"])
print("Evaluate: the mean square error is: ", mse)
print("Evaluate: the smape value is: ", smape)

