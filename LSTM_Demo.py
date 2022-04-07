import glob
import pandas as pd
import numpy as np

df = pd.concat([pd.read_csv(f, header=0) for f in glob.glob("dataset/*.csv")])

dt = df["Date Time"] = pd.to_datetime(df["SETTLEMENTDATE"])
df["Day"], df["Month"], df["Year"], df["Hour"], df["Minute"] = (
    dt.dt.day,
    dt.dt.month,
    dt.dt.year,
    dt.dt.hour,
    dt.dt.minute,
)

df["Demand"] = pd.to_numeric(df["TOTALDEMAND"], errors="coerce")

df.drop(["REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE", "Date Time"], axis=1, inplace=True)

x, y = [], []

for i in range(0, df.shape[0] - 48):
    x.append(df.iloc[i : i + 48, 5])
    y.append(df.iloc[i + 48, 5])

x, y = np.array(x), np.array(y)

y = np.reshape(y, (len(y), 1))
x = np.delete(x, list(range(1, x.shape[1], 2)), axis=1)
x = np.delete(x, list(range(1, x.shape[0], 2)), axis=0)
y = np.delete(y, list(range(1, y.shape[0], 2)), axis=0)

pd.DataFrame(x).to_csv("appended_Demand.csv")
pd.DataFrame(y).to_csv("appended_Demand1.csv")

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

x = scaler.fit_transform(x)
y = scaler.fit_transform(y)

x_train, x_test = x[:-480], x[-480:]
y_train, y_test = y[:-480], y[-480:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_train.shape[1], 1))

from keras.models import Sequential
from keras.layers import Dense, Activation, CuDNNLSTM, LSTM
from keras import optimizers

model = Sequential()
model.add(CuDNNLSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(CuDNNLSTM(50, return_sequences=False))
model.add(Dense(50, activation="relu"))
model.add(Dense(1))

from keras.callbacks import ModelCheckpoint, EarlyStopping

filepath = "models/{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{mae:.4f}.hdf5"
callbacks = [
    EarlyStopping(monitor="val_loss", patience=50),
    ModelCheckpoint(filepath, monitor="loss", save_best_only=True, mode="min"),
]

optimizers.adam_v2.Adam(learning_rate=0.0001)
model.compile(optimizer="adam", loss="mse", metrics=["mae"])
model.fit(x_train, y_train, validation_split=0.2, epochs=10, callbacks=callbacks, batch_size=8)
