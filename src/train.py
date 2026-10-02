import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

with open("params.yaml") as f:
    params = yaml.safe_load(f)["train"]

units         = params["units"]
dropout       = params["dropout"]
learning_rate = params["learning_rate"]
epochs        = params["epochs"]
batch_size    = params["batch_size"]
random_seed   = params["random_seed"]

tf.random.set_seed(random_seed)
np.random.seed(random_seed)

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy")
X_val   = np.load("data/processed/X_val.npy")
y_val   = np.load("data/processed/y_val.npy")

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(units, activation="relu"),
    Dropout(dropout),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer = Adam(learning_rate=learning_rate),
    loss      = "sparse_categorical_crossentropy",
    metrics   = ["accuracy"]
)

#model.summary()

history = model.fit(
    X_train, y_train,
    epochs          = epochs,
    batch_size      = batch_size,
    validation_data = (X_val, y_val),
    verbose         = 1
)

os.makedirs("models", exist_ok=True)

model.save("models/model.h5")

history_df = pd.DataFrame(history.history)
history_df.to_csv("models/history.csv", index=False)