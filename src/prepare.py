import os
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

np.random.seed(42)

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

os.makedirs("data/raw", exist_ok=True)

np.save("data/raw/X_train.npy", X_train)
np.save("data/raw/y_train.npy", y_train)
np.save("data/raw/X_test.npy",  X_test)
np.save("data/raw/y_test.npy",  y_test)

print("Saved raw arrays to data/raw/")
print("  data/raw/X_train.npy")
print("  data/raw/y_train.npy")
print("  data/raw/X_test.npy")
print("  data/raw/y_test.npy")