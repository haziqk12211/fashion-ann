import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

with open("params.yaml") as f:
    params = yaml.safe_load(f)["evaluate"]

batch_size = params["batch_size"]

class_names = [
    "T-shirt", "Trouser", "Pullover", "Dress",   "Coat",
    "Sandal",  "Shirt",   "Sneaker",  "Bag",      "Boot"
]

model  = tf.keras.models.load_model("models/model.h5")
X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy")

test_loss, test_acc = model.evaluate(X_test, y_test,
                                     batch_size=batch_size,
                                     verbose=0)

y_pred_prob = model.predict(X_test, batch_size=batch_size, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=class_names,
    cmap="Blues",
    ax=ax
)

plt.title("Confusion matrix")
plt.tight_layout()
os.makedirs("metrics", exist_ok=True)
plt.savefig("metrics/confusion_matrix.png", dpi=300)
plt.show()
plt.close()

report = classification_report(
    y_test, y_pred,
    target_names = class_names,
    output_dict  = True
)
metrics = {
    "test_loss"    : round(float(test_loss), 4),
    "test_accuracy": round(float(test_acc),  4),
    "per_class"    : {
        class_names[i]: {
            "precision": round(report[class_names[i]]["precision"], 4),
            "recall"   : round(report[class_names[i]]["recall"],    4),
            "f1_score" : round(report[class_names[i]]["f1-score"],  4)
        }
        for i in range(10)
    }
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
