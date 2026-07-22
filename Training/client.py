import sys
import flwr as fl
import pandas as pd
import numpy as np

from sklearn.utils.class_weight import compute_class_weight

from model import create_model


if len(sys.argv) != 2:

    print(
        "Usage: python client.py <client_id>"
    )

    sys.exit()


CLIENT_ID = int(sys.argv[1])

print(
    f"\nStarting Client {CLIENT_ID}\n"
)


X = pd.read_csv(
    f"client_{CLIENT_ID}_X.csv"
).values


y = pd.read_csv(
    f"client_{CLIENT_ID}_y.csv"
).values.ravel()


classes = np.unique(y)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y
)

class_weights = dict(
    zip(classes, weights)
)


model = create_model(
    X.shape[1]
)


class FlowerClient(
    fl.client.NumPyClient
):

    def get_parameters(
        self,
        config
    ):
        return model.get_weights()

    def fit(
        self,
        parameters,
        config
    ):

        model.set_weights(
            parameters
        )

        model.fit(
            X,
            y,
            epochs=10,
            batch_size=128,
            verbose=1,
            class_weight=class_weights
        )

        return (
            model.get_weights(),
            len(X),
            {}
        )

    def evaluate(
        self,
        parameters,
        config
    ):

        model.set_weights(
            parameters
        )

        loss, acc, prec, rec, auc = model.evaluate(
            X,
            y,
            verbose=0
        )

        print(
            f"\nClient {CLIENT_ID}"
        )

        print(
            f"Accuracy : {acc:.4f}"
        )

        print(
            f"Precision: {prec:.4f}"
        )

        print(
            f"Recall   : {rec:.4f}"
        )

        print(
            f"AUC      : {auc:.4f}\n"
        )

        return (
            float(loss),
            len(X),
            {
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "auc": float(auc)
            }
        )


fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient()
)