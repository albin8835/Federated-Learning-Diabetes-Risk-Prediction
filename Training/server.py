import flwr as fl
import pandas as pd
import numpy as np

from flwr.common import parameters_to_ndarrays

from model import create_model


class SaveModelStrategy(
    fl.server.strategy.FedAvg
):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures
    ):

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated is not None:

            parameters_aggregated, metrics = aggregated

            if server_round == 10:

                print(
                    "\nSaving Final Global Model..."
                )

                X = pd.read_csv(
                    "client_1_X.csv"
                ).values

                model = create_model(
                    X.shape[1]
                )

                weights = parameters_to_ndarrays(
                    parameters_aggregated
                )

                model.set_weights(
                    weights
                )

                model.save(
                    "global_model.keras"
                )

                print(
                    "global_model.keras saved"
                )

        return aggregated


strategy = SaveModelStrategy(

    fraction_fit=1.0,

    fraction_evaluate=1.0,

    min_fit_clients=4,

    min_evaluate_clients=4,

    min_available_clients=4
)

fl.server.start_server(

    server_address="127.0.0.1:8080",

    config=fl.server.ServerConfig(
        num_rounds=10
    ),

    strategy=strategy
)