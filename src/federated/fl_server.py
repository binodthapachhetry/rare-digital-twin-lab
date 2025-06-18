# src/federated/fl_server.py
import flwr as fl, json, pathlib

history = {"round": [], "mae": []}
def fit_config(rnd): return {}
def eval_config(rnd): return {}

class SaveHistory(fl.server.strategy.FedAvg):
    def aggregate_evaluate(self, rnd, res, failures):
        global history
        mae = super().aggregate_evaluate(rnd, res, failures)[0]
        history["round"].append(rnd); history["mae"].append(mae)
        pathlib.Path("output").mkdir(exist_ok=True)
        json.dump(history, open("output/flwr_history.json","w"))
        return mae, {}

def start_server():
    strategy = SaveHistory(fraction_fit=1.0,
                           fraction_evaluate=1.0,
                           min_fit_clients=3,
                           min_evaluate_clients=3,
                           min_available_clients=3,
                           on_fit_config_fn=fit_config,
                           on_evaluate_config_fn=eval_config)
    fl.server.start_server(server_address="127.0.0.1:8080",
                           config=fl.server.ServerConfig(num_rounds=5),
                           strategy=strategy)
