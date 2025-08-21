import numpy as np

from sklearn.model_selection import ParameterGrid

from modules.utils import generate, pretty_evaluate, precision, recall
from modules.stein import graph_inference
from modules.experiments.experiment import Experiment

class DASBoostExperiment(Experiment):
    def __init__(self, d_values, num_tests, s0, noise_type, graph_type, delta, k):
        self.d_values = d_values
        self.num_tests = num_tests
        self.s0 = s0
        self.noise_type = noise_type
        self.graph_type = graph_type
        self.delta = delta
        self.k = k
        self.output_file = f"../logs/exp/dasboost_{s0}_{d_values[-1]}.csv"
        self.logs = []
        self.columns = [
            'V', 'E', 'N', 'delta', 'cutoff', 'fn', 'fp', 'precision', 'recall', 'reversed', 'SHD', 'SID' , 'D_top', 'SCORE time [s]','Total time [s]'
        ]

    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'delta': self.delta, 'k': [self.k]}))

    def config_logs(self, run_logs, sid):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif self.columns[i] == 'delta':
                logs.append(round(m, 5))
            elif not sid and self.columns[i] == "SID":
                logs.append(None)
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        self.logs.append(logs)


    def dasboost(self, X, adj, eta_G, eta_H, delta, d, s0, N, sid, run_logs):
        """
        Run SCORE with DASBoost as pruning algorithm. Update logs
        """
        A_SCORE, top_order_SCORE, SCORE_time, tot_time =  graph_inference(X, eta_G, eta_H, pruning="DASBoost", delta=delta, K=self.k)
        fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, sid)
        precision_metric = precision(s0, fn, fp)
        recall_metric = recall(s0, fn)
        print(pretty_evaluate("DASBoost", delta, adj, A_SCORE, top_order_errors, SCORE_time, tot_time, sid, s0))
        run_logs.append([d, s0, N, delta, -1, fn, fp, precision_metric, recall_metric, rev, SHD, SID, top_order_errors, SCORE_time, tot_time])

        return A_SCORE, top_order_SCORE, SCORE_time, tot_time


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        delta = params['delta']
        s0 = self.set_s0(d)
        sid = self.compute_SID(d)

        run_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            X, adj = generate(d, s0, N, noise_type=self.noise_type, graph_type=self.graph_type, GP=True)
            self.dasboost(X, adj, eta_G, eta_H, delta, d, s0, N, sid, run_logs)

        self.config_logs(run_logs, sid)
        self.save_logs()
