import numpy as np
import pandas as pd
import time

from sklearn.model_selection import ParameterGrid
import torch, os
from pathlib import Path
from modules.utils import generate, pretty_evaluate, recall, precision
from modules.stein import cam_pruning
from modules.experiments.dasboost_experiment import DASBoostExperiment


def true_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if pred_adj[i][j] != 0 and true_adj[i][j] != 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res

def spur_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if pred_adj[i][j] != 0 and true_adj[i][j] == 0 and true_adj[j][i] == 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res

def fals_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if pred_adj[i][j] != 0 and true_adj[i][j] == 0 and true_adj[j][i] != 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res

def miss_edge(true_adj, pred_adj):
    d, _ = pred_adj.shape
    res = []
    for i in range(d):
        for j in range(d):
            if true_adj[i][j] != 0 and pred_adj[i][j] == 0 and pred_adj[j][i] == 0:
                if ([f'X{i+1}', f'X{j+1}'] not in res) and ([f'X{j+1}', f'X{i+1}'] not in res):
                    res.append([f'X{i+1}', f'X{j+1}'])
    return res

def evaluate(groundtruth, adj_mtx):
    etrue = true_edge(groundtruth, adj_mtx)
    espur = spur_edge(groundtruth, adj_mtx)
    efals = fals_edge(groundtruth, adj_mtx)
    emiss = miss_edge(groundtruth, adj_mtx)

    return len(etrue), len(espur), len(emiss), len(efals)


def load_data():
    folderpath = "../../../data/distributed/insurance/m3_d1_n10"
    dagpath = "../../../data/distributed/insurance/adj.txt"
    groundtruth = np.loadtxt(dagpath)

    silos = []
    if not Path(folderpath).exists():
        print("Folder", folderpath, "not exist!")
        exit()
    
    for file in sorted(os.listdir(folderpath)):
        filename = os.path.join(folderpath, file)
        silo_data = pd.read_csv(filename)
        silos.append(silo_data)

    data = pd.concat(silos, axis=0).sample(2000).to_numpy()
    # data = (data - data.mean(axis=0, keepdims=True))/(data.std(axis=0, keepdims=True))
    data = data + 0.1 * np.random.randn(data.shape[0],data.shape[1])
    return torch.from_numpy(data) *1., groundtruth


class DASExperiment(DASBoostExperiment):
    def __init__(self, d_values, num_tests, s0, noise_type, graph_type, delta, cam_cutoff, k):
        super().__init__(d_values, num_tests, s0, noise_type, graph_type, delta, k)
        
        self.cam_cutoff = cam_cutoff
        if k is None:
            self.dasboost_output = f"../logs/das/ER/dasboost_{s0}_{d_values[-1]}.csv"
            self.das_output = f"../logs/das/ER/das_{s0}_{d_values[-1]}.csv"
        else:
            self.dasboost_output = f"../logs/das/ER/dasboost_{k}_{s0}_{d_values[-1]}.csv"
            self.das_output = f"../logs/das/ER/das_{k}_{s0}_{d_values[-1]}.csv"

        self.dasboost_logs = []
        self.das_logs = []

    def get_params(self):
        return list(ParameterGrid({'d': self.d_values, 'delta': self.delta, 'k': [self.k]}))

    def save_logs(self, logtype):
        if logtype=="dasboost":
            df = pd.DataFrame(self.dasboost_logs, columns=self.columns)
            df.to_csv(self.dasboost_output)
        else:
            df = pd.DataFrame(self.das_logs, columns=self.columns)
            df.to_csv(self.das_output)
            

    def config_logs(self, run_logs, sid, logtype):
        mean_logs = np.mean(run_logs, axis=0)
        std_logs = np.std(run_logs, axis=0)
        logs = []
        for i in range(len(self.columns)):
            m = mean_logs[i]
            s = std_logs[i]
            if self.columns[i] in ["V", "E", "N"]:
                logs.append(f"{int(m)}")
            elif self.columns[i] in ['delta', 'cutoff']:
                logs.append(round(m, 5))
            elif not sid and self.columns[i] == "SID":
                logs.append(None)
            else:
                logs.append(f"{round(m, 2)} +- {round(s, 2)}")
        
        if logtype == "dasboost":
            self.dasboost_logs.append(logs)
        else:
            self.das_logs.append(logs)


    def das(self, X, adj, delta, d, s0, N, A_SCORE, top_order_SCORE, SCORE_time, dasboost_time, sid, run_logs):
        """
        Apply CAM pruning to adjacency matrix found by DASBoost pruning. Update logs
        """
        start = time.time()
        A_SCORE = cam_pruning(A_SCORE, X, self.cam_cutoff)
        das_time = dasboost_time + (time.time() - start)

        fn, fp, rev, SHD, SID, top_order_errors = self.metrics(A_SCORE, adj, top_order_SCORE, sid)
        precision_metric = precision(s0, fn, fp)
        recall_metric = recall(s0, fn)
        # print(pretty_evaluate("DAS", delta, adj, A_SCORE, top_order_errors, SCORE_time, das_time, sid, s0, K=self.k))
        run_logs.append([d, s0, N, delta, self.cam_cutoff, fn, fp, precision_metric, recall_metric, rev, SHD, SID, top_order_errors, SCORE_time, das_time])


    def run_config(self, params, N, eta_G, eta_H):
        d = params['d']
        delta = params['delta']
        s0 = self.set_s0(d)
        sid = self.compute_SID(d)

        dasboost_logs = []
        das_logs = []
        for k in range(self.num_tests):
            print(f"Iteration {k+1}/{self.num_tests}")
            # X, adj = generate(d, s0, N, noise_type=self.noise_type, graph_type = self.graph_type, GP=True)
            X, adj = load_data()
            # print(X)
            # exit()
            A_SCORE, top_order_SCORE, SCORE_time, tot_time = self.dasboost(X, adj, eta_G, eta_H, delta, d, s0, N, sid, dasboost_logs)
            self.das(X, adj, delta, d, s0, N, A_SCORE, top_order_SCORE, SCORE_time, tot_time, sid, das_logs)

            print("EVAL:", evaluate(adj, A_SCORE))
        # self.config_logs(dasboost_logs, sid, "dasboost")
        # self.save_logs("dasboost")

        # self.config_logs(das_logs, sid, "das")
        # self.save_logs("das")
