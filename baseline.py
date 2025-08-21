import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
from utils.utils import count_accuracy
import bnlearn as bn
from causallearn.utils.cit import fisherz , chisq
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
import time
from cdt.causality.graph import GIES
from SCORE.src.modules.algorithms.cd import DAS, SCORE
from CausalDisco.baselines import var_sort_regress, r2_sort_regress
from castle.algorithms import ICALiNGAM, DirectLiNGAM, GOLEM, GraNDAG, Notears, NotearsNonlinear, NotearsLowRank
from castle.algorithms import PC 
from castle.common.independence_tests import CITest

os.environ['CASTLE_BACKEND'] = "pytorch"

def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.csv") # Must have headers
    parser.add_argument("--groundtruth_path", type=str, default="graph.csv") # Must have headers, same as the one above
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--datatype", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--routine", type=str, choices=["PC", "GES", "GIES", 
                                                            "Notears", "NotearsNonlinear", "NotearsLowRank", "GOLEM",
                                                            "DAS", "SCORE",
                                                            "Varsort", "R2sort",
                                                            "ICALiNGAM", "DirectLiNGAM",
                                                            "GraNDAG"], default="PC")
    
    options = vars(parser.parse_args())
    return options


def load_data(options):
    data_df = pd.read_csv(options["data_path"])
    groundtruth = pd.read_csv(options["groundtruth_path"]).to_numpy()
    
    if not Path(options["output"]).exists():
        f = open(options["output"], "w")
        f.write("{},{},{},{},{},{},{},{},{}\n".format('data_path', 'routine', "P", "R", "F1", "S", "shd", "time", "CIT"))
        f.close()
        
    return data_df, groundtruth # must be (pd.df,np.ndarray, np.ndarray)


def reformat_causal_graph(cg_graph):
    d = cg_graph.G.graph.shape[0]
    new_cg = np.zeros((d, d), dtype=int)

    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            
            if cg_graph.G.graph[j, i] == 1 and cg_graph.G.graph[i, j] == -1:
                new_cg[i, j] = 1
                new_cg[j, i] = 0

            elif (cg_graph.G.graph[i, j] == -1 and cg_graph.G.graph[j, i] == -1) or \
                 (cg_graph.G.graph[i, j] == 1 and cg_graph.G.graph[j, i] == 1):
                new_cg[i, j] = -1
                new_cg[j, i] = 0
            else:
                new_cg[i, j] = 0
                new_cg[j, i] = 0
                
    new_cg_graph = cg_graph
    new_cg_graph.G.graph = new_cg
    return new_cg, new_cg_graph     #new_cg is np.array and new_cg_graph is Graph obj 



if __name__ == "__main__":
    options = read_opts()
    print("Running:", options)
    data_df, groundtruth = load_data(options)
    d = len(data_df.columns)
    
    for _ in range(options['repeat']):
        start = time.time()
        
        if options['routine'] == "PC":
            if options['datatype'] == "continuous":
                cg = pc(data=data_df.to_numpy(), alpha=0.3, indep_test='fisherz')
                adj_mtx, _ = reformat_causal_graph(cg)
            else:
                cg = pc(data=data_df.to_numpy(), alpha=0.05, indep_test='chisq')
                adj_mtx, _ = reformat_causal_graph(cg)
            
        
        elif options["routine"] == "FCI":
            g, edges = fci(data_df.to_numpy())
            adj_mtx , _ = reformat_causal_graph(g)
        
        
        elif options["routine"] == "GES":
            model = bn.structure_learning.fit(data_df, methodtype='hc', verbose=0)
            adj_mtx = model['adjmat'].to_numpy() * 1.0      # type:ignore
            
            
        elif options['routine'] == "GIES":
            obj = GIES()
            output = obj.predict(data_df)
            adj_mtx = np.zeros([d, d])
            for edge in output.edges:     # type:ignore
                source, target = edge
                source_id = int(source[1:]) - 1
                target_id = int(target[1:]) - 1
                adj_mtx[source_id][target_id] = 1
                if adj_mtx[target_id][source_id] == 1:
                    adj_mtx[source_id][target_id] = 0
                    adj_mtx[target_id][source_id] = 0
        
        
        # elif options['routine'] == "DAS":
        #     algorithm = DAS(data_df.to_numpy(), kwargs={'d': d, 'eta_G': 0.01, 'eta_H': 0.01, 
        #                                             'cam_cutoff': 0.01, 'pruning': 'CAM', 'threshold': 0.01, 'K': 10})
        #     adj_mtx = algorithm.inference()
        #     adj_mtx = (adj_mtx > 0) * 1.
            
        
        elif options['routine'] == "SCORE":
            algorithm = SCORE(data_df.to_numpy(), kwargs={'d': d, 'eta_G': 0.01, 'eta_H': 0.01, 
                                                    'cam_cutoff': 0.01, 'pruning': 'DAS', 'threshold': 0.01, 'pns': 10})
            adj_mtx = algorithm.inference()
            adj_mtx = (adj_mtx > 0) * 1.
            
            
        elif options['routine'] == "Varsort":
            adj_mtx = 1.0 * (var_sort_regress(data_df.to_numpy())!=0)
            
            
        elif options['routine'] == "R2sort":
            adj_mtx = 1.0 * (r2_sort_regress(data_df.to_numpy())!=0)
            
        
        if options['routine'] == 'ICALiNGAM':
            model = ICALiNGAM(thresh=0.3)
            model.learn(data_df.to_numpy())
            adj_mtx = model.causal_matrix
            
        
        if options['routine'] == 'Notears':
            model = Notears(w_threshold=0.3)
            model.learn(data_df.to_numpy())
            adj_mtx = model.causal_matrix
            
        
        if options['routine'] == 'NotearsNonlinear':
            model = NotearsNonlinear(w_threshold=0.3, device_type="gpu")
            model.learn(data_df.to_numpy())
            adj_mtx = model.causal_matrix
            
        
        if options['routine'] == 'NotearsLowRank':
            model = NotearsLowRank(w_threshold=0.3)
            model.learn(data_df.to_numpy(), rank=int(d/10))
            adj_mtx = model.causal_matrix
            
            
        if options['routine'] == 'DirectLiNGAM':
            model = DirectLiNGAM(thresh=0.3)
            model.learn(data_df.to_numpy())
            adj_mtx = model.causal_matrix
            
            
        if options['routine'] == 'GraNDAG':
            model = GraNDAG(input_dim=data_df.shape[1], iterations=1000, lr=0.005, device_type='gpu')
            model.learn(data_df.to_numpy())
            adj_mtx = model.causal_matrix
            
            
        if options['routine'] == 'GOLEM':
            model = GOLEM(num_iter=10000, graph_thres=0.3, device_type='gpu', learning_rate=0.005)
            model.learn(data_df.to_numpy())
            adj_mtx = model.causal_matrix
            
        
        finish = time.time()
        precision, recall, f1, S, shd = count_accuracy(B_true=groundtruth, B_est=adj_mtx)

        f = open(options["output"], "a")
        f.write("{},{},{},{},{},{},{},{}\n".format(
                options['data_path'], options['routine'],
                precision, recall, f1, S, shd, finish - start)
            )
        f.close()
        print("Writting results done!")