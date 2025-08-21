import numpy as np
from castle.metrics import MetricsDAG
from model import SAHCD
import time
from args import get_args
import pandas as pd
import sys 
import pandas as pd
sys.path.append('../../VerticalCausalDiscovery')
from utils.utils import count_accuracy
from baseline import read_opts
import os

"""
args:
--method:Algorithm for sub-graph.You can choose:ICALiNGAM(default),DirectLiNGAM,PC,Notears(GOLEM),GraNDAG
--pre_gate:Threshold for conditional independence(fisherz test) tests(default=0.8).Higher thresholds represent higher confidence requirements
--thresh:Threshold for sub-graph(default=0.3).
--golem_epoch:Iteration rounds of the GOLEM(default=5000).
--lr:Learning rate for GOLEM(default=0.05).
--pc_alpha:Parameter of PC(default =0.05).
--data_path:Parameter of data path(default=data/100/0).
The parameter of our algorithm is pre_gate.You can get detailed descriptions of other parameters from gcastle.
"""
if __name__ =='__main__':
    args=get_args()
    print(args.data_path)
    print(args)
    # data = np.load(args.data_path+'/data.npy')
    # Tdata = np.load(args.data_path+'/truth.npy')
    # Tdata= np.int64(Tdata != 0)
    
    data = pd.read_csv(args.data_path).to_numpy()
    Tdata = pd.read_csv(args.groundtruth_path).to_numpy()
    
    print(type(data) , type(Tdata))
    
    start=time.time()
    model=SAHCD(data,Tdata,args)
    print(f"start running {args.data_path}")
    model.run()
    print(f"finished running {args.data_path}")
    end=time.time()
    final_graph = model.global_graph
    print(type(final_graph) , type(Tdata))
    
    precision, recall, f1, S, shd = count_accuracy(B_true=Tdata, B_est=final_graph)

   
    if not os.path.exists(args.output):
        f = open(args.output, "w")
        f.write("{},{},{},{},{},{},{},{},{}\n".format('data_path', 'routine', "P", "R", "F1", "S", "shd", "time", "CIT"))
        f.close()
        
    f = open(args.output, "a")
    f.write("{},{},{},{},{},{},{},{},{}\n".format(
            args.data_path, args.method,
            precision, recall, f1, S, shd, end - start, model.CIT)
        )
    f.close()
    print("Writting results done!")
    
    # with open('result_base.txt', 'a', encoding='utf-8') as f:
    #     f.write(str(args) + '\t'+str(end-start)+'\t')
    #     f.write(str(MetricsDAG(model.global_graph, Tdata).metrics) + '\n')