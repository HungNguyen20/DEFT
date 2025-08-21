from modules.algorithms.cd import DAS, SCORE
from modules.utils import get_data
import argparse
import numpy as np
import os
import pandas as pd
from pathlib import Path
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def get_args():
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Data generation arguments
    parser.add_argument('--graph_type', '-gt', type=str, default='ER', help="Accepted values: ER, SF")
    parser.add_argument('-d', type=int, default=10, help="Number of causal variables")
    parser.add_argument('-s0', type=int, default=10, help="Number of expected edges")
    parser.add_argument('-N', type=int, default=1000, help="Sample size")
    parser.add_argument('-GP', default=True, action='store_false' ,help="TODO")
    # Hyperparameters
    parser.add_argument('--eta_G', type=float, default=0.001, help="Regularization coefficient 1st order")
    parser.add_argument('--eta_H', type=float, default=0.001, help="Regularization coefficient 2nd order")
    parser.add_argument('--cam_cutoff', type=float, default=0.001, help="CAM pruning hyperparameter")
    parser.add_argument('-K', type=int, default=20, help="Max number of candidate parents per node (DAS pruning)")
    parser.add_argument('--pns', type=int, default=None, help="Max number of candidate parents per node (CAM pruning)")
    # Others
    parser.add_argument('--pruning', '-p', type=str, default="DAS", help="Threshold for fast pruning (mine)")
    parser.add_argument('--threshold' ,'-t', type=float, default=0.05, help="Threshold for das edge selection")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (TODO: make results reproducible")
    parser.add_argument('--logs', '-l', default=False, action='store_true', help="Save logs data")
    # Read arguments from command line
    parser.add_argument("--datafolderpath", type=str, default="asia")
    parser.add_argument("--dagpath", type=str, default="asia")
    return parser.parse_args()

def load_data(options):
    folderpath = options.datafolderpath
    dagpath = options.dagpath
    groundtruth = np.loadtxt(dagpath)

    silos = []
    if not Path(folderpath).exists():
        print("Folder", folderpath, "not exist!")
        exit()
    
    for file in sorted(os.listdir(folderpath)):
        filename = os.path.join(folderpath, file)
        silo_data = pd.read_csv(filename)
        silos.append(silo_data)

    data = pd.concat(silos, axis=0)
    return torch.from_numpy(data.sample(500).to_numpy()) *1., groundtruth


args = get_args()
# graph_type = args.graph_type
# d = args.d
# s0 = args.s0
# N = args.N
# GP = args.GP

# X, A = get_data(graph_type, d, s0, N, GP)
X, A = load_data(args)
print(X.shape, A.shape)

# pruning = args.pruning
# if pruning == 'DAS' or pruning == "DASBoost":
algorithm = DAS(X, **vars(args))
# elif pruning == 'CAM':
#     algorithm = SCORE(X, A, **vars(args))

algorithm.inference()