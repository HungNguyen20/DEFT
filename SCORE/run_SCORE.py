from src.modules.algorithms.cd import SCORE
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data/100/0')
args = parser.parse_args()

data = np.load(args.data_path+'/data.npy')
Tdata = np.load(args.data_path+'/truth.npy')

algorithm = SCORE(data, kwargs={'d': data.shape[1], 'eta_G': 0.01, 'eta_H': 0.01, 
                                'cam_cutoff': 0.01, 'pruning': 'CAM', 'threshold': 0.01, 'pns': 10})

adj_mtx = algorithm.inference()
adj_mtx = (adj_mtx > 0) * 1.