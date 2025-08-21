import abc
from SCORE.src.modules.stein import num_errors
import torch

# TODO: Non ha senso distinzione tra K e pns in questo momento. Fix

class CausalDiscovery(metaclass=abc.ABCMeta):
    """
    Class to make inference using DAS algorithm on observational data
    """
    def __init__(self, X, kwargs):
        """
        Args:
            X: NxD matrx of the data
            A_truth: NxD adjacency ground truth of the graph
        """
        self.X = torch.from_numpy(X)
        # self.A_truth = A_truth

        self.d = kwargs['d']
        # self.s0 = kwargs['s0']
        self.eta_G = kwargs['eta_G']
        self.eta_H = kwargs['eta_H']
        self.cam_cutoff = kwargs['cam_cutoff']
        self.pruning = kwargs['pruning']
        self.threshold = kwargs['threshold']
        self.sid = bool(self.d <= 200)
        
    @abc.abstractmethod
    def algorithm_inference(self):
        raise NotImplementedError  

    @abc.abstractmethod
    def pretty_print(self, A, top_order_err, SCORE_time, tot_time):
        raise NotImplementedError  

    def inference(self):
        A, _, _, _ =  self.algorithm_inference()
        # top_order_err = num_errors(top_order_SCORE, self.A_truth)
        # pretty = self.pretty_print(A, top_order_err, SCORE_time, tot_time,)
        # print(pretty)
        return A