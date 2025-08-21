from SCORE.src.modules.algorithms.base import CausalDiscovery
from SCORE.src.modules.stein import graph_inference
from SCORE.src.modules.utils import pretty_evaluate

class DAS(CausalDiscovery):
    """
    Class to make inference using DAS algorithm on observational data
    """
    def __init__(self, X, kwargs):
        """
        Args:
            X: NxD matrx of the data
            A_truth: NxD adjacency ground truth of the graph
        """
        super().__init__(X, kwargs)
        self.K = kwargs['K']

    def algorithm_inference(self):
        return graph_inference(
            self.X, self.eta_G, self.eta_H, self.cam_cutoff, pruning=self.pruning, delta=self.threshold, K=self.K
        )

    def pretty_print(self, A, top_order_err, SCORE_time, tot_time):
        # return  pretty_evaluate(
        #     self.pruning, self.threshold, self.A_truth, A, top_order_err, SCORE_time, tot_time, self.sid, s0=self.s0, K=self.K
        # )
        return 

class SCORE(CausalDiscovery):
    """
    Class to make inference using DAS algorithm on observational data
    """
    def __init__(self, X, kwargs):
        """
        Args:
            X: NxD matrx of the data
            A_truth: NxD adjacency ground truth of the graph
        """
        super().__init__(X, kwargs)
        self.pns = kwargs['pns']

    def algorithm_inference(self):
        return graph_inference(
            self.X, self.eta_G, self.eta_H, self.cam_cutoff, pruning=self.pruning, pns = self.pns, K=10,
        )

    def pretty_print(self, A, top_order_err, SCORE_time, tot_time):
    #     return  pretty_evaluate(
    #         self.pruning, self.threshold, self.A_truth, A, top_order_err, SCORE_time, tot_time, self.sid, s0=self.s0, K=self.K
    #     )
        return