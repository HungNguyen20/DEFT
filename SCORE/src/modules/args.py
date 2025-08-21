import argparse

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
    return parser.parse_args()
