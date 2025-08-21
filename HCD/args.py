import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Demo of argparse')
    parser.add_argument('--method', type=str, default='ICALiNGAM')
    parser.add_argument('--pre_gate', type=float, default=0.8)
    parser.add_argument('--thresh', type=float, default=0.3)
    parser.add_argument('--golem_epoch', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--pc_alpha',type=float, default=0.05)
    # parser.add_argument('--data_path', type=str, default='data/100/0')
    parser.add_argument("--data_path", type=str, default="data.csv") # Must have headers
    parser.add_argument("--groundtruth_path", type=str, default="graph.csv") # Must have headers, same as the one above
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--datatype", type=str, default="continuous", choices=["continuous", "discrete"])
    args = parser.parse_args()
    return args