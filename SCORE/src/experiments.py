from modules.experiments.score_experiment import SCOREExperiment
from modules.experiments.das_experiment import DASExperiment


if __name__ == "__main__":
    """
    Run experiments and store logs
    """
    # General
    num_tests = 1

    # Data generation
    N = 1000
    noise_type = 'Gauss'
    graph_type = 'ER'

    # Regression parameters
    eta_G = 0.001
    eta_H = 0.001

    # Experiments parameters
    pruning = "DAS" # ["DAS", "CAM"]
    edges = ['d']
    d_values = [10]
    K = 20
    delta = [0.01]
    cam_cutoff = 0.001

    if pruning == "CAM":
        for s0 in edges:
            experiment = SCOREExperiment(d_values, num_tests, s0, noise_type, cam_cutoff, pns=K)
            experiment.run_experiment(N, eta_G, eta_H)

    elif pruning == "DAS":
        for s0 in edges:
            experiment = DASExperiment(d_values, num_tests, s0, noise_type, graph_type, delta, cam_cutoff, K)
            experiment.run_experiment(N, eta_G, eta_H)

    else:
        raise ValueError("Unknown pruning method")