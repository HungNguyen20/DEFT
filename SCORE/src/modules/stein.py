import torch
import time

from scipy.stats import ttest_ind
from cdt.utils.R import launch_R_script
from SCORE.src.modules.utils import *


def Stein_hess_diag(X, eta_G, eta_H, s = None):
    """
    Estimates the diagonal of the Hessian of log p_X at the provided samples points
    """
    # print("Here inside Stein_hess_diag")
    n, d = X.shape
    # print("Here into X_diff", X.shape)
    X_diff = X.unsqueeze(1) - X
    if s is None:
        # print("Here into D", X_diff.shape)
        D = torch.norm(X_diff, dim=2, p=2)
        # print("Here into s", D.shape)
        s = D.flatten().median()
    
    # print("Here into K", s.shape)
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    # print("Here into eisum", K.shape)
    nablaK = -torch.einsum('kij,ik->kj', X_diff, K) / s**2
    # print("Here into inverse", nablaK.shape)
    G = torch.matmul(torch.inverse(K + eta_G * torch.eye(n)), nablaK)
    # print("Here into eisum", G.shape)
    nabla2K = torch.einsum('kij,ik->kj', -1/s**2 + X_diff**2/s**4, K)
    # print("Here into inverse", nabla2K.shape)
    return -G**2 + torch.matmul(torch.inverse(K + eta_H * torch.eye(n)), nabla2K)


def Stein_hess_col(X_diff, G, K, v, s, eta, n):
    """
    See https://arxiv.org/pdf/2203.04413.pdf Section 2.2 and Section 3.2 (SCORE paper)
        Args:
            X_diff (tensor): X.unsqueeze(1)-X difference in the NxD matrix of the data X
            G (tensor): G stein estimator 
            K (tensor): evaluated gaussian kernel
            s (float): kernel width estimator
            eta (float): regularization coefficients
            n (int): number of input samples

        Return:
            Hess_v: estimator of the v-th column of the Hessian of log(p(X))
    """
    Gv = torch.einsum('i,ij->ij', G[:,v], G)
    nabla2vK = torch.einsum('ik,ikj,ik->ij', X_diff[:,:,v], X_diff, K) / s**4
    nabla2vK[:,v] -= torch.einsum("ik->i", K) / s**2
    Hess_v = -Gv + torch.matmul(torch.inverse(K + eta * torch.eye(n)), nabla2vK)

    return Hess_v


# Would it be better to comptue only row of interest at each iteration?
def Stein_hess_matrix(X, s, eta):
    """
    Compute the Stein Hessian estimator matrix for each sample in the dataset

    Args:
        X: N x D matrix of the data
        s: kernel width estimate
        eta: regularization coefficient

    Return:
        Hess: N x D x D hessian estimator of log(p(X))
    """
    n, d = X.shape
    
    X_diff = X.unsqueeze(1)-X
    K = torch.exp(-torch.norm(X_diff, dim=2, p=2)**2 / (2 * s**2)) / s
    
    nablaK = -torch.einsum('ikj,ik->ij', X_diff, K) / s**2
    G = torch.matmul(torch.inverse(K + eta * torch.eye(n)), nablaK)
    
    # Compute the Hessian by column stacked together
    Hess = Stein_hess_col(X_diff, G, K, 0, s, eta, n) # Hessian of col 0
    Hess = Hess[:, None, :]
    for v in range(1, d):
        Hess = torch.hstack([Hess, Stein_hess_col(X_diff, G, K, v, s, eta, n)[:, None, :]])
    
    return Hess

def compute_top_order(X, eta_G, eta_H, normalize_var=True, dispersion="var"):
    """
        Estimate the topological ordering of variables from observational data

        Args:
            X (tensor): N x D matrix of the data
            eta_G, eta_H (float):  regularization coefficients

        Return:
            order (tensor): the estimated topological ordering

    """
    # print("Here in compute_top_order()")
    _, d = X.shape
    order = []
    var = []
    active_nodes = list(range(d))
    for _ in range(d-1):
        # print("\tHere loop")
        H = Stein_hess_diag(X, eta_G, eta_H)
        # print("\tHere done computing H")
        if normalize_var:
            H = H / H.mean(axis=0)
        if dispersion == "var": # The one mentioned in score-matching paper (Arxiv: 2203.04413)
            l = int(H.var(axis=0).argmin())
        elif dispersion == "median":
            med = H.median(axis = 0)[0]
            l = int((H - med).abs().mean(axis=0).argmin())
        else:
            raise Exception("Unknown dispersion criterion")
        var.append(1/H[:, l].var(dim=0).item())
        order.append(active_nodes[l])
        active_nodes.pop(l)
        X = torch.hstack([X[:,0:l], X[:,l+1:]])
    
    # print("\tHere out loop")
    order.append(active_nodes[0])
    var.append(1/H[:, 0].var(dim=0).item())
    order.reverse()
    # print("\tHere out compute_top_order()")
    return order, var


def heuristic_kernel_width(X):
    """
    Estimator of width parameter for gaussian kernel

    Args:
        X (tensor): N x D matrix of the data

    Return: 
        s(float): estimate of the variance in the kernel
    """
    X_diff = X.unsqueeze(1)-X
    D = torch.norm(X_diff, dim=2, p=2)
    s = D.flatten().median()
    return s

def das_pruning(K, X, top_order, eta_G, delta, var):
    """
    Method for fast selection of K likely parents for each node using the jacobian of the score function
    The output of this
    Args:
        K: Like in CAM preliminary Neighbour Search, select only the K most probable parents, i.e. those with highest score derivative
        X: N x D matrix of the samples
        top_order: 1 x D vector of topoligical order. top_order[0] is source
        eta_g: regularizer coefficient
        delta: hyperparameter for threshold definition
        var: 1 x D vector of estimate of the variance of the noise terms for each variable

    Return:
        A (NxD tensor): the estimated adjacency matrix
    """
    if K is None:
        K = 0
    K = K+1 # To account for A[l, l] = 0
    n, d = X.shape
    remaining_nodes = list(range(d))
    s = heuristic_kernel_width(X.detach()) # This actually changes at each iteration 
    hess = Stein_hess_matrix(X, s, eta_G)

    A = np.zeros((d,d))
    for i in range(d-1):
        l = top_order[-(i+1)]
        l_index = remaining_nodes.index(l)

        hess_remaining = hess[:, remaining_nodes, :]
        hess_remaining = hess_remaining[:, :, remaining_nodes]
        hess_l = hess_remaining[:, l_index, :]
        hess_m = torch.abs(torch.median(hess_l*var[l_index], dim=0).values) # or mean
        
        K = min(K, len(remaining_nodes))
        topk_values, topk_indices = torch.topk(hess_m, K, sorted=True)
        argmin = topk_indices[torch.argmin(topk_values)]
        parents = []
        hess_l = torch.abs(hess_l)
        for j in range(K):
            node = topk_indices[j]
            if j <= 5 and top_order[node] != l: # do not filter first M=5 nodes, as CAM handles those quick
                parents.append(remaining_nodes[node])
            else:
                _, pval = ttest_ind(hess_l[:, node], hess_l[:, argmin], alternative="greater", equal_var=False) # works fine-ish. equal_var=False?
                if pval < delta:
                    if top_order[node] != l:
                        parents.append(remaining_nodes[node])


        A[parents, l] = 1
        A[l, l] = 0
        del remaining_nodes[l_index]
        del var[l_index]
    return A


def fullAdj2Order(A):
    order = list(A.sum(axis=1).argsort())
    order.reverse()
    return order


def cam_pruning(A, X, cutoff, prune_only=True, pns=False):
    save_path = "./"


    data_np = np.array(X.detach().cpu().numpy())
    data_csv_path = np_to_csv(data_np, save_path)
    dag_csv_path = np_to_csv(A, save_path) 

    arguments = dict()
    arguments['{PATH_DATA}'] = data_csv_path
    arguments['{PATH_DAG}'] = dag_csv_path
    arguments['{PATH_RESULTS}'] = os.path.join(save_path, "results.csv")
    arguments['{ADJFULL_RESULTS}'] = os.path.join(save_path, "adjfull.csv")
    arguments['{CUTOFF}'] = str(cutoff)
    arguments['{VERBOSE}'] = "FALSE"
    # print(arguments)

    if True: #prune_only:
        def retrieve_result():
            A = pd.read_csv(arguments['{PATH_RESULTS}']).values
            os.remove(arguments['{PATH_RESULTS}'])
            os.remove(arguments['{PATH_DATA}'])
            os.remove(arguments['{PATH_DAG}'])
            return A
        dag = launch_R_script("./SCORE/R_code/cam_pruning.R", arguments, output_function=retrieve_result, verbose=False)
        return dag
    # else:
    #     def retrieve_result():
    #         A = pd.read_csv(arguments['{PATH_RESULTS}']).values
    #         Afull = pd.read_csv(arguments['{ADJFULL_RESULTS}']).values
            
    #         return A, Afull
    #     dag, dagFull = launch_R_script("/Users/user/Documents/EPFL/PHD/Causality/score_based/CAM.R", arguments, output_function=retrieve_result)
    #     top_order = fullAdj2Order(dagFull)
    #     return dag, top_order
        
  
def graph_inference(X, eta_G=0.001, eta_H=0.001, cutoff=0.001, normalize_var=False, dispersion="var", pruning = 'DAS', delta=0, pns=None, K=None):
    """
    Estimate adjacency matrix A and topological ordering of the variable in X from sample from data. Return estimations and execution times.
    """
    start_time = time.time()
    # print(start_time)
    top_order, var = compute_top_order(X, eta_G, eta_H, normalize_var, dispersion)
    order_time = time.time() - start_time 
    print(order_time)
    
    start_time = time.time()
    if pruning == 'CAM':
        if pns is None:
            A = cam_pruning(full_DAG(top_order), X, cutoff)
        else: 
            A = cam_pruning(pns_(full_DAG(top_order), X, K, thresh=1), X, cutoff)
    elif pruning == "DAS": 
        A = cam_pruning(das_pruning(K, X, top_order, eta_G, delta, var), X, cutoff)
    elif pruning == "DASBoost":
        A = das_pruning(K, X, top_order, eta_G, delta, var)
    else:
        raise Exception("Unknown pruning method")

    tot_time = order_time + (time.time() - start_time) # top ordering + pruning time

    return A, top_order, order_time, tot_time


def sortnregress(X, cutoff=0.001):
    var_order = np.argsort(X.var(axis=0))
    
    return cam_pruning(full_DAG(var_order), X, cutoff), var_order


def num_errors(order, adj):
    err = 0
    for i in range(len(order)):
        err += adj[order[i+1:], order[i]].sum()
    return err
