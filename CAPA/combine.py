import numpy as np 


def edge_to_graph(d , edge_list): 
    final_graph = np.zeros((d, d))
    for e in edge_list:
        if final_graph[e[1] ,e[0]] == 1:
            final_graph[e[1] , e[0]] = -1
            final_graph[e[0] ,e[1]] = 0
        else:
            final_graph[e[0] , e[1]] = 1
    return final_graph

def count_accuracy(B_true, B_est):
    """
    Compute precision, recall, f1, S, and SHD.

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge

    Returns:
        precision: true positive / predicted positive
        recall: true positive / condition positive
        f1: 2 * precision * recall / (precision + recall)
        S: number of extra edges / predicted positive
        shd: undirected extra + undirected missing + reverse
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
    
    d = B_true.shape[0]

    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])

    # true positives
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])

    # predicted positives
    pred_size = len(pred) + len(pred_und)

    # condition positives
    cond_size = len(cond)

    # false positives (extra edges not in skeleton)
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])

    # extra for reversed edges
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)

    # precision, recall, f1
    precision = float(len(true_pos)) / max(pred_size, 1)
    recall = float(len(true_pos)) / max(cond_size, 1)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # S metric
    # num_extra = len(false_pos) + len(reverse)
    # S = float(num_extra) / max(pred_size, 1)

    # SHD
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    S = len(extra_lower) / max(pred_size, 1)

    return precision, recall, f1, S, shd


def vertical_data_seperation(data_df , node_idx):
    nodes = []
    for idx in node_idx:
        nodes.append(data_df.columns[idx])
    data = data_df[nodes]
    return data 


def merge_struct(struA , struB , global_adj):
    ...
    return global_adj

# def Plus_PC(partition_alg, data , stru_GT , maxCset , datatype):

#     if partition_alg == "SADA":
#         cut_set,nodeA,nodeB, _ = SADA(data , stru_GT)
#     elif partition_alg == "CAPA":
#         cut_set, nodeA, nodeB, _ = CAPA(data, stru_GT)

#     PA = np.unique(np.concatenate((nodeA, cut_set)))
#     PB = np.unique(np.concatenate((nodeB, cut_set)))

#     data_A = vertical_data_seperation(data , PA)
#     data_B = vertical_data_seperation(data , PB)

#     # -----------Run PC on dataA ---------------------------
#     if datatype == 'continuous':
#         pc_A = PC(variant='stable', alpha=0.05, ci_test='fisherz')
#     elif datatype == 'discrete':
#         pc_A = PC(variant='stable', alpha=0.05, ci_test='chi2') #need preprocess categorical data -> discrete numeric values?

#     pc_A.learn(data_A)
#     stru_A = pc_A.causal_matrix

#     # -----------Run PC on dataB ---------------------------
#     if datatype == 'continuous':
#         pc_B = PC(variant='stable', alpha=0.05, ci_test='fisherz')
#     elif datatype == 'discrete':
#         pc_B = PC(variant='stable', alpha=0.05, ci_test='chi2')

#     pc_B.learn(data_B)
#     stru_B = pc_B.causal_matrix


#     # ---------merge 2 local structure ----------------------
#     global_adj = np.zeros((stru_GT.shape[0] , stru_GT.shape[0]))
#     global_adj = merge_struct(struA=stru_A , struB=stru_B , global_adj=global_adj)



#     precision, recall, f1, S, shd = count_accuracy(stru_GT , global_adj)
#     return np.array([precision , recall , f1 , S , shd])



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



    

    










