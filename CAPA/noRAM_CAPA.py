import sys 
sys.path.append('../../VerticalCausalDiscovery')
from itertools import combinations
from combine import vertical_data_seperation
# from causallearn.utils.cit import CIT
import random 
import networkx as nx
from causallearn.utils.cit import fisherz, chi2
from causallearn.search.ConstraintBased.PC import pc
from oraclecit import CitOracle
import argparse
import pandas as pd
import os
import time
from combine import edge_to_graph, count_accuracy
from pathlib import Path
import numpy as np
from tqdm import tqdm
from causallearn.utils.cit import CIT



def merge_adj(structA, structB, V_set1, V_set2, realcit:CIT):
    edge_score = {}
    struct_ps = [structA , structB]
    Vset_ps = [V_set1 , V_set2]
            
    for edge in structA:
        edge_score[edge] = 1 - realcit(edge[0] , edge[1])
    
        
    for edge in structB:
        if edge_score.get(edge) is None and edge_score.get((edge[1] , edge[0])) is None:
            edge_score[edge] = 1 - realcit(edge[0] , edge[1])
        elif edge_score.get(edge) is None and edge_score.get((edge[1] , edge[0])) is not None:
            tmp_score = 1 -realcit(edge[0] , edge[1])
            if tmp_score > edge_score[(edge[1] , edge[0])]:
                edge_score.pop((edge_score[1] , edge_score[0]))
                edge_score[edge] = tmp_score 
    final_edges = edge_score.copy()
    edges_to_remove = []

    for edge, _ in edge_score.items():
        valid_in_any_structure = False
        for i in range(2):
            struct = struct_ps[i]
            V = Vset_ps[i]
            if edge[0] in V and edge[1] in V:
                if edge in struct or (edge[1], edge[0]) in struct:
                    valid_in_any_structure = True
                    break
        if not valid_in_any_structure:
            edges_to_remove.append(edge)

    # Remove after iteration
    for edge in edges_to_remove:
        final_edges.pop(edge)

            
    return list(final_edges.keys())





def heuristic_search_partitioning(V_set , M):
    '''
    V_set: list of variables 
    M: independence matrix , M[i][j] = 1 if V_set[i] and V_set[j] are independence in order k (conditional level of M)
    '''
    order_of_nodes = {}
    for index_vi in range(len(V_set)):
        order_of_nodes[index_vi] = sum(M[index_vi])
        # for index_vj in range(len(V_set)):
        #     order_of_nodes[i] += M[index_vi][index_vj]
        
    sorted_nodes = sorted(order_of_nodes.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_nodes)
    A = [] #original V1, with seed
    B = [sorted_nodes[-1][0]] #original V2 
    C = [] #original causal cut 
    D = [] #list of nodes contained in A join B, that is adjacent to C 
    V1 = []  #A joins C joins D 
    V2 = []  #B joins C joins D 
    for index_vi, _ in sorted_nodes:
        if index_vi in A or index_vi in B:
            continue
        # index_vi belongs to A if it is non adjacent to everynode in B 
        flagA = 0 
        belong_to_A = 0
        for index_vj in B:
            flagA += M[index_vi][index_vj]
        if flagA == len(B):
            A.append(index_vi)
            # print(f"{V_set[index_vi]} to A")
            belong_to_A = 1
        
        
        flagB = 0 
        belong_to_B = 0
        if belong_to_A == 0:
            for index_vj in A:
                flagB += M[index_vi][index_vj]
            if flagB == len(A):
                B.append(index_vi)
                # print(f"{V_set[index_vi]} to B")
                belong_to_B = 1
        
        if belong_to_A == 0 and belong_to_B == 0:
            C.append(index_vi)
            # print(f"{V_set[index_vi]} to C")
        
    print(f"A = {A} , B = {B} , C = {C}")
    for index_vi, _ in sorted_nodes:
        flag = 0 #if this flag doesn't remain 0 after all then V_set[index_vi] belongs to D 
        if index_vi in A or index_vi in B:
            for index_vj in C:
                if M[index_vi][index_vj] == 0: #meaning index_vi and index_vj are adjacent in M  
                    flag+=1 
        if flag!= 0:
            # print(f"{V_set[index_vi]} to D")
            D.append(index_vi)
    print(f"D  = {D}")
    A = [V_set[index] for index in A]
    B = [V_set[index] for index in B]
    C = [V_set[index] for index in C] 
    D = [V_set[index] for index in D]    
    V1 = list(set(A + C + D ))
    V2 = list(set(B + C + D))
    return V1 , V2

    
def optimization_causal_cut(V_set,  M):
    ...
    
                
        

def find_causal_cut(V_set, citobject:CitOracle, sigma = 3):
    
    #initialize the independent matrix of V_set (M having dimension d x d , with d=|V_set| )
    M = np.zeros((len(V_set) , len(V_set)))
    
    for order in range(sigma):
        print(f"Start construct {order} independent matrix")
        for index_vi in range(len(V_set)):
            for index_vj in range(index_vi +1 , len(V_set)):
                if M[index_vi][index_vj] == 0:
                    all_conditioning_node_sets = list(combinations(
                        [v for v in V_set if v != V_set[index_vi] and v != V_set[index_vj]], order))
                    for node_set in all_conditioning_node_sets:
                        
                        if citobject.query(V_set[index_vi] , V_set[index_vj] , Z = node_set ):
                            M[index_vi][index_vj] = 1 
                            M[index_vj][index_vi] = 1
                            break
        # print(M)
        V1, V2 = heuristic_search_partitioning(V_set= V_set , M = M )
        if max(len(V1) , len(V2)) != len(V_set):
            # print(f"found seperating set {V1} , {V2}")
            return V1 , V2
        # print(f"Cannot seperate with order {order}")
    # print(f"finding no seperating set: V1 = {V1}, V2 = {V2}")
    return V1, V2
            
                    
        
        

    
            

def extract_edge(cg_graph, V_set):
    edge_list=[]
    adj_mtx=cg_graph.G.graph
    for i in range(adj_mtx.shape[0]):
        for j in range(adj_mtx.shape[1]):
            if adj_mtx[i, j] ==1 and adj_mtx[j, i] ==0:
                edge_list.append((V_set[i], V_set[j]))
            elif adj_mtx[i,j] ==-1:
                edge_list.append((V_set[i],V_set[j]))
                edge_list.append((V_set[j],V_set[i]))
                
    return edge_list

def CAPA(dataset, V_set, groundtruth_struct , options, citobject ):
    #find causal partioning on V_set 
    
    if options['datatype'] =='continuous':
        ci_test= fisherz
        CoInT=CIT(data=dataset.to_numpy(), method='kci')
        
    elif options['datatype'] =='discrete':
        ci_test= chi2
        CoInT=CIT(data=dataset.to_numpy(), method='chi2')
        
    V1 , V2 = find_causal_cut(V_set, citobject=citobject, sigma = options['maxCset'])
    if max(len(V1) , len(V2)) == len(V_set):
        #return G by running algorithm on dataset 
        print(f"Running PC on set size {len(V_set)}")
        dataset_name = options['data_path'].split('/')[-3] + "-" + options["data_path"].split('/')[-2]
        os.makedirs("ablation", exist_ok=True)
        log_path = f"ablation/{dataset_name}.log"
        with open(log_path, 'a') as f:
            f.write(f"{len(V_set)},")
        data = vertical_data_seperation(data_df=dataset , node_idx= V_set)
        cg = pc(data = data.to_numpy() , alpha = 0.3 , indep_test = ci_test)
        edge_list = extract_edge(cg , V_set = V_set)
        # print(edge_list)
        return edge_list
    else:
        edge_list1 = CAPA(dataset , V1 , groundtruth_struct= groundtruth_struct , options=options, citobject=citobject )
        edge_list2 = CAPA(dataset , V2 , groundtruth_struct=groundtruth_struct , options=options , citobject= citobject) 
        # print(edge_list1 , edge_list2)
        return merge_adj(edge_list1, edge_list2, V_set1=V1, V_set2=V2, realcit= CoInT)

def read_opts():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.csv") # Must have headers
    parser.add_argument("--groundtruth_path", type=str, default="graph.csv") # Must have headers, same as the one above
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--routine", type=str, default="PC") # do not change
    parser.add_argument("--datatype", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--maxCset", type=int, default=3) 
    
    options=vars(parser.parse_args())
    return options



if __name__ =="__main__":
    options=read_opts()
    data=pd.read_csv(options['data_path'])
    stru_GT=pd.read_csv(options['groundtruth_path']).to_numpy()
    
    d = data.shape[1]
    
    if not Path(options["output"]).exists():
        f = open(options["output"], "w")
        f.write("{},{},{},{},{},{},{},{},{}\n".format('data_path', 'routine', "P", "R", "F1", "S", "shd", "time", "CIT"))
        f.close()
    
    for i in range(options['repeat']):
        citoracle=CitOracle(stru_GT)
        start_time=time.time()
        edge_list=CAPA(dataset=data, V_set=np.arange(d).tolist(),groundtruth_struct = stru_GT, options=options, citobject=citoracle)
        elapsed_time=time.time() - start_time
        final_graph=edge_to_graph(d, edge_list)
        precision, recall, f1, S, shd=count_accuracy(stru_GT, final_graph)
        f=open(options["output"], "a")
        f.write("{},{},{},{},{},{},{},{},{}\n".format(
                options['data_path'], options['routine'],
                precision, recall, f1, S, shd, elapsed_time, citoracle.num_query)
            )
        f.close()
        print("Writting results done!")