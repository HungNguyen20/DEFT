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


def find_all_paths_exclude_direct(v1, v2, edge_list):
    from collections import defaultdict

    # Build adjacency list
    graph=defaultdict(list)
    for u, v in edge_list:
        if not (u ==v1 and v ==v2):  # Exclude direct edge
            graph[u].append(v)

    all_paths=[]

    def dfs(current, path, visited):
        if current ==v2:
            all_paths.append(path[:])
            return
        visited.add(current)
        for neighbor in graph[current]:
            if neighbor not in visited:
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
        visited.remove(current)

    dfs(v1, [v1], set())
    return all_paths


def merge_adj(structA, structB, ground_truth_struct, realcit:CIT, citobject:CitOracle , oracle_dict):

    # Step 1: Combine and score edges
    edge_scores={}
    for edge in structA + structB:
        p_value = realcit(edge[0] , edge[1])
        edge_scores[(edge[0], edge[1])]= 1- p_value

    sorted_edges=sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
    candidate_edges=[e[0] for e in sorted_edges]

    # Step 2: Build DAG incrementally, avoiding cycles
    G=nx.DiGraph()
    for (u, v) in candidate_edges:
        G.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G):  # Cycle introduced
            G.remove_edge(u, v)  # Revert
            continue

    final_edges=list(G.edges)

    # Step 3: Remove conditionally independent redundant edges
    result_edges=final_edges[:]
    for (u, v) in final_edges:
        
        #using oracleCIT for eliminating redundant edges (as in paper)
        alt_paths=find_all_paths_exclude_direct(u, v, final_edges)  #find or path between u and v, excluding direct u->v 
        for path in alt_paths:
            cond_set=[n for n in path if n !=u and n !=v]
            if find_min_separating_set(cond_set, x_idx=u, y_idx=v, citobject= citobject , oracle_dict=oracle_dict) is not None:
                result_edges.remove((u, v))
                break
        # using groundtruth structure for eliminating redundant edges
        # if ground_truth_struct[u][v]==0:
        #     result_edges.remove((u,v))

    return result_edges


def find_min_separating_set(cond_set, x_idx, y_idx, citobject: CitOracle, oracle_dict):
    cond_set = sorted(cond_set)
    for k in range(3):
        for subset in combinations(cond_set, k):
            subset = sorted(subset)
            key1 = (x_idx, y_idx, tuple(subset))
            key2 = (y_idx , x_idx , tuple(subset))
            if key1 not in oracle_dict and key2 not in oracle_dict:
                result = citobject.query(A=x_idx, B=y_idx, Z=subset)
                oracle_dict[key1] = result
                oracle_dict[key2] = result
                if result:
                    # print(f"{x_idx} and {y_idx} separated by {subset}")
                    return list(subset)
            elif oracle_dict[key1] == True or oracle_dict[key2] == True:
                # print(f"{x_idx} and {y_idx} separated by {subset}")
                return list(subset)
    # print(f"{x_idx} and {y_idx} can not be separated")
    return None


def find_causal_cut(V_set, citobject:CitOracle, oracle_dict, k=5):
    print("Finding causal cut for V_set:", V_set)
    V_set=sorted(V_set)
    best_balance=0 
    best_V1=[]
    best_V2=[]
    best_C=[]
    all_pairs=list(combinations(V_set, 2))
    for _ in tqdm(range(k)):
        flag=0
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            oracle_dict[(u , v , tuple(sorted(set(V_set) - {u,v})))] = citobject.query(A=u, B=v, Z=list(set(V_set) - {u,v}))
            oracle_dict[(v, u , tuple(sorted(set(V_set) - {u,v})))] = citobject.query(A=u, B=v, Z=list(set(V_set) - {u,v}))
            if oracle_dict[(u , v , tuple(sorted(set(V_set) - {u,v})))]:
                cond_nodes=list(set(V_set) - {u,v})
                flag =1 
                break
        print(f"original pair {u} , {v}\n")
        
        if flag == 0:
            continue
        
        min_subset = find_min_separating_set(cond_set=cond_nodes,x_idx=u, y_idx=v, citobject=citobject, oracle_dict = oracle_dict)
        if min_subset is None:
            print("None in min subset")
            continue
        
        V1=[u]
        V2=[v]
        C=min_subset
        
        print(f"premier subset {C}\n")
        
        # print(V1, V2, C, CIT.query(u, v, Z=C))
        remaining_V=list(set(V_set) - set(V1) - set(V2) - set(C))
        for node in remaining_V:
            flag1=1
            flag2=1 
            for node1 in V1:
                tmp_sepset = find_min_separating_set(C, node, node1, citobject=citobject, oracle_dict= oracle_dict)
                if tmp_sepset is None:
                    flag2=0  #node can not be in V2 
            
            for node2 in V2:
                tmp_sepset = find_min_separating_set(C, node, node2, citobject=citobject , oracle_dict= oracle_dict)
                if tmp_sepset is None:
                    flag1=0 #node can not be in V1 
            
            if flag1 ==1 and flag2 ==0:
                print(f"{node} to V1")
                V1.append(node)
            elif flag1 ==0 and flag2 ==1:
                print(f"{node} to V2")
                V2.append(node)
            else:
                print(f"{node} to C")
                C.append(node)
                
        print(f"second subset {C}\n")

        for s in C[:]:
            flag1=1 
            flag2=1 
            for node1 in V1:
                tmp_sepset = find_min_separating_set(list(set(C) - {s}), node1, s, citobject , oracle_dict= oracle_dict) 
                if tmp_sepset is None:
                    flag2=0 
            for node2 in V2:
                tmp_sepset = find_min_separating_set(list(set(C) - {s}), node2, s, citobject , oracle_dict= oracle_dict) 
                if tmp_sepset is None:
                    flag1=0 
            if flag1 ==0 and flag2 ==1:
                C.remove(s)
                V2.append(s)
                print(f"{s} from C to V2")
            elif flag1 ==1 and flag2 ==0:
                C.remove(s)
                V1.append(s)
                print(f"{s} from C to V1")
                
        
        print(f"Last subset {C}\n")

        if best_balance < min(len(V1), len(V2)):
            best_balance=min(len(V1), len(V2))
            best_V1=V1 
            best_V2=V2 
            best_C=C
            print(f"best cut {V1}, {V2}, {C}")

    return best_V1, best_V2, best_C


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


def SADA(dataset, V_set, groundtruth_struct, options, citobject , oracle_dict):
    print("Here |V_set| =", len(V_set), " |V_set| < threshold =", len(V_set) < options['threshold'])
    
    if options['datatype'] =='continuous':
        ci_test=fisherz
        CoInT=CIT(data=dataset.to_numpy(), method='fisherz')
        
    elif options['datatype'] =='discrete':
        ci_test=chi2
        CoInT=CIT(data=dataset.to_numpy(), method='chi2')
        
    if len(V_set) ==0:
        print(f"|V_set| = 0")
        return [] 
    
    if len(V_set) < options['threshold']:
        print("Running PC on set size", len(V_set))
        dataset_name = options["data_path"].split('/')[-3] + "-" + options["data_path"].split('/')[-2]
        with open(f"ablation/{dataset_name}.log", "a") as f:
            f.write(f"{len(V_set)},")
            
        data=vertical_data_seperation(data_df=dataset, node_idx=V_set)
        cg=pc(data=data.to_numpy(), alpha=0.3, indep_test=ci_test)
        edge_list=extract_edge(cg, V_set=V_set)
        return edge_list

    V1, V2, C = find_causal_cut(V_set=V_set, citobject=citobject, oracle_dict=oracle_dict) 
    edge_list_1 =SADA(dataset=dataset, V_set=V1 + C, groundtruth_struct=groundtruth_struct, options=options, citobject=citobject , oracle_dict = oracle_dict)
    edge_list_2  =SADA(dataset=dataset, V_set=V2 + C,groundtruth_struct=groundtruth_struct, options=options, citobject=citobject , oracle_dict= oracle_dict) 
    
    return merge_adj(structA=edge_list_1, structB=edge_list_2, ground_truth_struct= groundtruth_struct, realcit=CoInT, citobject= citobject , oracle_dict = oracle_dict)



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
        oracle_dict = {}
        start_time=time.time()
        edge_list =SADA(dataset=data, V_set=np.arange(d).tolist(),groundtruth_struct = stru_GT, options=options, citobject=citoracle , oracle_dict = oracle_dict)
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