
import sys 
sys.path.append('../../VerticalCausalDiscovery')
import networkx as nx
from castle.algorithms import PC
from oraclecit import CitOracle
import argparse
import pandas as pd
from pathlib import Path
import time
import numpy as np
from time import time
from itertools import combinations 
from scipy.sparse.linalg import eigsh
import networkx as nx
from utils.utils import count_accuracy


class DependenceGraph:
    numNode=0 
    df: np.ndarray
    var : list
    adj : np.ndarray
     
    def __init__(self,df: pd.DataFrame) -> None:
        self.df=df.to_numpy() 
        self.df=self.df.swapaxes(0,1)
        self.numNode=len(df.columns) 
        self.adj=np.zeros((self.numNode,self.numNode))
        self.var=df.columns.to_list() 
        
    def addEdge(self,u: int,v: int)->None: 
        self.adj[u][v]=1 
        self.adj[v][u]=-1 
        
    def eraseEdge(self,u: int,v: int)->None: 
        if self.adj[v][u]==1: 
            self.adj[u][v]=-1 
        else : 
            self.adj[u][v]=0 
            self.adj[v][u]=0 


def construct_adjoint_graph(G: DependenceGraph):
    """
    Construct the adjoint graph G_A from the original graph G.
    Each undirected edge (u-v) becomes a node in G_A.
    There is an edge between two nodes in G_A if their corresponding edges in G share a node.
    """
    edge_nodes = []
    edge_to_index = {}
    
    for u in range(G.numNode):
        for v in range(u+1, G.numNode):
            if  G.adj[u][v]==2 :  # undirected edge
                edge_to_index[(u, v)] = len(edge_nodes)
                edge_nodes.append((u, v))
    
    num_edge_nodes = len(edge_nodes)
    G_A = np.zeros((num_edge_nodes, num_edge_nodes))

    for i, (u1, v1) in enumerate(edge_nodes):
        for j, (u2, v2) in enumerate(edge_nodes):
            if i >= j:
                continue
            if len(set([u1, v1]) & set([u2, v2])) > 0:
                G_A[i][j] = 1
                G_A[j][i] = 1

    return G_A, edge_nodes

def spectral_cut(G_A):
    """Apply spectral clustering to find the second smallest eigenvector"""
    D = np.diag(np.sum(G_A, axis=1))
    L = D-G_A
    _, vecs = eigsh(L, k=2, which='SM')
    v2 = vecs[:, 1]
    return v2

def cut_from_eigenvector(v2, G_A, edge_nodes):
    cluster1 = [i for i, v in enumerate(v2) if v < 0]
    cluster2 = [i for i, v in enumerate(v2) if v >= 0]

    cut_set = set()
    for i in cluster1:
        for j in cluster2:
            if G_A[i][j] == 1:
                if edge_nodes[i][0] == edge_nodes[j][0] or edge_nodes[i][0] == edge_nodes[j][1]: 
                    cut_set.add(edge_nodes[i][0]) 
                else: 
                    cut_set.add(edge_nodes[i][1])

    return cut_set

def convert_cut_to_node_sets(G: DependenceGraph, cut_set, edge_nodes):
    G_temp = nx.Graph()
    # Keep only edges whose indices are NOT in cut_set
    for i, (u, v) in enumerate(edge_nodes):
        if u not in cut_set and v not in cut_set:
            G_temp.add_edge(u, v)

    components = list(nx.connected_components(G_temp))
    if len(components) < 2:
        return list(components[0]), []

    # Largest two connected components
    components = sorted(components, key=len, reverse=True)
    A, B = list(components[0]), list(components[1])
    return A, B


def build_superstructure(G: DependenceGraph, citobject:CitOracle, k_par=2, alpha=0.05):
    n = G.numNode
    for i in range(n): 
        for j in range(i+1,n):
            if i!=j: 
                G.adj[i][j] = G.adj[j][i] = 2  # Initialize as undirected edge
    for depth in range(k_par+1): 
        # edge_removal=[] 
        done=True
        for X in range(n): 
            candidates=list(np.where(G.adj[X]!=0)[0])
            fullCandidates=np.array(list(citobject.parents[X])).reshape(-1)
            # print(fullCandidates)
            if len(candidates)<= depth: 
                continue 
            while len(candidates)>0:
                Y=candidates.pop() 
                if G.adj[X][Y]!=2: 
                    continue 
                
                SXY=np.delete(fullCandidates,np.where(fullCandidates==Y))
                done=False 
                if len(SXY)<depth:
                    continue
                subCandidates=combinations(SXY,depth) 
                for S in subCandidates: 
                    # print(S) 
                    # numTest=numTest+1 
                    if citobject.query(X,Y,S) : 
                        G.adj[X][Y]=G.adj[Y][X]=0 
                        break
        # for (X,Y) in edge_removal: 
        #     G.adj[X][Y]=G.adj[Y][X]=0 
        if done==True: 
            break 


def CausalPartitioning(G: DependenceGraph, citobject:CitOracle, k_par=2, alpha=0.05):
    build_superstructure(G, citobject, k_par)
    G_A, edge_nodes = construct_adjoint_graph(G)
    v2 = spectral_cut(G_A)
    cut_nodes = cut_from_eigenvector(v2, G_A, edge_nodes)
    A, B = convert_cut_to_node_sets(G, cut_nodes,edge_nodes)
    C = cut_nodes.copy()  # Nodes that are cut

    for u in cut_nodes:
        for v in range(G.numNode):
            if G.adj[u][v] == 2 and v not in cut_nodes:
                C.add(v) 
                
    for u in range(G.numNode): 
        if (u not in A) and (u not in B): 
            C.add(u)
            
    # print("Neighbor set and pruning set:")
    V1 = list(set(A) | C)
    V2 = list(set(B) | C)

    return V1, V2


def read_opts():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data.csv") # Must have headers
    parser.add_argument("--groundtruth_path", type=str, default="graph.csv") # Must have headers, same as the one above
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--routine", type=str, default="PC") # do not change
    parser.add_argument("--datatype", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--maxCset", type=int, default=2)
    
    options=vars(parser.parse_args())
    return options



if __name__=='__main__':
    options=read_opts()
    data=pd.read_csv(options['data_path'])
    stru_GT=pd.read_csv(options['groundtruth_path']).to_numpy()
    
    d = data.shape[1]
    if not Path(options["output"]).exists():
        f = open(options["output"], "w")
        f.write("{},{},{},{},{},{},{},{},{}\n".format('data_path', 'routine', "P", "R", "F1", "S", "shd", "time", "CIT"))
        f.close()

    citobject = CitOracle(stru_GT)

    pc_model = PC(variant='parallel', alpha=0.3, ci_test='fisherz')
    if options['datatype'] == 'discrete':
        pc_model = PC(variant='parallel', alpha=0.05, ci_test='chi2')
        
    for i in range(options['repeat']):
        start_time=time()
        G=DependenceGraph(data) 
        V1, V2 = CausalPartitioning(G, citobject, k_par=options['maxCset'])
        print(len(V1),V1)
        print(len(V2),V2)
         
        # Solving partition V1
        pc_model.learn(data=data.iloc[:, V1].to_numpy())
        adj_mtx1 = pc_model.causal_matrix
        
        # Solving partition V2
        pc_model.learn(data=data.iloc[:, V2].to_numpy())
        adj_mtx2 = pc_model.causal_matrix
        
        # Merging results
        final_graph = np.zeros((d, d))
        for i in V1:
            index_i = V1.index(i)
            for j in V1:
                index_j = V1.index(j)
                final_graph[i][j] = adj_mtx1[index_i][index_j]
                final_graph[j][i] = adj_mtx1[index_j][index_i]
        
        for i in V2:
            index_i = V2.index(i)
            for j in V2:
                index_j = V2.index(j)
                final_graph[i][j] = adj_mtx2[index_i][index_j]
                final_graph[j][i] = adj_mtx2[index_j][index_i]
        
        elapsed_time=time() - start_time
        print("Elapsed time: ", elapsed_time)
        precision, recall, f1, S, shd = count_accuracy(stru_GT, final_graph)
        
        f=open(options["output"], "a")
        f.write("{},{},{},{},{},{},{},{},{}\n".format(
                options['data_path'], options['routine'],
                precision, recall, f1, S, shd, elapsed_time, citobject.num_query)
            )
        f.close()
        print("Writting results done!")
    