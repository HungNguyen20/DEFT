import networkx as nx
from collections import deque
# from baseline import find_groundtruth_connectivity , seperate_node_into_group
import numpy as np
import pandas as pd 
import time 

class CitOracle:
    def __init__(self, adj_matrix):
        """
        Initializes the oracle with a DAG represented by an adjacency matrix.
        adj_matrix: numpy array (shape: [n, n]) where adj[i][j] = 1 implies i → j
        """
        self.n = len(adj_matrix)
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(self.n))
        for i in range(self.n):
            for j in range(self.n):
                if adj_matrix[i][j]:
                    self.G.add_edge(i, j)

        self.parents = {node: set(self.G.predecessors(node)) for node in self.G.nodes}
        self.children = {node: set(self.G.successors(node)) for node in self.G.nodes}

    def query(self, A, B, Z):
        """
        Checks whether node A is d-separated from node B given a set of observed nodes Z.
        A, B: node indices (int)
        Z: list or set of observed node indices
        Returns True if A ⫫ B | Z (i.e., A is d-separated from B given Z), else False
        """
        Z = set(Z)

        # Phase I: compute ancestors of Z
        ancestors = set()
        to_visit = set(Z)
        while to_visit:
            Y = to_visit.pop()
            if Y not in ancestors:
                ancestors.add(Y)
                to_visit.update(self.parents[Y])

        # Phase II: traverse active trails from A
        L = deque()
        L.append((A, 'up'))  # Start upward from A
        visited = set()
        reachable = set()

        while L:
            Y, d = L.popleft()
            if (Y, d) in visited:
                continue
            visited.add((Y, d))

            if Y not in Z:
                reachable.add(Y)

            if d == 'up':
                if Y not in Z:
                    for Zp in self.parents[Y]:
                        L.append((Zp, 'up'))
                    for Zc in self.children[Y]:
                        L.append((Zc, 'down'))
            elif d == 'down':
                if Y not in Z:
                    for Zc in self.children[Y]:
                        L.append((Zc, 'down'))
                if Y in ancestors:
                    for Zp in self.parents[Y]:
                        L.append((Zp, 'up'))

        return B not in reachable  # True if d-separated, False otherwise
    


    
    
    

    
    

if __name__ == "__main__":
    
    # example graph included in slide
    adj = [
        [0,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,1,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1],
        [0,0,0,0,0,0,1,1],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0]
    ]
    
    adj = np.loadtxt("data/n500_d500_e1500_dtiid_gtER_stgauss_seed42/graph.csv", delimiter=',', dtype=int)
    oracle = CitOracle(adj)

    
    
    #debugging 
    numbers = [i for i in range(500) if i not in {28 , 129 , 55, 320}]
    # print(oracle.query(262, 288 , Z = numbers)) #True 
    print(oracle.query(28, 55 , Z = numbers))
    
    print(adj[272][39] , adj[39][272])
    print(adj[28][55] , adj[28][129] , adj[320][55] , adj[320][129])
    
    
    # print(oracle.query(320, 55,  Z = [i for i in range(500) if i not in {28, 55, 320 , 129}]))
    # for a in [28, 320 , 55, 129]:
    #     for b in [28,320 ,55, 129]:
    #         if a!=b :
    #             print(a,b , adj[a][b])
    
    s1_288 = set()
    for node in range(500):
        if oracle.query(288, node, Z=[]) == False: 
            s1_288.add(node) 
    s2_262 = set()
    for node in range(500):
        if oracle.query(262, node, Z=[]) == False: 
            s2_262.add(node) 
            
    s3_423 = set()
    for node in range(500):
        if oracle.query(423, node, Z=[]) == False:
            s3_423.add(node)
    
    # result = (s2_262 == s3_423) and (len(s2_262) > len(s1_288)) and all(elem in s2_262 for elem in s1_288)
    # # print(result)
    print(all(elem in s2_262 for elem in s1_288) , "1")
    print(len(s2_262) > len(s1_288) , "2")
    print(s2_262 == s3_423, "3")
    # print(s1_288)
    # print(s2_262)
    
    # if all(node1 in s1_288 for node1 in s2_262):
        # print("very true")
        
    for node in s3_423:
        if node not in s1_288:
            print(node)
    
    
    
    
    
    
    
    