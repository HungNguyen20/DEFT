import networkx as nx
from collections import deque


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
        self.num_query = 0

    def query(self, A, B, Z):
        """
        Checks whether node A is d-separated from node B given a set of observed nodes Z.
        A, B: node indices (int)
        Z: list or set of observed node indices
        Returns True if A ⫫ B | Z (i.e., A is d-separated from B given Z), else False
        """
        self.num_query += 1
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

    
    
    
    
    