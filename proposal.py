from baseline import *
from oraclecit import CitOracle
from castle.algorithms import PC
os.environ['CASTLE_BACKEND'] = "pytorch"


def find_groundtruth_connectivity(all_vars: list, groundtruth):
    # Initialize: each variable connected to itself
    connectivity = {var: [var] for var in all_vars}
    oracle = CitOracle(groundtruth)
    num_CIT = 0
    
    for X in connectivity.keys():
        # Now exclude only already-connected vars (including X itself, now)
        other_vars = list(set(all_vars) - set(connectivity[X]))
        for Y in other_vars:
            if oracle.query(X, Y, Z=[]) == False:
                num_CIT += 1
                connectivity[X] = list(set(connectivity[X]) | set([Y]))
                connectivity[Y] = list(set(connectivity[Y]) | set([X]))
    return connectivity , num_CIT


def separate_node_into_group(connection):
    # Step 1: Group nodes by exact neighbor set
    signature_to_nodes = {}
    for node, neighbors in connection.items():
        signature = frozenset(neighbors)
        signature_to_nodes.setdefault(signature, []).append(node)

    # Step 2: Build unsorted groups and tau
    unsorted_groups = {}
    unsorted_group_tau = {}
    group_id = 0
    for signature, nodes in signature_to_nodes.items():
        unsorted_groups[group_id] = nodes
        unsorted_group_tau[group_id] = len(signature)
        group_id += 1

    # Step 3: Sort groups by tau ascending
    sorted_items = sorted(unsorted_groups.items(), key=lambda item: unsorted_group_tau[item[0]])
    groups = {}
    group_tau = {}
    for new_id, (old_id, members) in enumerate(sorted_items):
        groups[new_id] = members
        group_tau[new_id] = unsorted_group_tau[old_id]

    return groups, group_tau


def find_ancestor_of_group(connection, groups, groups_tau):
    ancestor = {group: [] for group in groups}  # Khởi tạo ancestor
    
    for group1 in groups:
        for group2 in groups:
            # Kiểm tra điều kiện: tau của group1 nhỏ hơn group2, và group1 có kết nối với group2
            if groups_tau[group1] < groups_tau[group2]:
                # Kiểm tra tất cả các node trong group1 có kết nối với bất kỳ node nào trong group2
                if all(node1 in connection[groups[group2][0]] for node1 in connection[groups[group1][0]]):
                    ancestor[group2].append(group1)
    return ancestor


def find_group_causal_graph(oracle, ancestor, groups):
    parent = {group: [] for group in groups}
    child = {group: [] for group in groups}
    graph = []
    num_CIT = 0 
    
    markov_blanket = {}
    for group in groups:
        for anc in ancestor[group]:
            edge = 0 
            C = set(parent[anc])
            # Step 2: For each c in child(p) ∩ ancestor(u)
            for c in set(child[anc]) & set(ancestor[group]):
                C.add(c)
                C.update(parent[c])  # Thêm parent(c) vào C
            C.discard(anc)
            C.discard(group)
            # Step 3: Gather all nodes in groups of C
            blocked_nodes = set()
            for c in C:
                blocked_nodes.update(groups[c])

            markov_blanket[f"{anc} {group}"] = C 
            for node1 in groups[group]:
                for node2 in groups[anc]:
                    if oracle.query(node1, node2, Z=list(blocked_nodes)) == False:
                        num_CIT+=1
                        edge = 1 
            if edge == 1:
                graph.append([anc, group])
                parent[group].append(anc)
                child[anc].append(group)
                
    return graph, parent, child, num_CIT


def vertical_data_seperation(data_df, groups, parent):
    group_data = {}
    markov_blanket = {}

    for group in groups.keys():
        node_indexes = []

        # Add parent group nodes
        for par in parent[group]:
            node_indexes.extend(groups[par])

        # Add current group nodes
        node_indexes.extend(groups[group])
        # Get the data slice
        nodes = []
        for idx in node_indexes:
            nodes.append(data_df.columns[idx])
        data = data_df[nodes]
        # Store it by group name
        group_data[group] = data
        markov_blanket[group] = node_indexes
        
    return group_data, markov_blanket


def merge_adj_mtx(final_graph, partial_adj, child_nodes, group_markov_blanket):
    copy_final_graph = final_graph.copy()
            
    for i, src in enumerate(group_markov_blanket):
        for j, tgt in enumerate(group_markov_blanket):
            if src in child_nodes and tgt in child_nodes:
                copy_final_graph[src, tgt] = partial_adj[i, j]
            elif src not in child_nodes and tgt in child_nodes:
                if partial_adj[i, j]!=0 or partial_adj[j, i]!= 0 :
                    copy_final_graph[src, tgt] = 1 
                    copy_final_graph[tgt, src] = 0 
    return copy_final_graph


if __name__ == "__main__":
    options = read_opts()
    print("Running:", options)
    data_df, groundtruth = load_data(options)
    
    print(groundtruth.shape)
    all_vars = list(range(data_df.shape[1]))
    d = len(all_vars)
    oracle = CitOracle(groundtruth)
    
    for _ in range(options['repeat']):
        start = time.time()
        # find GR 
        
        print("start finding GR")
        groundtruth_connect , num_CIT_1 = find_groundtruth_connectivity(all_vars, groundtruth)
        groups, group_tau  = separate_node_into_group(groundtruth_connect)
        ancestor = find_ancestor_of_group(groundtruth_connect, groups, group_tau)
        causal_graph_of_groups, parent_groups, child_groups , num_CIT_2 = find_group_causal_graph(oracle, ancestor, groups)
        group_data, markov_blanket = vertical_data_seperation(data_df, groups, parent_groups)
        print(f"finished finding GR, running routine {options['routine']}")
        
        # initialize final graph
        final_graph = np.zeros([d,d])
        
        # find partial causal graph and assemble GR with partial causal graphs 
        for group_id, data_id in zip(groups.keys(), group_data.keys()):
            group = groups[group_id]
            parent_group = parent_groups[group_id]
            data = group_data[data_id]
            tmp_markov_blanket = markov_blanket[group_id]
            if len(tmp_markov_blanket) == 1:
                continue
            
            if options['routine'] == "PC":
                if options['datatype'] == "continuous":
                    pc_model = PC(variant='stable', alpha=0.3, ci_test='fisherz')
                    pc_model.learn(data=data.to_numpy())
                    adj_mtx = pc_model.causal_matrix
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                else:
                    pc_model = PC(variant='stable', alpha=0.05, ci_test='chi2')
                    pc_model.learn(data=data.to_numpy())
                    adj_mtx = pc_model.causal_matrix
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                
            
            elif options['routine'] == "SCORE":
                if options['datatype'] == "continuous":
                    algorithm = SCORE(data.to_numpy(), kwargs={'d': data.shape[1], 'eta_G': 0.01, 'eta_H': 0.01, 
                                                            'cam_cutoff': 0.01, 'pruning': 'DAS', 'threshold': 0.01, 'pns': 10})
                    adj_mtx = algorithm.inference()
                    adj_mtx = (adj_mtx > 0) * 1.
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                else:
                    raise ValueError("SCORE only supports continuous data")
                
                
            elif options['routine'] == "Varsort":
                if options['datatype'] == "continuous":
                    adj_mtx = var_sort_regress(data.to_numpy())
                    adj_mtx = 1.0 * (adj_mtx!=0)
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                else:
                    raise ValueError("Varsort only supports continuous data")
            
            
            elif options['routine'] == 'ICALiNGAM':
                if options['datatype'] == "continuous":
                    model = ICALiNGAM(thresh=0.3)
                    model.learn(data.to_numpy())
                    adj_mtx = model.causal_matrix
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                else:
                    raise ValueError("ICALiNGAM only supports continuous data")
                
                
            elif options['routine'] == 'GraNDAG':
                if options['datatype'] == "continuous":
                    model = GraNDAG(input_dim=data.shape[1], 
                                    iterations= 1000,
                                    lr=0.01,
                                    device_type='gpu')
                    model.learn(data.to_numpy())
                    adj_mtx = model.causal_matrix
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                else:
                    raise ValueError("GraNDAG only supports continuous data")
                
                
            elif options['routine'] == 'GOLEM':
                if options['datatype'] == "continuous":
                    model = GOLEM(num_iter=500, graph_thres=0.3, checkpoint_iter=499,
                                  device_type='gpu', learning_rate=0.1)
                    model.learn(data.to_numpy())
                    adj_mtx = model.causal_matrix
                    final_graph = merge_adj_mtx(final_graph=final_graph, 
                                                partial_adj=adj_mtx, 
                                                child_nodes=group, 
                                                group_markov_blanket=tmp_markov_blanket)
                else:
                    raise ValueError("GOLEM only supports continuous data")
                
            else:
                raise ValueError("Unsupported routine")
        
        
        finish = time.time()
        precision, recall, f1, S, shd = count_accuracy(B_true=groundtruth, B_est=final_graph)

        f = open(options["output"], "a")
        f.write("{},{},{},{},{},{},{},{},{}\n".format(
                options['data_path'], options['routine'],
                precision, recall, f1, S, shd, finish - start, num_CIT_1+num_CIT_2)
            )
        f.close()
        print("Writting results done!")
        
        
        
        logging_folder_dir = os.path.dirname(options['data_path'])
        
        
        # Save to file
        log_connect_dir = os.path.join(logging_folder_dir , "true_connectivity.txt")
        with open(log_connect_dir, "w") as f:
            for var, neighbors in groundtruth_connect.items():
                f.write(f"{var}: {sorted(neighbors)}\n")
            f.write(f"\nNumber of CIT: {num_CIT_1}\n")
        print("Logging connectivity done!")
        
        log_groups_dir = os.path.join(logging_folder_dir , "true_group.txt")
        with open(log_groups_dir, "w") as f:
            for group_num in sorted(groups.keys()):
                f.write(f"Group {group_num} (tau = {group_tau[group_num]}): {sorted(groups[group_num])}\n")
        print("Logging groups information done!")
        
        log_anc_dir = os.path.join(logging_folder_dir , "true_IR.txt")
        with open(log_anc_dir, "w") as f:
            for group, ancestors in ancestor.items():
                f.write(f"{group}: {sorted(ancestors)}\n")
        print("Logging ancestors groups done!")
        
        
        log_GR_dir =os.path.join(logging_folder_dir , "true_GR.txt")
        with open(log_GR_dir, "w") as f:
            for edge in causal_graph_of_groups:
                f.write(f"{edge}\n")
            
            for element in markov_blanket:
                f.write(f"{element} : {markov_blanket[element]}\n")
            f.write(f"\nNumber of CIT: {num_CIT_2}\n")
        print("Logging GR information done!")
        
        
        
        
            
                
                
        
            
                
        
        
        
        