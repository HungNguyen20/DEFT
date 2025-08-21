import time 
import numpy as np 
import pandas as pd 
import argparse
# from combine import Plus_PC
import os 
from SADA import SADA 
import networkx as nx 
from combine import count_accuracy
from combine import edge_to_graph



Algorithm = ['CPA' , 'SADA' , 'CAPA' , 'HCD']



def read_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm" , type=str, default="CPA" , choices=["CPA" , "CAPA" , "SADA" , "HCD"])
    parser.add_argument("--data_path", type=str, default="data.csv") # Must have headers
    parser.add_argument("--groundtruth_path", type=str, default="graph.csv") # Must have headers, same as the one above
    parser.add_argument("--output", type=str, default="res.csv")
    parser.add_argument("--datatype", type=str, default="continuous", choices=["continuous", "discrete"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--routine", type=str, choices=["PC", "GES", "GIES", 
                                                            "Notears", "NotearsNonlinear", "NotearsLowRank", "GOLEM",
                                                            "DAS", "SCORE",
                                                            "Varsort", "R2sort",
                                                            "ICALiNGAM", "DirectLiNGAM",
                                                            "GraNDAG"], default="PC")
    parser.add_argument("--threshold" , type= int , default=10)
    parser.add_argument("--maxCset", type=int, default= 3) 
    
    options = vars(parser.parse_args())
    return options


def read_groundtruth(data_dir):

    data = pd.read_csv(f"{os.path.join(data_dir , 'data.csv')}")
    stru_GT = pd.read_csv(os.path.join(data_dir, 'groundtruth.csv')).to_numpy()
    return data, stru_GT


def categorize_data(data):
    ...
    return data 


if __name__ == "__main__":

    options = read_opts()
    results = {}
    results[f'{options["algorithm"]}'] = []
    for i in range(options['repeat']):
        
        data = pd.read_csv(options['data_path'])
        stru_GT = pd.read_csv(options['groundtruth_path']).to_numpy()

        if options['datatype'] == 'discrete':
            data = categorize_data(data)

        #Run partition and combine 
        
        if options['algorithm'] == 'SADA':
            start_time = time.time()
            edge_list = SADA(dataset=data , V_set= [i for i in range(stru_GT.shape[0])], stru_GT= stru_GT ,options = options)
            elapsed = time.time() - start_time

            final_graph = edge_to_graph(stru_GT , edge_list)
            precision, recall, f1, S, shd = count_accuracy(stru_GT , final_graph)
            result = [precision, recall, f1, S, shd]
            results[options['algorithm']].append(result)
        elif options== 'CAPA':
            ...
        
        print(f"{options['algorithm']} completed in {elapsed:.4f} seconds")
        print(result)



    
