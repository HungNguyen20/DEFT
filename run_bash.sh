python baseline.py --routine PC --data_path "data/win95pts/data_1000.csv" --groundtruth "data/win95pts/graph.csv" --output "res/bnlearn-baseline.csv" --datatype "discrete"
python proposal.py --routine PC --data_path "data/win95pts/data_1000.csv" --groundtruth "data/win95pts/graph.csv" --output "res/proposal-baseline.csv" --datatype "discrete"
