"""Configuration for experiments"""

import sys
# path to daggen code
DAGGEN_PATH = "../.."
sys.path.append(DAGGEN_PATH)
from daggen.models import GraphGRU, DAG, BatchDAG
from daggen.utils import do_generative_graph_modeling, dag_dataset_from_batchdag, build_empty_graph, collate_dags