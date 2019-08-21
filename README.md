# dag-gen

Code for generating DAGs. The RNN models for DAGs are based on the [GraphRNN](https://arxiv.org/abs/1802.08773) paper.

## Running tests
In the terminal:
```bash
./test
```

## to do
* Allow for scaling of edges during forward pass
* Speed up the DAG forward pass
    * First, benchmark it
    * maybe do topological sort after graph construction? Then each layer can be updated in parallel. 