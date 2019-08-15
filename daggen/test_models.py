import unittest
import torch
import numpy as np

class TestMLP(unittest.TestCase):

    def setUp(self):
        from .models import MLP
        self.layer_sizes1 = [10, 2]
        self.layer_sizes2 = [10, 5, 2]
        self.mlp1 = MLP(self.layer_sizes1)
        self.mlp2 = MLP(self.layer_sizes2)

    def test_sizes(self):
        self.assertEqual(self.mlp1.num_layers, 1)
        self.assertEqual(self.mlp2.num_layers, 2)

    def test_mlp_dimensions(self):
        x = torch.ones(13, 10)
        y1 = self.mlp1(x)
        y2 = self.mlp2(x)
        self.assertEqual(tuple(y1.shape), (13, 2))
        self.assertEqual(tuple(y2.shape), (13, 2))

class TestTwoLayerMLP(unittest.TestCase):

    def test_sizes(self):
        from .models import TwoLayerMLP
        mlp = TwoLayerMLP(5, 4, 2)
        self.assertEqual(mlp.num_layers, 2)

class TestScalarGraphGRU(unittest.TestCase):

    def setUp(self):
        from .models import ScalarGraphGRU
        hidden_size=16
        logits_hidden_size=4
        num_activations = 5
        self.test_graph_gru = ScalarGraphGRU(hidden_size, logits_hidden_size, num_activations)
        
    def test_build(self):
        from .models import ScalarGraphGRU
        hidden_size=16
        logits_hidden_size=4
        num_activations = 5
        gg = ScalarGraphGRU(hidden_size, logits_hidden_size, num_activations)

    def test_modules(self):
        for module in ['vertex_cell', 'edge_cell', 'activation_cell', 'vertex_logits', 'edge_logits', 'activation_logits']:
            self.assertTrue(hasattr(self.test_graph_gru, module))

    def test_params(self):
        params = list(self.test_graph_gru.named_parameters(recurse=False))
        self.assertTrue(len(params)==2)
        param_names = [p[0] for p in params]
        self.assertTrue('hidden_init' in param_names)
        self.assertTrue('vertex_input_init' in param_names)

    def test__get_samples_and_log_probs(self):
        l1 = torch.randn(10, 4)
        
        s, lps = self.test_graph_gru._get_samples_and_log_probs(l1)
        self.assertEqual(tuple(s.shape), (10,))
        self.assertTrue(max(s) < 4)
        
    def test__sample_graph_tensors_resolved_logprobs(self):
        batch_size=5
        max_vertices=4
        next_active_vertices, connections, activations, log_probs = self.test_graph_gru._sample_graph_tensors_resolved_logprobs(batch_size, max_vertices=max_vertices)

        max_vertices = next_active_vertices.size(1)
        self.assertEqual(tuple(log_probs.shape), (batch_size,max_vertices))

    def test_sample_graph_tensors(self):
        batch_size=5
        max_vertices=None
        next_active_vertices, connections, activations, log_probs = self.test_graph_gru.sample_graph_tensors(batch_size, max_vertices=max_vertices)
        #check that all outputs have the right shape
        max_vertices = next_active_vertices.size(1)
        for t in (next_active_vertices, activations):
            self.assertEqual(tuple(t.shape), (batch_size, max_vertices))
        self.assertEqual(tuple(connections.shape), (batch_size, max_vertices, max_vertices))
        self.assertEqual(tuple(log_probs.shape), (batch_size,))

        #active_vertices indicates whether another vertex ought to be added. At the final step this should be false for all graphs
        self.assertEqual(next_active_vertices[:, -1].sum().item(), 0)
        #also, a graph should never go from being inactive to active
        react = np.diff(next_active_vertices.to(dtype=torch.long).numpy(), axis=1) > 0
        self.assertEqual(np.sum(react), 0)

        #no vertex should have connections flowing into it from future vertices
        for i in range(max_vertices):
            self.assertEqual(connections[:, i, i:].sum().item(), 0)

        #activations entries should lie within the expected range
        self.assertTrue( 0 <= activations.min().item() and activations.max().item() < self.test_graph_gru.num_activations )

        #log probs are nonpositive
        self.assertEqual( (log_probs <=0).sum().item(), batch_size )

    def test_min_vertices(self):
        batch_size=5
        min_vertices=4
        max_vertices=4
        next_active_vertices, connections, activations, __ = self.test_graph_gru.sample_graph_tensors(batch_size, min_vertices=min_vertices,
                                                                                                                    max_vertices=max_vertices)
        for tensor in (next_active_vertices, connections, activations):
            self.assertTrue(tensor.size(1) == min_vertices)

class TestGraphGRU(unittest.TestCase):

    def setUp(self):
        from .models import GraphGRU
        
        self.num_input = 2
        self.num_output = 3
        hidden_size = 10
        logits_hidden_size = 10
        self.num_activations = 4
        self.graphgru = GraphGRU(self.num_input, self.num_output, hidden_size, logits_hidden_size, self.num_activations)

    def test__sample_graph_tensors_resolved(self):
        """ Check that vertex-resolved tensors have the expected shape"""
        batch_size = 2
        lengths, all_connections, activations, log_probs = self.graphgru._sample_graph_tensors_resolved(batch_size)
        maxnum = lengths.max().item()
        self.assertEqual(len(activations), maxnum + self.num_output)
        self.assertEqual(tuple(lengths.shape), (batch_size,))
        self.assertEqual(len(log_probs), len(activations))
        self.assertEqual(len(all_connections), len(activations))

    def test_sample_graph_tensors(self):
        """ Check that sampled graph tensors have the correct shape"""
        batch_size=5
        num_intermediate, activations, connections, log_probs = self.graphgru.sample_graph_tensors(batch_size)
        maxnum = num_intermediate.max().item()
        max_num_emitting = maxnum + self.num_input
        max_num_act = maxnum + self.num_output

        #check that the shapes come out right
        self.assertEqual(tuple(num_intermediate.shape), (batch_size,))
        self.assertEqual(tuple(log_probs.shape), (batch_size,))
        self.assertEqual(tuple(activations.shape), (batch_size, max_num_act))
        self.assertEqual(tuple(connections.shape), (batch_size, max_num_act, max_num_emitting))

        #check that all the graph tensors are "left-justified"
        for i in range(batch_size):
            ni=num_intermediate[i]
            num_act = ni + self.num_output
            num_act_unused = max_num_act - num_act
            num_emitting = ni + self.num_input
            num_emitting_unused = max_num_emitting - num_emitting

            self.assertEqual( (activations[i]==-1).sum().item(), num_act_unused)
            self.assertEqual( connections[i, num_act:, ...].sum().item(), 0)
            for j in range(num_act):
                self.assertEqual( connections[i, j, num_emitting:].sum().item(), 0)
    
    def test_sample_dags_with_log_probs(self):
        """ Check that BatchDAG sampling works"""
        batch_size = 2
        dags, log_probs = self.graphgru.sample_dags_with_log_probs(batch_size)
        self.assertEqual(tuple(log_probs.shape), (batch_size,))
        self.assertEqual(len(dags), batch_size)

    def test_sampling_size_constraints(self):
        """Check that constraints on min/max number of intermediate vertices are obeyed."""
        batch_size = 5
        num_int = 3
        num_intermediate, activations, connections, log_probs = self.graphgru.sample_graph_tensors(batch_size, max_intermediate_vertices=num_int, 
                                                                                                    min_intermediate_vertices=num_int)
        self.assertEqual(tuple(activations.shape), (batch_size, num_int + self.num_output))
        self.assertEqual((num_intermediate - num_int).abs().sum(), 0)

class TestBatchDAG(unittest.TestCase):

    def setUp(self):
        from .models import GraphGRU
        from .models import BatchDAG
        self.input_dim = 2
        self.output_dim = 3
        hidden_size = 2
        logits_hidden_size = 2
        self.num_activations = 4
        self.graphgru = GraphGRU(self.input_dim, self.output_dim, hidden_size, 
                                logits_hidden_size, self.num_activations)

        self.batch_size = 3
        num_intermediate, activations, connections, log_probs = self.graphgru.sample_graph_tensors(self.batch_size, 
                                                                                                      )
        self.num_intermediate = num_intermediate
        self.activations = activations
        self.connections = connections
        
        self.dag = BatchDAG(self.input_dim, self.output_dim, self.num_intermediate, 
                    self.connections, self.activations)


    def test_build(self):
        """Check that the BatchDAG builds"""
        pass

    def test_forward_shape(self):
        """Check that forward pass produces outputs of expected shape."""
        activation_functions = [lambda x: x, lambda x: -x, torch.relu, torch.cos]
        x = torch.randn(self.input_dim)
        y = self.dag.forward(x, activation_functions)
        self.assertEqual(tuple(y.shape), (self.batch_size, self.output_dim))
        x = torch.randn(3, self.input_dim)
        y = self.dag.forward(x, activation_functions)
        self.assertEqual(tuple(y.shape), (3, self.batch_size, self.output_dim))

    def test_forward(self):
        """ Check that batched BatchDAGs actually output the correct result for known examples."""
        from .models import BatchDAG
        input_dim = 2
        output_dim = 1
        activation_functions = [lambda x: x, lambda x : -x ]

        conns0 = torch.tensor([[1,1,0], [0, 1, 1]], dtype=torch.uint8)
        conns1 = torch.tensor([[1,1,0], [0,0,0]], dtype=torch.uint8)
        connections = torch.stack((conns0, conns1), dim=0)

        activations = torch.tensor([[1, 0], [0, -1]], dtype=torch.long)
        num_intermediate = torch.tensor([1, 0], dtype=torch.long)

        dag = BatchDAG(input_dim, output_dim, num_intermediate, connections, activations)

        x = torch.tensor([[1, 2], [0, 3]], dtype=torch.float)
        y = dag.forward(x, activation_functions)
        target = torch.tensor([[-1, 3], [0, 3]], dtype=torch.float).view(2, 2, 1)
        self.assertAlmostEqual((y - target).abs().sum().item(), 0)

    def test_build_graphviz(self):
        """ Check that digraphs build OK"""
        digraphs = self.dag.to_graphviz(['a', 'b', 'c', 'd'])
        self.assertEqual(len(digraphs), self.batch_size)

class TestDAG(unittest.TestCase):

    def setUp(self):
        from .models import DAG
        self.input_dim = 1
        self.output_dim = 2
        num_intermediate = 1
        connections = torch.tensor( [[1, 0], [ 1, 1 ], [1, 1]],dtype=torch.uint8)
        activations = torch.tensor([0, 0, 1], dtype=torch.long)
        
        self.dag = DAG(self.input_dim, self.output_dim, num_intermediate, connections, activations)


    def test_build(self):
        from .models import DAG
        input_dim = 1
        output_dim = 2
        num_intermediate = 1
        connections = torch.tensor( [[1, 0], [ 1, 1 ], [1, 1]],dtype=torch.uint8)
        activations = torch.tensor([0, 0, 1], dtype=torch.long)
        
        valid_dag = DAG(input_dim, output_dim, num_intermediate, connections, activations, check_valid=True)

        invalid_connections = torch.tensor( [[1, 1], [ 1, 1 ], [1, 1]],dtype=torch.uint8)
        with self.assertRaises(ValueError):
            invalid_dag = DAG(input_dim, output_dim, num_intermediate, invalid_connections, activations, check_valid=True)

    def test_forward(self):
        x = torch.tensor([0, 1]).view(2, 1)
        activation_choices = [lambda x: x, lambda x: torch.ones_like(x)]
        y = self.dag.forward(x, activation_choices)
        target = torch.tensor([ [0, 1], [2, 1] ], dtype=torch.float)
        self.assertAlmostEqual( (y - target).abs().sum().item(), 0)

    def test_to_graphviz(self):
        from graphviz import Digraph
        g = self.dag.to_graphviz(['a', 'b'])
        self.assertTrue(isinstance(g, Digraph))

if __name__ == "__main__":
    unittest.main()