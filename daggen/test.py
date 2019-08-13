import unittest
import torch
import numpy as np


class TestTorchDAG(unittest.TestCase):

    def setUp(self):
        from .models import TorchDAG

        self.activation_choices = [torch.relu, lambda x: .5 * x]
        batch_size = 2
        self.batch_size = batch_size
        max_v = 3
        connections = torch.zeros(batch_size, max_v, max_v,dtype=torch.uint8)
        connections[0, 1, 0] = 1
        connections[1, 1, 0] = 1
        connections[1, 2, :2] = 1
        active_vertices = torch.ones(batch_size, max_v, dtype=torch.uint8)
        active_vertices[0, 2] = 0

        activations = torch.zeros(batch_size, max_v,dtype=torch.long)
        activations[0, 1] = 1
        activations[1, 1] = 1
        activations[1, 2] = 1

        self.dag = TorchDAG(self.activation_choices, connections, activations, active_vertices)

    def test_init(self):
        self.assertEqual(len(self.activation_choices), self.dag.num_activations)

    def test_forward(self):
        x = torch.arange(self.batch_size+1, dtype=torch.float)
        with self.assertRaises(ValueError):
            y = self.dag.forward(x)

        x0 = torch.tensor(1.0)
        x = x0.expand(self.batch_size)
        y = self.dag.forward(x)
        self.assertEqual(tuple(y.shape), (self.batch_size,))
        
        target_0 = .5 * x0
        target_1 = .5 * (.5 * x0 + x0)
        target = torch.stack((target_0, target_1))
        self.assertAlmostEqual( (y-target).abs().sum().item(), 0)

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
        batch_size = 2
        lengths, all_connections, activations, log_probs = self.graphgru._sample_graph_tensors_resolved(batch_size)
        self.assertEqual(tuple(lengths.shape), (batch_size,))
        self.assertEqual(len(log_probs), len(activations))
        self.assertEqual(len(all_connections), len(activations))

if __name__ == "__main__":
    unittest.main()