import unittest
import torch


def tensor_diff(x, y):
    return (x - y).abs().sum().item()

class Test(unittest.TestCase):
    
    def assertTensorAlmostEqual(self, x, y):
        """Checks that two tensors are almost equal"""
        self.assertAlmostEqual(tensor_diff(x, y), 0)


class TestDAGUtils(Test):

    def test_is_valid_adjacency_matrix(self):
        """ Check whether adjacency tensors are correctly identified."""
        from daggen.utils import is_valid_adjacency_matrix
        input_dim = 2
        output_dim = 1
        num_intermediate = 1
        #ok
        conn1 = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.uint8)
        #too small
        conn2 = torch.zeros(output_dim, input_dim, dtype=torch.uint8)
        #inconsistent shape
        conn3 = torch.zeros(output_dim + 3, input_dim + 2, dtype = torch.uint8)
        # has non-ancestral connections
        conn4 = torch.tensor([[1, 0, 1], [1, 1, 1]], dtype=torch.uint8)
        # is not left-justified (embedded in dag space with 2 intermediate)
        conn5 = torch.tensor([[1, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.uint8)
        
        self.assertTrue(is_valid_adjacency_matrix(conn1, num_intermediate, input_dim, output_dim))
        for conn in (conn2, conn3, conn4, conn5):
            self.assertFalse(is_valid_adjacency_matrix(conn, num_intermediate, input_dim, output_dim))

    def test_build_graphviz(self):
        """ Check graphviz constructor."""
        from daggen.utils import build_graphviz
        input_dim = 2
        output_dim = 1
        num_intermediate = 1

        conns = torch.tensor([[1,1,0], [0, 1, 1]], dtype=torch.uint8)
        activations = torch.tensor([1, 0], dtype=torch.long)
        activation_labels = ['id', 'inv']
        dag = build_graphviz(input_dim, output_dim, num_intermediate, conns, activations, activation_labels)
        self.assertTrue(dag.directed)

class TestModelUtils(Test):

    def test_get_activation(self):
        from daggen.utils import get_activation
        f = get_activation('id')
        x = torch.ones(4,4)
        self.assertAlmostEqual((x - f(x)).abs().sum(), 0)
        f = get_activation('inv')
        self.assertAlmostEqual((x + f(x)).abs().sum(), 0)
        f = get_activation('gauss')
        self.assertAlmostEqual((f(x) - torch.tensor(-1.).exp() ).abs().sum(), 0)

    def test_right_justify(self):
        from daggen.utils import right_justify
        a = torch.tensor([[1, 1, 2, -1, -1],
                          [5, 6,7, 8, -1]])
        target = torch.tensor([[1, 0, 0, 1, 2],
                                 [5, 6, 0, 7,8]])
        output_dim =2
        lengths = [1, 2]
        b = right_justify(a, lengths, output_dim)
        self.assertTensorAlmostEqual(target, b)

    def test_to_resolved_tensors(self):
        from daggen.utils import to_resolved_tensors
        batch_size = 2
        num_in = 1
        num_out = 1
        num_itermediate = torch.tensor([1, 2])
        max_int = num_itermediate.max().item()
        connections = torch.zeros(batch_size, max_int + num_out, max_int + num_in, dtype=torch.uint8)
        connections[0, 0, 0] = 1
        connections[0, 1, 1] = 1
        connections[1, 0, 0] = 1
        connections[1, 1, 1] = 1
        connections[1, 2, 2] = 1
        activations =torch.tensor([[0, 1, -1], [1, 1, 2]])

        conns_resolved, activations_resolved = to_resolved_tensors(num_itermediate, connections, activations,
                                                                        num_in, num_out)
        self.assertEqual(len(conns_resolved), max_int + num_out)
        self.assertEqual(len(activations_resolved), max_int + num_out)
        self.assertTensorAlmostEqual(activations_resolved[0], torch.tensor([0, 1]))
        self.assertTensorAlmostEqual(activations_resolved[1], torch.tensor([0, 1]))
        self.assertTensorAlmostEqual(activations_resolved[2], torch.tensor([1, 2]))

        self.assertTensorAlmostEqual(conns_resolved[0], torch.tensor([[1], [1]], dtype=torch.uint8))
        self.assertTensorAlmostEqual(conns_resolved[1], torch.tensor([[0,0], [0,1]], dtype=torch.uint8))
        self.assertTensorAlmostEqual(conns_resolved[2], torch.tensor([[0,1,0], [0,0,1]], dtype=torch.uint8))

    def test_dag_dataset_from_tensors(self):
        from daggen.utils import dag_dataset_from_tensors
        batch_size = 2
        num_in = 1
        num_out = 1
        num_itermediate = torch.tensor([1, 2])
        max_int = num_itermediate.max().item()
        connections = torch.zeros(batch_size, max_int + num_out, max_int + num_in, dtype=torch.uint8)
        connections[0, 0, 0] = 1
        connections[0, 1, 1] = 1
        connections[1, 0, 0] = 1
        connections[1, 1, 1] = 1
        connections[1, 2, 2] = 1
        activations =torch.tensor([[0, 1, -1], [1, 1, 2]])
        ds = dag_dataset_from_tensors(num_in, num_out, 
                                      num_itermediate, connections, activations)
        self.assertEqual(len(ds), 2)
        self.assertTensorAlmostEqual(ds[0].connections, connections[0, ...])
        self.assertTensorAlmostEqual(ds[0].activations, activations[0, ...])

    def test_collate_dags(self):
        from daggen.models import BatchDAG
        from daggen.utils import collate_dags
        batch_size = 2
        num_in = 1
        num_out = 1
        num_itermediate = torch.tensor([1, 2])
        max_int = num_itermediate.max().item()
        connections = torch.zeros(batch_size, max_int + num_out, max_int + num_in, dtype=torch.uint8)
        connections[0, 0, 0] = 1
        connections[0, 1, 1] = 1
        connections[1, 0, 0] = 1
        connections[1, 1, 1] = 1
        connections[1, 2, 2] = 1
        activations =torch.tensor([[0, 1, -1], [1, 1, 2]])

        bd = BatchDAG(num_in, num_out, num_itermediate, connections, activations)
        bd2 = collate_dags([d for d in bd])
        self.assertTensorAlmostEqual(bd.connections, bd2.connections)
        self.assertTensorAlmostEqual(bd.activations, bd2.activations)

    def test_build_empty_graph(self):
        from daggen.utils import build_empty_graph
        input_dim = 2
        output_dim = 2
        num_intermediate = 4
        dag = build_empty_graph(input_dim, output_dim, num_intermediate)
        self.assertTensorAlmostEqual(dag.activations, torch.zeros(1, num_intermediate + output_dim, dtype=torch.long))
        self.assertAlmostEqual(dag.connections.sum().item(), 0)


    def test_build_fully_connected_graph(self):
        from daggen.utils import build_fully_connected_graph
        input_dim = 2
        output_dim = 2
        num_intermediate = 4
        dag = build_fully_connected_graph(input_dim, output_dim, num_intermediate)
        self.assertAlmostEqual(dag.connections.sum().item(), sum(input_dim + i for i in range(num_intermediate)) + 
                                                            output_dim * (num_intermediate + input_dim) )

# skip becuase this involves actual training...
@unittest.skip
class TestTrainingUtils(Test):

    def test_do_score_training(self):
        """ Check that the mechanics of score training are ok"""
        from daggen.models import GraphGRU
        from daggen.utils import do_score_training
        from torch.optim import SGD
        
        samples = 10
        batch_size = 3
        
        input_dim = output_dim = num_activations = 1
        hidden_size = logits_hidden_size = 1
        dag_model = GraphGRU(input_dim, output_dim, hidden_size, logits_hidden_size, num_activations)
        dag_model.activation_functions = [lambda x : x]
        score = lambda d : 1.0
        optimizer = SGD(dag_model.parameters(), lr=.1)

        nets, log_probs = dag_model.sample_dags_with_log_probs(1)
        batch_scores = do_score_training(dag_model, score, samples, batch_size, optimizer)

        self.assertEqual(len(batch_scores), 4)



if __name__ == "__main__":
    unittest.main()

