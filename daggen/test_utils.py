import unittest
import torch

class TestDAGChecks(unittest.TestCase):
    """Check whether DAG-testing functions work as expected."""

    def test_is_valid_adjacency_matrix(self):
        """ Check whether adjacency tensors are correctly identified."""
        from .utils import is_valid_adjacency_matrix
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

if __name__ == "__main__":
    unittest.main()

