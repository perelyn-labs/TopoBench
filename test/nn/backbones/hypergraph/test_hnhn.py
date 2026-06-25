"""Unit tests for HNHN backbone."""

import pytest
import torch

from topobench.nn.backbones.hypergraph.hnhn import HNHN


def _make_incidence_sparse(num_nodes, num_edges, entries):
    """Create a sparse COO incidence matrix from (node, edge) pairs."""
    indices = torch.tensor(entries, dtype=torch.long).t()  # [2, nnz]
    values = torch.ones(indices.size(1))
    return torch.sparse_coo_tensor(indices, values, (num_nodes, num_edges))


@pytest.fixture
def simple_hypergraph():
    """A simple hypergraph: 6 nodes, 3 hyperedges.

    Hyperedge 0: {0, 1, 2}
    Hyperedge 1: {2, 3, 4}
    Hyperedge 2: {4, 5}
    """
    entries = [
        [0, 0], [1, 0], [2, 0],  # hyperedge 0
        [2, 1], [3, 1], [4, 1],  # hyperedge 1
        [4, 2], [5, 2],          # hyperedge 2
    ]
    incidence = _make_incidence_sparse(6, 3, entries)
    x = torch.randn(6, 16)
    return x, incidence


class TestHNHN:
    """Test HNHN backbone model."""

    def test_initialization(self):
        """Test default and custom initialization."""
        model = HNHN(16, 32, num_layers=2, alpha=0.0, beta=0.0)
        assert model.hidden_channels == 32
        assert model.num_layers == 2
        assert len(model.lins_e) == 2
        assert len(model.lins_v) == 2

    def test_first_layer_dim(self):
        """Test that first layer handles in_channels != hidden_channels."""
        model = HNHN(16, 32, num_layers=2)
        # First lin_e: 16 -> 32, rest: 32 -> 32
        assert model.lins_e[0].in_features == 16
        assert model.lins_e[0].out_features == 32
        assert model.lins_e[1].in_features == 32

    def test_forward_basic(self, simple_hypergraph):
        """Test basic forward pass."""
        x, incidence = simple_hypergraph
        model = HNHN(16, 32, num_layers=2, dropout=0.0)
        x_v, x_e = model(x, incidence)
        assert x_v.shape == (6, 32)
        assert x_e.shape == (3, 32)
        assert not torch.isnan(x_v).any()
        assert not torch.isnan(x_e).any()

    def test_forward_with_alpha_beta(self, simple_hypergraph):
        """Test forward with non-zero normalization exponents."""
        x, incidence = simple_hypergraph
        for alpha in [-1.0, 0.0, 0.5, 1.0]:
            for beta in [-1.0, 0.0, 0.5, 1.0]:
                model = HNHN(16, 32, num_layers=1, alpha=alpha, beta=beta, dropout=0.0)
                x_v, x_e = model(x, incidence)
                assert x_v.shape == (6, 32)
                assert x_e.shape == (3, 32)
                assert not torch.isnan(x_v).any(), f"NaN with alpha={alpha}, beta={beta}"

    def test_forward_edge_index_format(self, simple_hypergraph):
        """Test forward with dense edge_index instead of sparse COO."""
        x, incidence = simple_hypergraph
        # Convert sparse to edge_index format
        edge_index, _ = torch_geometric_to_edge_index(incidence)

        model = HNHN(16, 32, num_layers=1, dropout=0.0)
        x_v, x_e = model(x, edge_index)
        assert x_v.shape == (6, 32)
        assert x_e.shape == (3, 32)

    def test_backward_pass(self, simple_hypergraph):
        """Test gradient computation."""
        x, incidence = simple_hypergraph
        x = x.clone().requires_grad_(True)
        model = HNHN(16, 32, num_layers=2, dropout=0.0)
        x_v, x_e = model(x, incidence)
        (x_v.sum() + x_e.sum()).backward()
        assert x.grad is not None
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad

    def test_deterministic_eval(self, simple_hypergraph):
        """Test deterministic outputs in eval mode."""
        x, incidence = simple_hypergraph
        model = HNHN(16, 32, num_layers=2, dropout=0.5)
        model.eval()
        x_v1, x_e1 = model(x, incidence)
        x_v2, x_e2 = model(x, incidence)
        assert torch.allclose(x_v1, x_v2)
        assert torch.allclose(x_e1, x_e2)

    @pytest.mark.parametrize("num_layers", [1, 2, 3, 4])
    def test_parametrized_layers(self, simple_hypergraph, num_layers):
        """Test different numbers of layers."""
        x, incidence = simple_hypergraph
        model = HNHN(16, 32, num_layers=num_layers, dropout=0.0)
        x_v, x_e = model(x, incidence)
        assert x_v.shape == (6, 32)
        assert x_e.shape == (3, 32)

    @pytest.mark.parametrize("hidden_channels", [8, 16, 32, 64])
    def test_parametrized_hidden_dims(self, simple_hypergraph, hidden_channels):
        """Test different hidden dimensions."""
        x, incidence = simple_hypergraph
        model = HNHN(16, hidden_channels, num_layers=1, dropout=0.0)
        x_v, x_e = model(x, incidence)
        assert x_v.shape == (6, hidden_channels)
        assert x_e.shape == (3, hidden_channels)

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.3, 0.5])
    def test_parametrized_dropout(self, simple_hypergraph, dropout):
        """Test different dropout rates."""
        x, incidence = simple_hypergraph
        model = HNHN(16, 32, num_layers=1, dropout=dropout)
        x_v, x_e = model(x, incidence)
        assert x_v.shape == (6, 32)

    def test_large_hypergraph(self):
        """Test with a larger random hypergraph."""
        num_nodes, num_edges = 100, 30
        # Random incidence: ~5 nodes per hyperedge
        entries = []
        for e in range(num_edges):
            nodes = torch.randint(0, num_nodes, (5,)).unique().tolist()
            entries.extend([n, e] for n in nodes)
        incidence = _make_incidence_sparse(num_nodes, num_edges, entries)
        x = torch.randn(num_nodes, 16)

        model = HNHN(16, 32, num_layers=2, dropout=0.0)
        x_v, x_e = model(x, incidence)
        assert x_v.shape == (num_nodes, 32)
        assert x_e.shape == (num_edges, 32)

    def test_aggregation_correctness(self):
        """Test that aggregation computes correct values for alpha=0, beta=0."""
        # 3 nodes, 1 hyperedge containing all 3
        entries = [[0, 0], [1, 0], [2, 0]]
        incidence = _make_incidence_sparse(3, 1, entries)
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

        model = HNHN(2, 2, num_layers=1, alpha=0.0, beta=0.0, dropout=0.0)
        # With alpha=0, beta=0: aggregation is mean
        # N->E: hyperedge gets mean of 3 node features
        # After lin_e + relu, then E->N: all nodes get same hyperedge feature
        # After lin_v + relu, all nodes should have same features
        model.eval()
        x_v, x_e = model(x, incidence)
        # All nodes receive the same hyperedge representation
        assert torch.allclose(x_v[0], x_v[1])
        assert torch.allclose(x_v[1], x_v[2])


# Helper to avoid import issues in the edge_index format test
def torch_geometric_to_edge_index(sparse_tensor):
    """Convert sparse tensor to edge_index."""
    import torch_geometric.utils
    return torch_geometric.utils.to_edge_index(sparse_tensor)
