"""HNHN backbone for TopoBench.

Implements HNHN from:
Dong et al., "HNHN: Hypergraph Networks with Hyperedge Neurons",
ICML 2020 Workshop on Graph Representation Learning and Beyond.
https://arxiv.org/abs/2006.12278

HNHN is a hypergraph convolution network that treats hyperedges as
first-class citizens with their own learned representations. Each layer
alternates between node-to-hyperedge and hyperedge-to-node aggregation,
with nonlinear activation applied to both, and configurable degree-based
normalization via hyperparameters alpha and beta.

Update equations per layer (Algorithm 1 from the paper):
    X_E = sigma(D_{E,l,beta}^{-1} A^T D_{V,r,beta} X_V . W_E + b_E)
    X_V = sigma(D_{V,l,alpha}^{-1} A D_{E,r,alpha} X_E . W_V + b_V)

where A is the incidence matrix [n_nodes, n_hyperedges].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.utils import scatter


class HNHN(nn.Module):
    """HNHN (Hypergraph Networks with Hyperedge Neurons) backbone.

    Alternates between node and hyperedge feature updates using the
    incidence matrix, with degree-based normalization controlled by
    alpha (hyperedge cardinality weighting) and beta (node degree
    weighting).

    Parameters
    ----------
    in_channels : int
        Input node feature dimension.
    hidden_channels : int
        Hidden and output embedding dimension.
    num_layers : int, optional
        Number of HNHN convolution layers. Default is 2.
    alpha : float, optional
        Hyperedge cardinality normalization exponent. When alpha > 0,
        larger hyperedges contribute more; when alpha < 0, less.
        Default is 0.0 (equal weighting).
    beta : float, optional
        Node degree normalization exponent. When beta > 0, higher-degree
        nodes contribute more; when beta < 0, less.
        Default is 0.0 (equal weighting).
    dropout : float, optional
        Dropout rate applied after each aggregation. Default is 0.5.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        alpha=0.0,
        beta=0.0,
        dropout=0.5,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.alpha = alpha
        self.beta = beta
        self.dropout_rate = dropout

        # Per-layer linear transforms for hyperedge and node updates
        self.lins_e = nn.ModuleList()
        self.lins_v = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_channels if i == 0 else hidden_channels
            self.lins_e.append(nn.Linear(d_in, hidden_channels))
            self.lins_v.append(nn.Linear(hidden_channels, hidden_channels))

    def _to_edge_index(self, incidence):
        """Convert incidence matrix to edge_index format if needed.

        Parameters
        ----------
        incidence : torch.Tensor
            Incidence matrix, either sparse COO [n, m] or edge_index [2, nnz].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, int]
            (V, E, num_hyperedges): node indices, hyperedge indices,
            and total number of hyperedges.
        """
        if incidence.layout == torch.sparse_coo:
            num_hyperedges = incidence.size(1)
            edge_index, _ = torch_geometric.utils.to_edge_index(incidence)
        else:
            edge_index = incidence
            num_hyperedges = (
                int(edge_index[1].max().item()) + 1
                if edge_index.numel() > 0
                else 0
            )
        return edge_index[0], edge_index[1], num_hyperedges

    def _aggregate_n2e(self, x_v, V, E, node_deg, num_edges):
        """Aggregate node features to hyperedges with beta-normalization.

        Implements: D_{E,l,beta}^{-1} A^T D_{V,r,beta} X_V

        Parameters
        ----------
        x_v : torch.Tensor
            Node features [num_nodes, d].
        V : torch.Tensor
            Node indices from incidence pairs.
        E : torch.Tensor
            Hyperedge indices from incidence pairs.
        node_deg : torch.Tensor
            Node degrees [num_nodes].
        num_edges : int
            Total number of hyperedges.

        Returns
        -------
        torch.Tensor
            Aggregated hyperedge features [num_edges, d].
        """
        if self.beta != 0.0:
            weight = node_deg[V].pow(self.beta)
            weighted = x_v[V] * weight.unsqueeze(1)
            agg = scatter(weighted, E, dim=0, dim_size=num_edges, reduce="sum")
            norm = scatter(weight, E, dim=0, dim_size=num_edges, reduce="sum")
            return agg / norm.unsqueeze(1).clamp(min=1e-6)
        return scatter(x_v[V], E, dim=0, dim_size=num_edges, reduce="mean")

    def _aggregate_e2n(self, x_e, V, E, edge_card, num_nodes):
        """Aggregate hyperedge features to nodes with alpha-normalization.

        Implements: D_{V,l,alpha}^{-1} A D_{E,r,alpha} X_E

        Parameters
        ----------
        x_e : torch.Tensor
            Hyperedge features [num_edges, d].
        V : torch.Tensor
            Node indices from incidence pairs.
        E : torch.Tensor
            Hyperedge indices from incidence pairs.
        edge_card : torch.Tensor
            Hyperedge cardinalities [num_edges].
        num_nodes : int
            Total number of nodes.

        Returns
        -------
        torch.Tensor
            Aggregated node features [num_nodes, d].
        """
        if self.alpha != 0.0:
            weight = edge_card[E].pow(self.alpha)
            weighted = x_e[E] * weight.unsqueeze(1)
            agg = scatter(weighted, V, dim=0, dim_size=num_nodes, reduce="sum")
            norm = scatter(weight, V, dim=0, dim_size=num_nodes, reduce="sum")
            return agg / norm.unsqueeze(1).clamp(min=1e-6)
        return scatter(x_e[E], V, dim=0, dim_size=num_nodes, reduce="mean")

    def forward(self, x_0, incidence_hyperedges):
        """Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features of shape [num_nodes, in_channels].
        incidence_hyperedges : torch.Tensor
            Incidence matrix (sparse COO [n, m]) or edge_index [2, nnz].

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Node embeddings [num_nodes, hidden_channels] and
            hyperedge embeddings [num_hyperedges, hidden_channels].
        """
        V, E, num_edges = self._to_edge_index(incidence_hyperedges)
        num_nodes = x_0.size(0)

        # Precompute degrees
        ones = torch.ones(V.size(0), device=V.device)
        node_deg = scatter(ones, V, dim=0, dim_size=num_nodes, reduce="sum")
        edge_card = scatter(ones, E, dim=0, dim_size=num_edges, reduce="sum")

        x_v = x_0
        x_e = None
        for i in range(self.num_layers):
            # Node -> Hyperedge (Eq. from Sec. 2.3)
            x_e = self._aggregate_n2e(x_v, V, E, node_deg, num_edges)
            x_e = self.lins_e[i](x_e)
            x_e = F.relu(x_e)
            x_e = F.dropout(x_e, p=self.dropout_rate, training=self.training)

            # Hyperedge -> Node (Eq. from Sec. 2.3)
            x_v = self._aggregate_e2n(x_e, V, E, edge_card, num_nodes)
            x_v = self.lins_v[i](x_v)
            x_v = F.relu(x_v)
            x_v = F.dropout(x_v, p=self.dropout_rate, training=self.training)

        return x_v, x_e
