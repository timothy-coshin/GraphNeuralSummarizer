import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv
from torch_scatter import scatter
from torch_geometric.data import Data, Batch
from sklearn.cluster import MiniBatchKMeans


# =============================================================================
# GNN Backbone Models
# =============================================================================

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr


class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1, edge_feature=True):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()

        if edge_feature:
            self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels // num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        else:
            self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels // num_heads, heads=num_heads, dropout=dropout))

        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            if edge_feature:
                self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels // num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
            else:
                self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels // num_heads, heads=num_heads, dropout=dropout))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        if edge_feature:
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels // num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        else:
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels // num_heads, heads=num_heads, dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr


# =============================================================================
# Graph Neural Summarizer (GNS)
#
# Implements Algorithm 1 from:
#   Kim & Kim, "Addressing information bottlenecks in graph augmented large
#   language models via graph neural summarization", Information Fusion, 2026.
#
# Architecture mapping (Algorithm 1 → code):
#   Line 1: GNNquery  → self.question_node_message_model
#   Line 2: GNNnode   → self.node_node_message_model
#   Line 3: Clustering → implicit via learnable token attention
#   Line 5: GNNpool   → self.node_graphtoken_message_model
#   Line 7: Projection → handled externally in gns_llm.py (self.projector)
# =============================================================================

class GNS(torch.nn.Module):
    """
    Graph Neural Summarizer: generates k query-aware prompt vectors
    from graph-structured data for integration with LLMs.
    
    Args:
        in_channels: Input feature dimension (from pretrained text encoder)
        hidden_channels: Hidden dimension for GNN layers
        out_channels: Output dimension
        num_layers: Number of GNN layers for node-node message passing
        dropout: Dropout rate
        num_heads: Number of attention heads (required for gt/gat)
        num_graph_token: Number of learnable graph tokens (k in the paper)
        edge_feature: Whether to use edge features in message passing
        gnn: GNN backbone type ('gt', 'gat', or 'gcn')
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout,
                 num_heads=-1, num_graph_token=8, edge_feature=True, gnn='gt'):
        super(GNS, self).__init__()

        if gnn not in ['gcn', 'gat', 'gt']:
            raise NotImplementedError('gnn must be one of: GraphTransformer (gt), GAT (gat), GCN (gcn)')

        # --- GNNquery: Query-aware node encoding (Algorithm 1, Line 1) ---
        # Performs message passing between query (virtual super node) and all nodes.
        # Edge features are not used here since virtual edges have no attributes.
        if gnn == 'gt':
            if num_heads == -1:
                raise ValueError("num_heads must be a positive integer!")
            self.question_node_message_model = GraphTransformer(
                in_channels=in_channels, hidden_channels=hidden_channels,
                out_channels=hidden_channels, dropout=dropout,
                num_layers=1, num_heads=num_heads, edge_feature=False
            )
            self.node_node_message_model = GraphTransformer(
                in_channels=hidden_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, dropout=dropout,
                num_layers=num_layers, num_heads=num_heads, edge_feature=edge_feature
            )
            self.node_graphtoken_message_model = GraphTransformer(
                in_channels=out_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, dropout=dropout,
                num_layers=1, num_heads=num_heads, edge_feature=False
            )
        elif gnn == 'gat':
            if num_heads == -1:
                raise ValueError("num_heads must be a positive integer!")
            self.question_node_message_model = GAT(
                in_channels=in_channels, hidden_channels=hidden_channels,
                out_channels=hidden_channels, dropout=dropout,
                num_layers=1, num_heads=num_heads
            )
            self.node_node_message_model = GAT(
                in_channels=hidden_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, dropout=dropout,
                num_layers=num_layers, num_heads=num_heads
            )
            self.node_graphtoken_message_model = GAT(
                in_channels=out_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, dropout=dropout,
                num_layers=1, num_heads=num_heads
            )
        elif gnn == 'gcn':
            self.question_node_message_model = GCN(
                in_channels=in_channels, hidden_channels=hidden_channels,
                out_channels=hidden_channels, dropout=dropout,
                num_layers=1, num_heads=num_heads
            )
            self.node_node_message_model = GCN(
                in_channels=hidden_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, dropout=dropout,
                num_layers=num_layers, num_heads=num_heads
            )
            self.node_graphtoken_message_model = GCN(
                in_channels=out_channels, hidden_channels=hidden_channels,
                out_channels=out_channels, dropout=dropout,
                num_layers=1, num_heads=num_heads
            )

        # Learnable graph tokens g_1, ..., g_k (Algorithm 1, Lines 4-6)
        self.num_graph_token = num_graph_token
        self.graph_token = torch.nn.Parameter(
            torch.randn(self.num_graph_token, hidden_channels), requires_grad=True
        )

        # MLP for projecting query embedding before insertion as virtual super node
        self.query_node_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels),
        )

    def _build_query_aware_graph(self, pyg_graph_list, text_embedds):
        """
        Insert query embedding as a virtual super node connected to all nodes.
        This enables GNNquery to perform query-aware message passing.
        
        Returns:
            Batch of graphs where each graph has an additional query node
            at index 0, connected to all other nodes via virtual edges.
        """
        device = next(self.parameters()).device
        pyg_graph_list = pyg_graph_list.to(device)
        text_embedding_tensor = text_embedds.to(device)
        text_embedding_tensor = self.query_node_mlp(text_embedding_tensor)

        num_graph = len(pyg_graph_list['ptr']) - 1
        x_list = []
        edge_list = []
        for i in range(num_graph):
            node_features = pyg_graph_list.x[pyg_graph_list['batch'] == i]
            # Prepend query embedding as virtual super node (index 0)
            x_list.append(torch.cat([text_embedding_tensor[i].unsqueeze(0), node_features], dim=0))
            # Create edges: query node (0) -> all other nodes (1, 2, ..., n)
            num_nodes = node_features.size(0)
            cur_edge = [[0] * num_nodes, list(range(1, num_nodes + 1))]
            edge_list.append(torch.tensor(cur_edge, dtype=torch.long))

        graphs = [Data(x=x_list[i], edge_index=edge_list[i]) for i in range(num_graph)]
        return Batch.from_data_list(graphs)

    def _build_graphtoken_graph(self, pyg_graph_list):
        """
        Build per-cluster pooling graphs (Algorithm 1, Lines 3-6).
        
        1. Apply k-means clustering on node embeddings to form k clusters.
        2. For each cluster S_i, create a subgraph where learnable token g_i
           is connected only to nodes in S_i via virtual edges.
        3. GNNpool then performs message passing within each cluster,
           allowing g_i to aggregate information from its assigned nodes.
        
        When num_nodes <= k, each node is assigned to its own cluster
        (one-to-one mapping) to avoid degenerate clustering.
        
        Returns:
            Batch of graphs where each graph has k learnable tokens
            (indices 0..k-1), each connected only to its cluster's nodes.
        """
        device = next(self.parameters()).device
        num_graph = len(pyg_graph_list['ptr']) - 1
        graph_list = []

        for i in range(num_graph):
            node_features = pyg_graph_list.x[pyg_graph_list['batch'] == i]
            num_nodes = node_features.size(0)

            # --- Algorithm 1, Line 3: Clustering ---
            if num_nodes == 0:
                # Empty graph: just use learnable tokens with no edges
                graph_list.append(Data(
                    x=self.graph_token.clone(),
                    edge_index=torch.zeros((2, 0), dtype=torch.long)
                ))
                continue

            if num_nodes <= self.num_graph_token:
                # Fewer nodes than tokens: assign each node to its own token
                cluster_labels = list(range(num_nodes))
                # Remaining tokens get no connections
                cluster_labels_full = cluster_labels  # length = num_nodes
            else:
                x_np = node_features.detach().cpu().numpy()
                kmeans = MiniBatchKMeans(
                    n_clusters=self.num_graph_token,
                    batch_size=256, n_init='auto', random_state=42
                )
                cluster_labels_full = kmeans.fit_predict(x_np).tolist()

            # --- Algorithm 1, Lines 4-6: Per-cluster GNNpool ---
            # Prepend k learnable graph tokens, then connect each g_i
            # only to nodes assigned to cluster i
            x_with_tokens = torch.cat(
                [self.graph_token.reshape(self.num_graph_token, -1), node_features], dim=0
            )

            edge_src = []  # node indices (offset by num_graph_token)
            edge_dst = []  # token indices (0..k-1)
            for node_idx, cluster_id in enumerate(cluster_labels_full):
                # node_idx in original features -> node_idx + k in x_with_tokens
                edge_src.append(self.num_graph_token + node_idx)
                edge_dst.append(cluster_id)

            if len(edge_src) > 0:
                edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            graph_list.append(Data(x=x_with_tokens, edge_index=edge_index))

        return Batch.from_data_list(graph_list).to(device)

    def forward(self, pyg_graph_list, text_embedds=None, query_aware=True, pooling='graph_token'):
        """
        Forward pass implementing the GNS algorithm.
        
        Args:
            pyg_graph_list: Batched PyG graph data
            text_embedds: Pre-computed query embeddings [batch_size, dim]
            query_aware: If True, apply query-aware encoding (Algorithm 1, Line 1)
            pooling: Pooling strategy
                - 'graph_token': GNS learnable token pooling (default, Algorithm 1)
                - 'mean': Mean pooling baseline (single vector)
                - 'sum': Sum pooling baseline (single vector)
        
        Returns:
            Tuple of (graph_embeddings, auxiliary_info)
            - graph_token: [batch_size, num_graph_token, hidden_dim]
            - mean/sum: [batch_size, 1, hidden_dim]
        """
        device = next(self.parameters()).device
        num_graph = len(pyg_graph_list['ptr']) - 1
        num_node_list = [
            (pyg_graph_list['batch'] == i).sum().item() for i in range(num_graph)
        ]

        # ---- Step 1: Query-aware encoding (Algorithm 1, Line 1) ----
        if query_aware:
            if text_embedds is None:
                raise ValueError("query_aware is True but text_embedds is None.")

            query_aware_batch = self._build_query_aware_graph(pyg_graph_list, text_embedds)
            query_aware_batch = query_aware_batch.to(device)

            # GNNquery: message passing between query super node and all nodes
            node_embedding, _ = self.question_node_message_model(
                x=query_aware_batch.x,
                adj_t=query_aware_batch.edge_index,
                edge_attr=None
            )
            # Residual connection
            node_embedding = node_embedding + query_aware_batch.x

            # Remove query virtual node (index 0) from each graph's embeddings
            cur_start = 0
            temp = []
            for num in num_node_list:
                temp.extend(node_embedding[cur_start + 1: cur_start + num + 1])
                cur_start += num + 1
            node_embedding = torch.stack(temp)

            # ---- Step 2: Node-node message passing (Algorithm 1, Line 2) ----
            node_embedding, _ = self.node_node_message_model(
                x=node_embedding,
                adj_t=pyg_graph_list.edge_index,
                edge_attr=pyg_graph_list.edge_attr
            )
            # Residual connection with original node features
            pyg_graph_list.x = node_embedding + pyg_graph_list.x
        else:
            # Without query awareness: only node-node message passing
            node_embedding, _ = self.node_node_message_model(
                x=pyg_graph_list.x,
                adj_t=pyg_graph_list.edge_index,
                edge_attr=pyg_graph_list.edge_attr
            )
            pyg_graph_list.x = node_embedding

        # ---- Step 3-6: Pooling ----
        if pooling == 'graph_token':
            # Learnable graph token pooling (Algorithm 1, Lines 3-6)
            # Each learnable token g_i aggregates information from nodes via GNNpool
            graph_token_batch = self._build_graphtoken_graph(pyg_graph_list)

            graph_token_embedding, _ = self.node_graphtoken_message_model(
                x=graph_token_batch.x,
                adj_t=graph_token_batch.edge_index,
                edge_attr=None
            )

            # Extract graph token embeddings (first k entries per graph)
            res_graph_token_embedding = []
            cur_start = 0
            for num in num_node_list:
                res_graph_token_embedding.append(
                    graph_token_embedding[cur_start: cur_start + self.num_graph_token]
                )
                cur_start += num + self.num_graph_token

            # [batch_size, num_graph_token, hidden_dim]
            return torch.stack(res_graph_token_embedding), ""

        elif pooling == 'mean':
            return self._global_pooling(pyg_graph_list, num_graph, reduce='mean'), ""

        elif pooling == 'sum':
            return self._global_pooling(pyg_graph_list, num_graph, reduce='sum'), ""

        else:
            raise ValueError('pooling must be one of: "graph_token", "sum", "mean".')

    def _global_pooling(self, pyg_graph_list, num_graph, reduce='mean'):
        """Global pooling with handling for empty graphs (zero nodes)."""
        # Detect empty graphs
        prev = pyg_graph_list['ptr'][0]
        zero_node_list = []
        for i, ptr in enumerate(pyg_graph_list['ptr'][1:]):
            if ptr == prev:
                zero_node_list.append(i)
            prev = ptr

        pooled_output = scatter(pyg_graph_list.x, pyg_graph_list.batch, dim=0, reduce=reduce)

        if len(zero_node_list) != 0:
            embedding_list = []
            cur = 0
            for i in range(num_graph):
                if i in zero_node_list:
                    embedding_list.append(torch.zeros_like(pooled_output[0]))
                else:
                    embedding_list.append(pooled_output[cur])
                    cur += 1
            pooled_output = torch.stack(embedding_list).reshape(num_graph, 1, -1)
        else:
            pooled_output = pooled_output.reshape(num_graph, 1, -1)

        return pooled_output


# =============================================================================
# Model Registry
# =============================================================================

load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
    'gns': GNS,
}
