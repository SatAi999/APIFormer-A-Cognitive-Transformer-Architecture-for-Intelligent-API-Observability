import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from src.data.synthetic_gen import UnifiedAPIEvent
from src.utils.logger import logger

class ServiceDependencyGraph:
    """Constructs microservice call chains and transactional graphs for service topology analytics."""
    
    def __init__(self):
        self.service_graph = nx.DiGraph()
        self.endpoint_graph = nx.DiGraph()
        
    def build_graphs(self, events: List[UnifiedAPIEvent]) -> None:
        """Parses events to populate node representations and edges (based on tracing trace_id and session sequences)."""
        logger.info("Building Service and Endpoint Dependency Graphs...")
        
        # 1. Trace-level mapping (Distributed tracing dependency)
        trace_groups = defaultdict(list)
        for e in events:
            if e.trace_id != "None":
                trace_groups[e.trace_id].append(e)
                
        for trace_id, t_events in trace_groups.items():
            # Sort trace events chronologically
            t_events = sorted(t_events, key=lambda x: x.timestamp)
            if len(t_events) > 1:
                # Gateway calls backend microservices inside the same trace
                caller = t_events[0]
                for callee in t_events[1:]:
                    # Service dependency
                    s_u, s_v = caller.service_name, callee.service_name
                    if self.service_graph.has_edge(s_u, s_v):
                        self.service_graph[s_u][s_v]['weight'] += 1
                    else:
                        self.service_graph.add_edge(s_u, s_v, weight=1)
                        
                    # Endpoint dependency
                    ep_u = f"{caller.http_method} {caller.endpoint}"
                    ep_v = f"{callee.http_method} {callee.endpoint}"
                    if self.endpoint_graph.has_edge(ep_u, ep_v):
                        self.endpoint_graph[ep_u][ep_v]['weight'] += 1
                    else:
                        self.endpoint_graph.add_edge(ep_u, ep_v, weight=1)
                        
        # 2. Session-level workflow transitions
        session_groups = defaultdict(list)
        for e in events:
            if e.session_id != "None":
                session_groups[e.session_id].append(e)
                
        for sess_id, s_events in session_groups.items():
            s_events = sorted(s_events, key=lambda x: x.timestamp)
            for i in range(len(s_events) - 1):
                # Transition from event i to event i+1
                e_curr = s_events[i]
                e_next = s_events[i+1]
                
                # We model transitions between endpoints in session sequences
                ep_curr = f"{e_curr.http_method} {e_curr.endpoint}"
                ep_next = f"{e_next.http_method} {e_next.endpoint}"
                
                if self.endpoint_graph.has_edge(ep_curr, ep_next):
                    self.endpoint_graph[ep_curr][ep_next]['weight'] += 1
                else:
                    self.endpoint_graph.add_edge(ep_curr, ep_next, weight=1)

        # Populate node defaults
        for node in self.service_graph.nodes():
            self.service_graph.nodes[node]['type'] = 'service'
        for node in self.endpoint_graph.nodes():
            self.endpoint_graph.nodes[node]['type'] = 'endpoint'
            
        logger.info(f"Graph Construction complete: Service Graph has {self.service_graph.number_of_nodes()} nodes, {self.service_graph.number_of_edges()} edges.")
        logger.info(f"Endpoint Graph has {self.endpoint_graph.number_of_nodes()} nodes, {self.endpoint_graph.number_of_edges()} edges.")

    def compute_graph_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculates topological characteristics (PageRank, degrees) to provide node embeddings."""
        if len(self.endpoint_graph) == 0:
            return {}
            
        pagerank = nx.pagerank(self.endpoint_graph, weight='weight')
        in_degree = dict(self.endpoint_graph.in_degree(weight='weight'))
        out_degree = dict(self.endpoint_graph.out_degree(weight='weight'))
        
        metrics = {}
        for node in self.endpoint_graph.nodes():
            metrics[node] = {
                "pagerank": float(pagerank.get(node, 0.0)),
                "in_degree": float(in_degree.get(node, 0.0)),
                "out_degree": float(out_degree.get(node, 0.0))
            }
        return metrics

    def get_gnn_structures(self, vocab: Dict[str, int]) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Prepares adjacency matrices and edge_index lists mapping to endpoint tokenizer vocabulary indices."""
        vocab_size = len(vocab)
        adj_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        edge_list = []
        
        for u, v, data in self.endpoint_graph.edges(data=True):
            # Strip method to find token index
            # The endpoints in endpoint_graph are stored as "GET /api/v1/products"
            # In APITokenizer, it fits endpoint URLs (e.g., "/api/v1/products"). Let's extract the endpoint name:
            u_clean = u.split(" ")[-1] if " " in u else u
            v_clean = v.split(" ")[-1] if " " in v else v
            
            u_idx = vocab.get(u_clean)
            v_idx = vocab.get(v_clean)
            
            if u_idx is not None and v_idx is not None:
                w = data.get('weight', 1.0)
                adj_matrix[u_idx, v_idx] = float(w)
                edge_list.append((u_idx, v_idx))
                
        # Normalize adjacency matrix rows
        row_sums = adj_matrix.sum(axis=1, keepdims=True)
        # Avoid divide by zero
        adj_matrix = np.divide(adj_matrix, row_sums, out=np.zeros_like(adj_matrix), where=row_sums!=0)
        
        return adj_matrix, edge_list
