import json
import networkx as nx
from utils import OpenAIInterface, GraphMerger
from utils.graph_utils import serialize_graph, save_graph_to_json, export_to_mermaid

class CausalGraph:
    def __init__(self, graph_merger=None):
        """
        Initialize causal graph data structure
        Args:
            graph_merger: Optional GraphMerger instance. If not provided, creates default one with OpenAI
        """
        self.nodes = {}  # Node dictionary
        self.edges = {}  # Edge dictionary
        self.qa_history = {}  # QA pair history
        self._graph = None  # Lazy initialization of NetworkX graph
        self.qa_counter = 0  # Counter for generating qa_ids
        self.node_counter = 0  # Counter for generating node_ids
        self.edge_counter = 0  # Counter for generating edge_ids
        
        # Initialize graph merger with default OpenAI interface if none provided
        if graph_merger is None:
            self.llm_interface = OpenAIInterface()
            self.graph_merger = GraphMerger(self.llm_interface)
        else:
            self.graph_merger = graph_merger
            self.llm_interface = graph_merger.llm

    def add_node(self, label, confidence=1.0, source_qa=None, node_id=None):
        """Add a node to the causal graph"""
        # Generate node ID if not provided
        if node_id is None:
            self.node_counter += 1
            node_id = f"n{self.node_counter}"
            
        self.nodes[node_id] = {
            "label": label,
            "confidence": confidence,
            "source_qa": source_qa or [],
            "incoming_edges": [],
            "outgoing_edges": []
        }
        self._graph = None  # Mark for graph rebuild
        return node_id
        
    def add_edge(self, source, target, confidence, evidence=None, modifier=None, edge_id=None):
        """Add an edge between two nodes in the causal graph"""
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node does not exist")
        
        # Generate edge ID if not provided
        if edge_id is None:
            self.edge_counter += 1
            edge_id = f"e{self.edge_counter}"
            
        positive = modifier > 0 if modifier is not None else True
        self.edges[edge_id] = {
            "source": source,
            "target": target,
            "aggregate_confidence": abs(confidence),
            "evidence": evidence or [],
            "modifier": modifier or confidence,
            "positive": positive
        }
        
        # Update edge lists in nodes
        self.nodes[source]["outgoing_edges"].append(edge_id)
        self.nodes[target]["incoming_edges"].append(edge_id)
        self._graph = None  # Mark for graph rebuild
        return edge_id
        
    @property
    def graph(self):
        """Get NetworkX graph representation (created on demand)"""
        if self._graph is None:
            self._build_graph()
        return self._graph
        
    def _build_graph(self):
        """Build NetworkX graph from internal data structure"""
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node_data in self.nodes.items():
            G.add_node(node_id, **node_data)
            
        # Add edges
        for edge_id, edge_data in self.edges.items():
            G.add_edge(
                edge_data["source"], 
                edge_data["target"],
                id=edge_id,
                confidence=edge_data["aggregate_confidence"],
                **{k: v for k, v in edge_data.items() if k not in ["source", "target"]}
            )
            
        self._graph = G
        
    def to_json(self, include_qa=True):
        """
        Export graph to JSON format
        Args:
            include_qa: Whether to include QA history in output
        Returns:
            Dictionary containing serialized graph data
        """
        return serialize_graph(self.nodes, self.edges, self.qa_history, include_qa)
        
    def save(self, base_filename):
        """
        Save graph data directly to JSON files
        Args:
            base_filename: Base name for output files
        Returns:
            Dictionary containing paths to saved files
        """
        return save_graph_to_json(self.nodes, self.edges, self.qa_history, base_filename)
        
    def export_mermaid(self, output_file):
        """Export graph to Mermaid diagram"""
        return export_to_mermaid(self.graph, output_file)

    def update_from_qa(self, qa_pair, qa_id=None):
        """
        Update graph structure from QA pair by extracting causal relationships
        """
        # Generate QA ID if not provided
        if qa_id is None:
            self.qa_counter += 1
            qa_id = f"qa_{str(self.qa_counter).zfill(3)}"

        # Extract relationships using LLM
        relationships = self.llm_interface.extract_causal_relationships(qa_pair["question"], qa_pair["answer"])
        
        # Record QA history with extracted pairs
        extracted_pairs = []
        for rel in relationships:
            extracted_pairs.append({
                "source": rel["source_concept"],
                "target": rel["target_concept"],
                "confidence": rel["confidence"],
                "positive_influence": rel["positive_influence"]
            })
            
        self.qa_history[qa_id] = {
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
            "extracted_pairs": extracted_pairs
        }
        
        # Update graph with each relationship
        for rel in relationships:
            source = rel["source_concept"]
            target = rel["target_concept"]
            
            # Skip self-loops
            if source == target:
                continue
            
            # Add nodes and edge
            source_id = self.add_node(source)
            target_id = self.add_node(target)
            
            # Create evidence for the edge
            evidence = [{
                "qa_id": qa_id,
                "confidence": rel["confidence"]
            }]
            
            # Calculate modifier based on positive/negative influence
            modifier = rel["confidence"] if rel["positive_influence"] else -rel["confidence"]
            
            self.add_edge(
                source=source_id,
                target=target_id,
                confidence=rel["confidence"],
                evidence=evidence,
                modifier=modifier
            )
        
        # Periodically check and merge nodes
        if self.qa_counter % 3 == 0:  # Every 5 QA pairs
            self.nodes, self.edges, self.qa_history = self.graph_merger.check_and_merge_nodes(
                self.nodes,
                self.edges,
                self.qa_history
            )
            print("All node labels:", ", ".join([node_data["label"] for node_data in self.nodes.values()]))
            self.export_mermaid("causal_graph.mmd")
            self._graph = None  # Mark for graph rebuild
