import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
from typing import Dict, Any, Optional

class CausalGraphVisualizer:
    def __init__(self, graph: nx.DiGraph):
        """Initialize visualizer"""
        self.graph = graph
        
    def plot_matplotlib(self, 
                       title: str = "Causal Relationship Graph",
                       figsize: tuple = (10, 8),
                       node_color: str = "lightblue",
                       edge_color: str = "gray",
                       with_labels: bool = True,
                       node_size: int = 3000,
                       font_size: int = 12,
                       marginals: Optional[Dict[str, Any]] = None):
        """Draw static graph using Matplotlib"""
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(self.graph)
        
        # Draw nodes and edges
        nx.draw(
            self.graph,
            pos,
            with_labels=with_labels,
            node_color=node_color,
            edge_color=edge_color,
            node_size=node_size,
            font_size=font_size,
            font_weight="bold"
        )
        
        # If marginal distribution info exists, add to node labels
        if marginals:
            labels = {}
            for node in self.graph.nodes():
                if node in marginals:
                    mean = marginals[node]["mean"]
                    std = marginals[node]["std"]
                    labels[node] = f"{node}\n(μ={mean:.2f}, σ={std:.2f})"
                else:
                    labels[node] = node
                    
            nx.draw_networkx_labels(
                self.graph,
                pos,
                labels,
                font_size=font_size
            )
            
        plt.title(title)
        plt.axis("off")
        return plt
        
    def create_html_graph(self,
                         output_path: str = "causal_graph.html",
                         height: str = "500px",
                         width: str = "100%",
                         bgcolor: str = "#ffffff",
                         font_color: str = "#000000",
                         marginals: Optional[Dict[str, Any]] = None):
        """Create interactive HTML graph"""
        net = Network(height=height, width=width, bgcolor=bgcolor, font_color=font_color)
        net.from_nx(self.graph)
        
        # Update node information
        if marginals:
            for node in net.nodes:
                if node["id"] in marginals:
                    mean = marginals[node["id"]]["mean"]
                    std = marginals[node["id"]]["std"]
                    node["title"] = f"Mean: {mean:.2f}\nStd Dev: {std:.2f}"
                    node["label"] = f"{node['id']}\n(μ={mean:.2f})"
                    
        # Set interaction options
        net.toggle_physics(True)
        net.show_buttons(filter_=["physics"])
        
        # Save file
        net.save_graph(output_path)
        return output_path
        
    def save_graph_image(self,
                        output_path: str = "causal_graph.png",
                        **kwargs):
        """Save graph image to file"""
        plt = self.plot_matplotlib(**kwargs)
        plt.savefig(output_path, bbox_inches="tight", dpi=300)
        plt.close()
        return output_path 