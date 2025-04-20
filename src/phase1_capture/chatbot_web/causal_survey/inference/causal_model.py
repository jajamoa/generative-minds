import pyro
import torch
import pyro.distributions as dist
from typing import List, Dict, Any
import networkx as nx

class CausalModel:
    def __init__(self):
        """Initialize causal model"""
        pyro.clear_param_store()
        self.graph = nx.DiGraph()
        self.observed_data = {}
        
    def _create_variable_name(self, text: str) -> str:
        """Create variable name"""
        return text.lower().replace(" ", "_")
        
    def add_causal_relation(self, cause: str, effect: str):
        """Add causal relationship to the graph"""
        cause_var = self._create_variable_name(cause)
        effect_var = self._create_variable_name(effect)
        
        self.graph.add_edge(cause_var, effect_var)
        
    def model(self):
        """Define probabilistic graphical model"""
        # Create variables for each node in the graph
        node_samples = {}
        
        # First process nodes without parents (root nodes)
        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                node_samples[node] = pyro.sample(
                    node,
                    dist.Beta(torch.tensor(2.0), torch.tensor(2.0))
                )
                
        # Then process nodes with parents
        for node in self.graph.nodes():
            if self.graph.in_degree(node) > 0:
                parents = list(self.graph.predecessors(node))
                parent_values = torch.stack([node_samples[p] for p in parents])
                
                # Calculate current node probability using parent values
                alpha = 0.2 + 0.5 * torch.mean(parent_values)
                node_samples[node] = pyro.sample(
                    node,
                    dist.Beta(
                        alpha * torch.ones(1),
                        (1 - alpha) * torch.ones(1)
                    )
                )
                
        return node_samples
        
    def infer(self, observed_data: Dict[str, Any]):
        """Perform Bayesian inference"""
        self.observed_data = {
            self._create_variable_name(k): torch.tensor(float(v))
            for k, v in observed_data.items()
        }
        
        # Condition the model
        conditioned_model = pyro.condition(self.model, data=self.observed_data)
        
        # Use importance sampling for inference
        importance = pyro.infer.Importance(conditioned_model, num_samples=1000)
        importance.run()
        
        # Get posterior distribution
        marginals = {}
        for node in self.graph.nodes():
            if node not in self.observed_data:
                marginal = pyro.infer.EmpiricalMarginal(importance, sites=node)
                marginals[node] = {
                    "mean": float(marginal.mean),
                    "std": float(marginal.variance.sqrt())
                }
                
        return marginals
        
    def update_from_dialogue(self, causal_relations: List[Dict[str, str]]):
        """Update causal model from dialogue"""
        for relation in causal_relations:
            self.add_causal_relation(relation["cause"], relation["effect"])
            
        # Set simple binary observations for observed variables
        observed_data = {
            relation["effect"]: 1.0
            for relation in causal_relations
        }
        
        return self.infer(observed_data) 