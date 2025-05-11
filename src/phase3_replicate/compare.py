import json
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import Importance, EmpiricalMarginal
import argparse

def load_belief_graph(json_file):
    """Load belief graph from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_network_structure(graph_data):
    """Extract nodes and edges from complex belief graph"""
    # Extract nodes with their labels
    nodes = {}
    for node_id, node_data in graph_data['nodes'].items():
        nodes[node_id] = {'label': node_data['label']}
    
    # Extract edges with their modifiers
    edges = {}
    for edge_id, edge_data in graph_data['edges'].items():
        source = edge_data['source']
        target = edge_data['target']
        modifier = edge_data['modifier']
        
        edges[edge_id] = {
            'source': source,
            'target': target,
            'modifier': modifier
        }
    
    return {'nodes': nodes, 'edges': edges}

def build_dag_structure(network_structure):
    """Build a directed acyclic graph structure"""
    # Initialize graph structure
    dag = {}
    for node_id in network_structure['nodes']:
        dag[node_id] = {'children': [], 'parents': []}
    
    # Add parent-child relationships
    for edge_id, edge in network_structure['edges'].items():
        parent = edge['source']
        child = edge['target']
        modifier = edge['modifier']
        
        # Convert modifier to probability
        prob = 1 + modifier if modifier < 0 else modifier
        
        # Add child to parent's children list
        if parent in dag:
            dag[parent]['children'].append((child, prob))
        
        # Add parent to child's parents list
        if child in dag:
            dag[child]['parents'].append((parent, prob))
    
    return dag

def define_bayesian_model(dag, intervention_node=None, intervention_prob=None, use_observe=False):
    """Define Pyro model for the Bayesian network
    
    Parameters:
    -----------
    dag: dict
        The directed acyclic graph structure
    intervention_node: str
        The node to intervene on
    intervention_prob: float
        The probability to set for the intervention
    use_observe: bool
        If True, use observe instead of sample for intervention (conditioning vs. intervention)
    """
    def model():
        # Dictionary to track sampled values
        values = {}
        
        # Find nodes with no parents (root nodes)
        root_nodes = [node for node, info in dag.items() if not info['parents']]
        
        # Sample root nodes
        for node in root_nodes:
            values[node] = pyro.sample(node, dist.Bernoulli(0.5))
        
        # Process remaining nodes in topological order
        visited = set(root_nodes)
        nodes_to_process = set(dag.keys()) - visited
        
        while nodes_to_process:
            for node in list(nodes_to_process):
                # Check if all parents have been processed
                parent_ids = [p[0] for p in dag[node]['parents']]
                if all(p in visited for p in parent_ids):
                    # Node can be processed
                    
                    # Handle intervention or observation
                    if node == intervention_node and intervention_prob is not None:
                        if use_observe:
                            # Use observe - this is conditioning (seeing)
                            # Create a fake observation with the desired probability
                            observation = torch.bernoulli(torch.tensor([intervention_prob]))
                            values[node] = pyro.sample(node, dist.Bernoulli(0.5), obs=observation)
                        else:
                            # Use sample - this is intervention (doing)
                            values[node] = pyro.sample(node, dist.Bernoulli(intervention_prob))
                    else:
                        # Calculate conditional probability based on parents
                        if not dag[node]['parents']:
                            prob = 0.5  # No parents
                        else:
                            # Start with base probability
                            prob = 0.5
                            
                            # Adjust based on each parent's value and relationship strength
                            for parent_id, parent_weight in dag[node]['parents']:
                                parent_value = values[parent_id]
                                # Adjust probability based on parent value and weight
                                prob = prob + (parent_value * parent_weight - 0.5) * 0.5
                                # Ensure probability stays in valid range
                                prob = max(0.1, min(0.9, prob))
                        
                        values[node] = pyro.sample(node, dist.Bernoulli(prob))
                    
                    # Mark as visited
                    visited.add(node)
                    nodes_to_process.remove(node)
        
        return values
    
    return model

def perform_intervention_or_conditioning(json_file, target_node, intervention_prob, num_samples=1000, use_observe=False):
    """Perform intervention or conditioning on a target node with specified probability
    
    Parameters:
    -----------
    json_file: str
        Path to the JSON file containing the belief graph
    target_node: str
        The node to intervene on or condition on
    intervention_prob: float
        The probability to set for the intervention or observation
    num_samples: int
        Number of samples to draw
    use_observe: bool
        If True, use observe (conditioning) instead of sample (intervention)
    """
    # Load and process belief graph
    graph_data = load_belief_graph(json_file)
    network_structure = extract_network_structure(graph_data)
    
    # Create label-to-id and id-to-label mappings
    node_id_map = {data['label']: node_id for node_id, data in network_structure['nodes'].items()}
    node_label_map = {node_id: data['label'] for node_id, data in network_structure['nodes'].items()}
    
    # Get the node ID for the target node
    target_node_id = None
    if target_node in node_id_map:
        target_node_id = node_id_map[target_node]
    elif target_node in network_structure['nodes']:
        target_node_id = target_node
    else:
        raise ValueError(f"Target node '{target_node}' not found in belief graph")
    
    # Build DAG structure
    dag = build_dag_structure(network_structure)
    
    # Define model with intervention or conditioning
    model = define_bayesian_model(dag, target_node_id, intervention_prob, use_observe=use_observe)
    
    # Run inference
    pyro.clear_param_store()
    importance = Importance(model, num_samples=num_samples)
    importance_samples = importance.run()
    
    # Extract marginal distributions for all nodes
    results = {}
    for node_id in dag:
        try:
            marginal = EmpiricalMarginal(importance_samples, sites=node_id)
            samples = marginal.sample((1000,))
            prob = samples.float().mean().item()
            
            # Use label for the result if available
            node_name = node_label_map.get(node_id, node_id)
            results[node_name] = prob
        except Exception as e:
            print(f"Error processing node {node_id}: {e}")
    
    return results

def compare_intervention_vs_conditioning(json_file, target_node, prob, num_samples=1000):
    """Compare the results of intervention (do) vs conditioning (see/observe)"""
    # Perform intervention (do-operator)
    intervention_results = perform_intervention_or_conditioning(
        json_file, target_node, prob, num_samples, use_observe=False
    )
    
    # Perform conditioning (observation)
    conditioning_results = perform_intervention_or_conditioning(
        json_file, target_node, prob, num_samples, use_observe=True
    )
    
    return intervention_results, conditioning_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run interventions or conditioning on belief graphs')
    parser.add_argument('--json', type=str, required=True, help='Path to JSON belief graph file')
    parser.add_argument('--node', type=str, required=True, help='Target node for intervention/conditioning')
    parser.add_argument('--prob', type=float, required=True, help='Intervention/conditioning probability (0-1)')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--compare', action='store_true', help='Compare intervention vs conditioning')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            # Compare both methods
            intervention_results, conditioning_results = compare_intervention_vs_conditioning(
                args.json,
                args.node,
                args.prob,
                args.samples
            )
            
            print(f"\nResults after intervention do({args.node}={args.prob}):")
            print("-" * 40)
            for node, prob in sorted(intervention_results.items()):
                print(f"{node}: {prob:.3f}")
            
            print(f"\nResults after conditioning see({args.node}={args.prob}):")
            print("-" * 40)
            for node, prob in sorted(conditioning_results.items()):
                print(f"{node}: {prob:.3f}")
            
            # Show differences
            print(f"\nDifference (intervention - conditioning):")
            print("-" * 40)
            all_nodes = set(intervention_results.keys()) | set(conditioning_results.keys())
            for node in sorted(all_nodes):
                int_prob = intervention_results.get(node, 0)
                cond_prob = conditioning_results.get(node, 0)
                diff = int_prob - cond_prob
                print(f"{node}: {diff:.3f}")
        else:
            # Only run intervention
            results = perform_intervention_or_conditioning(
                args.json,
                args.node,
                args.prob,
                args.samples,
                use_observe=False
            )
            
            if not results:
                print("No results were generated. Please check your model and node names.")
            else:
                print(f"\nResults after intervention do({args.node}={args.prob}):")
                print("-" * 40)
                for node, prob in sorted(results.items()):
                    print(f"{node}: {prob:.3f}")
    except Exception as e:
        print(f"Error: {e}")