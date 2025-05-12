import json
import numpy as np
import argparse
from qwen_extractor import QwenExtractor


def load_and_build_dag(json_file: str) -> tuple[dict, dict]:
    """
    Load graph structure from JSON file and build a DAG.

    Args:
        json_file: Path to the JSON file containing the graph structure

    Returns:
        tuple: (dag, node_labels) where
            - dag: dict with node_id as key and dict of parents/children as value
            - node_labels: dict mapping node_id to node label
    """
    # Load JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    # Create node label mapping
    node_labels = {node_id: node_data["label"] for node_id, node_data in nodes.items()}

    # Initialize DAG structure
    dag = {}
    for node_id in nodes:
        dag[node_id] = {
            "parents": [],  # List of (parent_id, modifier) tuples
            "children": [],  # List of (child_id, modifier) tuples
        }

    # Add edge information
    for edge_id, edge_data in edges.items():
        source = edge_data["source"]
        target = edge_data["target"]
        modifier = edge_data["modifier"]

        # Add parent-child relationships
        dag[target]["parents"].append((source, modifier))
        dag[source]["children"].append((target, modifier))

    print(f"DEBUG: DAG: {dag}")
    return dag, node_labels


def calculate_impact(parent_prob: float, modifier: float) -> float:
    """
    Calculate the impact with enhanced sensitivity to changes.
    """
    centered_prob = 2 * parent_prob - 1

    if modifier > 0:
        impact = centered_prob * modifier
    else:
        impact = -centered_prob * abs(modifier)

    return impact


def simulate_graph(
    dag: dict,
    num_samples: int = 1000,
    intervention_node: str = None,
    intervention_prob: float = None,
) -> dict:
    """
    Simulate the entire belief graph using continuous probability propagation.

    Args:
        dag: DAG structure containing nodes and their relationships
        num_samples: Number of samples to generate
        intervention_node: Optional node to intervene on
        intervention_prob: Probability to set for intervention node

    Returns:
        dict: Sampling results for all nodes including means and distributions
    """
    # Initialize storage for all node samples
    node_samples = {node: [] for node in dag}

    # Get topological order of nodes
    visited = set()
    topo_order = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        # Process all parents first
        for parent, _ in dag[node]["parents"]:
            if parent not in visited:
                dfs(parent)
        topo_order.append(node)

    # Build topological order
    for node in dag:
        if node not in visited:
            dfs(node)

    # Perform sampling
    for _ in range(num_samples):
        prob_values = {}

        for node in topo_order:
            if not dag[node]["parents"]:  # Root node
                if node == intervention_node:
                    prob_values[node] = intervention_prob
                else:
                    # NOTE: 0.1 for debugging, the actual value is 0.5
                    prob_values[node] = 0.1
            else:
                parent_impacts = []
                total_modifier = 0

                for parent_id, modifier in dag[node]["parents"]:
                    parent_prob = prob_values[parent_id]
                    impact = calculate_impact(parent_prob, modifier)
                    parent_impacts.append(impact)
                    total_modifier += abs(modifier)

                if len(parent_impacts) > 1:
                    weights = [abs(m) / total_modifier for _, m in dag[node]["parents"]]

                    combined_impact = sum(
                        i * w for i, w in zip(parent_impacts, weights)
                    )

                    sensitivity = 10.0
                    prob = 1 / (1 + np.exp(-sensitivity * combined_impact))
                else:
                    impact = parent_impacts[0]
                    prob = 1 / (1 + np.exp(-2.0 * impact))

                prob_values[node] = min(0.95, max(0.05, prob))

            # sample = np.random.binomial(n=1, p=prob_values[node])
            if not dag[node]["children"]:
                sample = np.random.binomial(n=1, p=prob_values[node])
            else:
                sample = prob_values[node]
            node_samples[node].append(sample)

    # Calculate statistics for each node
    results = {}
    for node in dag:
        samples = node_samples[node]
        node_results = {
            "mean": np.mean(samples),
            "std": np.std(samples),
            "samples": samples,
        }
        results[node] = node_results

    return results


def process_question_and_sample(
    question: str, json_file: str, num_samples: int = 1000
) -> dict:
    """
    Process a question and simulate the entire graph.
    """
    # Initialize extractor
    extractor = QwenExtractor()

    # Load graph and build DAG
    dag, node_labels = load_and_build_dag(json_file)

    # Create reverse mapping (label to node_id)
    label_to_id = {label: node_id for node_id, label in node_labels.items()}

    # # TODO: Implement actual extraction
    # node_label = "building_height_increase"
    # prob = 0.9
    # explanation = "BLABLABLA"

    intervention = extractor.extract_intervention(dag, question)
    if not intervention:
        raise ValueError("Could not determine intervention from question")
    node_label, prob, explanation = intervention

    # Get node ID for the intervention
    if node_label not in label_to_id:
        raise ValueError(f"Node '{node_label}' not found in belief graph")
    intervention_node = label_to_id[node_label]

    # Sample without intervention
    print("\nSimulating base probabilities...")
    base_results = simulate_graph(dag, num_samples)

    # Sample with intervention
    print(f"\nSimulating with intervention (setting {node_label} to {prob})...")
    intervention_results = simulate_graph(
        dag, num_samples, intervention_node=intervention_node, intervention_prob=prob
    )

    # Calculate changes
    changes = {
        node: intervention_results[node]["mean"] - base_results[node]["mean"]
        for node in dag
    }

    # Create id to label mapping for printing
    id_to_label = {node_id: label for label, node_id in label_to_id.items()}

    # Print simplified results
    print("\nIntervention Analysis")
    print("-" * 40)
    print(f"Question: {question}")
    print(f"Setting {node_label} to {prob:.3f}")
    print(f"Explanation: {explanation}")
    print()

    # First print stance node changes
    stance_nodes = [
        node for node, label in id_to_label.items() if "stance" in label.lower()
    ]
    if stance_nodes:
        stance_node = stance_nodes[0]
        stance_label = id_to_label[stance_node]
        print("Stance Change:")
        print("-" * 40)
        base = base_results[stance_node]["mean"]
        intervention = intervention_results[stance_node]["mean"]
        change = intervention - base
        print(f"{stance_label}:")
        print(f"  Before: {base:.3f}")
        print(f"  After:  {intervention:.3f}")
        print(f"  Change: {change:+.3f}")
        print()

    # Then print all other node changes
    print("All Node Changes:")
    print("-" * 40)
    for node in sorted(base_results.keys()):
        if node in stance_nodes:  # Skip stance node as it's already shown
            continue

        label = id_to_label[node]
        base = base_results[node]["mean"]
        intervention = intervention_results[node]["mean"]
        change = intervention - base

        print(f"{label}:")
        print(f"  Before: {base:.3f}")
        print(f"  After:  {intervention:.3f}")
        print(f"  Change: {change:+.3f}")

    return {
        "intervention": {
            "node": node_label,
            "probability": prob,
            "explanation": explanation,
        },
        "base_probabilities": base_results,
        "intervention_probabilities": intervention_results,
        "changes": changes,
    }


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Process questions about causal Bayesian networks"
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        help="Path to the JSON file containing the graph structure",
    )
    parser.add_argument(
        "--question", type=str, required=True, help="Question to analyze"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Process the question and get results
    try:
        results = process_question_and_sample(
            question=args.question, json_file=args.json, num_samples=args.num_samples
        )
    except Exception as e:
        print(f"Error processing question: {e}")
        return


if __name__ == "__main__":
    main()
