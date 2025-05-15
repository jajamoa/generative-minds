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

    return dag, node_labels


def build_cpt_from_modifier(modifier: float) -> dict:
    """
    Build Conditional Probability Table (CPT) based on modifier value.

    Args:
        modifier: Float between -1 and 1, indicating relationship strength and direction
            positive: positive correlation
            negative: negative correlation

    Returns:
        dict: CPT mapping parent state (bool) to child node's probability of being True
    """
    if modifier > 0:
        # Positive correlation
        cpt = {
            False: 1 - modifier,  # Lower probability when parent is False
            True: modifier,  # Higher probability when parent is True
        }
    else:
        # Negative correlation
        cpt = {
            False: abs(modifier),  # Higher probability when parent is False
            True: 1 - abs(modifier),  # Lower probability when parent is True
        }
    return cpt


def get_conditional_probability(parent_value: bool, modifier: float) -> float:
    """
    Get conditional probability from CPT based on parent value and modifier.

    Args:
        parent_value: Boolean value of parent node
        modifier: Float between -1 and 1

    Returns:
        float: Probability of child node being True
    """
    cpt = build_cpt_from_modifier(modifier)
    return cpt[bool(parent_value)]


def simulate_graph(
    dag: dict,
    num_samples: int = 1000,
    intervention_node: str = None,
    intervention_prob: float = None,
) -> dict:
    """
    Simulate the entire belief graph using forward sampling with Noisy-OR for multiple parents.

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

    def calculate_path_strength(node, visited=None):
        if visited is None:
            visited = set()

        if node in visited:
            return 0.0

        visited.add(node)

        if not dag[node]["parents"]:
            return 1.0

        path_strengths = []
        for parent, modifier in dag[node]["parents"]:
            parent_strength = calculate_path_strength(parent, visited.copy())
            path_strengths.append(abs(modifier) * parent_strength)

        return max(path_strengths) if path_strengths else 0.0

    path_strengths = {node: calculate_path_strength(node) for node in dag}

    # Perform sampling
    for _ in range(num_samples):
        sample_values = {}

        for node in topo_order:
            if not dag[node]["parents"]:  # Root node
                if node == intervention_node:
                    prob = intervention_prob
                else:
                    # TODO: for debugging only
                    prob = 0.1
                sample_values[node] = np.random.binomial(n=1, p=prob)
            else:
                parent_probs = []
                for parent_id, modifier in dag[node]["parents"]:
                    parent_value = sample_values[parent_id]
                    parent_prob = get_conditional_probability(
                        bool(parent_value), modifier
                    )
                    parent_probs.append(parent_prob)

                if len(parent_probs) > 1:
                    # use weighted average instead of Noisy-OR
                    weights = [abs(m) for _, m in dag[node]["parents"]]
                    weight_sum = sum(weights)
                    if weight_sum > 0:
                        weights = [w / weight_sum for w in weights]
                        combined_prob = sum(
                            p * w for p, w in zip(parent_probs, weights)
                        )
                    else:
                        combined_prob = sum(parent_probs) / len(parent_probs)
                else:
                    combined_prob = parent_probs[0]

                # remove path_compensation, use a more relaxed bound
                prob = min(0.99, max(0.01, combined_prob))

                sample_values[node] = np.random.binomial(n=1, p=prob)

            node_samples[node].append(sample_values[node])

    # Calculate statistics for each node
    results = {}
    for node in dag:
        samples = node_samples[node]
        node_results = {
            "mean": np.mean(samples),
            "std": np.std(samples),
            "samples": samples,  # Keep raw samples for additional analysis
        }

        # Calculate conditional probabilities if node has parents
        if dag[node]["parents"]:
            for parent_id, modifier in dag[node]["parents"]:
                parent_samples = node_samples[parent_id]

                # Calculate P(Node=1|Parent=0) and P(Node=1|Parent=1)
                parent_0_indices = [i for i, p in enumerate(parent_samples) if p == 0]
                parent_1_indices = [i for i, p in enumerate(parent_samples) if p == 1]

                if parent_0_indices:
                    p_1_given_0 = np.mean([samples[i] for i in parent_0_indices])
                else:
                    p_1_given_0 = None

                if parent_1_indices:
                    p_1_given_1 = np.mean([samples[i] for i in parent_1_indices])
                else:
                    p_1_given_1 = None

                node_results[f"P(1|{parent_id}=0)"] = p_1_given_0
                node_results[f"P(1|{parent_id}=1)"] = p_1_given_1

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

    # Extract intervention from question

    # intervention = extractor.extract_intervention(dag, question)
    # if not intervention:
    #     raise ValueError("Could not determine intervention from question")

    # TODO: Implement actual extraction
    node_label = "building_height_increase"
    prob = 0.9
    explanation = "BLABLABLA"

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

    # First print stance node changes (most important)
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
