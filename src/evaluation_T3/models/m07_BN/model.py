import json
import numpy as np
from .qwen_extractor import QwenExtractor
from typing import Dict, Tuple, Optional, List, Any
from models.base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING
from scipy.stats import entropy
import os

from .motif_library.graph_reconstruction import MotifBasedReconstructor
from .motif_library.motif_library import get_demographic_statistics, MotifLibrary


class BayesianNetwork(Census):
    def __init__(self, config: ModelConfig):
        """Initialize the belief network model."""
        super().__init__(config)
        self.extractor = QwenExtractor()
        self.dag = None
        self.node_labels = None
        self.label_to_id = None
        self.id_to_label = None

        # Get paths from config
        self.graph_path = getattr(
            self.config, "graph_path", "data/samples/sample_1.json"
        )
        self.motif_library_name = getattr(
            self.config,
            "motif_library_name",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "motif_library",
                "motif_library.json",
            ),
        )
        self.output_dir = getattr(
            self.config,
            "output_dir",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "graph_reconstruction"
            ),
        )

        # Get reconstruction parameters from config
        self.seed_node = getattr(self.config, "seed_node", "upzoning_stance")
        self.max_iterations = getattr(self.config, "max_iterations", 20)
        self.target_demographic = getattr(self.config, "target_demographic", None)
        self.demographic_weight = getattr(self.config, "demographic_weight", 0.3)

    def load_graph(self, json_file: str) -> tuple[dict, dict]:
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
        node_labels = {
            node_id: node_data["label"] for node_id, node_data in nodes.items()
        }

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

    def _calculate_impact(self, parent_prob: float, modifier: float) -> float:
        """Calculate the impact with enhanced sensitivity to changes."""
        centered_prob = 2 * parent_prob - 1
        if modifier > 0:
            impact = centered_prob * modifier
        else:
            impact = -centered_prob * abs(modifier)
        return impact

    def _get_topological_order(self) -> List[str]:
        """Get topological ordering of nodes."""
        visited = set()
        topo_order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for parent, _ in self.dag[node]["parents"]:
                if parent not in visited:
                    dfs(parent)
            topo_order.append(node)

        for node in self.dag:
            if node not in visited:
                dfs(node)
        return topo_order

    def simulate_intervention(
        self,
        intervention_node: Optional[str] = None,
        intervention_prob: Optional[float] = None,
        num_samples: int = 1000,
    ) -> Dict[str, Dict]:
        """Simulate the belief network with optional intervention."""
        node_samples = {node: [] for node in self.dag}
        topo_order = self._get_topological_order()

        for _ in range(num_samples):
            prob_values = {}

            for node in topo_order:
                if not self.dag[node]["parents"]:  # Root node
                    if node == intervention_node:
                        prob_values[node] = intervention_prob
                    else:
                        prob_values[node] = 0.5  # Base probability
                else:
                    parent_impacts = []
                    total_modifier = 0

                    for parent_id, modifier in self.dag[node]["parents"]:
                        parent_prob = prob_values[parent_id]
                        impact = self._calculate_impact(parent_prob, modifier)
                        parent_impacts.append(impact)
                        total_modifier += abs(modifier)

                    if len(parent_impacts) > 1:
                        weights = [
                            abs(m) / total_modifier
                            for _, m in self.dag[node]["parents"]
                        ]
                        combined_impact = sum(
                            i * w for i, w in zip(parent_impacts, weights)
                        )
                        sensitivity = 10.0
                        prob = 1 / (1 + np.exp(-sensitivity * combined_impact))
                    else:
                        impact = parent_impacts[0]
                        prob = 1 / (1 + np.exp(-2.0 * impact))

                    prob_values[node] = min(0.95, max(0.05, prob))

                # Only sample leaf nodes
                if not self.dag[node]["children"]:
                    sample = np.random.binomial(n=1, p=prob_values[node])
                else:
                    sample = prob_values[node]
                node_samples[node].append(sample)

        # Calculate statistics
        results = {}
        for node in self.dag:
            samples = node_samples[node]
            results[node] = {
                "mean": np.mean(samples),
                "std": np.std(samples),
                "samples": samples,
            }

        return results

    def _compute_kl_divergence(self, p: float, q: float) -> float:
        """
        Compute KL divergence for Bernoulli distributions

        Args:
            p: First probability
            q: Second probability

        Returns:
            KL divergence value
        """
        # Convert to distributions
        p_dist = np.array([p, 1 - p])
        q_dist = np.array([q, 1 - q])

        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        p_dist = np.clip(p_dist, epsilon, 1 - epsilon)
        q_dist = np.clip(q_dist, epsilon, 1 - epsilon)

        return entropy(p_dist, q_dist)

    def compute_node_contributions(
        self,
        base_results: Dict[str, Dict],
        intervention_results: Dict[str, Dict],
        stance_node: str,
    ) -> List[Tuple[str, float]]:
        """
        Compute each node's contribution to the stance change

        Args:
            base_results: Base simulation results
            intervention_results: Results after intervention
            stance_node: ID of the stance node

        Returns:
            List of (node_id, contribution_score, contribution_type) sorted by contribution
        """
        contributions = []

        # Get stance probabilities
        base_stance = base_results[stance_node]["mean"]
        intervention_stance = intervention_results[stance_node]["mean"]

        for node in self.dag:
            if node == stance_node:
                continue

            # Get node probabilities
            base_prob = base_results[node]["mean"]
            intervention_prob = intervention_results[node]["mean"]

            # Compute absolute change in probability
            prob_shift = abs(intervention_prob - base_prob)

            # Compute KL divergence contribution
            kl_div = self._compute_kl_divergence(intervention_prob, base_prob)

            # Combined score (weighted sum of probability shift and KL divergence)
            contribution_score = 0.7 * prob_shift + 0.3 * kl_div
            contributions.append((node, contribution_score))

        # Sort by contribution score in descending order
        contributions.sort(key=lambda x: x[1], reverse=True)

        max_score = max(score for _, score in contributions) if contributions else 1.0

        # Normalize scores
        normalized_contributions = [
            (node, score / max_score) for node, score in contributions
        ]

        return normalized_contributions

    def reconstruct_graph(self) -> None:
        """Reconstruct the graph from motif library."""
        # Default demographic if none found
        DEFAULT_DEMOGRAPHIC = "long_term_resident"

        try:
            # Get demographics from samples directory
            samples_dir = os.path.dirname(self.motif_library_name)
            demo_stats = get_demographic_statistics(samples_dir)

            print("\nDemographic Distribution:")
            print("-" * 40)
            if demo_stats and demo_stats["distribution"]:
                for demo, info in demo_stats["distribution"].items():
                    print(
                        f"{demo}: {info['count']} samples ({info['percentage']:.1f}%)"
                    )
                print()

                # Select target demographic if not provided
                if not self.target_demographic:
                    most_common = max(
                        demo_stats["distribution"].items(), key=lambda x: x[1]["count"]
                    )[0]
                    self.target_demographic = most_common
            else:
                print(f"No demographics found, using default: {DEFAULT_DEMOGRAPHIC}")
                self.target_demographic = DEFAULT_DEMOGRAPHIC

            print(f"Using demographic: {self.target_demographic}")

            # Load motif library with better error handling
            try:
                library = MotifLibrary.load_library(self.motif_library_name)
                if library is None:
                    print(
                        f"Warning: Could not load motif library from {self.motif_library_name}"
                    )
                    print("Creating empty motif library with default settings")
                    library = MotifLibrary()
                    library.demographic_distribution = {self.target_demographic: 1.0}
            except Exception as e:
                print(f"Error loading motif library: {str(e)}")
                print("Creating empty motif library with default settings")
                library = MotifLibrary()
                library.demographic_distribution = {self.target_demographic: 1.0}

            # Create reconstructor with better error handling
            reconstructor = MotifBasedReconstructor(
                library,
                similarity_threshold=getattr(self.config, "similarity_threshold", 0.3),
                node_merge_threshold=getattr(self.config, "node_merge_threshold", 0.8),
                target_demographic=self.target_demographic,
                demographic_weight=self.demographic_weight,
            )

            # Reconstruct graph
            reconstructed_graph = reconstructor.reconstruct_graph(
                self.seed_node, max_iterations=self.max_iterations, min_score=0.3
            )

            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)

            # Save outputs
            json_path = reconstructor.save_as_json(reconstructed_graph, self.output_dir)
            mmd_path = reconstructor.save_as_mmd(reconstructed_graph, self.output_dir)

            print(f"\nReconstructed graph has {len(reconstructed_graph.nodes())} nodes")
            print(f"Reconstructed graph has {len(reconstructed_graph.edges())} edges")
            print(f"Saved as JSON to: {json_path}")
            print(f"Saved as MMD to: {mmd_path}")

            print("-------------------------- Graph Reconstruction Complete --------------------------")

            # Load the reconstructed graph
            self.dag, self.node_labels = self.load_graph(json_path)

        except Exception as e:
            print(f"Error during graph reconstruction: {str(e)}")
            print("Creating minimal default graph")
            # Create a minimal default graph with just the seed node
            self.dag = {"node_0": {"parents": [], "children": []}}
            self.node_labels = {"node_0": self.seed_node}

    async def simulate_opinions(
        self, region: str, proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate opinions using Bayesian Network based on proposal details.

        Args:
            region: The target region name
            proposal: A dictionary containing the rezoning proposal details

        Returns:
            A dictionary with participant IDs as keys, containing opinions and reasons
        """
        # Auto-reconstruct and load graph if not loaded
        if self.dag is None:
            try:
                self.reconstruct_graph()
            except Exception as e:
                raise ValueError(f"Failed to reconstruct/load graph: {str(e)}")

        # Get proposal ID and convert to scenario ID
        proposal_id = proposal.get("proposal_id", "proposal_000")
        scenario_id = SCENARIO_MAPPING.get(
            proposal_id, "1.1"
        )  # Default to "1.1" if not found

        node_label, intervention_prob, explanation = (
            self.extractor.extract_intervention(
                self.dag, self._create_proposal_description(proposal)
            )
        )

        # Run simulations
        base_results = self.simulate_intervention()
        intervention_results = self.simulate_intervention(
            intervention_node=node_label,
            intervention_prob=intervention_prob,
        )

        # Calculate changes
        changes = {
            node: intervention_results[node]["mean"] - base_results[node]["mean"]
            for node in self.dag
        }

        # Get stance node
        stance_nodes = [
            node
            for node, label in self.node_labels.items()
            if "stance" in label.lower()
        ]
        stance_node = stance_nodes[0] if stance_nodes else None

        # Compute node contributions
        contributions = self.compute_node_contributions(
            base_results, intervention_results, stance_node
        )

        # Format results
        results = {}

        opinion_score = round(intervention_results[stance_node]["mean"] * 10)

        # Create a single agent response with filtered reasons
        agent_id = "bn_agent_001"
        results[agent_id] = {
            "opinions": {
                # NOTE: it is actually a Bernoulli distribution
                scenario_id: opinion_score
            },
            "reasons": {
                scenario_id: {
                    self.node_labels[node]: round(contrib_score * 4 + 1)
                    for node, contrib_score in contributions
                }
            },
        }

        return results

    def analyze_question(self, question: str) -> Dict:
        """
        Analyze a question and return the intervention results.

        Args:
            question: The question to analyze

        Returns:
            dict containing:
            - intervention details
            - base probabilities
            - intervention probabilities
            - changes in probabilities
            - stance probability
        """
        if self.dag is None:
            raise ValueError("Graph not loaded. Call load_graph() first.")

        # Extract intervention from question
        intervention = self.extractor.extract_intervention(self.dag, question)
        if not intervention:
            raise ValueError("Could not determine intervention from question")
        node_label, prob, explanation = intervention

        # Get node ID
        if node_label not in self.label_to_id:
            raise ValueError(f"Node '{node_label}' not found in belief graph")
        intervention_node = self.label_to_id[node_label]

        # Run simulations
        base_results = self.simulate_intervention()
        intervention_results = self.simulate_intervention(
            intervention_node=intervention_node, intervention_prob=prob
        )

        # Calculate changes
        changes = {
            node: intervention_results[node]["mean"] - base_results[node]["mean"]
            for node in self.dag
        }

        # Get stance node
        stance_nodes = [
            node
            for node, label in self.node_labels.items()
            if "stance" in label.lower()
        ]
        stance_node = stance_nodes[0] if stance_nodes else None

        stance_prob = None
        if stance_node:
            stance_prob = {
                "before": base_results[stance_node]["mean"],
                "after": intervention_results[stance_node]["mean"],
                "change": changes[stance_node],
            }

        return {
            "intervention": {
                "node": node_label,
                "probability": prob,
                "explanation": explanation,
            },
            "base_probabilities": base_results,
            "intervention_probabilities": intervention_results,
            "changes": changes,
            "stance": stance_prob,
        }


# Example usage:
if __name__ == "__main__":
    import argparse
    from motif_library.graph_reconstruction import MotifBasedReconstructor, MotifLibrary

    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Bayesian Network Model with Graph Reconstruction"
    )

    # Add arguments similar to graph_reconstruction.py
    parser.add_argument(
        "--seed_node",
        type=str,
        default="upzoning_stance",
        help="Seed node for graph reconstruction",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=20,
        help="Maximum iterations for graph reconstruction",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "graph_reconstruction"
        ),
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--motif_library",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "motif_library",
            "motif_library.json",
        ),
        help="Path to motif library file",
    )
    parser.add_argument(
        "--target_demographic", type=str, help="Target demographic for reconstruction"
    )
    parser.add_argument(
        "--demographic_weight",
        type=float,
        default=0.3,
        help="Weight for demographic scoring (0-1)",
    )

    parser.add_argument(
        "--question",
        type=str,
        help="Question to analyze (required if mode is 'analyze')",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        help="Path to existing graph (required if mode is 'analyze')",
    )

    args = parser.parse_args()

    # First analyze demographic distribution
    samples_dir = os.path.dirname(args.motif_library).replace("output", "")
    demo_stats = get_demographic_statistics(samples_dir)

    print("Demographic Distribution:")
    print("-" * 40)
    for demo, info in demo_stats["distribution"].items():
        print(f"{demo}: {info['count']} samples ({info['percentage']:.1f}%)")
    print()

    # Load motif library and create reconstructor
    library = MotifLibrary.load_library(args.motif_library)

    # Select target demographic
    target_demographic = args.target_demographic
    if not target_demographic:
        # Default to using the most common demographic
        most_common = max(
            demo_stats["distribution"].items(), key=lambda x: x[1]["count"]
        )[0]
        target_demographic = most_common
        print(f"Using most common demographic: {target_demographic}")

    # Create reconstructor
    reconstructor = MotifBasedReconstructor(
        library,
        similarity_threshold=0.3,
        node_merge_threshold=0.8,
        target_demographic=target_demographic,
        demographic_weight=args.demographic_weight,
    )

    # Reconstruct graph
    reconstructed_graph = reconstructor.reconstruct_graph(
        args.seed_node, max_iterations=args.max_iterations, min_score=0.3
    )

    # Save outputs
    json_path = reconstructor.save_as_json(reconstructed_graph, args.output_dir)
    mmd_path = reconstructor.save_as_mmd(reconstructed_graph, args.output_dir)

    print(f"\nReconstructed graph has {len(reconstructed_graph.nodes())} nodes")
    print(f"Reconstructed graph has {len(reconstructed_graph.edges())} edges")
    print(f"Saved as JSON to: {json_path}")
    print(f"Saved as MMD to: {mmd_path}")

    # Create and initialize model
    model = BayesianNetwork(ModelConfig())
    model.load_graph(json_path)

    # Analyze question
    results = model.analyze_question(args.question)

    # Print results
    print("\nAnalysis Results:")
    print("-" * 40)
    print(f"Intervention node: {results['intervention']['node']}")
    print(f"Intervention probability: {results['intervention']['probability']:.3f}")
    print(f"Explanation: {results['intervention']['explanation']}")

    if results["stance"]:
        print("\nStance Change:")
        print(f"Before: {results['stance']['before']:.3f}")
        print(f"After:  {results['stance']['after']:.3f}")
        print(f"Change: {results['stance']['change']:+.3f}")
