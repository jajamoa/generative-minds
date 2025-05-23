import json
import numpy as np
from .qwen_extractor import QwenExtractor
from typing import Dict, Tuple, Optional, List, Any
from models.base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING
from scipy.stats import entropy
import os


def ensure_evaluation_prefix(path: str) -> str:
    """Ensure path has src/evaluation prefix if not already present."""
    prefix = "src/evaluation"
    if not path.startswith(prefix) and not path.startswith("/"):
        return os.path.join(prefix, path)
    return path


responses_file_path = ensure_evaluation_prefix(
    "experiment/eval/data/sf_prolific_survey/causal_graph_responses_5.11_with_geo.json"
)


def find_agent_graph_data(agent_id: str, responses_file: str) -> Optional[Dict]:
    """
    Search for and extract graph data for a specific agent ID from the responses file.

    Args:
        agent_id (str): The ID of the agent to search for
        responses_file (str): Path to the responses JSON file

    Returns:
        Optional[Dict]: The graph data for the agent if found, None otherwise
    """
    responses_file = ensure_evaluation_prefix(responses_file)

    try:
        with open(responses_file, "r") as f:
            data = json.load(f)

        for entry in data:
            if entry.get("prolificId") == agent_id:
                graphs = entry.get("graphs", None)
                all_time_stamps = [graph.get("timestamp", None) for graph in graphs]
                if all_time_stamps in [None, []]:
                    continue
                latest_timestamp = max(all_time_stamps)
                latest_graph = next(
                    (
                        graph
                        for graph in graphs
                        if graph.get("timestamp") == latest_timestamp
                    ),
                    None,
                )
                if latest_graph is not None:
                    json_data = latest_graph.get("graphData", None)
                    assert isinstance(json_data, dict) or isinstance(
                        json_data, None
                    ), f"Graph data is not a dict: {json_data}"
                    if json_data is None:
                        continue
                    nodes = json_data.get("nodes", None)
                    assert isinstance(nodes, dict), f"Nodes are not a dict: {nodes}"
                    stance_nodes = [
                        node
                        for node, node_data in nodes.items()
                        if node_data.get("is_stance", False)
                    ]
                    return json_data, stance_nodes

        return None, None

    except Exception as e:
        print(f"Error reading responses file: {e}")
        return None, None


class BayesianNetwork(Census):
    def __init__(self, config: ModelConfig):
        """Initialize the belief network model."""
        self.config = config
        self.extractor = QwenExtractor()
        self.dag = None
        self.node_labels = None
        self.label_to_id = None
        self.id_to_label = None
        self.agent_data_file = ensure_evaluation_prefix(
            getattr(self.config, "agent_data_file", "data/samples/sample_1.json")
        )

        self.motif_library_name = ensure_evaluation_prefix(
            getattr(
                self.config,
                "motif_library_name",
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "motif_library",
                    "motif_library.json",
                ),
            )
        )

        agent_data_path = ensure_evaluation_prefix(
            getattr(
                self.config,
                "agent_data_file",
                "experiment/eval/data/sf_prolific_survey/responses_5.11_with_geo.json",
            )
        )
        self.agent_data_file = agent_data_path
        self.agent_ids = list(set(self._load_agent_ids(agent_data_path)))

    def _load_agent_ids(self, agent_data_path: str) -> list:
        """Load only agent IDs from the agent data file.

        Args:
            agent_data_path: Path to the agent data JSON file.

        Returns:
            List of agent IDs.
        """
        try:
            if os.path.exists(agent_data_path):
                with open(agent_data_path, "r", encoding="utf-8") as f:
                    raw_agents = json.load(f)
                return [
                    agent.get("id", f"agent_{i:03d}")
                    for i, agent in enumerate(raw_agents)
                ]
            else:
                print(f"WARNING: Agent data file not found: {agent_data_path}")
                return [f"agent_{i:03d}" for i in range(14)]  # Default to 14 agents
        except Exception as e:
            print(f"ERROR: Failed to load agent IDs: {str(e)}")
            return [f"agent_{i:03d}" for i in range(14)]  # Default to 14 agents

    def load_graph(self, raw_causal_graph: Dict) -> tuple[dict, dict]:
        """
        Load graph structure from raw causal graph data and build a DAG.

        Args:
            raw_causal_graph: Dictionary containing the graph structure with nodes and edges

        Returns:
            tuple: (dag, node_labels) where
                - dag: dict with node_id as key and dict of parents/children as value
                - node_labels: dict mapping node_id to node label
        """
        # Extract nodes and edges from raw graph
        nodes = raw_causal_graph.get("nodes", {})
        edges = raw_causal_graph.get("edges", {})

        # Create node label mapping
        node_labels = {
            node_id: node_data.get("label", f"node_{node_id}")
            for node_id, node_data in nodes.items()
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
            source = edge_data.get("source")
            target = edge_data.get("target")

            # Extract modifier from edge data
            modifier = max(min(edge_data.get("modifier", 0.0), 1.0), -1.0)

            # Ensure source and target exist
            if source in dag and target in dag:
                # Add parent-child relationships with modifier
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

        try:
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
        except Exception as e:
            print(f"Error simulating intervention: {e}")
            import pdb

            pdb.set_trace()
            return {}

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
        stance_node: str = None,
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
        if stance_node is None:
            raise ValueError("Stance node cannot be None")
        if stance_node not in base_results or stance_node not in intervention_results:
            raise ValueError(
                f"Stance node {stance_node} not found in simulation results"
            )

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

        # Get proposal ID and convert to scenario ID
        proposal_id = proposal.get("proposal_id", "proposal_000")
        scenario_id = SCENARIO_MAPPING.get(
            proposal_id, "1.1"
        )  # Default to "1.1" if not found

        results = {}

        print("-------------------------------------------------------")

        # TODO: for debug, only use the first 2 agents
        # for agent_id in self.agent_ids[:30]:

        for agent_id in self.agent_ids:

            raw_causal_graph, stance_nodes = find_agent_graph_data(
                agent_id, responses_file_path
            )

            if raw_causal_graph is None:
                print(f"WARNING: No causal graph found for agent {agent_id}")
                continue

            self.dag, self.node_labels = self.load_graph(raw_causal_graph)

            # remove the out degree of stance node
            # NOTE: this is a hack to make the stance node not affect the other nodes
            for node in stance_nodes:
                children_to_process = self.dag[node]["children"]

                # handle each child node
                for child_id, _ in children_to_process:
                    # remove the stance node from the child node's parents
                    self.dag[child_id]["parents"] = [
                        (parent_id, modifier)
                        for parent_id, modifier in self.dag[child_id]["parents"]
                        if parent_id != node
                    ]

                # clear the children of the stance node
                self.dag[node]["children"] = []

            retried = 0
            while retried < 5:
                try:
                    node_label, intervention_prob, explanation, expected_effects = (
                        self.extractor.extract_intervention(
                            {"dag": self.dag, "node_labels": self.node_labels},
                            self._create_proposal_description(proposal),
                        )
                    )
                    break
                except Exception as e:
                    retried += 1
                    print(f"Error extracting intervention: {e}")
                    continue

            # Run simulations
            base_results = self.simulate_intervention()
            intervention_results = self.simulate_intervention(
                intervention_node=node_label,
                intervention_prob=intervention_prob,
            )

            stance_node = stance_nodes[0] if stance_nodes else None

            # Compute node contributions
            contributions = self.compute_node_contributions(
                base_results, intervention_results, stance_node
            )

            opinion_score = round(intervention_results[stance_node]["mean"] * 10)

            # Create a single agent response with filtered reasons
            results[agent_id] = {
                "opinions": {
                    # NOTE: it is actually a Bernoulli distribution
                    scenario_id: opinion_score
                    # scenario_id: 1
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
    model = BayesianNetwork()
    model.load_graph("data/samples/sample_1.json")

    question = "What happens to traffic congestion when we increase building height?"
    results = model.analyze_question(question)

    # Print stance results
    if results["stance"]:
        print("\nStance Change:")
        print(f"Before: {results['stance']['before']:.3f}")
        print(f"After:  {results['stance']['after']:.3f}")
        print(f"Change: {results['stance']['change']:+.3f}")
