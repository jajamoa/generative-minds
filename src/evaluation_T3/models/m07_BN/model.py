import json
import numpy as np
from .qwen_extractor import QwenExtractor
from typing import Dict, Tuple, Optional, List, Any
from models.base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING
from scipy.stats import entropy
import os
import networkx as nx
from .motif_library.graph_reconstruction import MotifBasedReconstructor
from .motif_library.motif_library import get_demographic_statistics, MotifLibrary


def ensure_evaluation_prefix(path: str) -> str:
    """Ensure path has src/evaluation_T3 prefix if not already present."""
    prefix = "src/evaluation_T3"
    if not path.startswith(prefix) and not path.startswith("/"):
        return os.path.join(prefix, path)
    return path


class BayesianNetwork(Census):
    def __init__(self, config: ModelConfig):
        """Initialize the belief network model."""
        super().__init__(config)
        self.extractor = QwenExtractor()
        self.label_to_id = None
        self.id_to_label = None

        # Get paths from config
        self.graph_path = ensure_evaluation_prefix(
            getattr(self.config, "graph_path", "data/samples/sample_1.json")
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
        self.output_dir = ensure_evaluation_prefix(
            getattr(
                self.config,
                "output_dir",
                os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "graph_reconstruction"
                ),
            ),
        )

        # Get reconstruction parameters from config
        self.seed_node = getattr(self.config, "seed_node", "upzoning_stance")
        self.max_iterations = getattr(self.config, "max_iterations", 20)
        self.target_demographic = getattr(self.config, "target_demographic", None)
        self.demographic_weight = getattr(self.config, "demographic_weight", 0.3)

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

    def load_graph(
        self, json_file: str = None, graph: nx.DiGraph = None
    ) -> tuple[dict, dict]:
        """Load graph structure from either JSON file or NetworkX DiGraph.

        Args:
            json_file: Path to the JSON file containing the graph structure
            graph: NetworkX DiGraph object

        Returns:
            tuple: (dag, node_labels) where
                - dag: dict with node_id as key and dict of parents/children as value
                - node_labels: dict mapping node_id to node label

        Raises:
            ValueError: If neither json_file nor graph is provided
        """
        if not json_file and not graph:
            raise ValueError("Either json_file or graph must be provided")

        if json_file:
            # Load from JSON file
            with open(json_file, "r") as f:
                data = json.load(f)

            nodes = data["nodes"]
            edges = data["edges"]

            # Create node label mapping
            node_labels = {
                node_id: node_data["label"] for node_id, node_data in nodes.items()
            }

            # Initialize DAG structure
            dag = {
                node_id: {
                    "parents": [],  # List of (parent_id, modifier) tuples
                    "children": [],  # List of (child_id, modifier) tuples
                }
                for node_id in nodes
            }

            # Add edge information
            for edge_id, edge_data in edges.items():
                source = edge_data["source"]
                target = edge_data["target"]
                modifier = edge_data.get("modifier", 1.0)  # 默认为正向影响

                # Add parent-child relationships
                dag[target]["parents"].append((source, modifier))
                dag[source]["children"].append((target, modifier))

        else:
            # Load from NetworkX DiGraph
            # Create node label mapping
            node_labels = nx.get_node_attributes(graph, "label")
            if not node_labels:
                node_labels = {node: str(node) for node in graph.nodes()}

            # Initialize DAG structure
            dag = {
                node: {
                    "parents": [],
                    "children": [],
                }
                for node in graph.nodes()
            }

            # Add edge information
            for source, target, data in graph.edges(data=True):
                modifier = data.get("modifier", 1.0)
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

    def _get_topological_order(self, dag: Dict) -> List[str]:
        """Get topological ordering of nodes."""
        visited = set()
        topo_order = []

        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for parent, _ in dag[node]["parents"]:
                if parent not in visited:
                    dfs(parent)
            topo_order.append(node)

        for node in dag:
            if node not in visited:
                dfs(node)
        return topo_order

    def simulate_intervention(
        self,
        dag: Dict = None,
        node_labels: Dict = None,
        intervention_node: Optional[str] = None,
        intervention_prob: Optional[float] = None,
        base_prob: Optional[float] = 0.5,
        num_samples: int = 1000,
    ) -> Dict[str, Dict]:
        """Simulate the belief network with optional intervention."""
        node_samples = {node: [] for node in dag}
        topo_order = self._get_topological_order(dag)

        for _ in range(num_samples):
            prob_values = {}

            for node in topo_order:
                if not dag[node]["parents"]:  # Root node
                    if node == intervention_node:
                        prob_values[node] = intervention_prob
                    else:
                        prob_values[node] = base_prob  # Base probability
                else:
                    parent_impacts = []
                    total_modifier = 0

                    for parent_id, modifier in dag[node]["parents"]:
                        parent_prob = prob_values[parent_id]
                        impact = self._calculate_impact(parent_prob, modifier)
                        parent_impacts.append(impact)
                        total_modifier += abs(modifier)

                    if len(parent_impacts) > 1:
                        weights = [
                            abs(m) / total_modifier for _, m in dag[node]["parents"]
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
                if not dag[node]["children"]:
                    sample = np.random.binomial(n=1, p=prob_values[node])
                else:
                    sample = prob_values[node]
                node_samples[node].append(sample)

        # Calculate statistics
        results = {}
        for node in dag:
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
        dag: Dict,
        node_labels: Dict,
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

        for node in dag:
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

    def reconstruct_graph(self) -> Dict:
        """Reconstruct the graph from motif library using agent's complete demographics."""
        try:
            # Parse target demographic if it's a string (from json.dumps)
            if isinstance(self.target_demographic, str):
                try:
                    agent_profile = json.loads(self.target_demographic)
                except json.JSONDecodeError:
                    agent_profile = {"type": self.target_demographic}
            else:
                agent_profile = self.target_demographic

            print("\nUsing Agent Demographics:")
            print("-" * 40)
            for key, value in agent_profile.items():
                print(f"{key}: {value}")
            print()

            # Load motif library with better error handling
            try:
                library = MotifLibrary.load_library(self.motif_library_name)
                if library is None:
                    print(
                        f"Warning: Could not load motif library from {self.motif_library_name}"
                    )
                    print("Creating empty motif library with default settings")
                    library = MotifLibrary()
                    library.demographic_distribution = {str(agent_profile): 1.0}
            except Exception as e:
                print(f"Error loading motif library: {str(e)}")
                print("Creating empty motif library with default settings")
                library = MotifLibrary()
                library.demographic_distribution = {str(agent_profile): 1.0}

            # Create reconstructor with agent's complete demographic profile
            reconstructor = MotifBasedReconstructor(
                library,
                similarity_threshold=getattr(self.config, "similarity_threshold", 0.3),
                node_merge_threshold=getattr(self.config, "node_merge_threshold", 0.8),
                target_demographic=agent_profile,  # input agent profile
                demographic_weight=self.demographic_weight,
            )

            # Reconstruct graph
            reconstructed_graph = reconstructor.reconstruct_graph(
                self.seed_node, max_iterations=self.max_iterations, min_score=0.3
            )

            print(f"\nReconstructed graph has {len(reconstructed_graph.nodes())} nodes")
            print(f"Reconstructed graph has {len(reconstructed_graph.edges())} edges")

            print(
                "-------------------------- Graph Reconstruction Complete --------------------------"
            )

            return reconstructed_graph

        except Exception as e:
            print(f"Error during graph reconstruction: {str(e)}")
            print("Creating minimal default graph")
            dag = {"node_0": {"parents": [], "children": []}}
            node_labels = {"node_0": self.seed_node}

        return dag, node_labels

    def _graph_to_json(self, dag: Dict, node_labels: Dict) -> Dict[str, Any]:
        """Convert DAG to JSON format.

        Args:
            dag: Dictionary representation of DAG

        Returns:
            Dictionary in JSON format with nodes and edges
        """
        # Create graph data in JSON format
        graph_data = {
            "nodes": {
                node_id: {
                    "id": node_id,
                    "label": node_labels[node_id],
                    "type": "concept",
                    "in_degree": len(dag[node_id]["parents"]),  # calculate in-degree
                    "out_degree": len(dag[node_id]["children"]),  # calculate out-degree
                }
                for node_id in dag
            },
            "edges": {},
        }

        # Add edges
        edge_id = 0
        for node_id, node_data in dag.items():
            for child, modifier in node_data["children"]:
                graph_data["edges"][f"e{edge_id}"] = {
                    "id": f"e{edge_id}",
                    "source": node_id,
                    "target": child,
                    "type": "causal",
                    "modifier": modifier,
                }
                edge_id += 1

        return graph_data

    async def simulate_opinions(
        self, region: str, proposal: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate opinions using Bayesian Network for all agents.

        Args:
            region: The target region name
            proposal: A dictionary containing the rezoning proposal details

        Returns:
            A dictionary with all agents' IDs as keys, containing opinions, reasons, and graph
        """
        # Get proposal ID and convert to scenario ID
        proposal_id = proposal.get("proposal_id", "proposal_000")
        scenario_id = SCENARIO_MAPPING.get(
            proposal_id, "1.1"  # Default to "1.1" if not found
        )

        # Load all agents' data
        try:
            with open(self.agent_data_file, "r") as f:
                agents_data = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load agent data: {str(e)}")

        results = {}

        # # TODO: for debug, only use the first 2 agents
        # for agent_data in agents_data[:2]:

        # Process each agent
        for agent_data in agents_data:

            agent_id = agent_data["id"]
            try:
                # Set agent's demographic profile as target
                agent_profile = agent_data.get("agent", {})
                # Use all fields in agent profile as demographic information

                self.target_demographic = json.dumps(agent_profile, sort_keys=True)

                # Reconstruct graph for this specific agent
                graph = self.reconstruct_graph()

                dag, node_labels = self.load_graph(graph=graph)
                graph_json = self._graph_to_json(dag, node_labels)

                # Extract intervention from proposal
                node_label, intervention_prob, explanation, expected_effects = (
                    self.extractor.extract_intervention(
                        {"dag": dag, "node_labels": node_labels},
                        self._create_proposal_description(proposal),
                    )
                )

                # Run simulations

                # baseline
                base_results = self.simulate_intervention(
                    dag=dag, node_labels=node_labels
                )
                # intervention
                intervention_results = self.simulate_intervention(
                    dag=dag,
                    node_labels=node_labels,
                    intervention_node=node_label,
                    intervention_prob=intervention_prob,
                )

                # Get stance node
                stance_nodes = [
                    node
                    for node, label in node_labels.items()
                    if "stance" in label.lower()
                ]
                stance_node = stance_nodes[0] if stance_nodes else None
                # Compute node contributions
                contributions = self.compute_node_contributions(
                    dag=dag,
                    node_labels=node_labels,
                    base_results=base_results,
                    intervention_results=intervention_results,
                    stance_node=stance_node,
                )
                # Calculate opinion score
                opinion_score = round(intervention_results[stance_node]["mean"] * 10)

                # Store results for this agent
                results[agent_id] = {
                    "opinions": {scenario_id: opinion_score},
                    "reasons": {
                        scenario_id: {
                            node_labels[node]: round(contrib_score * 4 + 1)
                            for node, contrib_score in contributions
                        }
                    },
                    "graph": graph_json,
                    "demographic": agent_profile,  # save the complete demographic information
                }
                # print("-------------------------------------")
                # print(results)
                # print("-------------------------------------")

            except Exception as e:
                print(f"Error processing agent {agent_id}: {str(e)}")
                continue

        return results
