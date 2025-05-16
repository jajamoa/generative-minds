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
import asyncio
import time


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

        # Fix the motif library path
        default_motif_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "motif_library",
            "motif_library.json",
        )

        motif_library_path = getattr(
            self.config, "motif_library_name", default_motif_path
        )
        if isinstance(motif_library_path, dict):
            motif_library_path = default_motif_path
        self.motif_library_name = ensure_evaluation_prefix(motif_library_path)

        # Make sure the motif library file exists
        if not os.path.exists(self.motif_library_name):
            raise FileNotFoundError(
                f"Motif library not found at {self.motif_library_name}"
            )

        self.output_dir = ensure_evaluation_prefix(
            getattr(self.config, "output_dir", "graph_reconstruction")
        )

        # Get reconstruction parameters from config
        self.seed_node = getattr(self.config, "seed_node", "support_for_upzoning")
        self.max_iterations = getattr(self.config, "max_iterations", 20)
        self.demographic_weight = getattr(self.config, "demographic_weight", 0.3)

        # Load agent data
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

    async def process_single_agent(
        self, agent_data: Dict, proposal: Dict
    ) -> Tuple[str, Optional[Dict]]:
        """Process a single agent with better error handling."""
        agent_id = agent_data.get("id", "unknown_agent")
        print(f"\n[DEBUG] Processing agent {agent_id}")
        start_time = time.time()

        try:
            async with self.semaphore:
                # Get agent's demographic information
                agent_profile = agent_data.get("agent", {})
                if not agent_profile:
                    print(f"[WARNING] No demographic information for agent {agent_id}")
                    agent_profile = {}  # Use empty profile instead of returning None

                # Reconstruct graph
                try:
                    dag, node_labels = self.reconstruct_graph(agent_profile)
                except Exception as e:
                    print(
                        f"[ERROR] Graph reconstruction failed for agent {agent_id}: {str(e)}"
                    )
                    dag, node_labels = self._create_minimal_graph()

                # Extract intervention
                try:
                    extract_result = await self._extract_intervention_with_retry(
                        dag, node_labels, proposal
                    )
                    if not extract_result:
                        raise ValueError("Failed to extract intervention")
                    node_label, intervention_prob, explanation, expected_effects = (
                        extract_result
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Intervention extraction failed for agent {agent_id}: {str(e)}"
                    )
                    node_label = self.seed_node
                    intervention_prob = 0.5
                    explanation = "Fallback intervention"
                    expected_effects = {}

                # Run simulations
                try:
                    base_results = self.simulate_intervention(
                        dag=dag, node_labels=node_labels
                    )
                    intervention_results = self.simulate_intervention(
                        dag=dag,
                        node_labels=node_labels,
                        intervention_node=node_label,
                        intervention_prob=intervention_prob,
                    )
                except Exception as e:
                    print(f"[ERROR] Simulation failed for agent {agent_id}: {str(e)}")
                    # Create minimal simulation results
                    base_results = {
                        self.seed_node: {"mean": 0.5, "std": 0.1, "samples": [0.5]}
                    }
                    intervention_results = {
                        self.seed_node: {"mean": 0.5, "std": 0.1, "samples": [0.5]}
                    }

                # Calculate final results
                try:
                    stance_node = next(
                        (
                            node
                            for node, label in node_labels.items()
                            if "stance" in label.lower()
                        ),
                        self.seed_node,
                    )
                    contributions = self.compute_node_contributions(
                        dag=dag,
                        node_labels=node_labels,
                        base_results=base_results,
                        intervention_results=intervention_results,
                        stance_node=stance_node,
                    )
                    opinion_score = round(
                        intervention_results[stance_node]["mean"] * 10
                    )
                except Exception as e:
                    print(
                        f"[ERROR] Final calculations failed for agent {agent_id}: {str(e)}"
                    )
                    contributions = [(self.seed_node, 0.5)]
                    opinion_score = 5

                # Prepare final output
                proposal_id = proposal.get("proposal_id", "proposal_000")
                scenario_id = SCENARIO_MAPPING.get(proposal_id, "1.1")

                return agent_id, {
                    "opinions": {scenario_id: opinion_score},
                    "reasons": {
                        scenario_id: {
                            node_labels.get(node, str(node)): round(
                                contrib_score * 4 + 1
                            )
                            for node, contrib_score in contributions
                        }
                    },
                    "graph": self._graph_to_json(dag, node_labels),
                    "demographic": agent_profile,
                }

        except Exception as e:
            print(f"[ERROR] Unexpected error processing agent {agent_id}: {str(e)}")
            return agent_id, self._generate_fallback_result(agent_id, proposal)

    def reconstruct_graph(self, agent_demographic: Dict) -> Tuple[Dict, Dict]:
        """
        Reconstruct the graph from motif library using agent's demographic information.

        Args:
            agent_demographic: Dictionary containing agent's demographic information

        Returns:
            Tuple[Dict, Dict]: (dag, node_labels) where
                - dag: Dictionary representation of the graph structure
                - node_labels: Dictionary mapping node IDs to their labels
        """
        try:
            print(f"\n[DEBUG] Starting graph reconstruction")
            print(
                f"[DEBUG] Agent demographic: {json.dumps(agent_demographic, indent=2)}"
            )
            print(f"[DEBUG] Motif library path: {self.motif_library_name}")

            # Load motif library
            try:
                library = MotifLibrary.load_library(self.motif_library_name)
                print(f"[DEBUG] Successfully loaded motif library")
                print(f"[DEBUG] Number of motif groups: {len(library.semantic_motifs)}")
                print(
                    f"[DEBUG] Total individual motifs: {sum(len(motifs) for motifs in library.semantic_motifs.values())}"
                )
            except Exception as e:
                print(f"[ERROR] Failed to load motif library: {str(e)}")
                raise

            # Create reconstructor with agent's specific demographic
            reconstructor = MotifBasedReconstructor(
                library,
                similarity_threshold=0.3,
                node_merge_threshold=0.8,
                target_demographic=agent_demographic,
                demographic_weight=self.demographic_weight,
            )

            # Reconstruct graph
            print(
                f"[DEBUG] Starting graph reconstruction with seed node: {self.seed_node}"
            )
            reconstructed_graph = reconstructor.reconstruct_graph(
                self.seed_node, max_iterations=self.max_iterations, min_score=0.3
            )

            if not reconstructed_graph or reconstructed_graph.number_of_nodes() == 0:
                print("[WARNING] Empty graph generated, falling back to minimal graph")
                return self._create_minimal_graph()

            # Convert NetworkX graph to dictionary format
            dag = {}
            node_labels = {}

            # Add nodes and their labels
            for node in reconstructed_graph.nodes():
                label = reconstructed_graph.nodes[node].get("label", str(node))
                node_labels[node] = label
                dag[node] = {
                    "parents": [],
                    "children": [],
                    "attributes": reconstructed_graph.nodes[node],
                }

            # Add edges with their attributes
            for u, v, data in reconstructed_graph.edges(data=True):
                modifier = data.get("modifier", 1.0)
                confidence = data.get("confidence", 0.0)

                # Add to children list of source node
                dag[u]["children"].append(
                    {"node": v, "modifier": modifier, "confidence": confidence}
                )

                # Add to parents list of target node
                dag[v]["parents"].append(
                    {"node": u, "modifier": modifier, "confidence": confidence}
                )

            print(f"[DEBUG] Graph reconstruction completed successfully")
            print(f"[DEBUG] Nodes: {len(dag)}")
            print(
                f"[DEBUG] Edges: {sum(len(data['children']) for data in dag.values())}"
            )

            # Validate the reconstructed graph
            if self._validate_graph(dag, node_labels):
                return dag, node_labels
            else:
                print(
                    "[WARNING] Invalid graph generated, falling back to minimal graph"
                )
                return self._create_minimal_graph()

        except Exception as e:
            print(f"[ERROR] Critical error in graph reconstruction: {str(e)}")
            import traceback

            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return self._create_minimal_graph()

    def _create_minimal_graph(self) -> Tuple[Dict, Dict]:
        """Create a minimal valid graph when reconstruction fails."""
        minimal_dag = {
            self.seed_node: {
                "parents": [],
                "children": [],
                "attributes": {"label": self.seed_node},
            }
        }
        minimal_labels = {self.seed_node: self.seed_node}
        return minimal_dag, minimal_labels

    def _validate_graph(self, dag: Dict, node_labels: Dict) -> bool:
        """
        Validate the reconstructed graph.

        Args:
            dag: Dictionary representation of the graph
            node_labels: Dictionary of node labels

        Returns:
            bool: True if graph is valid, False otherwise
        """
        try:
            # Check if seed node exists
            if self.seed_node not in dag:
                print(f"[ERROR] Seed node {self.seed_node} not in graph")
                return False

            # Check for disconnected components
            nodes = set(dag.keys())
            visited = set()

            def dfs(node):
                visited.add(node)
                for child in dag[node]["children"]:
                    child_node = child["node"]
                    if child_node not in visited:
                        dfs(child_node)

            dfs(self.seed_node)

            if visited != nodes:
                print("[ERROR] Graph contains disconnected components")
                return False

            # Check for cycles
            visited = set()
            rec_stack = set()

            def has_cycle(node):
                visited.add(node)
                rec_stack.add(node)

                for child in dag[node]["children"]:
                    child_node = child["node"]
                    if child_node not in visited:
                        if has_cycle(child_node):
                            return True
                    elif child_node in rec_stack:
                        return True

                rec_stack.remove(node)
                return False

            if has_cycle(self.seed_node):
                print("[ERROR] Graph contains cycles")
                return False

            return True

        except Exception as e:
            print(f"[ERROR] Graph validation failed: {str(e)}")
            return False

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
        """Simulate opinions using Bayesian Network for all agents."""
        proposal_id = proposal.get("proposal_id", "proposal_000")
        scenario_id = SCENARIO_MAPPING.get(proposal_id, "1.1")

        try:
            with open(self.agent_data_file, "r") as f:
                agents_data = json.load(f)

            # HACK: choose first 5 agents
            agents_data = agents_data[:5]

        except Exception as e:
            print(f"[ERROR] Failed to load agent data: {str(e)}")
            agents_data = [
                {"id": f"agent_{i:03d}"} for i in range(10)
            ]  # Fallback to dummy agents

        results = {}
        self.semaphore = asyncio.Semaphore(10)

        print(
            f"\n[{time.strftime('%H:%M:%S')}] Starting simulation with {len(agents_data)} agents"
        )
        start_time = time.time()

        tasks = [
            self.process_single_agent(agent_data, proposal)
            for agent_data in agents_data
        ]
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results with better error handling
        for agent_id, opinion_data in completed_results:
            if isinstance(opinion_data, Exception):
                print(
                    f"[ERROR] Agent {agent_id} failed with error: {str(opinion_data)}"
                )
                # Generate fallback result instead of skipping
                results[agent_id] = self._generate_fallback_result(agent_id, proposal)
            elif opinion_data is None:
                print(f"[WARNING] Agent {agent_id} returned None, using fallback")
                results[agent_id] = self._generate_fallback_result(agent_id, proposal)
            else:
                results[agent_id] = opinion_data

        end_time = time.time()
        print(
            f"\n[{time.strftime('%H:%M:%S')}] Processed {len(results)} agents in {end_time - start_time:.2f}s"
        )

        return results

    def _generate_fallback_result(self, agent_id: str, proposal: Dict) -> Dict:
        """Generate a fallback result when agent processing fails."""
        proposal_id = proposal.get("proposal_id", "proposal_000")
        scenario_id = SCENARIO_MAPPING.get(proposal_id, "1.1")

        # Create minimal graph
        minimal_dag, minimal_labels = self._create_minimal_graph()
        graph_json = self._graph_to_json(minimal_dag, minimal_labels)

        return {
            "opinions": {scenario_id: 5},  # Neutral opinion
            "reasons": {scenario_id: {self.seed_node: 3}},  # Neutral contribution
            "graph": graph_json,
            "demographic": {},  # Empty demographic
        }

    async def _extract_intervention_with_retry(
        self, dag: Dict, node_labels: Dict, proposal: Dict
    ) -> Optional[Tuple[str, float, str, Dict]]:
        """Extract intervention with retries."""
        max_retries = 3
        retry_delay = 1

        for retry in range(max_retries):
            try:
                # Create proposal description
                proposal_desc = self._create_proposal_description(proposal)

                # Extract intervention
                result = await self.extractor.extract_intervention(
                    {"dag": dag, "node_labels": node_labels}, proposal_desc
                )

                if result:
                    return result

                if retry < max_retries - 1:
                    print(f"Retrying intervention extraction (attempt {retry + 1})")
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                print(f"Extraction attempt {retry + 1} failed: {str(e)}")
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay)

        return None

    def _create_proposal_description(self, proposal: Dict) -> str:
        """Create a description of the proposal for intervention extraction."""
        return f"""Proposal: {proposal.get('title', 'Unknown')}
Description: {proposal.get('description', 'No description available')}
Impact: {proposal.get('impact', 'Unknown impact')}"""
