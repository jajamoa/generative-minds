"""CBN Agent for processing belief attribution tasks"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from llm_utils import QwenLLM
from node_similarity import compute_node_similarity
from build_cbn_from_motifs import merge_motifs_to_graph


class CBNAgent:
    """Agent that uses causal Bayesian networks for belief attribution"""

    def __init__(
        self,
        model: str = "qwen-plus",
        cbn_path: Optional[str] = None,
        motifs_dir: Optional[str] = None,
    ):
        """Initialize CBN agent

        Args:
            model: Name of LLM model to use
            cbn_path: Path to causal Bayesian network definition (JSON). If None, will use motifs_dir to build on-the-fly.
            motifs_dir: Directory containing motif library (per-participant folders). If provided, builds CBNs dynamically from story.
        """
        self.llm = QwenLLM(model=model)
        self.dynamic_build = motifs_dir is not None or not cbn_path
        self.motifs_dir = Path(motifs_dir) if motifs_dir else None
        self._motif_library: Optional[List[Dict[str, Any]]] = None
        # Track last dynamically built CBN for saving/inspection
        self.last_built_cbn_graph: Optional[Dict[str, Any]] = None
        self.last_built_text: str = ""
        self.last_built_id: Optional[str] = None

        # Static CBNs mode
        self.cbns_by_id = {}
        self.default_cbn = None
        if not self.dynamic_build and cbn_path:
            with open(cbn_path) as f:
                cbn_data = json.load(f)

            if isinstance(cbn_data, list) and len(cbn_data) > 0:
                for session in cbn_data:
                    if "graphs" in session and len(session["graphs"]) > 0:
                        prolific_id = session.get("prolificId", "unknown")
                        first_graph = session["graphs"][0]
                        if "graphData" in first_graph:
                            graph_data = first_graph["graphData"]
                            self.cbns_by_id[prolific_id] = graph_data
                            # Set first valid CBN as default
                            if self.default_cbn is None:
                                self.default_cbn = graph_data

                if self.default_cbn is None:
                    raise ValueError("No valid CBN found in data")
            else:
                # Fallback: assume it's already in the correct format
                self.default_cbn = cbn_data
                self.cbns_by_id["default"] = cbn_data

    def select_cbn(
        self, prolific_id: Optional[str] = None
    ) -> Tuple[Dict[str, Any], bool]:
        """Select CBN based on prolific_id

        Args:
            prolific_id: Participant's prolific ID to find specific CBN

        Returns:
            Tuple of (CBN graphData, found_match: bool)
        """
        if prolific_id and prolific_id in self.cbns_by_id:
            return self.cbns_by_id[prolific_id], True

        # Fallback to default CBN
        return self.default_cbn, False

    def _ensure_motif_library(self) -> None:
        if self._motif_library is not None:
            return
        library: List[Dict[str, Any]] = []
        if not self.motifs_dir or not self.motifs_dir.exists():
            self._motif_library = []
            return
        for subdir in self.motifs_dir.iterdir():
            if not subdir.is_dir():
                continue
            json_file = subdir / f"{subdir.name}.json"
            if json_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        motifs = data.get("motifs", []) or []
                        for m in motifs:
                            # Attach owner id for traceability
                            m_copy = dict(m)
                            m_copy["_owner"] = data.get("prolific_id", subdir.name)
                            library.append(m_copy)
                except Exception:
                    continue
        self._motif_library = library

    def _score_motif_against_text(self, motif: Dict[str, Any], text: str) -> float:
        labels = list((motif.get("node_labels") or {}).values())
        if not labels or not text:
            return 0.0
        sims: List[float] = []
        for lbl in labels:
            try:
                sims.append(compute_node_similarity(str(lbl), str(text)))
            except Exception:
                continue
        if not sims:
            return 0.0
        sims.sort(reverse=True)
        topk = sims[:2] if len(sims) >= 2 else sims
        return sum(topk) / len(topk)

    def _build_cbn_from_text(
        self, text: str, k: int = 12, threshold: float = 0.5
    ) -> Dict[str, Any]:
        self._ensure_motif_library()
        selected: List[Dict[str, Any]] = []
        if not self._motif_library:
            return {"nodes": {}, "edges": {}}
        scored = []
        for m in self._motif_library:
            score = self._score_motif_against_text(m, text)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, m in scored:
            if len(selected) >= k:
                break
            if score >= threshold or len(selected) < max(5, k // 2):
                selected.append(m)
        nodes, edges = merge_motifs_to_graph(selected)
        built = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {"selected_motifs": len(selected)},
        }
        # Persist last built for external saving
        self.last_built_cbn_graph = built
        self.last_built_text = text
        return built

    def get_cbn_for_vqa(self, vqa: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
        if self.dynamic_build:
            # Build text from story/context
            context_qas = vqa.get("context_qas", []) or []
            # Use only Story context to ensure SEE/NO share the same CBN
            story_text = ""
            for qa in context_qas:
                q = str(qa.get("question", ""))
                a = str(qa.get("answer", ""))
                if q.lower().strip().startswith("story") and a:
                    story_text = a
                    break
            # Fallback: if no explicit Story, join non-Observation entries
            if not story_text:
                parts: List[str] = []
                for qa in context_qas:
                    q = str(qa.get("question", ""))
                    a = str(qa.get("answer", ""))
                    if "observation" in q.lower():
                        continue
                    if q or a:
                        parts.append(f"Q: {q} A: {a}")
                story_text = " ".join(parts)
            built = self._build_cbn_from_text(story_text)
            # Track id for saving
            self.last_built_id = vqa.get("prolific_id")
            return built, True
        else:
            pid = vqa.get("prolific_id")
            return self.select_cbn(pid)

    def translate_to_do_operation(
        self,
        question: str,
        demographics: Dict[str, Any],
        context_qas: list,
        include_demographics: bool,
        include_context: bool,
        temperature: float = 0.1,
        debug: bool = False,
        prolific_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Translate question to do() operation based on CBN variables"""

        # Build context with explicit Story/Observation sections for clarity
        story_text = ""
        observation_text = ""
        other_qas: List[str] = []
        if include_context and context_qas:
            for qa in context_qas:
                q = str(qa.get("question", ""))
                a = str(qa.get("answer", ""))
                ql = q.lower()
                if "story" in ql and not story_text:
                    story_text = a
                elif "observation" in ql and not observation_text:
                    observation_text = a
                else:
                    if q or a:
                        other_qas.append(f"Q: {q} A: {a}")

        context_lines: List[str] = []
        if include_demographics and demographics:
            context_lines.append(
                "Demographics: "
                + ", ".join([f"{k}: {v}" for k, v in demographics.items()])
            )
        if story_text:
            context_lines.append(f"Story: {story_text}")
        if observation_text:
            context_lines.append(f"Observation: {observation_text}")
        if other_qas:
            context_lines.append("Conversation: " + " ".join(other_qas))
        context = "\n".join(context_lines)

        # Get the appropriate CBN for this participant (dynamic or static)
        current_cbn, _ = self.get_cbn_for_vqa(
            {"prolific_id": prolific_id, "context_qas": context_qas}
        )

        # Get CBN variables with labels
        nodes = current_cbn.get("nodes", {})
        cbn_variables = []
        node_id_to_label = {}

        for node_id, node_data in nodes.items():
            label = node_data.get("label", node_id)
            cbn_variables.append(label)
            node_id_to_label[label] = node_id

        prompt = f"""Context:\n{context}\n\nQuestion: {question}\nAvailable CBN Variables: {cbn_variables}\n\nTranslate this question into a causal intervention (do operation).\nWhich variable should be intervened on and what value should it be set to?\nFormat: {{"variable": "variable_name", "value": 0.8}}"""

        if debug:
            print(f"\n=== CBN Agent: Translating to do() operation ===")

        response = self.llm.generate_response(
            prompt,
            system_message="You are translating questions to causal interventions. Return only JSON.",
            temperature=temperature,
            return_json=True,
            debug=debug,
        )

        if debug:
            print(f"Do operation: {response}")

        # Convert label back to node_id if response contains a label
        if response and "variable" in response:
            variable_name = response["variable"]
            # If it's a label, convert to node_id
            if variable_name in node_id_to_label:
                response["variable"] = node_id_to_label[variable_name]

        # Fallback with proper node_id
        fallback_variable = (
            node_id_to_label.get(cbn_variables[0]) if cbn_variables else "unknown"
        )
        return response or {"variable": fallback_variable, "value": 0.5}

    def run_cbn_inference(
        self,
        do_operation: Dict[str, Any],
        debug: bool = False,
        prolific_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Run CBN inference with do() operation"""

        # Get the appropriate CBN for this participant (dynamic or static)
        if self.dynamic_build and self.last_built_cbn_graph:
            current_cbn = self.last_built_cbn_graph
        else:
            current_cbn, _ = self.get_cbn_for_vqa(
                {"prolific_id": prolific_id, "context_qas": []}
            )

        # Initialize beliefs
        beliefs = {}
        nodes = current_cbn.get("nodes", {})
        edges = current_cbn.get("edges", {})

        # Set all nodes to default values first
        for node in nodes:
            beliefs[node] = 0.5  # neutral belief

        # Apply do() operation - fix the intervened variable
        intervention_var = do_operation.get("variable")
        raw_value = do_operation.get("value", 0.5)
        # Coerce non-numeric values into numeric [0.05, 0.95]
        value_map = {
            "yes": 0.9,
            "true": 0.9,
            "present": 0.9,
            "occur": 0.9,
            "occurs": 0.9,
            "high": 0.9,
            "no": 0.1,
            "false": 0.1,
            "absent": 0.1,
            "not": 0.1,
            "low": 0.1,
        }
        if isinstance(raw_value, (int, float)):
            intervention_value = float(raw_value)
        elif isinstance(raw_value, str):
            rv = raw_value.strip().lower()
            intervention_value = value_map.get(rv, 0.5)
        else:
            intervention_value = 0.5

        if intervention_var in nodes:
            beliefs[intervention_var] = intervention_value

        # Forward propagation through the network
        # Simple approach: iterate through edges and update beliefs
        for _ in range(3):  # Multiple passes for convergence
            updated = False
            for edge_id, edge in edges.items():
                source = edge.get("source")
                target = edge.get("target")
                modifier = edge.get("modifier", 1.0)

                # Skip if target is the intervention variable (it's fixed)
                if target == intervention_var:
                    continue

                if source in beliefs and target in beliefs:
                    # Calculate influence
                    source_prob = beliefs[source]
                    # Simple causal influence: target influenced by source
                    influence = source_prob * modifier
                    new_value = min(0.95, max(0.05, 0.5 + (influence - 0.5) * 0.7))

                    if abs(new_value - beliefs[target]) > 0.01:
                        beliefs[target] = new_value
                        updated = True

            if not updated:
                break

        if debug:
            print(f"\n=== CBN Agent: Running CBN inference ===")
            print(f"Intervention: {intervention_var} = {intervention_value}")
            print(f"CBN inference result: {beliefs}")

        return beliefs

    def select_answer(
        self,
        vqa: Dict[str, Any],
        cbn_state: Dict[str, float],
        temperature: float = 0.1,
        debug: bool = False,
    ) -> str:
        """Select answer based on updated CBN state"""

        question = vqa.get("task_question", "")
        task_type = vqa.get("task_type", "belief_attribution")

        # Format CBN state for LLM
        cbn_summary = ", ".join([f"{var}: {val:.2f}" for var, val in cbn_state.items()])

        if task_type == "belief_update":
            # Handle scale prediction for belief_update
            scale = vqa.get("scale", [1, 10])

            prompt = f"""Question: {question}
Current CBN State: {cbn_summary}
Scale: {scale[0]} to {scale[1]}

Based on the causal network state, predict a number on the scale from {scale[0]} to {scale[1]}.
Return only the number."""

            if debug:
                print(f"\n=== CBN Agent: Predicting scale value (belief_update) ===")

            response = self.llm.generate_response(
                prompt,
                system_message="You are predicting scale values based on causal inference results. Return only a number.",
                temperature=temperature,
                debug=debug,
            )

            # Extract number from response
            if response:
                import re

                numbers = re.findall(r"\b\d+\b", response.strip())
                if numbers:
                    try:
                        predicted_value = int(numbers[0])
                        # Clamp to scale range
                        predicted_value = max(scale[0], min(scale[1], predicted_value))
                        return str(predicted_value)
                    except ValueError:
                        pass

            # Fallback: return middle of scale
            return str((scale[0] + scale[1]) // 2)

        else:
            # Handle multiple choice for belief_attribution without LLM: build from opinion backward
            answer_options = vqa.get("answer_options", {})

            def score_option_backward(option_text: str, k: int = 8) -> float:
                # Use node-label similarity as weights and aggregate current beliefs
                current_cbn = (
                    self.last_built_cbn_graph
                    if self.last_built_cbn_graph
                    else {"nodes": {}}
                )
                node_id_to_label = {}
                for nid, ndata in (current_cbn.get("nodes") or {}).items():
                    node_id_to_label[nid] = ndata.get("label", nid)

                sims: List[Tuple[str, float]] = []  # (node_id, sim)
                for nid, label in node_id_to_label.items():
                    try:
                        s = compute_node_similarity(str(option_text), str(label))
                        sims.append((nid, max(0.0, s)))
                    except Exception:
                        continue

                if not sims:
                    return 0.0

                sims.sort(key=lambda x: x[1], reverse=True)
                top = sims[:k]
                # Normalize weights
                total_w = sum(w for _, w in top) or 1.0
                norm = [(nid, w / total_w) for nid, w in top]
                # Aggregate belief: baseline 0.5, shift by weighted deviations
                value = 0.5
                for nid, w in norm:
                    b = float(cbn_state.get(nid, 0.5))
                    value += (b - 0.5) * w
                # Clamp
                return max(0.05, min(0.95, value))

            if answer_options:
                option_scores = {
                    key: score_option_backward(text)
                    for key, text in answer_options.items()
                }
                best_key = max(option_scores.items(), key=lambda kv: kv[1])[0]
                if debug:
                    print(f"Option scores (backward build): {option_scores}")
                return best_key

            # Fallback: if no options provided, default to 'A'
            return "A"

    def process_query(
        self,
        vqa: Dict[str, Any],
        demographics: Dict[str, Any],
        context_qas: list,
        include_demographics: bool,
        include_context: bool,
        temperature: float = 0.1,
        debug: bool = False,
    ) -> str:
        """Process query using CBN causal inference pipeline"""

        if debug:
            print(f"\n{'='*80}")
            print(f"CBN AGENT: Starting causal inference pipeline")
            print(f"Task Type: {vqa.get('task_type', 'unknown')}")
            print(f"Question: {vqa.get('task_question', '')}")
            print(f"{'='*80}")

        # Get prolific_id from vqa data for CBN selection
        prolific_id = vqa.get("prolific_id")

        # Step 1: Select/Build CBN based on story (dynamic) or prolific_id (static)
        selected_cbn, found_match = self.get_cbn_for_vqa(vqa)

        # Import Colors for colored output
        try:
            from llm_utils import Colors
        except ImportError:
            # Fallback if Colors not available
            class Colors:
                RED = "\033[91m"
                GREEN = "\033[92m"
                RESET = "\033[0m"

                @staticmethod
                def format(text, color):
                    return f"{color}{text}{Colors.RESET}"

        if debug:
            print(f"\n=== Step 1: CBN Selected ===")
            print(f"Prolific ID: {prolific_id}")
            if found_match and self.dynamic_build:
                match_status = Colors.format("✓ Built CBN from story", Colors.GREEN)
            elif found_match:
                match_status = Colors.format("✓ Found specific CBN", Colors.GREEN)
            else:
                match_status = Colors.format(
                    "⚠ Using default CBN (fallback)", Colors.RED
                )
            print(f"CBN Match: {match_status}")
            print(f"CBN nodes: {len(selected_cbn.get('nodes', {}))}")
            print(f"CBN edges: {len(selected_cbn.get('edges', {}))}")

        # Step 2: Translate question to do() operation
        task_question = vqa.get("task_question", "")
        do_operation = self.translate_to_do_operation(
            task_question,
            demographics,
            context_qas,
            include_demographics,
            include_context,
            temperature,
            debug,
            prolific_id,
        )

        # Step 3: Run CBN inference with do() operation
        cbn_state = self.run_cbn_inference(do_operation, debug, prolific_id)

        # Step 4: Select answer based on CBN state
        response = self.select_answer(vqa, cbn_state, temperature, debug)

        if debug:
            print(f"\n=== CBN Agent: Final Result ===")
            print(f"Final Answer: {response}")
            print(f"{'='*80}")

        return response
