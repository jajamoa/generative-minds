import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import dashscope
from typing import Dict, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class QwenExtractor:
    """
    Extract intervention operations from questions about causal Bayesian networks
    """

    def __init__(self, api_key=None, model="qwen-plus"):
        # Try to load API key from environment
        parent_env_path = Path(__file__).parent.parent.parent / ".env"
        parent_env_local_path = Path(__file__).parent.parent.parent / ".env.local"

        if parent_env_local_path.exists():
            load_dotenv(dotenv_path=parent_env_local_path)
        elif parent_env_path.exists():
            load_dotenv(dotenv_path=parent_env_path)

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DashScope API key is required")

        self.model = model

    def extract_intervention(
        self, graph_info: Dict, question: str
    ) -> Optional[Tuple[str, float, str]]:
        """
        Extract intervention from question based on causal graph structure

        Args:
            graph_info: Dictionary containing:
                - dag: {node_id: {'parents': [(parent_id, modifier)],
                                'children': [(child_id, modifier)]}}
                - node_labels: {node_id: label}
            question: Question text to analyze

        Returns:
            Tuple of (node_id, intervention_value, explanation) or None
        """
        causal_graph = graph_info["dag"]
        node_labels = graph_info["node_labels"]

        # Get nodes and their labels
        nodes_with_labels = {
            node_id: node_labels.get(node_id, node_id)
            for node_id in causal_graph.keys()
        }

        edges = []
        for node_id, node_data in causal_graph.items():
            for child, modifier in node_data["children"]:
                edges.append(
                    {
                        "source": node_id,
                        "source_label": node_labels.get(node_id, node_id),
                        "target": child,
                        "target_label": node_labels.get(child, child),
                        "modifier": modifier,
                    }
                )

        prompt = f"""
Given a causal Bayesian network about urban development impacts with these nodes:

Nodes:
{', '.join(f"{node_id} ({label})" for node_id, label in nodes_with_labels.items())}

Relationships:
{self._format_key_relationships(edges)}

For this question:
{question}

Determine:
1. Which node should be intervened on (do-operator)
2. What probability value to set it to (between 0 and 1)
3. Why this intervention makes sense given the causal structure

Consider:
- Root nodes (no incoming edges) are better intervention targets
- Intervention value: 
  - "increase a lot" → 0.8-1.0
  - "moderate increase" → 0.6-0.8
  - "slight increase" → 0.5-0.6
  - "decrease" → 0.0-0.4
  - "maintain/no change" → 0.5

Return JSON:
{{
    "intervention_node": "exact node id",
    "intervention_value": "some value between 0 and 1",
    "explanation": "why this intervention makes sense",
    "expected_effects": ["list of likely affected downstream nodes"]
}}
"""

        try:
            print("------------- PROMPT -------------")
            GREEN = "\033[92m"
            END = "\033[0m"
            print(f"{GREEN}{prompt}{END}")
            print("------------- PROMPT -------------")
            response = dashscope.Generation.call(
                api_key=self.api_key,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze causal networks to determine appropriate interventions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                result_format="message",
                temperature=0.1,
            )

            print("------------- RESPONSE -------------")
            YELLOW = "\033[93m"
            END = "\033[0m"
            print(f"{YELLOW}{response.output.choices[0].message.content}{END}")
            print("------------- RESPONSE -------------")

            if response.status_code == 200:
                content = response.output.choices[0].message.content
                result = json.loads(self._clean_json_string(content))

                if result.get("intervention_node"):
                    node = result["intervention_node"]
                    value = result.get("intervention_value", 0.5)
                    explanation = result.get("explanation", "")
                    expected_effects = result.get("expected_effects", [])

                    # Validate node exists
                    if node in nodes_with_labels:
                        logger.info(f"Found intervention: do({node}) = {value}")
                        logger.info(f"Explanation: {explanation}")
                        logger.info(
                            f"Expected effects on: {', '.join(expected_effects)}"
                        )
                        return node, value, explanation, expected_effects

            return None

        except Exception as e:
            logger.error(f"Error extracting intervention: {e}")
            return None

    def _format_key_relationships(self, edges: List[Dict]) -> str:
        """Format edge information for prompt"""
        relationships = []
        for edge in edges:
            modifier = edge["modifier"]
            direction = "increases" if modifier > 0 else "decreases"
            relationships.append(
                f"- {edge['source']} ({edge['source_label']}) {direction} {edge['target']} ({edge['target_label']}) (strength: {abs(modifier):.1f})"
            )
        return "\n".join(relationships) + "\n"

    def _clean_json_string(self, json_str):
        """Clean JSON string"""
        start = json_str.find("{")
        end = json_str.rfind("}") + 1
        if start >= 0 and end > start:
            return json_str[start:end]
        return json_str


def main():
    # Load sample graph
    with open("data/samples/sample_1.json") as f:
        causal_graph = json.load(f)

    extractor = QwenExtractor()

    # Example questions based on the sample graph
    questions = [
        "What would happen to quality of life if we significantly increase building heights?",
        "How would reducing traffic congestion affect air quality and health impacts?",
        "What if we maintain current levels of sunlight access to public spaces?",
        "If we completely change the community character, how would it affect cultural heritage?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        result = extractor.extract_intervention(causal_graph, question)
        if result:
            node, value, explanation = result
            print(f"Intervention: do({node}) = {value}")
            print(f"Explanation: {explanation}")
        else:
            print("No clear intervention identified")


if __name__ == "__main__":
    main()
