import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
import dashscope
from typing import Dict, Optional, Tuple, List
import aiohttp
import asyncio

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

    async def extract_intervention(
        self, graph_info: Dict, question: str
    ) -> Optional[Tuple[str, float, str, List[str]]]:
        """
        Extract intervention from question based on causal graph structure (async version)

        Args:
            graph_info: Dictionary containing:
                - dag: {node_id: {'parents': [(parent_id, modifier)],
                                'children': [(child_id, modifier)]}}
                - node_labels: {node_id: label}
            question: Question text to analyze

        Returns:
            Tuple of (node_id, intervention_value, explanation, expected_effects) or None
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
            response = await asyncio.to_thread(
                dashscope.Generation.call,
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

            # 获取并解析 JSON 内容
            content = response["output"]["choices"][0]["message"]["content"]
            # print(f"\n=== Raw API Response ===\n{content}\n===================")
            GREEN = "\033[92m"
            END = "\033[0m"
            print(f"{GREEN}A call to the API has been made{END}")

            try:
                # 清理并解析 JSON 字符串
                json_str = self._clean_json_string(content)
                # print(f"\n=== Cleaned JSON String ===\n{json_str}\n===================")

                result = json.loads(json_str)
                # print(f"\n=== Parsed JSON Result ===\n{result}\n===================")

                if result.get("intervention_node"):
                    node = result["intervention_node"]
                    value = float(result.get("intervention_value", 0.5))
                    explanation = result.get("explanation", "")
                    expected_effects = result.get("expected_effects", [])

                    # 验证节点存在
                    if node in graph_info["node_labels"]:
                        logger.info(f"Found intervention: do({node}) = {value}")
                        logger.info(f"Explanation: {explanation}")
                        logger.info(
                            f"Expected effects on: {', '.join(expected_effects)}"
                        )
                        return node, value, explanation, expected_effects
                    else:
                        logger.error(f"Node {node} not found in graph")
                        return None

            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error at position {e.pos}: {e.msg}")
                logger.error(f"Problem line: {e.doc.splitlines()[e.lineno-1]}")
                return None
            except Exception as e:
                logger.error(f"Error processing API response: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"Error calling API: {str(e)}")
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

    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string by removing markdown code blocks and finding the JSON object"""
        # Remove markdown code block markers and any text after the JSON object
        lines = json_str.split("\n")
        json_lines = []
        in_json = False

        for line in lines:
            # Skip comment lines
            if "//" in line:
                line = line.split("//")[0].rstrip()

            if line.strip() == "```json":
                in_json = True
                continue
            elif line.strip() == "```":
                break
            elif line.strip().startswith("{") or in_json:
                in_json = True
                if line.strip():  # 只添加非空行
                    json_lines.append(line)

        if not json_lines:  # If no code block found, try to find JSON directly
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                return json_str[start:end]

        return "\n".join(json_lines)


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
            node, value, explanation, expected_effects = result
            print(f"Intervention: do({node}) = {value}")
            print(f"Explanation: {explanation}")
            print(f"Expected effects on: {', '.join(expected_effects)}")
        else:
            print("No clear intervention identified")


if __name__ == "__main__":
    main()
