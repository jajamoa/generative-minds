import openai
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import json

class LLMInterface(ABC):
    """Abstract interface for LLM services"""
    @abstractmethod
    def analyze_nodes_for_merging(self, nodes):
        """Analyze nodes and suggest merges"""
        pass

    def extract_causal_relationships(self, question: str, answer: str) -> List[Dict[str, Any]]:
        """Extract causal relationships from QA pair"""
        raise NotImplementedError

class OpenAIInterface(LLMInterface):
    """OpenAI implementation of LLM interface"""
    def __init__(self, client=None):
        self.client = client or self._create_default_client()
    
    @staticmethod
    def _create_default_client():
        """Create default OpenAI client using environment variables"""
        import os
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")
        return openai.OpenAI(api_key=api_key)
    
    def analyze_nodes_for_merging(self, nodes):
        """
        Analyze nodes and suggest merges using OpenAI
        Args:
            nodes: Dictionary of nodes with their data
        Returns:
            List of merge groups, each containing node_ids, merged_label, and reason
        """
        node_list = "\n".join([f"{id}: {data['label']}" for id, data in nodes.items()])
        
        system_message = """
        Analyze nodes and suggest merges. Return JSON array:
        {
          "merge_groups": [
            {
              "node_ids": ["id1", "id2"],
              "merged_label": "string",
              "reason": "string"
            }
          ]
        }
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Analyze these nodes for potential merges:\n{node_list}"}
                ],
                temperature=0.1
            )
            
            result = response.choices[0].message.content
            try:
                parsed_result = json.loads(result)
                return parsed_result.get("merge_groups", [])
            except json.JSONDecodeError:
                print(f"Error parsing LLM response as JSON: {result}")
                return []
        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return []

    def extract_causal_relationships(self, question: str, answer: str) -> List[Dict[str, Any]]:
        """
        Extract causal relationships from QA pair using OpenAI
        Returns list of relationships with source, target, confidence, and influence type
        """
        system_message = """
        Identify causal relationships in the text. Extract concepts and how they influence each other.
        For each causal relationship, provide:
        1. Source concept: The causing factor
        2. Target concept: The affected factor
        3. Relationship: Whether the influence is positive (increases/supports) or negative (decreases/opposes)
        4. Confidence: Your confidence in this causal relationship (0.0 to 1.0)
        
        Format as JSON:
        {
          "relationships": [
            {
              "source_concept": "concept_name",
              "target_concept": "concept_name", 
              "positive_influence": true/false,
              "confidence": 0.0-1.0
            }
          ]
        }
        
        If no clear causal relationships exist, return {"relationships": []}.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Question: {question}\nAnswer: {answer}"}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            return parsed.get("relationships", [])
            
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            return [] 