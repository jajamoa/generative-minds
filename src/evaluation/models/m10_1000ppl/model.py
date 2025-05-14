import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING

class M10_1000ppl(Census):
    """A model that generates opinions using transcript and expert reflection data."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize model components and set up data paths.
        
        Args:
            config: Model configuration containing settings.
        """
        super().__init__(config)
        
        # Set up data paths using Path for better cross-platform compatibility
        base_dir = Path(__file__).parent
        self.transcript_dir = Path(config.transcript_dir) if hasattr(config, 'transcript_dir') else base_dir / "data" / "processed_transcript"
        self.expert_reflection_dir = Path(config.expert_reflection_dir) if hasattr(config, 'expert_reflection_dir') else base_dir / "data" / "political_expert_reflection"
        
        print(f"DEBUG M10_1000ppl.__init__: transcript_dir={self.transcript_dir}")
        print(f"DEBUG M10_1000ppl.__init__: expert_reflection_dir={self.expert_reflection_dir}")

    def _load_transcript(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load transcript data for a specific agent.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            The transcript data dictionary if found, None otherwise.
        """
        transcript_file = self.transcript_dir / f"{agent_id}.json"
        
        try:
            if transcript_file.exists():
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    transcript_data = json.load(f)
                    print(f"DEBUG: Successfully loaded transcript for agent {agent_id}")
                    return transcript_data
            else:
                print(f"WARNING: No transcript file found for agent {agent_id} at {transcript_file}")
                return None
        except Exception as e:
            print(f"ERROR: Failed to load transcript for agent {agent_id}: {str(e)}")
            return None

    def _load_expert_reflection(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load expert reflection data for a specific agent.
        
        Args:
            agent_id: The ID of the agent.
            
        Returns:
            The expert reflection data if found, None otherwise.
        """
        reflection_file = self.expert_reflection_dir / f"{agent_id}.json"
        try:
            if reflection_file.exists():
                with open(reflection_file, 'r', encoding='utf-8') as f:
                    reflection_data = json.load(f)
                    print(f"DEBUG: Successfully loaded expert reflection for agent {agent_id}")
                    return reflection_data
            else:
                print(f"WARNING: No expert reflection found for agent {agent_id} at {reflection_file}")
                return None
        except Exception as e:
            print(f"ERROR: Failed to load expert reflection for agent {agent_id}: {str(e)}")
            return None
                
    async def _generate_opinion(self, 
                              agent: Dict[str, Any], 
                              proposal: Dict[str, Any],
                              proposal_desc: str,
                              region: str) -> Dict[str, Any]:
        """Generate opinion using transcript and expert reflection data.
        
        Args:
            agent: The agent data dictionary (only used for ID).
            proposal: The proposal dictionary.
            proposal_desc: Human-readable proposal description.
            region: The target region name.
            
        Returns:
            A dictionary containing the generated opinion and reasons.
        """
        agent_id = agent.get("id")
        
        # Load both transcript and expert reflection
        transcript_data = self._load_transcript(agent_id)
        expert_reflection_data = self._load_expert_reflection(agent_id)
        
        # Only proceed if we have both data sources
        if not transcript_data or not expert_reflection_data:
            print(f"ERROR: Missing data for agent {agent_id}. Transcript: {bool(transcript_data)}, Reflection: {bool(expert_reflection_data)}")
            return self._generate_fallback_opinion()
        
        # Generate opinion using both data sources
        prompt = self._build_opinion_prompt_with_transcript(
            transcript_data,
            expert_reflection_data,
            proposal_desc,
            region
        )
        
        # Generate and parse response
        temp = self.temperature
        response = await self.llm.generate(
            prompt,
            temperature=temp,
            max_tokens=self.max_tokens
        )
        
        rating, reason_scores = self._parse_opinion_response(response)
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        
        return {
            "opinions": {
                scenario_id: rating
            },
            "reasons": {
                scenario_id: reason_scores
            }
        }

    def _build_opinion_prompt_with_transcript(self,
                                            transcript: Dict[str, Any],
                                            expert_reflection: Dict[str, Any],
                                            proposal_desc: str,
                                            region: str) -> str:
        """Build a prompt using transcript and expert reflection data.
        
        Args:
            transcript: The transcript data dictionary.
            expert_reflection: The expert reflection data.
            proposal_desc: Human-readable proposal description.
            region: The target region name.
            
        Returns:
            A string containing the complete prompt.
        """
        # Extract QA pairs from transcript
        qa_pairs = transcript.get("transcript", [])
        
        # Build context from transcript responses
        context = "Interview Transcript:\n\n"
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if question and answer:
                context += f"Q: {question}\nA: {answer}\n\n"
        
        # Extract expert reflections
        expert_insights = "Expert Analysis:\n"
        if "political scientist" in expert_reflection:
            reflections = expert_reflection["political scientist"]
            expert_insights += "\n".join(reflections) + "\n\n"
        
        # Build the complete prompt
        prompt = f"""Based on the following interview transcript and expert analysis of a community member in {region}, evaluate this housing policy proposal:

{context}

{expert_insights}

Proposal to evaluate:
{proposal_desc}

Consider how this proposal might affect the community across multiple dimensions. Rate EACH of the following aspects on a scale of 1-5, where:
1 = Very Negative Impact
2 = Somewhat Negative Impact
3 = Neutral/No Impact
4 = Somewhat Positive Impact
5 = Very Positive Impact

Also provide an overall opinion rating from 1-10 where:
1-2 = Strongly Oppose
3-4 = Oppose
5-6 = Neutral
7-8 = Support
9-10 = Strongly Support

Required Response Format:
Rating: [1-10]
Reasons:
A: [1-5] (Housing supply and availability)
B: [1-5] (Affordability for low/middle-income residents)
C: [1-5] (Neighborhood character impact)
D: [1-5] (Infrastructure capacity)
E: [1-5] (Economic development)
F: [1-5] (Environmental impact)
G: [1-5] (Transit access)
H: [1-5] (Displacement risk)
I: [1-5] (Equity and social justice)
J: [1-5] (Public amenities)
K: [1-5] (Property values)
L: [1-5] (Historical preservation)

Base your evaluation ONLY on the provided interview transcript and expert analysis.
Format your response EXACTLY as shown above, with one rating (1-10) and twelve reason scores (1-5 each).
"""
        return prompt

    def _generate_fallback_opinion(self) -> Dict[str, Any]:
        """Generate a neutral fallback opinion when data is missing."""
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        return {
            "opinions": {
                scenario_id: 5  # Neutral rating
            },
            "reasons": {
                scenario_id: {code: 3 for code in REASON_MAPPING.values()}  # All neutral
            }
        } 