import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from ..base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING

class Transcript(Census):
    """A model that generates opinions using OpenAI API and transcript data."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize model components and set up transcript data paths.
        
        Args:
            config: Model configuration containing settings.
        """
        super().__init__(config)
        
        # Set up transcript data path from config or default
        self.transcript_dir = config.transcript_dir if hasattr(config, 'transcript_dir') else os.path.join(os.path.dirname(__file__), "data", "processed_transcript")
        
        print(f"DEBUG Transcript.__init__: transcript_dir={self.transcript_dir}")

    def _load_transcript(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load transcript data for a specific agent by matching filename with agent ID.
        
        Args:
            agent_id: The ID of the agent from agents_with_geo.json.
            
        Returns:
            The transcript data dictionary if found, None otherwise.
        """
        # Directly construct the transcript file path using agent_id
        transcript_file = os.path.join(self.transcript_dir, f"{agent_id}.json")
        
        try:
            if os.path.exists(transcript_file):
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

    async def _generate_opinion(self, 
                              agent: Dict[str, Any], 
                              proposal: Dict[str, Any],
                              proposal_desc: str,
                              region: str) -> Dict[str, Any]:
        """Generate opinion using transcript data and LLM.
        
        Args:
            agent: The agent data dictionary.
            proposal: The proposal dictionary.
            proposal_desc: Human-readable proposal description.
            region: The target region name.
            
        Returns:
            A dictionary containing the generated opinion and reasons.
        """
        agent_id = agent.get("id")
        transcript_data = self._load_transcript(agent_id)
        
        if transcript_data:
            # Build prompt using transcript data
            prompt = self._build_opinion_prompt_with_transcript(
                transcript_data,
                proposal_desc,
                region
            )
        else:
            # Fallback to base prompt if no transcript data
            prompt = self._build_opinion_prompt(agent, proposal_desc, region)
        
        # Generate and parse response
        temp = self.temperature
        response = await self.llm.generate(
            prompt,
            temperature=temp,
            max_tokens=self.max_tokens
        )
        
        rating, reason_scores = self._parse_opinion_response(response)
        
        # Get scenario ID
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
                                            proposal_desc: str,
                                            region: str) -> str:
        """Build a prompt incorporating transcript data.
        
        Args:
            transcript: The transcript data dictionary.
            proposal_desc: Human-readable proposal description.
            region: The target region name.
            
        Returns:
            A string containing the complete prompt.
        """
        # Extract QA pairs from transcript
        qa_pairs = transcript.get("transcript", [])
        
        # Build context from transcript responses
        context = "Based on the interview responses:\n\n"
        
        # Add relevant QA pairs to context
        for qa in qa_pairs:
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if question and answer:
                context += f"Q: {question}\nA: {answer}\n\n"
        
        # Combine with base prompt structure
        prompt = f"""As someone who provided the following responses about housing in {region}:

{context}

Please evaluate this housing policy proposal:

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

Consider the provided interview responses when evaluating each aspect.
Format your response EXACTLY as shown above, with one rating (1-10) and twelve reason scores (1-5 each).
"""
        return prompt 