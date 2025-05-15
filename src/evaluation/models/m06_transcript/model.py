import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from ..base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING
from ..m03_census.utils.spatial_utils import calculate_distance_to_affected_area

class Transcript(Census):
    """A model that generates opinions using transcript data and supports multiple LLM providers."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize model components and set up transcript data paths.
        
        Args:
            config: Model configuration containing settings.
        """
        super().__init__(config)
        
        # Set up transcript data path from config or default
        self.transcript_dir = config.transcript_dir if hasattr(config, 'transcript_dir') else os.path.join(os.path.dirname(__file__), "data", "processed_transcript")
        
        print(f"DEBUG Transcript.__init__: transcript_dir={self.transcript_dir}")
        
        # Inherit batch_size from config or use default
        self.batch_size = getattr(self.config, "batch_size", 5)

    async def simulate_opinions(self, region: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using parallel processing with transcript data.
        
        Args:
            region: The target region name.
            proposal: A dictionary containing the rezoning proposal details.
            
        Returns:
            A dictionary with multiple participants containing opinions and reasons.
        """
        self.current_proposal_id = proposal.get("proposal_id", None)
        print(f"DEBUG simulate_opinions: Processing proposal_id={self.current_proposal_id}")
        
        # Get scenario ID
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        
        # Prepare readable description of the proposal
        proposal_desc = self._create_proposal_description(proposal)
        print(f"DEBUG: Generated proposal description: {proposal_desc[:100]}...")
        
        results = {}
        
        # Process agents in batches
        for i in range(0, len(self.agent_data), self.batch_size):
            batch = self.agent_data[i:i + self.batch_size]
            tasks = []
            
            # Create tasks for each agent in the batch
            for agent in batch:
                participant_id = agent.get("id")
                if not participant_id:
                    continue
                    
                task = self._process_single_agent(
                    agent,
                    proposal,
                    proposal_desc,
                    region,
                    scenario_id
                )
                tasks.append(task)
            
            # Run batch of tasks concurrently
            batch_results = await asyncio.gather(*tasks)
            
            # Update results with batch results
            for agent, result in zip(batch, batch_results):
                participant_id = agent.get("id")
                if participant_id:
                    results[participant_id] = result
        
        return results

    async def _process_single_agent(self,
                                  agent: Dict[str, Any],
                                  proposal: Dict[str, Any],
                                  proposal_desc: str,
                                  region: str,
                                  scenario_id: str) -> Dict[str, Any]:
        """Process a single agent's opinion generation with transcript data.
        
        Args:
            agent: The agent data dictionary.
            proposal: The proposal dictionary.
            proposal_desc: The proposal description.
            region: The target region name.
            scenario_id: The scenario ID.
            
        Returns:
            A dictionary containing the agent's opinions and reasons.
        """
        try:
            agent_id = agent.get("id")
            transcript_data = self._load_transcript(agent_id)
            
            # Calculate distance to affected area
            agent_coords = agent.get("coordinates", {})
            agent_lat = agent_coords.get("lat")
            agent_lon = agent_coords.get("lng")
            
            distance_km = None
            if agent_lat is not None and agent_lon is not None:
                distance_km = calculate_distance_to_affected_area(
                    agent_lat, agent_lon, 
                    proposal.get("cells", {})
                )
                print(f"DEBUG: Agent {agent_id} is {distance_km:.2f}km from affected area")
            
            if transcript_data:
                # Build prompt using transcript data
                prompt = self._build_opinion_prompt_with_transcript(
                    transcript_data,
                    proposal_desc,
                    region,
                    agent,
                    distance_km
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
            
            return {
                "opinions": {
                    scenario_id: rating
                },
                "reasons": {
                    scenario_id: reason_scores
                }
            }
            
        except Exception as e:
            print(f"ERROR: Opinion generation failed for participant {agent_id}: {str(e)}")
            return self._generate_fallback_opinion(scenario_id)

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
        
        # Calculate distance to affected area
        agent_coords = agent.get("coordinates", {})
        agent_lat = agent_coords.get("lat")
        agent_lon = agent_coords.get("lng")
        
        distance_km = None
        if agent_lat is not None and agent_lon is not None:
            distance_km = calculate_distance_to_affected_area(
                agent_lat, agent_lon, 
                proposal.get("cells", {})
            )
            print(f"DEBUG: Agent {agent_id} is {distance_km:.2f}km from affected area")
        
        if transcript_data:
            # Build prompt using transcript data
            prompt = self._build_opinion_prompt_with_transcript(
                transcript_data,
                proposal_desc,
                region,
                agent,
                distance_km
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
                                            region: str,
                                            agent: Dict[str, Any],
                                            distance_km: Optional[float] = None) -> str:
        """Build a prompt incorporating transcript data.
        
        Args:
            transcript: The transcript data dictionary.
            proposal_desc: Human-readable proposal description.
            region: The target region name.
            agent: The agent data dictionary containing geo information.
            distance_km: Optional distance from the affected area in kilometers.
            
        Returns:
            A string containing the complete prompt.
        """
        # Extract QA pairs from transcript
        qa_pairs = transcript.get("transcript", [])
        
        # Build context from transcript responses and location info
        context = "Based on your interview responses and location:\n\n"
        
        # Add distance information if available
        if distance_km is not None:
            context += f"You live {distance_km:.2f} kilometers from the affected area.\n\n"
        
        # Add neighborhood context if available
        geo_content = agent.get("geo_content", {})
        if geo_content.get("narrative"):
            context += f"Your neighborhood context:\n{geo_content['narrative']}\n\n"
        
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

Consider your location context and interview responses when evaluating each aspect.
Format your response EXACTLY as shown above, with one rating (1-10) and twelve reason scores (1-5 each).
"""
        print(f"DEBUG: Generated prompt: {prompt}")
        return prompt 