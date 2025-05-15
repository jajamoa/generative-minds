import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from ..base import ModelConfig
from ..m03_census.model import Census, REASON_MAPPING, SCENARIO_MAPPING
from ..m03_census.utils.spatial_utils import calculate_distance_to_affected_area, create_proposal_description

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
        
        # Inherit batch_size from config or use default
        self.batch_size = getattr(self.config, "batch_size", 5)

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
                
    async def simulate_opinions(self, region: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using parallel processing with transcript and expert reflection data."""
        self.current_proposal_id = proposal.get("proposal_id", None)
        print(f"DEBUG simulate_opinions: Processing proposal_id={self.current_proposal_id}")
        
        # Get scenario ID
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        
        # Prepare proposal description
        proposal_desc = create_proposal_description(proposal)
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
        """Process a single agent's opinion generation with transcript and expert reflection."""
        try:
            agent_id = agent.get("id")
            
            # Load both transcript and expert reflection
            transcript_data = self._load_transcript(agent_id)
            expert_reflection_data = self._load_expert_reflection(agent_id)
            
            # Calculate distance if coordinates are available
            distance_km = None
            if agent:
                coords = agent.get("coordinates", {})
                lat = coords.get("lat")
                lon = coords.get("lng")
                if lat is not None and lon is not None:
                    distance_km = calculate_distance_to_affected_area(
                        lat, lon,
                        proposal.get("cells", {})
                    )
                    print(f"DEBUG: Agent {agent_id} is {distance_km:.2f}km from affected area")
            
            # Only proceed if we have both data sources
            if not transcript_data or not expert_reflection_data:
                print(f"ERROR: Missing data for agent {agent_id}. Transcript: {bool(transcript_data)}, Reflection: {bool(expert_reflection_data)}")
                return self._generate_fallback_opinion()
            
            # Generate opinion using both data sources
            prompt = self._build_opinion_prompt_with_transcript(
                transcript_data,
                expert_reflection_data,
                proposal_desc,
                agent,
                distance_km,
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
            
            return {
                "opinions": {
                    scenario_id: rating
                },
                "reasons": {
                    scenario_id: reason_scores
                }
            }
            
        except Exception as e:
            print(f"ERROR: Opinion generation failed for agent {agent_id}: {str(e)}")
            return self._generate_fallback_opinion()

    def _build_opinion_prompt_with_transcript(self,
                                            transcript: Dict[str, Any],
                                            expert_reflection: Dict[str, Any],
                                            proposal_desc: str,
                                            agent: Dict[str, Any],
                                            distance_km: float,
                                            region: str) -> str:
        """Build a prompt using transcript and expert reflection data.
        
        Args:
            transcript: The transcript data dictionary.
            expert_reflection: The expert reflection data.
            proposal_desc: Human-readable proposal description.
            agent: The agent data containing geolocation content.
            distance_km: Distance from agent to affected area in kilometers.
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

        # Add location context if available
        location_context = ""
        if agent:
            geo_content = agent.get("geo_content", {})
            if geo_content.get("narrative"):
                location_context += f"\nYour neighborhood context:\n{geo_content['narrative']}"
        
        if distance_km is not None:
            location_context += f"\nYour distance from the affected area: {distance_km:.2f} kilometers"
        
        # Extract expert reflections
        expert_insights = "Expert Analysis:\n"
        if "political scientist" in expert_reflection:
            reflections = expert_reflection["political scientist"]
            expert_insights += "\n".join(reflections) + "\n\n"
        
        # Build the complete prompt
        prompt = f"""Based on the following interview transcript and expert analysis of a community member in {region}, evaluate this housing policy proposal:{location_context}

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

Base your evaluation on the provided interview transcript, expert analysis, and your location context.
Format your response EXACTLY as shown above, with one rating (1-10) and twelve reason scores (1-5 each).
"""
        print(f"DEBUG: Generated prompt: {prompt}")
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