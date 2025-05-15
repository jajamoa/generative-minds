import os
import random
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict

from ..base import BaseModel, ModelConfig
from ..m03_census.components.llm import OpenAILLM, create_llm
from ..m03_census.model import REASON_MAPPING, SCENARIO_MAPPING
from ..m03_census.utils.spatial_utils import (
    calculate_distance_to_affected_area,
    create_proposal_description
)

class NaiveBaseline(BaseModel):
    """A naive baseline using only proposal/grid info, no demographics."""

    def __init__(self, config: ModelConfig = None):
        """Initialize model components.
        
        Args:
            config: Model configuration containing settings.
        """
        super().__init__(config)
        
        # Get LLM configuration
        llm_provider = getattr(self.config, "llm_provider", "openai")
        llm_model = getattr(self.config, "llm_model", None)
        
        # Initialize LLM using factory
        try:
            self.llm = create_llm(provider=llm_provider, model=llm_model)
        except Exception as e:
            print(f"WARNING: Failed to initialize {llm_provider} LLM: {str(e)}")
            print("Falling back to OpenAI GPT-3.5")
            self.llm = create_llm(provider="openai", model="gpt-3.5-turbo")
        
        # Get custom LLM parameters if provided
        self.temperature = getattr(self.config, "temperature", 0.7)
        self.max_tokens = getattr(self.config, "max_tokens", 800)
        self.batch_size = getattr(self.config, "batch_size", 5)  # Number of parallel requests
        print(f"llm_provider: {llm_provider}, llm_model: {llm_model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}, batch_size: {self.batch_size}")
        # Load agent IDs from the agent data file
        default_agent_file = os.path.join(os.path.dirname(__file__), "..", "m03_census", "census_data", "agents_with_geo.json")
        agent_data_path = getattr(self.config, "agent_data_file", default_agent_file)
        
        # If it's a relative path, make it relative to src/evaluation
        if not os.path.isabs(agent_data_path):
            evaluation_dir = Path(__file__).parent.parent.parent
            agent_data_path = os.path.join(evaluation_dir, agent_data_path)
        
        self.agent_ids = self._load_agent_ids(agent_data_path)
        
        # Track which proposal we're currently processing
        self.current_proposal_id = None

    def _load_agent_ids(self, agent_data_path: str) -> list:
        """Load only agent IDs from the agent data file.
        
        Args:
            agent_data_path: Path to the agent data JSON file.
            
        Returns:
            List of agent IDs.
        """
        try:
            if os.path.exists(agent_data_path):
                with open(agent_data_path, 'r', encoding='utf-8') as f:
                    raw_agents = json.load(f)
                return [agent.get("id", f"agent_{i:03d}") for i, agent in enumerate(raw_agents)]
            else:
                print(f"WARNING: Agent data file not found: {agent_data_path}")
                return [f"agent_{i:03d}" for i in range(14)]  # Default to 14 agents
        except Exception as e:
            print(f"ERROR: Failed to load agent IDs: {str(e)}")
            return [f"agent_{i:03d}" for i in range(14)]  # Default to 14 agents

    async def simulate_opinions(self,
                              region: str,
                              proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using parallel processing.
        
        Args:
            region: The target region name.
            proposal: A dictionary containing the rezoning proposal details.
        
        Returns:
            A dictionary with multiple participants containing opinions and reasons.
        """
        # Extract proposal ID and get scenario ID
        self.current_proposal_id = proposal.get("proposal_id", None)
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        print(f"DEBUG simulate_opinions: Processing proposal_id={self.current_proposal_id}")
        
        # Prepare proposal description
        proposal_desc = create_proposal_description(proposal)
        print(f"DEBUG: Generated proposal description: {proposal_desc[:100]}...")
        
        results = {}
        
        # Process agents in batches
        for i in range(0, len(self.agent_ids), self.batch_size):
            batch = self.agent_ids[i:i + self.batch_size]
            tasks = []
            
            # Create tasks for each agent in the batch
            for participant_id in batch:
                task = self._process_single_agent(
                    participant_id,
                    proposal,
                    proposal_desc,
                    region,
                    scenario_id
                )
                tasks.append(task)
            
            # Run batch of tasks concurrently
            batch_results = await asyncio.gather(*tasks)
            
            # Update results with batch results
            for participant_id, result in zip(batch, batch_results):
                results[participant_id] = result
        
        return results

    async def _process_single_agent(self,
                                  participant_id: str,
                                  proposal: Dict[str, Any],
                                  proposal_desc: str,
                                  region: str,
                                  scenario_id: str) -> Dict[str, Any]:
        """Process a single agent's opinion generation.
        
        Args:
            participant_id: The ID of the participant.
            proposal: The proposal dictionary.
            proposal_desc: The proposal description.
            region: The target region name.
            scenario_id: The scenario ID.
            
        Returns:
            A dictionary containing the agent's opinions and reasons.
        """
        try:
            # Load agent coordinates from agent data file
            agent_data = self._load_agent_data(participant_id)
            agent_coords = agent_data.get("coordinates", {}) if agent_data else {}
            agent_lat = agent_coords.get("lat")
            agent_lon = agent_coords.get("lng")
            
            distance_km = None
            if agent_lat is not None and agent_lon is not None:
                distance_km = calculate_distance_to_affected_area(
                    agent_lat, agent_lon, 
                    proposal.get("cells", {})
                )
            
            # Build prompt and generate response with different temperature for variety
            prompt = self._build_prompt(proposal_desc, region, agent_data, distance_km)
            temp = min(0.9, self.temperature + random.uniform(-0.2, 0.2))  # Add some randomness to temperature
            
            response = await self.llm.generate(
                prompt, 
                temperature=temp,
                max_tokens=self.max_tokens
            )
            
            # Parse the response
            rating, reason_scores = self._parse_response(response)
            print(f"DEBUG: Generated opinion for {participant_id}: rating={rating}")
            
            return {
                "opinions": {
                    scenario_id: rating
                },
                "reasons": {
                    scenario_id: reason_scores
                }
            }
            
        except Exception as e:
            print(f"ERROR: Opinion generation failed for participant {participant_id}: {str(e)}")
            return self._generate_fallback_opinion_single(scenario_id)

    def _create_proposal_description(self, proposal: Dict[str, Any]) -> str:
        """Generate a richer, geo-aware description for a rezoning proposal."""
        return create_proposal_description(proposal)

    def _build_prompt(self, proposal_desc: str, region: str, agent: Dict[str, Any] = None, distance_km: float = None) -> str:
        """Build a prompt for generating opinions on a housing policy proposal.
        
        Args:
            proposal_desc: A human-readable description of the proposal.
            region: The target region name.
            agent: The agent data containing geolocation content.
            distance_km: Distance from agent to affected area in kilometers.
            
        Returns:
            A string containing the prompt for the LLM.
        """
        context = f"""As an impartial evaluator, assess this housing policy proposal for {region}."""
        
        # Add location context if available
        if agent:
            geo_content = agent.get("geo_content", {})
            if geo_content.get("narrative"):
                context += f"\n\nYour neighborhood context:\n{geo_content['narrative']}"
        
        if distance_km is not None:
            context += f"\n Your distance from the affected area: {distance_km:.2f} kilometers"
        
        prompt = f"""{context}

Proposed Housing Policy Changes:
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

Consider:
1. Overall housing supply and affordability
2. Infrastructure and services capacity
3. Community character and quality of life
4. Economic and environmental impacts
5. Social equity and accessibility

Format your response EXACTLY as shown above, with one rating (1-10) and twelve reason scores (1-5 each).
"""
        print(f"DEBUG: Generated prompt: {prompt}")
        return prompt
    
    def _parse_response(self, response: str) -> tuple:
        """Parse the LLM response to extract rating and reason scores.
        
        Args:
            response: The response from the LLM.
            
        Returns:
            A tuple of (rating, reason_scores).
        """
        rating = 5  # Default neutral rating
        reason_scores = {
            "A": 3, "B": 3, "C": 3, "D": 3, "E": 3,
            "F": 3, "G": 3, "H": 3, "I": 3, "J": 3,
            "K": 3, "L": 3
        }  # Default neutral scores
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip()
                
                # Extract overall rating
                if line.lower().startswith("rating:"):
                    try:
                        rating_str = line.split(":", 1)[1].strip()
                        rating = int(rating_str)
                        rating = max(1, min(10, rating))  # Ensure 1-10 range
                    except:
                        pass
                
                # Extract reason scores
                elif ":" in line and line[0] in reason_scores:
                    try:
                        reason_code = line[0]
                        score_str = line.split(":", 1)[1].strip().split()[0]
                        score = int(score_str)
                        score = max(1, min(5, score))  # Ensure 1-5 range
                        reason_scores[reason_code] = score
                    except:
                        pass
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            # Keep default values if parsing fails
        
        return rating, reason_scores
    
    def _generate_fallback_opinion_single(self, scenario_id: str) -> Dict[str, Any]:
        """Generate a fallback random opinion for a single participant.
        
        Args:
            scenario_id: The ID of the scenario.
            
        Returns:
            A dictionary with random opinions and reasons.
        """
        # Generate random rating between 1 and 10
        rating = random.randint(3, 9)
        
        # Generate random scores for each reason (1-5)
        reason_scores = {}
        for code in REASON_MAPPING.values():
            reason_scores[code] = random.randint(2, 4)
        
        return {
            "opinions": {
                scenario_id: rating
            },
            "reasons": {
                scenario_id: reason_scores
            }
        }

    def _load_agent_data(self, agent_id: str) -> Dict[str, Any]:
        """Load full agent data for a specific agent ID.
        
        Args:
            agent_id: The ID of the agent to load data for.
            
        Returns:
            Dictionary containing agent data or None if not found.
        """
        try:
            default_agent_file = os.path.join(os.path.dirname(__file__), "..", "m03_census", "census_data", "agents_with_geo.json")
            agent_data_path = getattr(self.config, "agent_data_file", default_agent_file)
            
            if not os.path.isabs(agent_data_path):
                evaluation_dir = Path(__file__).parent.parent.parent
                agent_data_path = os.path.join(evaluation_dir, agent_data_path)
                
            if os.path.exists(agent_data_path):
                with open(agent_data_path, 'r', encoding='utf-8') as f:
                    agents = json.load(f)
                    for agent in agents:
                        if agent.get("id") == agent_id:
                            return agent
        except Exception as e:
            print(f"ERROR: Failed to load agent data: {str(e)}")
        return None 