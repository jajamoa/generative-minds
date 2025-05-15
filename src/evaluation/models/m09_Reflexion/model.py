import os
import random
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import asyncio
from ..base import ModelConfig
from ..m05_naive.model import NaiveBaseline
from ..m03_census.components.llm import OpenAILLM, create_llm
from ..m03_census.model import REASON_MAPPING, SCENARIO_MAPPING
from ..m03_census.utils.spatial_utils import calculate_distance_to_affected_area, create_proposal_description

class Reflexion(NaiveBaseline):
    """A Reflexion model using only proposal/grid info, no demographics."""

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

    async def simulate_opinions(self, region: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using parallel processing with reflection mechanism."""
        self.current_proposal_id = proposal.get("proposal_id", None)
        print(f"DEBUG simulate_opinions: Processing proposal_id={self.current_proposal_id}")
        
        # Get scenario ID
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        
        # Prepare proposal description
        proposal_desc = create_proposal_description(proposal)
        print(f"DEBUG: Generated proposal description: {proposal_desc[:100]}...")
        
        # Load agent data
        agent_data_path = os.path.join(os.path.dirname(__file__), "..", "m03_census", "census_data", "agents_with_geo.json")
        try:
            with open(agent_data_path, 'r', encoding='utf-8') as f:
                agent_data = json.load(f)
                agent_dict = {agent["id"]: agent for agent in agent_data}
        except Exception as e:
            print(f"WARNING: Failed to load agent data: {str(e)}")
            agent_dict = {}

        results = {}
        
        # Process agents in batches
        for i in range(0, len(self.agent_ids), self.batch_size):
            batch = self.agent_ids[i:i + self.batch_size]
            tasks = []
            
            # Create tasks for each agent in the batch
            for participant_id in batch:
                task = self._process_single_agent(
                    participant_id,
                    agent_dict.get(participant_id),
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
                                  agent: Optional[Dict[str, Any]],
                                  proposal: Dict[str, Any],
                                  proposal_desc: str,
                                  region: str,
                                  scenario_id: str) -> Dict[str, Any]:
        """Process a single agent's opinion generation with reflection."""
        try:
            # Load complete agent data including geo content
            agent_data = self._load_agent_data(participant_id)
            
            # Calculate distance if coordinates available
            distance_km = None
            if agent_data:
                coords = agent_data.get("coordinates", {})
                lat = coords.get("lat")
                lon = coords.get("lng")
                if lat is not None and lon is not None:
                    distance_km = calculate_distance_to_affected_area(
                        lat, lon,
                        proposal.get("cells", {})
                    )
                    print(f"DEBUG: Agent {participant_id} is {distance_km:.2f}km from affected area")

            # Initial evaluation with location context
            prompt = self._build_prompt(proposal_desc, region, agent=agent_data, distance_km=distance_km)
            temp = min(0.9, self.temperature + random.uniform(-0.2, 0.2))
            response_initial = await self.llm.generate(prompt, temperature=temp, max_tokens=self.max_tokens)
            rating_initial, reason_scores_initial = self._parse_response(response_initial)

            # Reflection phase
            reflection_prompt = f"""You just wrote the following evaluation:
{response_initial.strip()}

Now reflect on your reasoning. Did you miss any important impacts, contradict yourself, or overlook any downsides? Suggest specific improvements."""
            
            reflection = await self.llm.generate(reflection_prompt, temperature=0.7, max_tokens=300)

            # Revised evaluation with the same location context
            revised_prompt = f"""Original Evaluation:
{response_initial.strip()}

Reflection:
{reflection.strip()}

Now revise your evaluation to address these reflections. Use exactly the same format as the original evaluation."""
            
            response_revised = await self.llm.generate(revised_prompt, temperature=temp, max_tokens=self.max_tokens)
            rating_final, reason_scores_final = self._parse_response(response_revised)

            print(f"DEBUG: Final opinion for {participant_id}: rating={rating_final}")
            
            return {
                "opinions": {
                    scenario_id: rating_final
                },
                "reasons": {
                    scenario_id: reason_scores_final
                }
            }
            
        except Exception as e:
            print(f"ERROR: Opinion generation failed for participant {participant_id}: {str(e)}")
            return self._generate_fallback_opinion_single(scenario_id)

    def _build_prompt(self, proposal_desc: str, region: str, agent: Dict[str, Any] = None, distance_km: float = None) -> str:
        """Build a prompt for generating opinions on a housing policy proposal."""
        
        # Add location context if available
        location_context = ""
        if agent:
            geo_content = agent.get("geo_content", {})
            if geo_content.get("narrative"):
                location_context += f"\nYour neighborhood context:\n{geo_content['narrative']}"
        
        if distance_km is not None:
            location_context += f"\nYour distance from the affected area: {distance_km:.2f} kilometers"

        prompt = f"""You are an independent evaluator assigned to assess the following housing policy proposal in {region}.{location_context}

    Proposal Description:
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
        """Generate a fallback random opinion for a single participant."""
        rating = random.randint(3, 9)
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