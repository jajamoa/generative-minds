import os
import random
import json
from pathlib import Path
from typing import Dict, Any

from ..base import BaseModel, ModelConfig
from ..m03_census.components.llm import OpenAILLM
from ..m03_census.model import REASON_MAPPING, SCENARIO_MAPPING

class NaiveBaseline(BaseModel):
    """A naive baseline using only proposal/grid info, no demographics."""

    def __init__(self, config: ModelConfig = None):
        """Initialize model components.
        
        Args:
            config: Model configuration containing settings.
        """
        super().__init__(config)
        self.llm = OpenAILLM()
        
        # Get custom OpenAI parameters if provided
        self.temperature = getattr(self.config, "temperature", 0.7)
        self.max_tokens = getattr(self.config, "max_tokens", 800)
        
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
        """Simulate opinions using OpenAI based only on proposal information.
        
        Args:
            region: The target region name.
            proposal: A dictionary containing the rezoning proposal details.
        
        Returns:
            A dictionary with multiple participants containing opinions and reasons.
        """
        # Extract proposal ID from metadata if available
        self.current_proposal_id = proposal.get("proposal_id", None)
        print(f"DEBUG simulate_opinions: Processing proposal_id={self.current_proposal_id}")
        
        # Get scenario ID
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        
        # Prepare readable description of the proposal
        proposal_desc = self._create_proposal_description(proposal)
        print(f"DEBUG: Generated proposal description: {proposal_desc[:100]}...")
        
        results = {}
        
        # Generate opinions for each agent ID
        for participant_id in self.agent_ids:
            try:
                # Build prompt and generate response with different temperature for variety
                prompt = self._build_prompt(proposal_desc, region)
                temp = min(0.9, self.temperature + random.uniform(-0.2, 0.2))  # Add some randomness to temperature
                
                response = await self.llm.generate(
                    prompt, 
                    temperature=temp,
                    max_tokens=self.max_tokens
                )
                
                # Parse the response
                rating, reason_scores = self._parse_response(response)
                print(f"DEBUG: Generated opinion for {participant_id}: rating={rating}")
                
                results[participant_id] = {
                    "opinions": {
                        scenario_id: rating
                    },
                    "reasons": {
                        scenario_id: reason_scores
                    }
                }
                
            except Exception as e:
                print(f"ERROR: Opinion generation failed for participant {participant_id}: {str(e)}")
                # Generate fallback opinion for this participant
                results[participant_id] = self._generate_fallback_opinion_single(scenario_id)
        
        return results
    
    def _create_proposal_description(self, proposal: Dict[str, Any]) -> str:
        """Create a human-readable description of a rezoning proposal.
        
        Args:
            proposal: A dictionary containing proposal details.
            
        Returns:
            A string describing the key elements of the proposal.
        """
        # Extract basic information
        height_limits = proposal.get("heightLimits", {})
        default_height = height_limits.get("default", "varies")
        grid_config = proposal.get("gridConfig", {})
        cell_size = grid_config.get("cellSize", 100)
        
        # Count zones by category
        cells = proposal.get("cells", {})
        zone_counts = {}
        
        try:
            for cell_id, cell in cells.items():
                category = cell.get("category", "unknown")
                height = cell.get("heightLimit", default_height)
                
                key = f"{category}_{height}"
                zone_counts[key] = zone_counts.get(key, 0) + 1
        except Exception as e:
            print(f"ERROR: Failed to count zones: {str(e)}")
            return f"Rezoning proposal with {cell_size}m cells. Default height limit: {default_height} feet."
        
        # Create description
        desc = f"Rezoning proposal with {cell_size}m cells and {len(cells)} modified zones. "
        desc += f"Default height limit: {default_height} feet. "
        
        # Add zone category information if available
        if zone_counts:
            desc += "Zones include: "
            zone_info = []
            for key, count in zone_counts.items():
                try:
                    category, height = key.split("_")
                    zone_info.append(f"{count} {category} zones (height: {height}ft)")
                except:
                    zone_info.append(f"{count} zones of type {key}")
            desc += ", ".join(zone_info[:3])  # Limit to 3 zone types to keep it concise
            if len(zone_info) > 3:
                desc += f", and {len(zone_info) - 3} more zone types"
        
        return desc
    
    def _build_prompt(self, proposal_desc: str, region: str) -> str:
        """Build a prompt for generating opinions on a housing policy proposal.
        
        Args:
            proposal_desc: A human-readable description of the proposal.
            region: The target region name.
            
        Returns:
            A string containing the prompt for the LLM.
        """
        prompt = f"""As an impartial evaluator, assess this housing policy proposal for {region}.

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