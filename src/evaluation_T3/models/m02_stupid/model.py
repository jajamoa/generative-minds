import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..base import BaseModel, ModelConfig
from .components.llm import OpenAILLM
from .components.agent_generator import AgentGenerator

# Default grid bounds (San Francisco area)
DEFAULT_GRID_BOUNDS = {
    "north": 37.8120,
    "south": 37.7080,
    "east": -122.3549,
    "west": -122.5157
}

class StupidAgentModel(BaseModel):
    """A simple model using OpenAI API and random coordinates"""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize model components"""
        super().__init__(config)
        self.llm = OpenAILLM()
        self.agent_generator = AgentGenerator()
    
    async def simulate_opinions(self,
                              region: str,
                              proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using OpenAI and random coordinates"""
        # Get grid bounds from proposal or use defaults
        grid_bounds = (proposal.get("grid_config", {})
                      .get("bounds", DEFAULT_GRID_BOUNDS))
        
        # Generate agents with random coordinates
        raw_agents = self.agent_generator.generate_agents(
            num_agents=self.config.population,
            grid_bounds=grid_bounds
        )
        
        # Generate opinions and comments using OpenAI
        agents = []
        opinion_counts = {"support": 0, "oppose": 0, "neutral": 0}
        key_themes = {
            "support": set(),
            "oppose": set()
        }
        
        for i, raw_agent in enumerate(raw_agents):
            opinion, comment, themes = await self._generate_opinion_and_comment(raw_agent, proposal)
            opinion_counts[opinion] += 1
            
            # Find nearest cell
            agent_lat = raw_agent['coordinates']['lat']
            agent_lng = raw_agent['coordinates']['lng']
            nearest_cell = None
            min_distance = float('inf')
            nearest_cell_id = None
            
            for cell_id, cell in proposal['cells'].items():
                bbox = cell['bbox']
                cell_lat = (bbox['north'] + bbox['south']) / 2
                cell_lng = (bbox['east'] + bbox['west']) / 2
                
                distance = ((agent_lat - cell_lat) ** 2 + (agent_lng - cell_lng) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    nearest_cell = cell
                    nearest_cell_id = cell_id
            
            # Convert agent format to match ground truth
            agent = {
                "id": i + 1,
                "agent": {
                    "age": self._convert_age(raw_agent["agent"]["age"]),
                    "income_level": self._convert_income(raw_agent["agent"]["income"]),
                    "education_level": self._convert_education(raw_agent["agent"]["education"]),
                    "occupation": self._convert_occupation(raw_agent["agent"]["occupation"]),
                    "gender": raw_agent["agent"]["gender"]
                },
                "location": {
                    "lat": raw_agent["coordinates"]["lat"],
                    "lng": raw_agent["coordinates"]["lng"]
                },
                "cell_id": nearest_cell_id,
                "opinion": opinion,
                "comment": comment
            }
            agents.append(agent)
            
            # Collect themes
            if themes:
                key_themes[opinion].update(themes)
        
        # Return results with raw counts
        return {
            "summary": opinion_counts,
            "comments": agents,
            "key_themes": {
                "support": list(key_themes["support"]),
                "oppose": list(key_themes["oppose"])
            }
        }
    
    def _convert_age(self, age: int) -> int:
        """Convert age format"""
        # age is already an integer from agent_generator
        return age
    
    def _convert_income(self, income: str) -> str:
        """Convert income format"""
        income_map = {
            "0-25000": "low_income",
            "25000-50000": "low_income",
            "50000-75000": "middle_income",
            "75000-100000": "middle_income",
            "100000+": "high_income"
        }
        return income_map.get(income, "middle_income")
    
    def _convert_education(self, education: str) -> str:
        """Convert education format"""
        education_map = {
            "high school": "high_school",
            "some college": "some_college",
            "bachelor's": "bachelor",
            "master's": "postgraduate",
            "doctorate": "postgraduate"
        }
        return education_map.get(education, "bachelor")
    
    def _convert_occupation(self, occupation: str) -> str:
        """Convert occupation format"""
        occupation_map = {
            "student": "student",
            "professional": "white_collar",
            "service": "service",
            "retired": "retired",
            "other": "other"
        }
        return occupation_map.get(occupation, "white_collar")
    
    async def _generate_opinion_and_comment(self, agent: Dict[str, Any], proposal: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Generate opinion and comment for an agent using OpenAI"""
        # 找到最近的 cell
        agent_lat = agent['coordinates']['lat']
        agent_lng = agent['coordinates']['lng']
        nearest_cell = None
        min_distance = float('inf')
        
        for cell_id, cell in proposal['cells'].items():
            bbox = cell['bbox']
            cell_lat = (bbox['north'] + bbox['south']) / 2
            cell_lng = (bbox['east'] + bbox['west']) / 2
            
            distance = ((agent_lat - cell_lat) ** 2 + (agent_lng - cell_lng) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest_cell = cell

        prompt = f"""Given a rezoning proposal and a resident's information, generate their opinion and a brief comment.

Proposal Details:
- Nearest Rezoning Area: A {nearest_cell['category']} zone with height limit changed to {nearest_cell['height_limit']} feet
- Distance from Resident: {min_distance:.4f} degrees (approximately {min_distance * 111:.1f} km)
- Default Height Limit: {proposal['height_limits']['default']} feet

Resident Information:
- Location: ({agent['coordinates']['lat']}, {agent['coordinates']['lng']})
- Age: {agent['agent']['age']}
- Income: {agent['agent']['income']}
- Education: {agent['agent']['education']}
- Occupation: {agent['agent']['occupation']}
- Gender: {agent['agent']['gender']}

Consider how the height limit change and distance from the rezoning area might affect the resident's daily life, property value, and community character.

Generate:
1. Opinion (support/oppose/neutral)
2. A brief comment explaining their stance (1-2 sentences)
3. Key themes in the comment (2-3 keywords)

Format: opinion|comment|theme1,theme2,theme3"""

        response = await self.llm.generate(prompt)
        try:
            parts = response.strip().split("|")
            if len(parts) >= 3:
                opinion, comment, themes = parts[:3]
                themes = [theme.strip() for theme in themes.split(",")]
            else:
                opinion, comment = parts[:2]
                themes = []
            
            opinion = opinion.strip().lower()
            
            # Validate opinion
            if opinion not in {"support", "oppose", "neutral"}:
                opinion = random.choice(["support", "oppose", "neutral"])
            
            return opinion, comment.strip(), themes
        except Exception as e:
            # Fallback to random opinion if LLM response is invalid
            opinion = random.choice(["support", "oppose", "neutral"])
            comment = f"Error processing response: {str(e)}"
            return opinion, comment, [] 