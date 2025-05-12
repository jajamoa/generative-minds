import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List

from ..base import BaseModel, ModelConfig
from .components.llm import OpenAILLM
from ..m03_census.model import Census
from .prompts import get_prompt_first_layer, get_prompt_second_layer

# Default grid bounds (San Francisco area)
DEFAULT_GRID_BOUNDS = {
    "north": 37.8120,
    "south": 37.7080,
    "east": -122.3549,
    "west": -122.5157
}

class CensusTwoLayer(Census):
    def __init__(self, config: ModelConfig = None):
        super().__init__(config)


    async def _generate_opinion_and_comment(self, agent: Dict[str, Any], proposal: Dict[str, Any]) -> Tuple[str, str, List[str]]:
        """Generate opinion and comment for an agent using OpenAI.
        
        This function builds a prompt using agent and proposal details,
        then calls the LLM to generate an opinion (support/oppose/neutral),
        a brief comment, and key themes.
        
        Returns:
            A tuple of (opinion, comment, themes).
        """
        # get prompts for gerating intermediate thoughts
        prompts_first_layer = get_prompt_first_layer(agent, proposal)

        # generate intermediate thoughts
        intermediate_thoughts = {}
        for dependency, prompt in prompts_first_layer.items():
            intermediate_thoughts[dependency] = await self.llm.generate(prompt)

        prompts_second_layer = get_prompt_second_layer(intermediate_thoughts)
        response= await self.llm.generate(prompts_second_layer)
        
        try:
            parts = response.strip().split("|")
            if len(parts) >= 3:
                opinion, comment, themes = parts[:3]
                themes = [theme.strip() for theme in themes.split(",")]
            else:
                opinion, comment = parts[:2]
                themes = []
            opinion = opinion.strip().lower()
            if opinion not in {"support", "oppose", "neutral"}:
                opinion = random.choice(["support", "oppose", "neutral"])
            return opinion, comment.strip(), themes
        except Exception as e:
            opinion = random.choice(["support", "oppose", "neutral"])
            comment = f"Error processing response: {str(e)}"
            return opinion, comment, []
