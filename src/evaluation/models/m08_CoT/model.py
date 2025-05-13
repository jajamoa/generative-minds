import os
import random
import json
import requests
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from ..base import BaseModel, ModelConfig
from ..m03_census.components.llm import OpenAILLM
from ..m03_census.model import REASON_MAPPING, SCENARIO_MAPPING

class CoT(BaseModel):
    """A CoT model using only proposal/grid info, no demographics."""

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
        """Generate a richer, geo-aware description for a rezoning proposal."""
        
        # Extract basic proposal information
        height_limits = proposal.get("heightLimits", {})
        default_height = height_limits.get("default", 0)
        grid_config = proposal.get("gridConfig", {})
        cell_size = grid_config.get("cellSize", 100)
        cells = proposal.get("cells", {})
        
        # Create a dictionary to group cells by their characteristics
        zone_info = defaultdict(lambda: {
            'cells': [],
            'coordinates': [],
            'count': 0
        })
        
        # Process each cell and group by category and height
        for cell_id, cell in cells.items():
            category = cell.get("category", "unknown")
            height = cell.get("heightLimit", default_height)
            
            # Create a unique identifier for this zone type
            zone_type = (category, height)  # Use tuple instead of string
            
            try:
                bbox = cell.get("bbox", {})
                if bbox:
                    lat_center = (bbox["north"] + bbox["south"]) / 2
                    lon_center = (bbox["east"] + bbox["west"]) / 2
                    
                    zone_info[zone_type]['cells'].append(cell)
                    zone_info[zone_type]['coordinates'].append((lat_center, lon_center))
                    zone_info[zone_type]['count'] += 1
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not process cell {cell_id}: {str(e)}")
                continue
        
        # Start with overview
        total_blocks = len(cells)
        desc = f"This rezoning proposal affects {total_blocks} city blocks in San Francisco"
        if total_blocks > 0:
            desc += f", each approximately {cell_size} meters square"
        desc += ". "
        
        # Describe zones by type
        zone_descriptions = []
        for (category, height), info in zone_info.items():
            try:
                num_blocks = info['count']
                if num_blocks == 0:
                    continue
                
                # Calculate approximate area
                area_sqm = num_blocks * (cell_size * cell_size)
                area_acres = area_sqm * 0.000247105  # Convert to acres
                
                # Get central coordinates for this zone
                coords = info['coordinates']
                if coords:
                    central_lat = sum(c[0] for c in coords) / len(coords)
                    central_lon = sum(c[1] for c in coords) / len(coords)
                    
                    # Try to get neighborhood name for this zone
                    neighborhood = self._get_neighborhood_name(central_lat, central_lon)
                    
                    zone_desc = (
                        f"{num_blocks} blocks ({area_acres:.1f} acres) "
                        f"zoned for {category.replace('_', ' ')} "
                        f"with a height limit of {height} feet"
                    )
                    if neighborhood:
                        zone_desc += f" in the {neighborhood} area"
                    zone_descriptions.append(zone_desc)
            except Exception as e:
                print(f"Warning: Could not process zone {category}_{height}: {str(e)}")
                continue
        
        if zone_descriptions:
            desc += "The proposal includes: " + "; ".join(zone_descriptions[:3])
            if len(zone_descriptions) > 3:
                desc += f"; and {len(zone_descriptions) - 3} additional zoning changes"
            desc += ". "
        
        # Add impact summary
        desc += (
            "This rezoning would affect local housing capacity, neighborhood character, "
            "and urban development patterns. "
            "The height limits are designed to balance housing needs with neighborhood context. "
        )
        
        # Add transit context if available
        transit_info = self._get_nearby_transit({f"{cat}_{h}": coords 
                                               for (cat, h), info in zone_info.items() 
                                               for coords in [info['coordinates']] if coords})
        if transit_info:
            desc += transit_info
        
        # Print final description for debugging
        print(f"DEBUG: Final proposal description:\n{desc.strip()}")
        return desc.strip()

    def _get_neighborhood_name(self, lat: float, lon: float) -> Optional[str]:
        """Get neighborhood name from coordinates using Google Maps API."""
        try:
            api_key = os.getenv("GOOGLE_MAPS_API_KEY")
            if not api_key:
                return None
            
            params = {
                'latlng': f'{lat},{lon}',
                'result_type': 'neighborhood',
                'key': api_key
            }
            
            response = requests.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params=params
            ).json()
            
            for result in response.get('results', []):
                for component in result.get('address_components', []):
                    if 'neighborhood' in component.get('types', []):
                        return component['long_name']
            return None
        except Exception as e:
            print(f"Warning: Failed to get neighborhood name: {str(e)}")
            return None

    def _get_nearby_transit(self, coordinates_by_zone: Dict[str, List[Tuple[float, float]]]) -> Optional[str]:
        """Get nearby transit information using Google Places API."""
        try:
            api_key = os.getenv("GOOGLE_MAPS_API_KEY")
            if not api_key:
                return None
            
            # Get central point of all coordinates
            all_coords = [coord for coords in coordinates_by_zone.values() for coord in coords]
            if not all_coords:
                return None
            
            central_lat = sum(c[0] for c in all_coords) / len(all_coords)
            central_lon = sum(c[1] for c in all_coords) / len(all_coords)
            
            params = {
                'location': f'{central_lat},{central_lon}',
                'radius': 800,  # 800 meters ~ 0.5 miles
                'type': 'transit_station',
                'key': api_key
            }
            
            response = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params
            ).json()
            
            stations = [place['name'] for place in response.get('results', [])[:3]]
            if stations:
                return f"The area is served by public transit including {', '.join(stations)}. "
            return None
        except Exception as e:
            print(f"Warning: Failed to get transit information: {str(e)}")
            return None
    
    def _build_prompt(self, proposal_desc: str, region: str) -> str:
        """Build a CoT-enabled prompt using only proposal and region (no demographics)."""

        prompt = f"""You are an independent evaluator assigned to assess the following housing policy proposal in {region}.

    Proposal Description:
    {proposal_desc}

    Imagine you're one of several residents who each bring a unique but unspecified perspective. As an evaluator, please think through the following areas step by step before forming your opinion.

    Let's break it down:

    1. **Housing Supply & Affordability**:
    - Would this proposal increase the number of available housing units?
    - Might it improve affordability for lower or middle-income residents?

    2. **Neighborhood & Community Character**:
    - Would the physical changes (e.g., building height) alter the look and feel of neighborhoods?
    - Could it affect sunlight, street life, or existing community bonds?

    3. **Infrastructure & Services**:
    - Can local infrastructure support the changes (e.g., schools, transit, utilities)?
    - Might more development strain or improve public services?

    4. **Economic, Environmental & Equity Effects**:
    - Would this spur job creation or economic activity?
    - Could it lead to environmental benefits or harms?
    - Would it promote fairer outcomes across different populations?

    5. **Transit Access, Public Amenities & Preservation**:
    - How might this affect access to public transit?
    - Would it improve or reduce access to public amenities?
    - Are there concerns around historical or cultural preservation?

    After considering all of the above, please provide:

    - An **overall rating** (1-10) of the proposal's impact on the city.
    - A set of **reason codes A-L** with individual scores (1-5) for specific impact dimensions.

    **Scoring Scales:**
    - Overall Rating (1-10):  
    1-2 = Strongly Oppose  
    3-4 = Oppose  
    5-6 = Neutral  
    7-8 = Support  
    9-10 = Strongly Support

    - Reason Scores (A-L):  
    1 = Very Negative Impact  
    2 = Somewhat Negative Impact  
    3 = Neutral  
    4 = Somewhat Positive Impact  
    5 = Very Positive Impact

    **Required Output Format (use this exactly):**
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