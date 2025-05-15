import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter, defaultdict
import requests
import googlemaps
from math import sqrt, radians, sin, cos, asin, pi

from ..base import BaseModel, ModelConfig
from .components.llm import OpenAILLM
from .utils.spatial_utils import (
    calculate_haversine_distance,
    calculate_distance_to_affected_area,
    get_neighborhood_name,
    get_nearby_transit,
    create_proposal_description
)


# Default grid bounds (San Francisco area)
DEFAULT_GRID_BOUNDS = {
    "north": 37.8120,
    "south": 37.7080,
    "east": -122.3549,
    "west": -122.5157
}

# Reason mapping
REASON_MAPPING = {
    "Housing supply and availability": "A",
    "Affordability for low- and middle-income residents": "B",
    "Impact on neighborhood character": "C",
    "Infrastructure and services capacity": "D",
    "Economic development and job creation": "E",
    "Environmental concerns": "F",
    "Transit and transportation access": "G",
    "Displacement of existing residents": "H",
    "Equity and social justice": "I",
    "Public space and amenities": "J",
    "Property values and investment": "K",
    "Historical preservation": "L"
}

# Scenario mapping (for translating proposal IDs to scenario IDs)
SCENARIO_MAPPING = {
    "proposal_000": "1.1",
    "proposal_001": "1.2",
    "proposal_002": "1.3",
    "proposal_003": "2.1",
    "proposal_004": "2.2",
    "proposal_005": "2.3",
    "proposal_006": "3.1",
    "proposal_007": "3.2",
    "proposal_008": "3.3"
}

class Census(BaseModel):
    """A model that generates opinions using census demographic data."""

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
        
        # Load agent data
        default_agent_file = os.path.join(os.path.dirname(__file__), "census_data", "agents_with_geo.json")
        agent_data_path = getattr(self.config, "agent_data_file", default_agent_file)
        
        # If it's a relative path, make it relative to src/evaluation
        if not os.path.isabs(agent_data_path):
            evaluation_dir = Path(__file__).parent.parent.parent
            agent_data_path = os.path.join(evaluation_dir, agent_data_path)
        
        self.agent_data = self._load_agent_data(agent_data_path)
        self.agent_ids = [agent.get("id", f"agent_{i:03d}") for i, agent in enumerate(self.agent_data)]
        
        # Track which proposal we're currently processing
        self.current_proposal_id = None

    def _load_agent_data(self, agent_data_path: str) -> list:
        """Load agent data from JSON file.
        
        Args:
            agent_data_path: Path to the agent data JSON file.
            
        Returns:
            List of agent data dictionaries.
        """
        try:
            if os.path.exists(agent_data_path):
                with open(agent_data_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                print(f"WARNING: Agent data file not found: {agent_data_path}")
                return []
        except Exception as e:
            print(f"ERROR: Failed to load agent data: {str(e)}")
            return []

    async def simulate_opinions(self, region: str, proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using OpenAI with census demographic data.
        
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
        proposal_desc = create_proposal_description(proposal)
        print(f"DEBUG: Generated proposal description: {proposal_desc[:100]}...")
        
        results = {}
        
        # Generate opinions for each agent
        for agent in self.agent_data:
            participant_id = agent.get("id")
            if not participant_id:
                continue
                
            try:
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
                
                # Generate opinion
                opinion = await self._generate_opinion(agent, proposal, proposal_desc, region)
                results[participant_id] = opinion
                
            except Exception as e:
                print(f"ERROR: Opinion generation failed for participant {participant_id}: {str(e)}")
                results[participant_id] = self._generate_fallback_opinion(scenario_id)
        
        return results

    def _create_proposal_description(self, proposal: Dict[str, Any]) -> str:
        """Generate a richer, geo-aware description for a rezoning proposal."""
        return create_proposal_description(proposal)

    def _generate_mock_results(self) -> Dict[str, Any]:
        """Generate mock results for testing/debugging purposes."""
        print("DEBUG: Generating mock results for testing")
        
        scenario_id = SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
        results = {}
        
        # Generate 5 mock agents
        for i in range(5):
            agent_id = f"mock_agent_{i:03d}"
            rating = random.randint(3, 9)
            num_reasons = random.randint(1, 3)
            reason_codes = random.sample(list(REASON_MAPPING.values()), num_reasons)
            
            results[agent_id] = {
                "opinions": {
                    scenario_id: rating
                },
                "reasons": {
                    scenario_id: reason_codes
                }
            }
        
        return results

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
    
    def _calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the Haversine distance between two points on Earth.
        
        Args:
            lat1: Latitude of first point in degrees
            lon1: Longitude of first point in degrees
            lat2: Latitude of second point in degrees
            lon2: Longitude of second point in degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        
        return c * r

    def _calculate_distance_to_affected_area(self, agent_lat: float, agent_lon: float, cells: Dict[str, Any]) -> float:
        """Calculate the minimum distance from an agent to any affected cell in the proposal.
        
        Args:
            agent_lat: Agent's latitude
            agent_lon: Agent's longitude
            cells: Dictionary of cells from the proposal
            
        Returns:
            Minimum distance in kilometers
        """
        min_distance = float('inf')
        
        for cell in cells.values():
            bbox = cell.get('bbox', {})
            if not bbox:
                continue
            
            # Calculate center of the cell
            cell_lat = (bbox['north'] + bbox['south']) / 2
            cell_lon = (bbox['east'] + bbox['west']) / 2
            
            # Calculate distance to this cell
            distance = self._calculate_haversine_distance(agent_lat, agent_lon, cell_lat, cell_lon)
            min_distance = min(min_distance, distance)
        print(f"DEBUG: Agent {agent_lat}, {agent_lon} is {min_distance:.2f}km from affected area")
        return min_distance

    async def _generate_opinion(self, 
                              agent: Dict[str, Any], 
                              proposal: Dict[str, Any],
                              proposal_desc: str,
                              region: str) -> Dict[str, Any]:
        """Generate opinion using agent demographics and location."""
        try:
            # Calculate distance to affected area
            agent_coords = agent.get("coordinates", {})
            agent_lat = agent_coords.get("lat")
            agent_lon = agent_coords.get("lng")
            
            distance_km = None
            if agent_lat is not None and agent_lon is not None:
                distance_km = self._calculate_distance_to_affected_area(
                    agent_lat, agent_lon, 
                    proposal.get("cells", {})
                )
                print(f"DEBUG: Agent {agent.get('id')} is {distance_km:.2f}km from affected area")
            
            # Build prompt with distance information
            prompt = self._build_opinion_prompt(agent, proposal_desc, region, distance_km)
            
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
        except Exception as e:
            print(f"ERROR: Opinion generation failed: {str(e)}")
            return self._generate_fallback_opinion(SCENARIO_MAPPING.get(self.current_proposal_id, "1.1"))

    def _build_opinion_prompt(self, 
                             agent: Dict[str, Any], 
                             proposal_desc: str,
                             region: str,
                             distance_km: Optional[float] = None) -> str:
        """Build a prompt for generating opinions."""
        # Extract agent demographics
        agent_info = agent.get("agent", {})
        geo_content = agent.get("geo_content", {})
        
        # Build demographic context
        context = f"""As a resident of {region} with the following characteristics:
- Age: {agent_info.get('age')}
- Housing: {agent_info.get('householder type')}
- Transportation: {agent_info.get('means of transportation')}
- Income: {agent_info.get('income')}
- Occupation: {agent_info.get('occupation')}
- Marital Status: {agent_info.get('marital status')}
- Has Children Under 18: {agent_info.get('has children under 18')}
- Neighborhood: {geo_content.get('neighborhood', 'Unknown')}
"""

        # Add distance information if available
        if distance_km is not None:
            context += f"- Distance from affected area: {distance_km:.2f} kilometers\n"
        
        # Add neighborhood context
        if geo_content.get("narrative"):
            context += f"\nYour neighborhood context:\n{geo_content['narrative']}\n"

        # Build the complete prompt
        prompt = f"""{context}

Please evaluate this housing policy proposal:

{proposal_desc}

Consider how this proposal might affect you, your neighborhood, and the broader community across multiple dimensions. Rate EACH of the following aspects on a scale of 1-5, where:
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

Consider your personal circumstances, neighborhood context, and distance from the affected area when evaluating each aspect.
Format your response EXACTLY as shown above, with one rating (1-10) and twelve reason scores (1-5 each).
"""
        return prompt
    
    def _parse_opinion_response(self, response: str) -> Tuple[int, Dict[str, int]]:
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
    
    def _generate_fallback_opinion(self, scenario_id: str) -> Dict[str, Any]:
        """Generate a fallback random opinion and reasons for a scenario.
        
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
