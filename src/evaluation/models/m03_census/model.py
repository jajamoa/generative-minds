import json
import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter, defaultdict
import requests
import googlemaps
from math import sqrt

from ..base import BaseModel, ModelConfig
from .components.llm import OpenAILLM


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
    """A model that generates opinions using OpenAI API and agent data from a JSON file."""
    
    def __init__(self, config: ModelConfig = None):
        """Initialize model components and set the path to the agent data JSON file.
        
        Args:
            config: Model configuration containing settings such as population and agent_data_file.
        """
        super().__init__(config)
        self.llm = OpenAILLM()
        
        # Get custom OpenAI parameters if provided
        self.temperature = getattr(self.config, "temperature", 0.7)
        self.max_tokens = getattr(self.config, "max_tokens", 800)
        
        # Load the agent file
        default_agent_file = os.path.join(os.path.dirname(__file__), "census_data", "agents_with_geo.json")
        agent_data_path = getattr(self.config, "agent_data_file", default_agent_file)
        
        # If it's a relative path, make it relative to src/evaluation
        if not os.path.isabs(agent_data_path):
            # Get the evaluation directory (two levels up from current file)
            evaluation_dir = Path(__file__).parent.parent.parent
            agent_data_path = os.path.join(evaluation_dir, agent_data_path)
        
        self.agent_data_file = agent_data_path
        print(f"DEBUG Census.__init__: agent_data_file={self.agent_data_file}")
        
        # Track which proposal we're currently processing (for scenario ID mapping)
        self.current_proposal_id = None
    
    async def simulate_opinions(self,
                               region: str,
                               proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate opinions using OpenAI based on agent information from a JSON file.
        
        Args:
            region: The target region name.
            proposal: A dictionary containing the rezoning proposal details.
        
        Returns:
            A dictionary with participant IDs as keys, each containing opinions and reasons.
            Output format:
            {
                "<participant_id>": {
                    "opinions": {
                        "<scenario_id>": <rating_1_to_10>,
                        ...
                    },
                    "reasons": {
                        "<scenario_id>": ["A", "C", "D"],
                        ...
                    }
                },
                ...
            }
        """
        # Extract proposal ID from metadata if available
        self.current_proposal_id = proposal.get("proposal_id", None)
        print(f"DEBUG simulate_opinions: Processing proposal_id={self.current_proposal_id}")
        
        # Extract grid bounds and height limits info
        grid_bounds = proposal.get("gridConfig", {}).get("bounds", DEFAULT_GRID_BOUNDS)
        height_limits = proposal.get("heightLimits", {})
        
        # Prepare readable description of the proposal
        proposal_desc = self._create_proposal_description(proposal)
        print(f"DEBUG: Generated proposal description: {proposal_desc[:100]}...")
        
        # Verify agent_data_file exists
        if not os.path.exists(self.agent_data_file):
            print(f"ERROR: Agent data file not found: {self.agent_data_file}")
            # Generate mock data for testing/debugging
            return self._generate_mock_results()
        
        # Load agents from JSON file
        print(f"DEBUG: Loading agents from: {self.agent_data_file}")
        try:
            with open(self.agent_data_file, 'r', encoding='utf-8') as f:
                raw_agents = json.load(f)
            
            print(f"DEBUG: Loaded {len(raw_agents)} agents")
        except Exception as e:
            print(f"ERROR: Failed to load agents: {str(e)}")
            # Generate mock data for testing/debugging
            return self._generate_mock_results()
        
        results = {}
        
        # Process each agent (limit to 3 for testing if needed)
        # raw_agents = raw_agents[:3]  # Uncomment to process only 3 agents for testing
        
        for i, raw_agent in enumerate(raw_agents):
            participant_id = raw_agent.get("id")
            if not participant_id:
                participant_id = f"agent_{i:03d}"
                
            print(f"DEBUG: Processing agent {i+1}/{len(raw_agents)}: {participant_id}")
            
            # Generate opinion and reasons for this proposal
            try:
                opinion_data = await self._generate_opinion(
                    raw_agent, 
                    proposal,
                    proposal_desc,
                    region
                )
                results[participant_id] = opinion_data
            except Exception as e:
                print(f"ERROR: Failed to generate opinion for agent {participant_id}: {str(e)}")
                # Generate fallback data for this agent
                results[participant_id] = self._generate_fallback_opinion(
                    SCENARIO_MAPPING.get(self.current_proposal_id, "1.1")
                )
        
        print(f"DEBUG: Completed processing {len(results)} agents")
        return results
    
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
    
    async def _generate_opinion(self, 
                              agent: Dict[str, Any], 
                              proposal: Dict[str, Any],
                              proposal_desc: str,
                              region: str) -> Dict[str, Any]:
        """Generate opinion and reasons for a proposal for a specific agent.
        
        Args:
            agent: A dictionary containing agent demographic data.
            proposal: A dictionary containing the rezoning proposal details.
            proposal_desc: A human-readable description of the proposal.
            region: The target region name.
            
        Returns:
            A dictionary with opinions and reasons.
        """
        # Get scenario ID from proposal ID or use a default
        scenario_id = "1.1"  # Default scenario ID
        if self.current_proposal_id and self.current_proposal_id in SCENARIO_MAPPING:
            scenario_id = SCENARIO_MAPPING[self.current_proposal_id]
        
        print(f"DEBUG: Generating opinion for scenario_id={scenario_id}")
        
        # Build prompt based on proposal and agent details
        prompt = self._build_opinion_prompt(agent, proposal_desc, region)
        print(f"DEBUG: Prompt length: {len(prompt)} characters")
        
        # Generate response from LLM
        try:
            response = await self.llm.generate(
                prompt, 
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            print(f"DEBUG: Received response of length {len(response)} characters")
        except Exception as e:
            print(f"ERROR: LLM generation failed: {str(e)}")
            return self._generate_fallback_opinion(scenario_id)
        
        try:
            # Parse the response to extract rating and reason scores
            rating, reason_scores = self._parse_opinion_response(response)
            print(f"DEBUG: Extracted rating={rating}, reason_scores={reason_scores}")
            
            # Format into the expected output structure
            return {
                "opinions": {
                    scenario_id: rating
                },
                "reasons": {
                    scenario_id: reason_scores
                }
            }
        except Exception as e:
            print(f"ERROR: Failed to parse response: {str(e)}")
            return self._generate_fallback_opinion(scenario_id)
    
    def _build_opinion_prompt(self, 
                             agent: Dict[str, Any], 
                             proposal_desc: str,
                             region: str) -> str:
        """Build a prompt for generating opinions on a housing policy proposal.
        
        Args:
            agent: A dictionary containing agent demographic data.
            proposal_desc: A human-readable description of the proposal.
            region: The target region name.
            
        Returns:
            A string containing the prompt for the LLM.
        """
        # Handle possible different agent data formats
        agent_data = {}
        if "agent" in agent and isinstance(agent["agent"], dict):
            agent_data = agent["agent"]
        elif isinstance(agent, dict):
            agent_data = agent
        
        geo = agent.get("geo_content", {})
        geo_narrative = geo.get("narrative", "")
        neighborhood = geo.get("neighborhood", "unknown")
        
        # Extract more detailed demographic information
        housing_status = agent_data.get("householder type", "unknown")
        rent_burden = agent_data.get("Gross rent", "unknown")
        mobility = agent_data.get("Geo Mobility", "unknown")
        transportation = agent_data.get("means of transportation", "unknown")
        marital_status = agent_data.get("marital status", "unknown")
        has_children = agent_data.get("has children under 18", False)
        children_age = agent_data.get("children age range", "No Children")
        
        prompt = f"""As a resident of {region} living in the {neighborhood} neighborhood, evaluate this housing policy proposal based on your personal circumstances and local context.

Your Personal Profile:
- Age: {agent_data.get('age', 'unknown')} years old
- Income: {agent_data.get('income', 'unknown')}
- Occupation: {agent_data.get('occupation', 'unknown')}
- Housing: {housing_status} with {rent_burden} of income spent on housing
- Mobility History: {mobility}
- Transportation: {transportation}
- Family: {marital_status}, {children_age if has_children else 'no children'}

Your Neighborhood Context:
{geo_narrative}

Proposed Housing Policy Changes:
{proposal_desc}

Consider how this proposal might affect you and your community across multiple dimensions. Rate EACH of the following aspects on a scale of 1-5, where:
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
1. Your personal housing situation and needs
2. Your daily transportation and commute patterns
3. Your neighborhood's character and amenities
4. Local infrastructure and services
5. Economic impacts on you and your community
6. Environmental and quality of life effects

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
