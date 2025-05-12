import random
from typing import Dict, Any, List

class AgentGenerator:
    """Generate agents with random coordinates and demographics"""
    
    def __init__(self):
        """Initialize demographic options"""
        self.demographic_options = {
            "income": ["0-25000", "25000-50000", "50000-75000", "75000-100000", "100000+"],
            "education": ["high school", "some college", "bachelor's", "master's", "doctorate"],
            "occupation": ["student", "professional", "service", "retired", "other"],
            "gender": ["male", "female", "other"],
            "religion": ["christian", "jewish", "muslim", "buddhist", "hindu", "none", "other"],
            "race": ["white", "black", "asian", "hispanic", "other"]
        }
        
        # Age ranges for weighted random generation
        self.age_ranges = [
            (18, 24, 0.15),  # 15% probability
            (25, 34, 0.25),  # 25% probability
            (35, 44, 0.20),  # 20% probability
            (45, 54, 0.15),  # 15% probability
            (55, 64, 0.15),  # 15% probability
            (65, 85, 0.10)   # 10% probability
        ]
    
    def _generate_random_age(self) -> int:
        """
        Generate a random age based on demographic distribution
        Returns an integer age between 18 and 85
        """
        # Choose an age range based on weights
        ranges, weights = zip(*[(r[:2], r[2]) for r in self.age_ranges])
        selected_range = random.choices(ranges, weights=weights)[0]
        
        # Generate a random age within the selected range
        return random.randint(selected_range[0], selected_range[1])

    def generate_agents(self, num_agents: int, grid_bounds: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Generate agents with random coordinates and demographics
        
        Args:
            num_agents: Number of agents to generate
            grid_bounds: Grid boundaries (north, south, east, west)
            
        Returns:
            List of agent dictionaries
        """
        agents = []
        for i in range(num_agents):
            # Generate random coordinates within bounds
            lat = random.uniform(grid_bounds["south"], grid_bounds["north"])
            lng = random.uniform(grid_bounds["west"], grid_bounds["east"])
            
            # Generate random demographics
            demographics = {
                attr: random.choice(options)
                for attr, options in self.demographic_options.items()
            }
            
            # Generate specific age
            demographics["age"] = self._generate_random_age()
            
            agents.append({
                "id": i,
                "coordinates": {
                    "lat": round(lat, 6),
                    "lng": round(lng, 6)
                },
                "agent": demographics
            })
        
        return agents 