# src/simulation/simulate.py
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, List
import random

class SimulationEngine:
    def __init__(self):
        current_dir = Path(__file__).parent.parent
        data_path = current_dir / "data/reasons.json"
        self.reason_dictionary = json.load(open(data_path))
        
    def simulate(self,
                region: str,
                population: int,
                proposal: Dict[str, Any],
                demographics: Dict[str, Any],
                num_samples: int = 30) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Simulate opinions for a proposal
        
        Args:
            region: Target region
            population: Total population size
            proposal: Proposal details
            demographics: Demographic information
            num_samples: Number of sample agents to generate
            
        Returns:
            Tuple of (opinion distribution, sample agents)
        """
        # generate distribution of opinions
        opinion_distribution = self._simulate_opinion_distribution(population, proposal, demographics)
        
        # generate sample agents (deterministic based on demographics)
        sample_agents = self._generate_sample_agents(demographics, N=num_samples)
        
        # generate sample comments for each agent
        sample_agents = self._generate_sample_comments(sample_agents, proposal)
        
        return opinion_distribution, sample_agents
        
    def _generate_sample_agents(self, demographics, N=30):
        # generate sample agents based on demographics
        # attributes include age, income, education, occupation, gender, religion, and race.
        def sample_attribute(attr_name, demographics):
            # sample an attribute based on demographics
            dist_data = demographics[attr_name + "_distribution"]
            options = list(dist_data.keys())
            p = list(dist_data.values())
            p = np.array(p) / np.sum(p)
            
            if attr_name == "age":
                # For age, generate a specific number within the selected range
                age_range = np.random.choice(options, p=p)
                start_age, end_age = map(int, age_range.split("-"))
                return random.randint(start_age, end_age)
            else:
                return np.random.choice(options, p=p)
        
        attributes = ["age", "income", "education", "occupation", "gender", "religion", "race"]
        sample_agents = {}
        for i in range(N):
            agent = {}
            for attr in attributes:
                agent[attr] = sample_attribute(attr, demographics)
            sample_agents[i] = {"id": i, "agent": agent}
        return sample_agents
    
    
    def _generate_sample_comments(self, sample_agents, proposal):
        def generate_opinion_and_comment(attributes, proposal):
            # generate a comment based on attributes and proposal
            opinion = "support"  # or "oppose"
            comment = "hii"
            return opinion, comment
        
        # generate sample comments for each agent
        for agent_id, agent in sample_agents.items():
            attributes = agent["agent"]
            opinion, comment = generate_opinion_and_comment(attributes, proposal)
            sample_agents[agent_id]["opinion"] = opinion
            sample_agents[agent_id]["comment"] = comment
            return sample_agents
            
            
    def _simulate_opinion_distribution(self, population, proposal, demographics):
        # generate distribution among supporters, opponents, and neutrals
        # Example: 60% supporters, 30% opponents, 10% neutrals
        dist = self._generate_distribution(proposal, demographics)
        dist_supporter = self._generate_supporter_distribution(proposal, demographics)
        dist_opponent = self._generate_opponent_distribution(proposal, demographics)
        
        # write the result in a dictionary and return it
        result = {}
        num_supporters = int(population * dist["support"])
        num_opponents = int(population * dist["oppose"])
        num_neutral = population - (num_supporters + num_opponents)
        result["summary_statistics"] = {
            "support": num_supporters,
            "oppose": num_opponents,
            "neutral": num_neutral
        }
        result["supporter_reasons"] = dist_supporter
        result["opponent_reasons"] = dist_opponent
        return result
    
    def _generate_distribution(self, proposal, demographics):
        # generate distribution among supporters, opponents, and neutrals
        # Example: 60% supporters, 30% opponents, 10% neutrals
        dist = {"support": 0.6, "oppose": 0.3, "neutral": 0.1}
        assert abs(sum(dist.values()) - 1) < 1e-6, "Distribution must sum to 1"
        return dist
    
    def _generate_supporter_distribution(self, proposal, demographics):
        # generate distribution of reasons among supporters
        reasons = self.reason_dictionary["support_reasons"]
        dist = {reason: 1 / len(reasons) for reason in reasons}
        return dist
        
    def _generate_opponent_distribution(self, proposal, demographics):
        # generate distribution of reasons among opponents
        reasons = self.reason_dictionary["oppose_reasons"]
        dist = {reason: 1 / len(reasons) for reason in reasons}
        return dist