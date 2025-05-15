from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import asyncio

class ModelConfig:
    """Configuration for a simulation model."""
    
    def __init__(self, population: int = 100, **kwargs):
        """Initialize model configuration.
        
        Args:
            population: Number of agents to simulate
            **kwargs: Additional model-specific configuration parameters
        """
        self.population = population
        
        # Store all additional configuration parameters
        print(f"DEBUG ModelConfig: Initializing with population={population} and {len(kwargs)} additional parameters")
        for key, value in kwargs.items():
            print(f"DEBUG ModelConfig: Setting {key}={value}")
            setattr(self, key, value)

class BaseModel(ABC):
    """Base interface for all opinion simulation models"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize the model with configuration
        
        Args:
            config: Model configuration. If None, uses default configuration.
        """
        self.config = config or ModelConfig()
    
    @abstractmethod
    async def simulate_opinions(self, 
                              region: str,
                              proposal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate opinions for a given proposal in a region
        
        Args:
            region: Target region name
            proposal: Proposal details including title and description
            
        Returns:
            Opinion distribution summary
        """
        pass 
        
    async def simulate_opinions_batch(self,
                                     region: str,
                                     proposals: List[Dict[str, Any]],
                                     concurrency_limit: int = 4) -> Dict[str, Dict[str, Any]]:
        """
        Simulate opinions for multiple proposals in parallel
        
        Args:
            region: Target region name
            proposals: List of proposal details
            concurrency_limit: Maximum number of proposals to process concurrently
            
        Returns:
            Dictionary mapping proposal_ids to opinion results
        """
        results = {}
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency_limit)
        
        async def process_with_semaphore(proposal):
            async with semaphore:
                proposal_id = proposal.get("proposal_id", "unknown")
                try:
                    result = await self.simulate_opinions(region, proposal)
                    return proposal_id, result
                except Exception as e:
                    print(f"Error in batch processing proposal {proposal_id}: {str(e)}")
                    return proposal_id, {"error": str(e)}
        
        # Create tasks for each proposal
        tasks = [process_with_semaphore(proposal) for proposal in proposals]
        
        # Run all tasks concurrently and collect results
        for proposal_id, result in await asyncio.gather(*tasks):
            results[proposal_id] = result
            
        return results 