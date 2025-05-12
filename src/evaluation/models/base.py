from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

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