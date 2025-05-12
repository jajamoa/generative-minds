"""
Simplified data utilities for experiments.
Contains basic data structures and management functions.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import json
from datetime import datetime
import os
import shutil

# Simple dictionary-based data structures instead of Pydantic models
def create_zoning_proposal(data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a zoning proposal from raw data without validation"""
    return data

class DataManager:
    """Manages experiment data storage and retrieval."""
    
    def __init__(self, base_dir: str = "src/experiment"):
        """Initialize data manager with directory structure."""
        self.base_dir = Path(base_dir)
        self.eval_dir = self.base_dir / "eval"
        self.data_dir = self.eval_dir / "data"
        self.log_dir = self.base_dir / "log"
        self._init_directories()
    
    def _init_directories(self):
        """Create necessary directory structure if not exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def create_experiment(self, name: str, model_name: str) -> Tuple[Path, str]:
        """Create experiment directory with unique ID.
        
        Returns:
            Tuple[Path, str]: (experiment directory path, experiment ID)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{name}_{model_name}_{timestamp}"
        exp_dir = self.log_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir, exp_id
    
    def save_metadata(self, exp_dir: Path, metadata: Dict[str, Any]) -> None:
        """Save experiment metadata including runtime information and parameters."""
        with open(exp_dir / "experiment_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def load_ground_truth(self, gt_file: str) -> Dict[str, Any]:
        """Load ground truth data from file."""
        # Allow paths relative to data directory
        gt_path = self.data_dir / gt_file
        if not gt_path.exists():
            return {}
        
        with open(gt_path) as f:
            return json.load(f)
    
    def copy_ground_truth(self, gt_file: str, exp_dir: Path, proposal_id: str) -> Path:
        """Copy ground truth file to experiment directory.
        
        Args:
            gt_file: Path to ground truth file, relative to data directory
            exp_dir: Experiment directory to copy to
            proposal_id: Identifier for the proposal
            
        Returns:
            Path: Path to the copied ground truth file, or None if not found
        """
        gt_path = self.data_dir / gt_file
        if not gt_path.exists():
            return None
        
        # Copy to experiment directory
        gt_dest = exp_dir / f"{proposal_id}_ground_truth.json"
        shutil.copy2(gt_path, gt_dest)
        return gt_dest
    
    def save_experiment_result(self, 
                             exp_dir: Path,
                             proposal: Dict[str, Any],
                             result: Dict[str, Any],
                             proposal_id: str,
                             model_name: str) -> Tuple[Path, Path]:
        """Save experiment input and output.
        
        Returns:
            Tuple[Path, Path]: Paths to the saved input and output files
        """
        # Debug information
        print(f"DEBUG save_experiment_result: proposal_id={proposal_id}, model_name={model_name}")
        print(f"DEBUG save_experiment_result: result type={type(result)}")
        
        # Save input proposal
        input_path = exp_dir / f"{proposal_id}_input.json"
        with open(input_path, "w") as f:
            json.dump(proposal, f, indent=2)
        
        # Save output result
        output_path = exp_dir / f"{proposal_id}_output.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
            
        # Also save agent data separately for easy access
        if "comments" in result:
            agents_path = exp_dir / f"{proposal_id}_agents.json"
            with open(agents_path, "w") as f:
                json.dump(result["comments"], f, indent=2)
        
        # Always return exactly two values (input_path, output_path)        
        return input_path, output_path 