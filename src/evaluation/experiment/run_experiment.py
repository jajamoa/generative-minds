#!/usr/bin/env python3
"""
Main experiment runner for opinion simulation experiments.
"""
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime
import yaml
import traceback

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.base import BaseModel, ModelConfig
from models.m01_basic.model import BasicSimulationModel
from models.m02_stupid.model import StupidAgentModel
from models.m03_census.model import Census
from models.m04_census_twolayer.model import CensusTwoLayer
from models.m05_naive.model import NaiveBaseline
from models.m06_transcript.model import Transcript
from models.m08_CoT.model import CoT
from models.m09_Reflexion.model import Reflexion
from experiment.eval.utils.data_utils import DataManager, create_zoning_proposal
from experiment.eval.evaluators import evaluate_experiment_dir

AVAILABLE_MODELS = {
    "basic": BasicSimulationModel,
    "stupid": StupidAgentModel,
    "census": Census,
    "twolayer": CensusTwoLayer,
    "naive": NaiveBaseline,
    "transcript": Transcript,
    "CoT": CoT,
    "Reflexion": Reflexion
}

def get_project_root() -> Path:
    """Get the absolute path of the project root directory."""
    current_file = Path(__file__).resolve()
    for parent in [current_file, *current_file.parents]:
        if (parent / 'src').exists():
            return parent
    raise RuntimeError("Could not find project root directory")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run experiment with protocol")
    parser.add_argument(
        "--protocol",
        type=str,
        required=True,
        help="Path to experiment protocol YAML file"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on existing experiment outputs"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        help="Path to existing experiment directory (for --eval-only)"
    )
    return parser.parse_args()

def load_protocol(protocol_path: str) -> dict:
    """Load experiment protocol from YAML file."""
    with open(protocol_path) as f:
        protocol = yaml.safe_load(f)
    return protocol

async def run_experiment(protocol: dict, eval_only: bool = False, experiment_dir: str = None):
    """Run experiment based on protocol."""
    # Get project root
    project_root = get_project_root()
    
    # Initialize data manager
    data_manager = DataManager(base_dir=str(project_root / "src/evaluation/experiment"))
    
    if eval_only and experiment_dir:
        # Use existing experiment directory
        exp_dir = Path(experiment_dir)
        exp_id = exp_dir.name
        print(f"\nRunning evaluation on existing experiment: {exp_id}")
    else:
        # Create new experiment directory
        exp_dir, exp_id = data_manager.create_experiment(protocol["name"], protocol["model"])
        
        # Save protocol for reproducibility
        with open(exp_dir / "protocol.yaml", "w") as f:
            yaml.dump(protocol, f, default_flow_style=False)
        
        # Initialize model with model_config if specified in protocol
        model_class = AVAILABLE_MODELS[protocol["model"]]
        model_config = protocol.get("model_config", {})
        config = ModelConfig(population=protocol["population"], **model_config)
        print(f"Initializing model with config: {config.__dict__}")
        model = model_class(config)
        
        # Run experiment
        print(f"\nRunning experiment: {exp_id}")
        print(f"Model: {protocol['model']}")
        print(f"Population size: {protocol['population']}")
        print(f"Number of proposals: {len(protocol['input']['proposals'])}")
        
        start_time = datetime.now()
        
        for i, proposal_file in enumerate(protocol["input"]["proposals"]):
            proposal_id = f"proposal_{i:03d}"
            print(f"\nProcessing {proposal_id} ({proposal_file})...")
            
            try:
                # Load proposal
                input_file = data_manager.data_dir / proposal_file
                print(f"DEBUG: Looking for proposal file at: {input_file}")
                
                if not input_file.exists():
                    print(f"ERROR: Proposal file not found: {input_file}")
                    raise FileNotFoundError(f"Proposal file not found: {input_file}")
                
                with open(input_file) as f:
                    data = json.load(f)
                    proposal = create_zoning_proposal(data)
                
                # Add proposal_id to the proposal for reference in the model
                proposal["proposal_id"] = proposal_id
                
                print(f"DEBUG: Running simulation with proposal: {proposal_id}, region: {protocol.get('region', 'san_francisco')}")
                
                # Run simulation
                result = await model.simulate_opinions(
                    region=protocol.get("region", "san_francisco"),
                    proposal=proposal
                )
                
                print(f"DEBUG: Simulation completed. Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                
                # Copy ground truth files if provided in protocol
                if "evaluation" in protocol and "ground_truth" in protocol["evaluation"]:
                    gt_file = protocol["evaluation"]["ground_truth"]
                    if gt_file:
                        gt_dest = data_manager.copy_ground_truth(gt_file, exp_dir, proposal_id)
                        if gt_dest:
                            print(f"Copied ground truth to {gt_dest}")
                        else:
                            print(f"Warning: Ground truth file not found: {gt_file}")
                
                # Save results
                print(f"DEBUG: Saving results for {proposal_id}")
                try:
                    result_paths = data_manager.save_experiment_result(
                        exp_dir=exp_dir,
                        proposal=proposal,
                        result=result,
                        proposal_id=proposal_id,
                        model_name=protocol["model"]
                    )
                    
                    # Handle different return types from save_experiment_result
                    if isinstance(result_paths, tuple) and len(result_paths) == 2:
                        input_path, output_path = result_paths
                        print(f"✓ Results saved for {proposal_id}")
                        print(f"  - Input: {input_path}")
                        print(f"  - Output: {output_path}")
                    else:
                        print(f"✓ Results saved for {proposal_id}")
                        print(f"  - Result paths: {result_paths}")
                except Exception as save_error:
                    print(f"Error saving results: {str(save_error)}")
                    print(f"DEBUG: {traceback.format_exc()}")
                
            except Exception as e:
                print(f"Error processing {proposal_id}: {str(e)}")
                print(f"DEBUG: {traceback.format_exc()}")
        
        # Save experiment metadata
        end_time = datetime.now()
        metadata = protocol.copy()
        metadata["runtime"] = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        }
        data_manager.save_metadata(exp_dir, metadata)
        
        print(f"\nExperiment completed: {exp_id}")
        print(f"Duration: {(end_time - start_time).total_seconds():.2f} seconds")
        print(f"Results saved in: {exp_dir}")
    
    # Run evaluation if specified in protocol
    if "evaluation" in protocol and "evaluators" in protocol["evaluation"]:
        run_evaluation(exp_dir, protocol)

def run_evaluation(exp_dir: Path, protocol: dict):
    """Run evaluation on experiment results using evaluator module."""
    print("\nRunning evaluation...")
    
    # Get evaluators from protocol
    evaluator_names = protocol["evaluation"]["evaluators"]
    print(f"Running evaluators: {', '.join(evaluator_names)}")
    
    try:
        # Run evaluation on experiment directory
        results = evaluate_experiment_dir(exp_dir, evaluator_names)
        
        # Save evaluation results
        eval_results_path = exp_dir / "evaluation_results.json"
        with open(eval_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation completed. Results saved to {eval_results_path}")
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print(f"DEBUG: {traceback.format_exc()}")

async def main():
    args = parse_args()
    protocol = load_protocol(args.protocol)
    await run_experiment(protocol, args.eval_only, args.experiment_dir)

if __name__ == "__main__":
    asyncio.run(main()) 