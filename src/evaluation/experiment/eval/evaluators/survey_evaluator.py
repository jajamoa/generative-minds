"""
Survey opinion evaluator module.

This module provides evaluation metrics for comparing agent opinion predictions 
with ground truth from survey data. It supports evaluating both opinion scores 
and reason selections using various metrics including MAE and Wasserstein distance.
"""
import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Type, Set
import numpy as np
from scipy.stats import wasserstein_distance

class Evaluator:
    """Base evaluator interface"""
    
    def __init__(self, name: str):
        self.name = name
    
    def evaluate(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate predictions against ground truth"""
        raise NotImplementedError()

from scipy.stats import wasserstein_distance
import numpy as np
from typing import Dict, Any

class OpinionScoreEvaluator(Evaluator):
    """Evaluates opinion score accuracy using both MAE and 1 - Wasserstein distance"""
    
    def __init__(self):
        super().__init__("opinion_score")
    
    def evaluate(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate opinion score accuracy"""
        results = {
            "mean_absolute_error": 0.0,
            "wasserstein_similarity": 0.0,  # Changed from 'distance' to 'similarity'
            "region_errors": {},
            "region_wasserstein_similarity": {},
            "correlation": 0.0
        }
        
        all_gt_scores = []
        all_pred_scores = []
        errors = []
        region_errors = {}
        region_scores = {}  # For Wasserstein similarity calculation
        
        for user_id, gt_user in ground_truth.items():
            if user_id in predicted:
                pred_user = predicted[user_id]
                
                for region_id, gt_score in gt_user.get("opinions", {}).items():
                    pred_score = pred_user.get("opinions", {}).get(region_id)
                    
                    if pred_score is not None:
                        all_gt_scores.append(gt_score)
                        all_pred_scores.append(pred_score)
                        
                        error = abs(gt_score - pred_score)
                        errors.append(error)
                        
                        if region_id not in region_errors:
                            region_errors[region_id] = []
                            region_scores[region_id] = {"gt": [], "pred": []}
                        region_errors[region_id].append(error)
                        region_scores[region_id]["gt"].append(gt_score)
                        region_scores[region_id]["pred"].append(pred_score)
        
        if errors:
            results["mean_absolute_error"] = sum(errors) / len(errors)
        
        # Overall Wasserstein similarity
        if all_gt_scores and all_pred_scores:
            dist = wasserstein_distance(all_gt_scores, all_pred_scores)
            results["wasserstein_similarity"] = 1.0 / (1.0 + dist)

        # Per-region Wasserstein similarity
        for region_id, error_list in region_errors.items():
            if error_list:
                results["region_errors"][region_id] = sum(error_list) / len(error_list)
                region_dist = wasserstein_distance(
                    region_scores[region_id]["gt"],
                    region_scores[region_id]["pred"]
                )
                results["region_wasserstein_similarity"][region_id] = 1.0 - min(region_dist, 1.0)
        
        if len(all_gt_scores) > 1:
            try:
                correlation = np.corrcoef(all_gt_scores, all_pred_scores)[0, 1]
                results["correlation"] = float(correlation)
            except:
                results["correlation"] = float('nan')
        
        return results

class ReasonMatchEvaluator(Evaluator):
    """Evaluates reason rating accuracy using Wasserstein distance over Likert scores"""
    
    def __init__(self):
        super().__init__("reason_match")
    
    def evaluate(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        results = {
            "wasserstein_similarity": 0.0,
            "region_wasserstein": {}
        }
        
        wasserstein_scores = []
        region_scores = {}

        for user_id, gt_user in ground_truth.items():
            if user_id not in predicted:
                continue
            pred_user = predicted[user_id]

            for region_id, gt_reasons in gt_user.get("reasons", {}).items():
                pred_reasons = pred_user.get("reasons", {}).get(region_id)
                if pred_reasons is None:
                    continue

                # Ensure same reason keys
                reason_keys = sorted(set(gt_reasons.keys()) | set(pred_reasons.keys()))
                gt_vector = [gt_reasons.get(k, 0) for k in reason_keys]
                pred_vector = [pred_reasons.get(k, 0) for k in reason_keys]

                wd = wasserstein_distance(gt_vector, pred_vector)
                wasserstein_scores.append(wd)

                if region_id not in region_scores:
                    region_scores[region_id] = []
                region_scores[region_id].append(wd)

        if wasserstein_scores:
            avg_wd = sum(wasserstein_scores) / len(wasserstein_scores)
            # Convert distance to similarity: higher = better (bounded between 0 and 1)
            results["wasserstein_similarity"] = 1.0 / (1.0 + avg_wd)

        for region_id, distances in region_scores.items():
            if distances:
                avg_region_wd = sum(distances) / len(distances)
                results["region_wasserstein"][region_id] = 1.0 / (1.0 + avg_region_wd)

        return results
    
# class ReasonMatchEvaluator(Evaluator):
#     """Evaluates reason selection accuracy using both Jaccard similarity and Wasserstein distance"""
    
#     def __init__(self):
#         super().__init__("reason_match")
    
#     def evaluate(self, predicted: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
#         """Evaluate reason selection accuracy"""
#         results = {
#             "jaccard_similarity": 0.0,
#             "wasserstein_similarity": 0.0,
#             "region_similarities": {},
#             "region_wasserstein": {},
#             "most_common_correct": {},
#             "most_common_incorrect": {}
#         }
        
#         similarities = []
#         wasserstein_scores = []
#         region_similarities = {}
#         region_wasserstein = {}
#         correct_reasons = {}
#         incorrect_reasons = {}
        
#         for user_id, gt_user in ground_truth.items():
#             if user_id in predicted:
#                 pred_user = predicted[user_id]
                
#                 for region_id, gt_reasons in gt_user.get("reasons", {}).items():
#                     pred_reasons = pred_user.get("reasons", {}).get(region_id)
                    
#                     if pred_reasons is not None:
#                         # Calculate traditional Jaccard similarity
#                         gt_set = set(gt_reasons)
#                         pred_set = set(pred_reasons)
                        
#                         intersection = len(gt_set.intersection(pred_set))
#                         union = len(gt_set.union(pred_set))
                        
#                         similarity = intersection / union if union > 0 else 1.0
#                         similarities.append(similarity)
                        
#                         # Calculate Wasserstein distance for each reason
#                         reason_wasserstein = []
#                         for reason_code in set(gt_reasons.keys()) | set(pred_reasons.keys()):
#                             gt_score = gt_reasons.get(reason_code, 0)
#                             pred_score = pred_reasons.get(reason_code, 0)
#                             if isinstance(gt_score, (int, float)) and isinstance(pred_score, (int, float)):
#                                 reason_wasserstein.append(abs(gt_score - pred_score))
                        
#                         if reason_wasserstein:
#                             avg_wasserstein = sum(reason_wasserstein) / len(reason_wasserstein)
#                             wasserstein_scores.append(avg_wasserstein)
                            
#                             if region_id not in region_wasserstein:
#                                 region_wasserstein[region_id] = []
#                             region_wasserstein[region_id].append(avg_wasserstein)
                        
#                         if region_id not in region_similarities:
#                             region_similarities[region_id] = []
#                         region_similarities[region_id].append(similarity)
                        
#                         # Track correct and incorrect predictions
#                         for reason, score in pred_reasons.items():
#                             if reason in gt_reasons and abs(gt_reasons[reason] - score) <= 1:
#                                 if region_id not in correct_reasons:
#                                     correct_reasons[region_id] = {}
#                                 if reason not in correct_reasons[region_id]:
#                                     correct_reasons[region_id][reason] = 0
#                                 correct_reasons[region_id][reason] += 1
#                             else:
#                                 if region_id not in incorrect_reasons:
#                                     incorrect_reasons[region_id] = {}
#                                 if reason not in incorrect_reasons[region_id]:
#                                     incorrect_reasons[region_id][reason] = 0
#                                 incorrect_reasons[region_id][reason] += 1
        
#         if similarities:
#             results["jaccard_similarity"] = sum(similarities) / len(similarities)
        
#         if wasserstein_scores:
#             results["wasserstein_similarity"] = 1.0 / (1.0 + sum(wasserstein_scores) / len(wasserstein_scores))
        
#         for region_id, sim_list in region_similarities.items():
#             if sim_list:
#                 results["region_similarities"][region_id] = sum(sim_list) / len(sim_list)
#                 if region_id in region_wasserstein and region_wasserstein[region_id]:
#                     avg_wasserstein = sum(region_wasserstein[region_id]) / len(region_wasserstein[region_id])
#                     results["region_wasserstein"][region_id] = 1.0 / (1.0 + avg_wasserstein)
        
#         for region_id, reasons in correct_reasons.items():
#             sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
#             results["most_common_correct"][region_id] = sorted_reasons[:3] if sorted_reasons else []
        
#         for region_id, reasons in incorrect_reasons.items():
#             sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)
#             results["most_common_incorrect"][region_id] = sorted_reasons[:3] if sorted_reasons else []
        
#         return results

# Registry of available evaluators
EVALUATOR_REGISTRY = {
    "opinion_score": OpinionScoreEvaluator,
    "reason_match": ReasonMatchEvaluator
}

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load JSON file and return its contents"""
    with open(file_path, 'r') as f:
        return json.load(f)

def transform_output_to_survey_format(output_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform model output format to match survey format"""
    transformed = {}
    
    # Map opinion string to numeric scale
    opinion_mapping = {
        "oppose": 3,
        "neutral": 5,
        "support": 8
    }
    
    for comment in output_data.get("comments", []):
        user_id = comment.get("id", "")
        opinion_value = comment.get("opinion", "neutral")
        cell_id = comment.get("cell_id", "1.1")
        
        score = opinion_mapping.get(opinion_value, 5)
        
        if user_id not in transformed:
            transformed[user_id] = {
                "opinions": {},
                "reasons": {}
            }
        
        transformed[user_id]["opinions"][cell_id] = score
        
        if "reasons" in comment:
            transformed[user_id]["reasons"][cell_id] = comment["reasons"]
    
    return transformed

def evaluate_files(output_file: Path, ground_truth_file: Path, evaluator_names: List[str]) -> Dict[str, Any]:
    """Evaluate output against ground truth using specified evaluators"""
    results = {}
    
    try:
        # Load data
        output_data = load_json_file(output_file)
        ground_truth_data = load_json_file(ground_truth_file)
        
        # Transform output if needed
        if "comments" in output_data:
            predicted_data = transform_output_to_survey_format(output_data)
        else:
            predicted_data = output_data
        
        # Run evaluators
        for name in evaluator_names:
            if name in EVALUATOR_REGISTRY:
                evaluator = EVALUATOR_REGISTRY[name]()
                results[evaluator.name] = evaluator.evaluate(predicted_data, ground_truth_data)
            else:
                print(f"Warning: Unknown evaluator '{name}'")
        
    except Exception as e:
        print(f"Error evaluating: {str(e)}")
        results["error"] = str(e)
    
    return results

def run_evaluators(
    output_files: List[Path], 
    ground_truth_files: List[Path], 
    evaluator_names: List[str]
) -> Dict[str, Any]:
    """Run specified evaluators on multiple file pairs"""
    results = {}
    
    if not evaluator_names:
        print("No evaluators specified")
        return results
    
    # Match output files with ground truth files
    for output_file in output_files:
        proposal_id = output_file.name.split('_output.json')[0]
        gt_file = next((f for f in ground_truth_files if f.name.startswith(proposal_id)), None)
        
        if gt_file:
            # Run evaluation
            eval_results = evaluate_files(output_file, gt_file, evaluator_names)
            results[proposal_id] = eval_results
        else:
            print(f"No matching ground truth for {output_file}")
    
    return results

def evaluate_experiment_dir(experiment_dir: Path, evaluator_names: List[str]) -> Dict[str, Any]:
    """Evaluate all output files in an experiment directory"""
    results = {}
    
    # Find output and ground truth files
    output_files = list(experiment_dir.glob("*_output.json"))
    ground_truth_files = list(experiment_dir.glob("*_ground_truth.json"))
    
    if not output_files:
        print("No output files found in experiment directory.")
        return results
    
    if not ground_truth_files:
        print("No ground truth files found in experiment directory.")
        return results
    
    # Match output files with ground truth files
    for output_file in output_files:
        proposal_id = output_file.stem.rsplit("_output", 1)[0]
        gt_file = next((f for f in ground_truth_files if f.stem.startswith(proposal_id)), None)
        
        if gt_file:
            print(f"Evaluating {proposal_id}...")
            # Run evaluation
            proposal_results = evaluate_files(output_file, gt_file, evaluator_names)
            results[proposal_id] = proposal_results
            
            # Print summary
            for evaluator_name, metrics in proposal_results.items():
                if evaluator_name == "opinion_score":
                    print(f"  Opinion Score MAE: {metrics.get('mean_absolute_error', 'N/A'):.4f}")
                    print(f"  Opinion Correlation: {metrics.get('correlation', 'N/A'):.4f}")
                    print(f"  Opinion Wasserstein Similarity: {metrics.get('wasserstein_similarity', 'N/A'):.4f}")
                elif evaluator_name == "reason_match":
                    # print(f"  Reason Match Similarity: {metrics.get('jaccard_similarity', 'N/A'):.4f}")
                    print(f"  Reason Match Wasserstein: {metrics.get('wasserstein_similarity', 'N/A'):.4f}")

        else:
            print(f"No matching ground truth file for {output_file.name}")
    
    return results

def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(description="Evaluate model outputs against ground truth")
    
    # Input options - either file pair or experiment directory
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", "-o", help="Path to output JSON file")
    group.add_argument("--experiment-dir", "-d", help="Path to experiment directory")
    
    # Ground truth required only for file evaluation
    parser.add_argument("--ground-truth", "-g", help="Path to ground truth JSON file (required with --output)")
    
    # Other options
    parser.add_argument("--evaluators", "-e", nargs="+", default=list(EVALUATOR_REGISTRY.keys()),
                      help=f"Evaluators to run (available: {', '.join(EVALUATOR_REGISTRY.keys())})")
    parser.add_argument("--save", "-s", help="Path to save results JSON")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.output and not args.ground_truth:
        parser.error("--ground-truth is required when using --output")
    
    # Run evaluation based on input type
    if args.experiment_dir:
        # Evaluate experiment directory
        experiment_dir = Path(args.experiment_dir)
        if not experiment_dir.exists() or not experiment_dir.is_dir():
            print(f"Error: Experiment directory not found: {args.experiment_dir}")
            return 1
        
        results = evaluate_experiment_dir(experiment_dir, args.evaluators)
        
        # Determine save path
        save_path = Path(args.save) if args.save else experiment_dir / "evaluation_results.json"
        
    else:
        # Evaluate single file pair
        output_file = Path(args.output)
        ground_truth_file = Path(args.ground_truth)
        
        if not output_file.exists():
            print(f"Error: Output file not found: {args.output}")
            return 1
        
        if not ground_truth_file.exists():
            print(f"Error: Ground truth file not found: {args.ground_truth}")
            return 1
        
        results = evaluate_files(output_file, ground_truth_file, args.evaluators)
        
        # Print summary for single file evaluation
        print("\nEvaluation Summary:")
        for evaluator_name, metrics in results.items():
            if evaluator_name == "opinion_score":
                print(f"Opinion Score MAE: {metrics.get('mean_absolute_error', 'N/A'):.4f}")
                print(f"Opinion Correlation: {metrics.get('correlation', 'N/A'):.4f}")
                print(f"Opinion Wasserstein Similarity: {metrics.get('wasserstein_similarity', 'N/A'):.4f}")
            elif evaluator_name == "reason_match":
                print(f"Reason Match Similarity: {metrics.get('jaccard_similarity', 'N/A'):.4f}")
        
        # Determine save path
        save_path = Path(args.save) if args.save else None
    
    # Save results if path is specified
    if save_path:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {save_path}")
    elif not args.experiment_dir:
        # Print detailed results for single file if not saving
        print("\nDetailed Results:")
        print(json.dumps(results, indent=2))
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 