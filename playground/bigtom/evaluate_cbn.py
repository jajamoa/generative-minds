import json
import time
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from llm_utils import Colors
from cbn_agent import CBNAgent
from build_cbn_from_motifs import generate_mermaid_graph
import csv


def process_single_question(
    cbn_agent,
    vqa,
    include_demographics,
    include_context,
    temperature,
    debug=False,
    swap_data=None,
    max_retries=3,
    dynamic_save_dir: str | None = None,
):
    """Process a single question and return the result"""
    try:
        # Use swapped data if swap experiment is enabled
        if swap_data:
            demographics = swap_data.get("demographics", {})
            context_qas = swap_data.get("context_qas", [])
        else:
            demographics = vqa.get("demographics", {})
            context_qas = vqa.get("context_qas", [])

        # Check CBN matching status for display
        prolific_id = vqa.get("prolific_id")
        if getattr(cbn_agent, "dynamic_build", False):
            cbn_status = Colors.format("✓ Built CBN from story", Colors.GREEN)
        else:
            if hasattr(cbn_agent, "cbns_by_id") and prolific_id:
                found_match = prolific_id in cbn_agent.cbns_by_id
                if found_match:
                    cbn_status = Colors.format(
                        f"✓ Using specific CBN for {prolific_id}", Colors.GREEN
                    )
                else:
                    cbn_status = Colors.format(
                        f"⚠ Using default CBN (no match for {prolific_id})", Colors.RED
                    )
            else:
                cbn_status = Colors.format(
                    "⚠ Using default CBN (no prolific_id)", Colors.RED
                )

        if not debug:  # Show CBN status in non-debug mode
            print(f"CBN: {cbn_status}")

        # Determine task type
        task_type = vqa.get("task_type", "belief_attribution")

        if task_type == "belief_attribution":
            # Use CBN agent to process the query instead of direct LLM call
            generated_response = cbn_agent.process_query(
                vqa=vqa,
                demographics=demographics,
                context_qas=context_qas,
                include_demographics=include_demographics,
                include_context=include_context,
                temperature=temperature,
                debug=debug,
            )

            # Save dynamically built CBN if available
            try:
                if (
                    getattr(cbn_agent, "dynamic_build", False)
                    and getattr(cbn_agent, "last_built_cbn_graph", None)
                    and dynamic_save_dir
                ):
                    pid = vqa.get("prolific_id") or getattr(
                        cbn_agent, "last_built_id", "unknown"
                    )
                    # Normalize id so see/no share the same base id for saving
                    base_pid = (
                        pid.replace("_see", "").replace("_no", "")
                        if isinstance(pid, str)
                        else pid
                    )
                    out_dir = Path(dynamic_save_dir)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # File stems
                    stem = f"{base_pid}_cbn"
                    graph = cbn_agent.last_built_cbn_graph
                    # Save JSON
                    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
                        json.dump(graph, f, indent=2, ensure_ascii=False)
                    # Save Mermaid and MD
                    nodes = graph.get("nodes", {})
                    edges = graph.get("edges", {})
                    mermaid = generate_mermaid_graph(nodes, edges)
                    with open(out_dir / f"{stem}.mmd", "w", encoding="utf-8") as f:
                        f.write(mermaid)
                    with open(out_dir / f"{stem}.md", "w", encoding="utf-8") as f:
                        f.write("```mermaid\n")
                        f.write(mermaid)
                        f.write("\n```.\n")
            except Exception:
                pass

            # Extract the answer dynamically from available options
            generated_answer = None
            if generated_response:
                response_upper = generated_response.upper()
                answer_options = vqa["answer_options"]
                option_keys = list(answer_options.keys())

                # Try to find unique option key
                found_options = []
                for key in option_keys:
                    if key.upper() in response_upper:
                        found_options.append(key.upper())

                if len(found_options) == 1:
                    generated_answer = found_options[0]
                else:
                    # Try to get first option key that appears
                    for char in response_upper:
                        if char in [k.upper() for k in option_keys]:
                            generated_answer = char
                            break

            # Get the correct answer
            correct_answer = vqa["answer"]

            # Evaluate correctness
            is_correct = generated_answer == correct_answer

        elif task_type == "belief_update":
            # Use CBN agent to process the query
            generated_response = cbn_agent.process_query(
                vqa=vqa,
                demographics=demographics,
                context_qas=context_qas,
                include_demographics=include_demographics,
                include_context=include_context,
                temperature=temperature,
                debug=debug,
            )

            # Extract numeric answer
            generated_answer = None
            if generated_response:
                # Try to extract number from response
                import re

                numbers = re.findall(r"\b\d+\b", generated_response.strip())
                if numbers:
                    try:
                        generated_answer = int(numbers[0])
                    except ValueError:
                        pass

            # Get the correct answer
            correct_answer = vqa["user_answer"]

            # For belief_update, we evaluate based on exact match or close proximity
            is_correct = False
            if generated_answer is not None and correct_answer is not None:
                # Exact match or within 1 point for ordinal scales
                scale = vqa.get("scale", [1, 10])
                scale_range = scale[1] - scale[0]
                tolerance = (
                    1 if scale_range <= 5 else 2
                )  # Stricter tolerance for smaller scales
                is_correct = abs(generated_answer - correct_answer) <= tolerance

        else:
            raise ValueError(f"Unknown task type: {task_type}")

        return {
            "vqa": vqa,
            "generated_answer": generated_answer,
            "generated_response": generated_response,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "context_qas_count": len(context_qas),
            "task_type": task_type,
        }

    except Exception as e:
        return {
            "vqa": vqa,
            "generated_answer": None,
            "generated_response": None,
            "correct_answer": vqa.get("answer") or vqa.get("user_answer", ""),
            "is_correct": False,
            "context_qas_count": len(vqa.get("context_qas", [])),
            "task_type": vqa.get("task_type", "unknown"),
            "error": str(e),
        }


def create_swap_mapping(dataset):
    """Create mapping to swap data between consecutive prolific IDs"""
    prolific_ids = []
    id_to_data = {}

    for vqa in dataset:
        pid = vqa.get("prolific_id", "")
        if pid and pid not in id_to_data:
            prolific_ids.append(pid)
            id_to_data[pid] = {
                "demographics": vqa.get("demographics", {}),
                "context_qas": vqa.get("context_qas", []),
            }

    # Create swap mapping: each ID maps to next ID's data
    swap_mapping = {}
    for i, pid in enumerate(prolific_ids):
        next_pid = prolific_ids[(i + 1) % len(prolific_ids)]
        swap_mapping[pid] = id_to_data[next_pid]

    return swap_mapping


def evaluate_belief_inference(
    benchmark_path,
    cbn_path=None,
    model="qwen-plus",
    temperature=0,
    include_demographics=True,
    include_context=True,
    max_workers=3,
    debug=False,
    swap_experiment=False,
    limit=None,
    motifs_dir=None,
):
    """
    Evaluate Theory of Mind belief inference questions using CBN agent
    """
    # Initialize CBN agent
    cbn_agent = CBNAgent(model=model, cbn_path=cbn_path, motifs_dir=motifs_dir)
    # Where to save dynamically built CBNs
    dynamic_save_dir = None
    if motifs_dir:
        import os

        # Put eval artifacts under results/normal/eval/dynamic_cbns
        base_eval_dir = os.path.join("results", "normal", "eval")
        dynamic_save_dir = os.path.join(base_eval_dir, "dynamic_cbns")
        os.makedirs(dynamic_save_dir, exist_ok=True)

    # Load benchmark data (support JSONL or BigToM CSV)
    vqa_dataset = []
    if benchmark_path.lower().endswith(".csv"):
        with open(benchmark_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter=";")
            for row_idx, cols in enumerate(reader, start=1):
                if not cols or not any((c or "").strip() for c in cols):
                    continue

                def safe(i: int) -> str:
                    return (
                        cols[i].strip() if len(cols) > i and cols[i] is not None else ""
                    )

                story = safe(0)
                see_cond = safe(1)
                no_cond = safe(2)
                opt_a = safe(3)
                opt_b = safe(4)
                q_action = safe(7) or "What will they do?"
                a_will_see = safe(10)
                a_will_no = safe(13)

                if not story:
                    continue

                # Helper to compute correct letter by matching will-answer against options
                def match_correct_letter(will_answer: str) -> str | None:
                    if will_answer and opt_a and will_answer == opt_a:
                        return "A"
                    if will_answer and opt_b and will_answer == opt_b:
                        return "B"
                    return None

                # SEE scenario
                if see_cond:
                    vqa_dataset.append(
                        {
                            "prolific_id": f"bigtom_row{row_idx:04d}_see",
                            "task_type": "belief_attribution",
                            "task_question": q_action,
                            "answer_options": {"A": opt_a or "", "B": opt_b or ""},
                            "answer": match_correct_letter(a_will_see),
                            "demographics": {},
                            "context_qas": [
                                {"question": "Story context", "answer": story},
                                {
                                    "question": "Observation condition",
                                    "answer": see_cond,
                                },
                            ],
                        }
                    )

                # NO-SEE scenario
                if no_cond:
                    vqa_dataset.append(
                        {
                            "prolific_id": f"bigtom_row{row_idx:04d}_no",
                            "task_type": "belief_attribution",
                            "task_question": q_action,
                            "answer_options": {"A": opt_a or "", "B": opt_b or ""},
                            "answer": match_correct_letter(a_will_no),
                            "demographics": {},
                            "context_qas": [
                                {"question": "Story context", "answer": story},
                                {
                                    "question": "Observation condition",
                                    "answer": no_cond,
                                },
                            ],
                        }
                    )
    else:
        with open(benchmark_path, "r") as file:
            content = file.read().strip()

            # Handle formatted JSON objects separated by newlines
            json_objects = []
            current_object = ""
            brace_count = 0

            for line in content.split("\n"):
                current_object += line + "\n"
                brace_count += line.count("{") - line.count("}")

                if brace_count == 0 and current_object.strip():
                    try:
                        json_obj = json.loads(current_object.strip())
                        json_objects.append(json_obj)
                        current_object = ""
                    except json.JSONDecodeError:
                        continue

            vqa_dataset = json_objects

    # Apply limit if specified
    if limit is not None and limit > 0:
        vqa_dataset = vqa_dataset[:limit]
        print(f"Limiting evaluation to first {limit} questions")

    # Separate data by task type and difficulty
    task_types = set()
    difficulty_datasets = {"simple": [], "medium": [], "hard": [], "all": []}

    for vqa in vqa_dataset:
        task_type = vqa.get("task_type", "belief_attribution")
        task_types.add(task_type)

        # For belief_attribution, use context_length as difficulty
        if task_type == "belief_attribution":
            difficulty = vqa.get("context_length", "unknown")
            if difficulty not in difficulty_datasets:
                difficulty_datasets[difficulty] = []
            difficulty_datasets[difficulty].append(vqa)

        # For belief_update, group all together since difficulty doesn't apply the same way
        elif task_type == "belief_update":
            difficulty_datasets["all"].append(vqa)

    # Remove empty difficulty levels
    difficulty_datasets = {k: v for k, v in difficulty_datasets.items() if v}

    # Create swap mapping if swap experiment is enabled
    swap_mapping = create_swap_mapping(vqa_dataset) if swap_experiment else None

    # Results for each difficulty
    all_results = {}

    print(
        f"Evaluating {len(vqa_dataset)} questions across {len(difficulty_datasets)} groups..."
    )
    print(f"Task types found: {', '.join(task_types)}")
    if motifs_dir:
        print(f"CBN mode: dynamic build from motifs_dir={motifs_dir}")
    else:
        print(f"CBN file: {cbn_path or 'None'}")

    for difficulty, dataset in difficulty_datasets.items():
        if not dataset:
            continue

        print(f"\n{'='*60}")
        if difficulty == "all":
            # Get task type for this group
            sample_task_type = dataset[0].get("task_type", "unknown")
            print(f"EVALUATING TASK TYPE: {sample_task_type.upper()}")
        else:
            print(f"EVALUATING DIFFICULTY LEVEL: {difficulty.upper()}")
        print(f"{'='*60}")
        print(f"Questions in this group: {len(dataset)}")

        correct = 0
        total = 0
        answers = []
        absolute_errors = []  # For MAE calculation

        # Process questions (parallel or sequential based on debug mode)
        if debug:
            # Sequential processing for debug mode
            results = []
            for i, vqa in enumerate(dataset):
                print(
                    f"\n{Colors.format('[DEBUG]', Colors.BOLD + Colors.YELLOW)} Processing question {Colors.format(str(i+1), Colors.CYAN)}/{Colors.format(str(len(dataset)), Colors.CYAN)} in {Colors.format(difficulty, Colors.GREEN)} difficulty"
                )

                # Get swap data if experiment is enabled
                swap_data = None
                if swap_experiment and swap_mapping:
                    pid = vqa.get("prolific_id", "")
                    swap_data = swap_mapping.get(pid)

                result = process_single_question(
                    cbn_agent,
                    vqa,
                    include_demographics,
                    include_context,
                    temperature,
                    debug,
                    swap_data,
                    max_retries=5,
                    dynamic_save_dir=dynamic_save_dir,
                )
                results.append(result)
        else:
            # Sequential processing for CBN agent (no parallel support for now)
            results = []
            for i, vqa in enumerate(tqdm(dataset, desc=f"Processing {difficulty}")):
                # Get swap data if experiment is enabled
                swap_data = None
                if swap_experiment and swap_mapping:
                    pid = vqa.get("prolific_id", "")
                    swap_data = swap_mapping.get(pid)

                result = process_single_question(
                    cbn_agent,
                    vqa,
                    include_demographics,
                    include_context,
                    temperature,
                    debug,
                    swap_data,
                    max_retries=5,
                    dynamic_save_dir=dynamic_save_dir,
                )
                results.append(result)

                # Add small delay between calls
                time.sleep(0.1)

        # Process results in order
        for i, result in enumerate(results):
            if result is None:
                continue

            vqa = result["vqa"]
            generated_answer = result["generated_answer"]
            generated_response = result["generated_response"]
            correct_answer = result["correct_answer"]
            is_correct = result["is_correct"]
            context_qas_count = result["context_qas_count"]

            if is_correct:
                correct += 1
                answers.append(1)
            else:
                answers.append(0)

            # Calculate absolute error for numeric tasks
            if (
                vqa.get("task_type") == "belief_update"
                and generated_answer is not None
                and correct_answer is not None
            ):
                abs_error = abs(generated_answer - correct_answer)
                absolute_errors.append(abs_error)

            total += 1

            # Print details for verbose mode (only if not in debug mode)
            if not debug:
                group_label = (
                    difficulty.upper()
                    if difficulty != "all"
                    else vqa.get("task_type", "unknown").upper()
                )
                print(f"\n[{group_label}] Question {total}:")
                print(f"Task Type: {vqa.get('task_type', 'unknown')}")
                print(f"Context QAs: {context_qas_count}")
                print(f"Task: {vqa['task_question']}")
                print(f"Generated: {generated_answer} (Response: {generated_response})")
                print(f"Correct: {correct_answer}")
                print(f"Result: {'✓' if is_correct else '✗'}")

                # Show different fields based on task type
                if vqa.get("task_type") == "belief_attribution":
                    print(
                        f"Source QA: {vqa.get('source_qa', {}).get('question', 'N/A')}"
                    )
                elif vqa.get("task_type") == "belief_update":
                    print(f"Question Type: {vqa.get('question_type', 'N/A')}")
                    if "reason_text" in vqa:
                        print(f"Reason: {vqa['reason_text']}")

                if "error" in result:
                    print(f"Error: {result['error']}")
                print("-" * 50)

        # Calculate and store results for this difficulty
        accuracy = correct / total if total > 0 else 0
        mae = sum(absolute_errors) / len(absolute_errors) if absolute_errors else None

        all_results[difficulty] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy,
            "answers": answers,
            "mae": mae,
            "absolute_errors": absolute_errors,
        }

        print(f"\n{'='*50}")
        if difficulty == "all":
            sample_task_type = dataset[0].get("task_type", "unknown")
            print(f"RESULTS FOR {sample_task_type.upper()} TASK")
        else:
            print(f"RESULTS FOR {difficulty.upper()} DIFFICULTY")
        print(f"{'='*50}")
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct}")
        print(f"Accuracy: {accuracy:.2%}")
        if mae is not None:
            print(f"MAE (Mean Absolute Error): {mae:.3f}")
        print(
            f"Context QAs per question: {len(dataset[0]['context_qas']) if dataset else 0}"
        )

    # Display summary results
    print(f"\n{'='*60}")
    print(f"SUMMARY RESULTS ACROSS ALL DIFFICULTIES")
    print(f"{'='*60}")

    total_all = sum(result["total"] for result in all_results.values())
    correct_all = sum(result["correct"] for result in all_results.values())
    accuracy_all = correct_all / total_all if total_all > 0 else 0

    # Calculate overall MAE for belief_update tasks
    all_absolute_errors = []
    for result in all_results.values():
        if result["absolute_errors"]:
            all_absolute_errors.extend(result["absolute_errors"])

    overall_mae = (
        sum(all_absolute_errors) / len(all_absolute_errors)
        if all_absolute_errors
        else None
    )

    print(f"Overall accuracy: {accuracy_all:.2%} ({correct_all}/{total_all})")
    if overall_mae is not None:
        print(f"Overall MAE: {overall_mae:.3f}")
    print(f"\nBreakdown by group:")
    for group_name in all_results:
        result = all_results[group_name]
        if group_name == "all":
            task_type = difficulty_datasets[group_name][0].get("task_type", "unknown")
            group_label = f"{task_type}"
        else:
            group_label = group_name

        context_size = (
            len(difficulty_datasets[group_name][0]["context_qas"])
            if difficulty_datasets[group_name]
            else 0
        )
        mae_str = f", MAE: {result['mae']:.3f}" if result["mae"] is not None else ""
        print(
            f"  {group_label.capitalize():<12} ({context_size:2d} context): {result['accuracy']:.2%} ({result['correct']}/{result['total']}){mae_str}"
        )

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Theory of Mind belief inference benchmark with CBN Agent"
    )
    parser.add_argument(
        "--benchmark", type=str, required=True, help="Path to the benchmark JSONL file"
    )
    parser.add_argument(
        "--cbn",
        type=str,
        default=None,
        help="(Optional) Path to a static CBN JSON. If omitted and --motifs_dir is provided, CBNs are built dynamically.",
    )
    parser.add_argument(
        "--motifs_dir",
        type=str,
        default=None,
        help="Directory of motif library (per-participant folders). If set, CBNs will be built on-the-fly from story/context.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="qwen-plus",
        choices=[
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen2.5-72b-instruct",
            "qwen2.5-32b-instruct",
            "qwen2.5-14b-instruct",
            "qwen2.5-7b-instruct",
            "qwen2.5-3b-instruct",
            "qwen2.5-1.5b-instruct",
            "qwen2.5-0.5b-instruct",
            "meta-llama/llama-3.3-70b-instruct",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ],
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="Temperature for generation"
    )
    parser.add_argument(
        "--no-demographics",
        action="store_true",
        help="Exclude demographics from prompt",
    )
    parser.add_argument(
        "--no-context", action="store_true", help="Exclude context QAs from prompt"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of parallel workers for API calls (not used in CBN mode)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: sequential processing with full prompt/response display",
    )
    parser.add_argument(
        "--swap-experiment",
        action="store_true",
        help="Use next participant's demographics/context for prediction (extreme test)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only test the first N questions (default: 5)",
    )

    args = parser.parse_args()

    try:
        results = evaluate_belief_inference(
            benchmark_path=args.benchmark,
            cbn_path=args.cbn,
            model=args.model,
            temperature=args.temperature,
            include_demographics=not args.no_demographics,
            include_context=not args.no_context,
            max_workers=args.max_workers,
            debug=args.debug,
            swap_experiment=args.swap_experiment,
            limit=args.limit,
            motifs_dir=args.motifs_dir,
        )

        # Save results to results directory
        import os

        # Put final results under results/normal/eval
        results_dir = "results/normal/eval"
        os.makedirs(results_dir, exist_ok=True)

        # Clean model name and extract benchmark name for file naming
        clean_model_name = args.model.replace("/", "-")
        benchmark_name = args.benchmark.split("/")[-1].split(".")[0]
        results_filename = os.path.join(
            results_dir, f"cbn_results_{clean_model_name}.json"
        )
        with open(results_filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to {results_filename}")

    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
