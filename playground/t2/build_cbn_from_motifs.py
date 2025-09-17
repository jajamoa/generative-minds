#!/usr/bin/env python3
"""
Build Cognitive Belief Networks (CBN) from extracted motifs for each participant.
This script converts motifs from individual participants into CBN format compatible with evaluate_cbn.py.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import argparse
from collections import defaultdict
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Import our node similarity module
from node_similarity import (
    get_consistent_node_id, 
    is_stance_node, 
    find_similar_node,
    compute_node_similarity
)

def get_node_id(label: str) -> str:
    """Generate a consistent node ID from a label."""
    # Use first 8 chars of hash to ensure uniqueness while keeping IDs short
    hash_val = hashlib.md5(label.encode()).hexdigest()[:8]
    return f"n_{hash_val}"

def get_edge_id(source_id: str, target_id: str) -> str:
    """Generate a consistent edge ID from source and target node IDs."""
    combined = f"{source_id}_{target_id}"
    hash_val = hashlib.md5(combined.encode()).hexdigest()[:6]
    return f"e_{hash_val}"

def merge_motifs_to_graph(motifs: List[Dict]) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Merge multiple 3-node motifs into a unified graph using semantic similarity.
    Returns nodes and edges dictionaries in CBN format.
    """
    # Track all unique nodes and edges
    nodes = {}
    edges = {}
    
    # Track node evidence (which QAs mention each node)
    node_evidence = defaultdict(list)
    edge_evidence = defaultdict(list)
    
    for motif in motifs:
        node_labels = motif.get("node_labels", {})
        motif_edges = motif.get("edges", [])
        sources = motif.get("sources", [])
        
        # Map local node IDs (n1, n2, n3) to global node IDs using similarity
        local_to_global_id = {}
        local_to_label = {}
        
        for local_id, label in node_labels.items():
            if label:  # Skip empty labels
                local_to_label[local_id] = label
                
                # Find similar existing node or create new one
                node_id = get_consistent_node_id(label, nodes, similarity_threshold=0.8)
                local_to_global_id[local_id] = node_id
                
                # Add node if not exists
                if node_id not in nodes:
                    nodes[node_id] = {
                        "label": label,
                        "aggregate_confidence": 0.8,  # Default confidence
                        "importance": 0.7,  # Default importance
                        "frequency": 0,
                        "is_stance": is_stance_node(label),  # Use semantic detection
                        "incoming_edges": [],
                        "outgoing_edges": [],
                        "evidence": [],
                        "status": "anchor"
                    }
                else:
                    # Update label to the most representative one if this one is better
                    if is_stance_node(label) and not is_stance_node(nodes[node_id]["label"]):
                        nodes[node_id]["label"] = label
                        nodes[node_id]["is_stance"] = True
                
                # Update frequency
                nodes[node_id]["frequency"] += 1
                
                # Add evidence from sources
                for source in sources:
                    qa_id = source.split("_")[-1] if "_" in source else source
                    evidence_entry = {
                        "qa_id": qa_id,
                        "confidence": 0.8,
                        "importance": 0.7
                    }
                    if evidence_entry not in nodes[node_id]["evidence"]:
                        nodes[node_id]["evidence"].append(evidence_entry)
        
        # Process edges using the mapped global node IDs
        for edge in motif_edges:
            if len(edge) == 2:
                src_local, tgt_local = edge
                if src_local in local_to_global_id and tgt_local in local_to_global_id:
                    src_id = local_to_global_id[src_local]
                    tgt_id = local_to_global_id[tgt_local]
                    src_label = local_to_label[src_local]
                    tgt_label = local_to_label[tgt_local]
                    edge_id = get_edge_id(src_id, tgt_id)
                    
                    # Add edge if not exists
                    if edge_id not in edges:
                        edges[edge_id] = {
                            "source": src_id,
                            "target": tgt_id,
                            "source_label": nodes[src_id]["label"],  # Use the canonical label
                            "target_label": nodes[tgt_id]["label"],  # Use the canonical label
                            "direction": "positive",  # Default to positive
                            "strength": 0.7,  # Default strength
                            "modifier": 1.0,
                            "aggregate_confidence": 0.8,
                            "evidence": [],
                            "explanation": f"{nodes[src_id]['label']} influences {nodes[tgt_id]['label']}"
                        }
                        
                        # Update node edge lists
                        if edge_id not in nodes[src_id]["outgoing_edges"]:
                            nodes[src_id]["outgoing_edges"].append(edge_id)
                        if edge_id not in nodes[tgt_id]["incoming_edges"]:
                            nodes[tgt_id]["incoming_edges"].append(edge_id)
                    
                    # Add evidence from sources
                    for source in sources:
                        qa_id = source.split("_")[-1] if "_" in source else source
                        evidence_entry = {
                            "qa_id": qa_id,
                            "confidence": 0.8,
                            "original_modifier": 1.0
                        }
                        if evidence_entry not in edges[edge_id]["evidence"]:
                            edges[edge_id]["evidence"].append(evidence_entry)
                    
                    # Update edge strength if we have multiple mentions
                    edge_mention_count = len(edges[edge_id]["evidence"])
                    edges[edge_id]["strength"] = min(0.7 + (edge_mention_count - 1) * 0.1, 1.0)
    
    # Update node importance based on connectivity
    for node_id, node_data in nodes.items():
        connectivity = len(node_data["incoming_edges"]) + len(node_data["outgoing_edges"])
        node_data["importance"] = min(0.7 + connectivity * 0.05, 1.0)
        
        # Update aggregate confidence based on evidence count
        if node_data["evidence"]:
            avg_confidence = sum(e["confidence"] for e in node_data["evidence"]) / len(node_data["evidence"])
            node_data["aggregate_confidence"] = avg_confidence
    
    return nodes, edges

def generate_mermaid_graph(nodes: Dict, edges: Dict) -> str:
    """
    Generate Mermaid diagram from CBN nodes and edges.
    """
    lines = ["graph TD"]
    
    # Add nodes with styling
    for node_id, node_data in nodes.items():
        label = node_data["label"].replace('"', "'")
        confidence = node_data.get("aggregate_confidence", 0.8)
        
        # Style nodes based on their properties
        if node_data.get("is_stance", False):
            # Stance nodes - diamond shape with bold
            lines.append(f'    {node_id}{{{{{label}}}}}')
            lines.append(f'    style {node_id} fill:#ff9999,stroke:#333,stroke-width:3px')
        elif len(node_data["incoming_edges"]) == 0:
            # Root nodes - rounded rectangle
            lines.append(f'    {node_id}["{label}"]')
            lines.append(f'    style {node_id} fill:#99ccff,stroke:#333,stroke-width:2px')
        elif len(node_data["outgoing_edges"]) == 0:
            # Leaf nodes - double border
            lines.append(f'    {node_id}[["{label}"]]')
            lines.append(f'    style {node_id} fill:#ccffcc,stroke:#333,stroke-width:2px')
        else:
            # Intermediate nodes - regular rectangle
            lines.append(f'    {node_id}["{label}"]')
            
    # Add edges with labels
    for edge_id, edge_data in edges.items():
        source = edge_data["source"]
        target = edge_data["target"]
        strength = edge_data.get("strength", 0.7)
        direction = edge_data.get("direction", "positive")
        
        # Edge style based on strength
        if strength >= 0.9:
            arrow = "==>"
        elif strength >= 0.7:
            arrow = "-->"
        else:
            arrow = "-.->"
            
        # Add direction indicator
        edge_label = f"{strength:.2f}"
        if direction == "negative":
            edge_label = f"-{edge_label}"
            
        lines.append(f'    {source} {arrow}|{edge_label}| {target}')
    
    # Add legend
    lines.extend([
        "",
        "    %% Legend",
        "    subgraph Legend",
        "    L1[Root Node]",
        "    L2[Intermediate Node]", 
        "    L3[[Leaf Node]]",
        "    L4{Stance Node}",
        "    end",
        "    style L1 fill:#99ccff",
        "    style L3 fill:#ccffcc",
        "    style L4 fill:#ff9999"
    ])
    
    return "\n".join(lines)

def build_cbn_for_participant(participant_dir: Path, prolific_id: str) -> Tuple[Dict, str]:
    """
    Build a CBN structure for a single participant from their motifs.
    Returns CBN dict and Mermaid diagram string.
    """
    # Read the participant's motifs
    motif_file = participant_dir / f"{prolific_id}.json"
    if not motif_file.exists():
        return None, None
    
    with open(motif_file, 'r') as f:
        data = json.load(f)
    
    motifs = data.get("motifs", [])
    if not motifs:
        return None, None
    
    # Get demographics from first motif (they should all be the same)
    demographics = motifs[0].get("demographics", {}) if motifs else {}
    
    # Merge motifs into a graph
    nodes, edges = merge_motifs_to_graph(motifs)
    
    # Enhanced stance node identification
    # Priority 1: Nodes already identified as stance through semantic analysis
    # Priority 2: Leaf nodes (no outgoing edges) that represent final outcomes
    # Priority 3: Nodes with "support for" patterns
    
    stance_candidates = []
    
    for node_id, node_data in nodes.items():
        label = node_data["label"]
        stance_score = 0.0
        
        # High priority: semantically identified stance nodes
        if node_data.get("is_stance", False):
            stance_score += 10.0
            
        # Medium priority: leaf nodes that could be outcomes
        if len(node_data["outgoing_edges"]) == 0:
            stance_score += 5.0
            
        # Boost for specific patterns indicating final beliefs
        if any(pattern in label.lower() for pattern in ['support for', 'support', 'belief', 'opinion']):
            stance_score += 3.0
            
        # Boost for high connectivity (importance)
        stance_score += node_data["importance"] * 2.0
        
        # Boost for frequency (how often mentioned)
        stance_score += min(node_data["frequency"] * 0.5, 2.0)
        
        if stance_score >= 5.0:  # Threshold for stance consideration
            stance_candidates.append((node_id, node_data, stance_score))
    
    # Sort by stance score and mark top candidates
    stance_candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Mark stance nodes, ensuring we have at least one
    stance_count = 0
    for node_id, node_data, score in stance_candidates:
        if stance_count < 3 and (score >= 8.0 or stance_count == 0):  # At least one stance node
            nodes[node_id]["is_stance"] = True
            nodes[node_id]["status"] = "stance"
            stance_count += 1
        else:
            # Ensure semantic stance detection is preserved
            if node_data.get("is_stance", False):
                nodes[node_id]["status"] = "stance"
    
    # Generate Mermaid diagram
    mermaid_diagram = generate_mermaid_graph(nodes, edges)
    
    # Build the CBN structure
    cbn = {
        "sessionId": f"session_motif_{prolific_id}",
        "prolificId": prolific_id,
        "status": "completed",
        "graphs": [
            {
                "_id": f"graph_{prolific_id}",
                "sessionId": f"session_motif_{prolific_id}",
                "prolificId": prolific_id,
                "qaPairId": "motif_synthesis",
                "graphData": {
                    "agent_id": prolific_id,
                    "nodes": nodes,
                    "edges": edges,
                    "metadata": {
                        "total_motifs": len(motifs),
                        "unique_nodes": len(nodes),
                        "unique_edges": len(edges),
                        "demographics": demographics,
                        "topic": data.get("topic", "healthcare")
                    }
                },
                "createdAt": "2025-01-16T00:00:00.000Z",
                "updatedAt": "2025-01-16T00:00:00.000Z"
            }
        ],
        "createdAt": "2025-01-16T00:00:00.000Z",
        "updatedAt": "2025-01-16T00:00:00.000Z"
    }
    
    return cbn, mermaid_diagram

def process_single_participant(participant_info: Tuple[Path, str, Path]) -> Optional[Dict]:
    """
    Process a single participant for parallel execution.
    Returns processed data or None if failed.
    """
    participant_dir, prolific_id, output_dir = participant_info
    
    try:
        # Build CBN for this participant
        cbn, mermaid_diagram = build_cbn_for_participant(participant_dir, prolific_id)
        
        if cbn:
            # Create participant output directory
            participant_output_dir = output_dir / prolific_id
            participant_output_dir.mkdir(exist_ok=True)
            
            # Save CBN JSON
            cbn_file = participant_output_dir / f"{prolific_id}_cbn.json"
            with open(cbn_file, 'w') as f:
                json.dump([cbn], f, indent=2)  # Wrap in array like sample_cbn.json
            
            # Save Mermaid diagram (.mmd file)
            mmd_file = participant_output_dir / f"{prolific_id}_cbn.mmd"
            with open(mmd_file, 'w') as f:
                f.write(mermaid_diagram)
            
            # Save Markdown file with embedded Mermaid
            md_file = participant_output_dir / f"{prolific_id}_cbn.md"
            with open(md_file, 'w') as f:
                f.write(f"# Cognitive Belief Network for Participant {prolific_id}\n\n")
                f.write(f"**Topic**: {cbn['graphs'][0]['graphData']['metadata']['topic']}\n")
                f.write(f"**Total Motifs**: {cbn['graphs'][0]['graphData']['metadata']['total_motifs']}\n")
                f.write(f"**Unique Nodes**: {cbn['graphs'][0]['graphData']['metadata']['unique_nodes']}\n")
                f.write(f"**Unique Edges**: {cbn['graphs'][0]['graphData']['metadata']['unique_edges']}\n\n")
                f.write("## Graph Visualization\n\n")
                f.write("```mermaid\n")
                f.write(mermaid_diagram)
                f.write("\n```\n\n")
                f.write("## Node Types\n\n")
                f.write("- **Root Nodes** (Blue): Starting points with no incoming edges\n")
                f.write("- **Intermediate Nodes** (White): Nodes with both incoming and outgoing edges\n")
                f.write("- **Leaf Nodes** (Green): Endpoints with no outgoing edges\n")
                f.write("- **Stance Nodes** (Red): High-importance belief endpoints\n\n")
                f.write("## Edge Strength\n\n")
                f.write("- Solid thick arrow (==>): Strong connection (≥0.9)\n")
                f.write("- Solid arrow (-->): Medium connection (≥0.7)\n")
                f.write("- Dashed arrow (-.->): Weak connection (<0.7)\n")
            
            return {
                'prolific_id': prolific_id,
                'nodes': len(cbn['graphs'][0]['graphData']['nodes']),
                'edges': len(cbn['graphs'][0]['graphData']['edges']),
                'success': True
            }
        else:
            return {'prolific_id': prolific_id, 'success': False, 'error': 'No motifs found'}
            
    except Exception as e:
        return {'prolific_id': prolific_id, 'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="Build CBN from participant motifs")
    parser.add_argument(
        "--input_dir",
        default="playground/t2/results/motifs_from_transcripts",
        help="Directory containing participant motif folders"
    )
    parser.add_argument(
        "--output_dir",
        default="playground/t2/results/participant_cbns",
        help="Directory to save individual CBN JSON files"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of participants to process"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    root_dir = Path(__file__).parent.parent.parent
    input_dir = root_dir / args.input_dir
    output_dir = root_dir / args.output_dir
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all participant directories
    participant_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if args.limit:
        participant_dirs = participant_dirs[:args.limit]
    
    print(f"Processing {len(participant_dirs)} participants...")
    print(f"Using {'serial' if args.no_parallel else 'parallel'} processing with {args.workers if not args.no_parallel else 1} worker(s)")
    
    successful = 0
    failed = 0
    
    # Prepare participant info for processing
    participant_infos = [(participant_dir, participant_dir.name, output_dir) for participant_dir in participant_dirs]
    
    if args.no_parallel:
        # Serial processing with progress bar
        for participant_info in tqdm(participant_infos, desc="Processing participants"):
            result = process_single_participant(participant_info)
            if result and result['success']:
                successful += 1
                print(f"✓ {result['prolific_id']}: {result['nodes']} nodes, {result['edges']} edges")
            else:
                failed += 1
                error_msg = result['error'] if result else "Unknown error"
                print(f"✗ {participant_info[1]}: {error_msg}")
    else:
        # Parallel processing with progress bar
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Submit all tasks
            future_to_info = {
                executor.submit(process_single_participant, info): info[1] 
                for info in participant_infos
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_info), total=len(participant_infos), desc="Processing participants"):
                prolific_id = future_to_info[future]
                try:
                    result = future.result()
                    if result and result['success']:
                        successful += 1
                        tqdm.write(f"✓ {result['prolific_id']}: {result['nodes']} nodes, {result['edges']} edges")
                    else:
                        failed += 1
                        error_msg = result['error'] if result else "Unknown error"
                        tqdm.write(f"✗ {prolific_id}: {error_msg}")
                except Exception as e:
                    failed += 1
                    tqdm.write(f"✗ {prolific_id}: {str(e)}")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Successfully processed: {successful} participants")
    print(f"  Failed: {failed} participants")
    print(f"  Output directory: {output_dir.relative_to(root_dir)}")
    
    # Create a combined CBN file with all participants
    if successful > 0:
        print(f"\nCreating combined CBN file...")
        all_cbns = []
        
        for participant_dir in output_dir.iterdir():
            if participant_dir.is_dir():
                cbn_files = list(participant_dir.glob("*_cbn.json"))
                if cbn_files:
                    with open(cbn_files[0], 'r') as f:
                        cbns = json.load(f)
                        all_cbns.extend(cbns)
        
        # Save to both output directory and main results directory
        combined_file = output_dir / "all_participants_cbn.json"
        with open(combined_file, 'w') as f:
            json.dump(all_cbns, f, indent=2)
        
        # Also save to main results directory for easy access
        main_results_dir = root_dir / "playground" / "t2" / "results"
        main_results_dir.mkdir(exist_ok=True)
        main_cbn_file = main_results_dir / "motif_based_cbn.json"
        with open(main_cbn_file, 'w') as f:
            json.dump(all_cbns, f, indent=2)
        
        print(f"  ✓ Created combined CBN file: {combined_file.relative_to(root_dir)}")
        print(f"  ✓ Created main CBN file: {main_cbn_file.relative_to(root_dir)}")
        print(f"  ✓ Total participants in combined file: {len(all_cbns)}")

if __name__ == "__main__":
    main()
