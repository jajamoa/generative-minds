#!/usr/bin/env python3
"""
Extract common motifs from transcripts using minimal engineering.
This script analyzes motifs from all transcripts, finds common patterns,
and generates visualizations for each pattern type.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional
import hashlib
import argparse

# Import node similarity functions
from node_similarity import compute_node_similarity, find_similar_node


class MotifPattern:
    """Represents a motif pattern type"""

    CHAIN = "chain"  # A -> B -> C
    FORK = "fork"  # A -> {B, C}
    COLLIDER = "collider"  # {A, B} -> C

    @staticmethod
    def detect_pattern(nodes: List[str], edges: List[List[str]]) -> Optional[str]:
        """Detect the pattern type of a motif"""
        if len(nodes) != 3 or len(edges) < 2:
            return None

        # Build adjacency info
        out_edges = defaultdict(set)
        in_edges = defaultdict(set)

        for src, dst in edges:
            out_edges[src].add(dst)
            in_edges[dst].add(src)

        # Check for chain: A -> B -> C
        for b in nodes:
            if len(in_edges[b]) == 1 and len(out_edges[b]) == 1:
                a = list(in_edges[b])[0]
                c = list(out_edges[b])[0]
                if a != c and a in nodes and c in nodes:
                    return MotifPattern.CHAIN

        # Check for fork: A -> {B, C}
        for a in nodes:
            if len(out_edges[a]) >= 2:
                targets = list(out_edges[a])
                if all(t in nodes for t in targets[:2]):
                    return MotifPattern.FORK

        # Check for collider: {A, B} -> C
        for c in nodes:
            if len(in_edges[c]) >= 2:
                sources = list(in_edges[c])
                if all(s in nodes for s in sources[:2]):
                    return MotifPattern.COLLIDER

        return None


class CommonMotif:
    """Represents a common motif pattern found across multiple transcripts"""

    def __init__(self, pattern_type: str, canonical_labels: Dict[str, str]):
        self.pattern_type = pattern_type
        self.canonical_labels = canonical_labels  # Canonical node labels
        self.instances = []  # List of (transcript_id, motif_data)
        self.node_clusters = defaultdict(list)  # Node position -> list of actual labels

    def add_instance(self, transcript_id: str, motif_data: dict):
        """Add an instance of this common motif"""
        self.instances.append((transcript_id, motif_data))

        # Track node label variations
        for node_id, label in motif_data["node_labels"].items():
            self.node_clusters[node_id].append(label)

    def get_frequency(self) -> int:
        """Get the frequency of this motif"""
        return len(self.instances)

    def get_transcript_count(self) -> int:
        """Get the number of unique transcripts this motif appears in"""
        return len(set(tid for tid, _ in self.instances))


def load_all_motifs(motifs_dir: Path) -> Dict[str, List[dict]]:
    """Load all motif data from the transcripts directory"""
    all_motifs = {}

    for subdir in motifs_dir.iterdir():
        if not subdir.is_dir():
            continue

        json_file = subdir / f"{subdir.name}.json"
        if json_file.exists():
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if "motifs" in data:
                        all_motifs[data["prolific_id"]] = data["motifs"]
            except Exception as e:
                print(f"Error loading {json_file}: {e}")

    return all_motifs


def normalize_motif_structure(motif: dict) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Normalize a motif to a canonical form for comparison.
    Returns (pattern_type, normalized_edges)
    """
    pattern_type = MotifPattern.detect_pattern(motif["nodes"], motif["edges"])
    if not pattern_type:
        return None, []

    # Normalize edges to canonical form based on pattern
    edges = [(e[0], e[1]) for e in motif["edges"]]

    if pattern_type == MotifPattern.CHAIN:
        # Find the chain order: start -> middle -> end
        out_degree = defaultdict(int)
        in_degree = defaultdict(int)
        for src, dst in edges:
            out_degree[src] += 1
            in_degree[dst] += 1

        start = [n for n in motif["nodes"] if in_degree[n] == 0][0]
        middle = [
            n for n in motif["nodes"] if in_degree[n] == 1 and out_degree[n] == 1
        ][0]
        end = [n for n in motif["nodes"] if out_degree[n] == 0][0]

        return (
            pattern_type,
            [("n1", "n2"), ("n2", "n3")],
            {start: "n1", middle: "n2", end: "n3"},
        )

    elif pattern_type == MotifPattern.FORK:
        # Find the source node
        out_degree = defaultdict(int)
        for src, dst in edges:
            out_degree[src] += 1

        source = max(motif["nodes"], key=lambda n: out_degree[n])
        targets = sorted([dst for src, dst in edges if src == source])[:2]

        node_map = {source: "n1", targets[0]: "n2", targets[1]: "n3"}
        return pattern_type, [("n1", "n2"), ("n1", "n3")], node_map

    elif pattern_type == MotifPattern.COLLIDER:
        # Find the target node
        in_degree = defaultdict(int)
        for src, dst in edges:
            in_degree[dst] += 1

        target = max(motif["nodes"], key=lambda n: in_degree[n])
        sources = sorted([src for src, dst in edges if dst == target])[:2]

        node_map = {sources[0]: "n1", sources[1]: "n2", target: "n3"}
        return pattern_type, [("n1", "n3"), ("n2", "n3")], node_map

    return None, []


def cluster_similar_motifs(
    all_motifs: Dict[str, List[dict]], similarity_threshold: float = 0.7
) -> List[CommonMotif]:
    """
    Cluster similar motifs across all transcripts.
    Uses node_similarity to match similar concepts.
    """
    # Group motifs by pattern type
    pattern_groups = defaultdict(list)

    for transcript_id, motifs in all_motifs.items():
        for motif in motifs:
            pattern_type, norm_edges, node_map = normalize_motif_structure(motif)
            if pattern_type:
                # Map labels to normalized positions
                norm_labels = {}
                for orig_id, norm_id in node_map.items():
                    norm_labels[norm_id] = motif["node_labels"][orig_id]

                pattern_groups[pattern_type].append(
                    {
                        "transcript_id": transcript_id,
                        "motif": motif,
                        "norm_labels": norm_labels,
                        "norm_edges": norm_edges,
                    }
                )

    # Cluster similar motifs within each pattern type
    common_motifs = []

    for pattern_type, motif_list in pattern_groups.items():
        # Use simple clustering based on node similarity
        clusters = []

        for motif_data in motif_list:
            norm_labels = motif_data["norm_labels"]

            # Find matching cluster
            best_cluster = None
            best_score = 0.0

            for cluster in clusters:
                # Calculate average similarity across all node positions
                total_sim = 0.0
                for node_id in ["n1", "n2", "n3"]:
                    if node_id in norm_labels and node_id in cluster.canonical_labels:
                        sim = compute_node_similarity(
                            norm_labels[node_id], cluster.canonical_labels[node_id]
                        )
                        total_sim += sim

                avg_sim = total_sim / 3
                if avg_sim > best_score and avg_sim >= similarity_threshold:
                    best_score = avg_sim
                    best_cluster = cluster

            if best_cluster:
                best_cluster.add_instance(
                    motif_data["transcript_id"], motif_data["motif"]
                )
            else:
                # Create new cluster
                new_cluster = CommonMotif(pattern_type, norm_labels)
                new_cluster.add_instance(
                    motif_data["transcript_id"], motif_data["motif"]
                )
                clusters.append(new_cluster)

        common_motifs.extend(clusters)

    # Sort by frequency
    common_motifs.sort(key=lambda m: m.get_frequency(), reverse=True)

    return common_motifs


def generate_mermaid_visualization(motif: CommonMotif) -> str:
    """Generate Mermaid diagram for a common motif"""
    lines = ["flowchart LR"]

    # Add nodes
    node_mapping = {"n1": "A", "n2": "B", "n3": "C"}

    for node_id, label in motif.canonical_labels.items():
        mermaid_id = node_mapping[node_id]
        # Escape special characters in labels
        safe_label = label.replace('"', '\\"')
        lines.append(f'{mermaid_id}["{safe_label}"]')

    # Add edges based on pattern type
    if motif.pattern_type == MotifPattern.CHAIN:
        lines.append("A --> B")
        lines.append("B --> C")
    elif motif.pattern_type == MotifPattern.FORK:
        lines.append("A --> B")
        lines.append("A --> C")
    elif motif.pattern_type == MotifPattern.COLLIDER:
        lines.append("A --> C")
        lines.append("B --> C")

    # Add style
    lines.append("")
    lines.append("style A fill:#e1f5fe")
    lines.append("style B fill:#e1f5fe")
    lines.append("style C fill:#e1f5fe")

    return "\n".join(lines)


def analyze_node_variations(motif: CommonMotif) -> Dict[str, List[Tuple[str, int]]]:
    """Analyze variations in node labels for each position"""
    variations = {}

    for node_id, labels in motif.node_clusters.items():
        # Count label frequencies
        label_counts = Counter(labels)

        # Sort by frequency
        sorted_labels = sorted(label_counts.items(), key=lambda x: -x[1])
        variations[node_id] = sorted_labels[:5]  # Top 5 variations

    return variations


def generate_report(common_motifs: List[CommonMotif], output_dir: Path):
    """Generate comprehensive report with visualizations"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics
    summary = {
        "total_common_motifs": len(common_motifs),
        "pattern_distribution": Counter(m.pattern_type for m in common_motifs),
        "motifs": [],
    }

    # Create visualizations for top motifs
    for i, motif in enumerate(common_motifs[:20]):  # Top 20 most common
        motif_id = f"motif_{i+1:03d}"

        # Generate motif data
        motif_data = {
            "id": motif_id,
            "pattern_type": motif.pattern_type,
            "canonical_labels": motif.canonical_labels,
            "frequency": motif.get_frequency(),
            "transcript_count": motif.get_transcript_count(),
            "node_variations": analyze_node_variations(motif),
            "example_transcripts": [tid for tid, _ in motif.instances[:5]],
        }

        summary["motifs"].append(motif_data)

        # Generate Mermaid visualization
        mermaid_content = generate_mermaid_visualization(motif)
        mermaid_file = output_dir / f"{motif_id}.mmd"
        with open(mermaid_file, "w") as f:
            f.write(mermaid_content)

        # Generate detailed report
        report_lines = [
            f"# Common Motif {i+1}",
            f"\n**Pattern Type**: {motif.pattern_type.title()}",
            f"\n**Frequency**: {motif.get_frequency()} instances across {motif.get_transcript_count()} transcripts",
            f"\n## Canonical Structure",
            f"```mermaid",
            mermaid_content,
            "```",
            f"\n## Node Variations",
        ]

        for node_id, variations in motif_data["node_variations"].items():
            report_lines.append(f"\n### Position {node_id}:")
            for label, count in variations:
                report_lines.append(f'- "{label}" ({count} times)')

        report_lines.append(f"\n## Example Transcripts")
        for tid in motif_data["example_transcripts"]:
            report_lines.append(f"- {tid}")

        report_file = output_dir / f"{motif_id}.md"
        with open(report_file, "w") as f:
            f.write("\n".join(report_lines))

    # Save summary JSON
    summary_file = output_dir / "common_motifs_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Generate main report
    main_report = [
        "# Common Motifs Analysis Report",
        f"\n## Summary",
        f"- Total common motifs found: {summary['total_common_motifs']}",
        f"- Pattern distribution:",
    ]

    for pattern, count in summary["pattern_distribution"].items():
        main_report.append(f"  - {pattern.title()}: {count}")

    main_report.append(f"\n## Top 20 Most Common Motifs")

    for i, motif_data in enumerate(summary["motifs"]):
        main_report.append(f"\n### {i+1}. {motif_data['pattern_type'].title()} Pattern")
        main_report.append(f"- Frequency: {motif_data['frequency']} instances")
        main_report.append(f"- Appears in {motif_data['transcript_count']} transcripts")
        main_report.append(
            f"- [Details](./{motif_data['id']}.md) | [Diagram](./{motif_data['id']}.mmd)"
        )

    main_report_file = output_dir / "README.md"
    with open(main_report_file, "w") as f:
        f.write("\n".join(main_report))

    print(f"Report generated in {output_dir}")
    print(f"- Main report: {main_report_file}")
    print(f"- Summary data: {summary_file}")
    print(f"- Individual motif reports and diagrams in {output_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Cluster and summarize common motifs across extracted results"
    )
    parser.add_argument(
        "--motifs_dir",
        default="results/motifs_from_bigtom",
        help="Directory containing per-participant motif folders (relative to script dir)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/common_motifs_analysis",
        help="Directory to write the common motifs analysis (relative to script dir)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.7,
        help="Threshold for clustering node label similarity (0..1)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    motifs_dir = base_dir / args.motifs_dir
    if not motifs_dir.exists():
        fallback = base_dir / "results" / "motifs_from_transcripts"
        if fallback.exists():
            print(f"Warning: {motifs_dir} not found. Falling back to {fallback}")
            motifs_dir = fallback
        else:
            raise FileNotFoundError(f"Motifs directory not found: {motifs_dir}")

    output_dir = base_dir / args.output_dir

    print(f"Loading motifs from: {motifs_dir}")
    all_motifs = load_all_motifs(motifs_dir)
    print(f"Loaded motifs from {len(all_motifs)} participants")

    print("\nClustering similar motifs...")
    common_motifs = cluster_similar_motifs(
        all_motifs, similarity_threshold=args.similarity_threshold
    )
    print(f"Found {len(common_motifs)} common motif patterns")

    print("\nGenerating report and visualizations...")
    generate_report(common_motifs, output_dir)

    # Print top 10 most common motifs
    print("\nTop 10 Most Common Motifs:")
    for i, motif in enumerate(common_motifs[:10]):
        print(
            f"{i+1}. {motif.pattern_type.title()} - "
            f"{motif.get_frequency()} instances in "
            f"{motif.get_transcript_count()} participants"
        )
        for node_id, label in sorted(motif.canonical_labels.items()):
            print(f"   {node_id}: {label}")
        print()


if __name__ == "__main__":
    main()
