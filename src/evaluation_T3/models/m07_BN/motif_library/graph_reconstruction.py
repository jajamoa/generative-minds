"""
Motif-Based Causal Graph Reconstruction

This module implements an algorithm to reconstruct causal graphs using motifs as building blocks.
The reconstruction starts from a seed node (typically upzoning_stance) and grows the graph
by iteratively adding best-matching motifs.
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Set, List, Tuple, Optional, Union
from motif_library import MotifLibrary, get_demographic_statistics
from semantic_similarity import SemanticSimilarityEngine
import os
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import argparse
import json


class MotifBasedReconstructor:
    def __init__(
        self,
        motif_library: MotifLibrary,
        similarity_threshold: float = 0.4,
        node_merge_threshold: float = 0.8,
        target_demographic: Union[str, Dict, List] = None,
        demographic_weight: float = 0.3,
    ):
        """
        Initialize the reconstructor.

        Args:
            motif_library: Loaded MotifLibrary instance
            similarity_threshold: Threshold for semantic similarity matching
            node_merge_threshold: Threshold for merging similar nodes
            target_demographic: Target demographic(s) - can be string, dict, or list
            demographic_weight: Weight for demographic scoring
        """
        self.motif_library = motif_library
        self.similarity_threshold = similarity_threshold
        self.node_merge_threshold = node_merge_threshold
        self.similarity_engine = SemanticSimilarityEngine(use_wordnet=True)
        self.target_demographic = target_demographic
        self.demographic_weight = demographic_weight
        self.demographic_scores = self._calculate_demographic_scores()

    def _calculate_demographic_scores(self):
        """Calculate demographic similarity scores with finer-grained feature matching."""
        scores = {}

        # if no demographic information, return empty dict
        if not self.motif_library.demographic_distribution:
            return scores

        # if no target demographic or format is wrong, return empty dict
        if not self.target_demographic or not isinstance(self.target_demographic, dict):
            return scores

        # get all features of target demographic
        target_features = set()
        for key, value in self.target_demographic.items():
            if isinstance(value, (list, set)):
                target_features.update(value)
            else:
                target_features.add(str(value))

        for demo, count in self.motif_library.demographic_distribution.items():
            frequency_score = 1.0 - (
                count / sum(self.motif_library.demographic_distribution.values())
            )
            demo_features = set(str(demo).split("_"))
            matching_features = len(target_features.intersection(demo_features))
            feature_score = matching_features / max(
                len(target_features), len(demo_features)
            )
            scores[demo] = 0.3 * frequency_score + 0.7 * feature_score

        return scores

    def _demographic_similarity(self, motif: nx.DiGraph, target_demo: Dict) -> float:
        """
        Calculate demographic similarity with stricter matching
        """
        if not target_demo:
            return 0.0

        motif_demo = motif.graph.get("motif_metadata", {}).get("demographics", {})
        if not motif_demo:
            return 0.0

        # 定义关键特征及其权重
            key_features = {
            "householder type": 0.25,  # 增加房主/租户状态的权重
            "means of transportation": 0.2,
            "income": 0.2,
            "has children under 18": 0.15,
            "Geo Mobility": 0.2,
        }

        total_score = 0
        total_weight = 0

        for feature, weight in key_features.items():
            target_value = str(target_demo.get(feature, "")).lower()
            motif_value = str(motif_demo.get(feature, "")).lower()

            if feature == "income":
                # 收入匹配使用范围比较
                score = self._compare_income_ranges(target_value, motif_value)
            elif feature == "has children under 18":
                # 布尔值精确匹配
                score = 1.0 if target_value == motif_value else 0.0
        else:
                # 其他特征使用字符串相似度
                score = 1.0 if target_value == motif_value else 0.0
                if score == 0.0 and (
                    target_value in motif_value or motif_value in target_value
                ):
                    score = 0.5

        total_score += score * weight
        total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _compare_income_ranges(self, target_income: str, motif_income: str) -> float:
        """
        Compare income ranges and return similarity score
        """
        income_levels = [
            "0-$24,999",
            "$25,000-$49,999",
            "$50,000-$74,999",
            "$75,000-$99,999",
            "$100,000-$124,999",
            "$125,000-$149,999",
            "$150,000-$199,999",
            ">$200,000",
        ]

        try:
            target_idx = next(
                i for i, level in enumerate(income_levels) if level in target_income
            )
            motif_idx = next(
                i for i, level in enumerate(income_levels) if level in motif_income
            )
            return 1.0 - abs(target_idx - motif_idx) / len(income_levels)
        except:
            return 0.0

    def find_motif_candidates(
        self, frontier_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str]]:
        """
        Find candidate motifs that could extend from a frontier node.
        Now includes semantic matching for finding more relevant motifs.
        """
        candidates = []

        # Get frontier node label
        frontier_label = current_graph.nodes[frontier_node].get("label", frontier_node)

        # Search through all motif groups in library
        for group_key, motifs in self.motif_library.semantic_motifs.items():
            for motif in motifs:
                # Try matching each node in the motif to the frontier node
                for motif_node in motif.nodes():
                    motif_label = motif.nodes[motif_node].get("label", "")

                    # Calculate semantic similarity
                    similarity = self.similarity_engine.node_similarity(
                        motif_label, frontier_label
                    )

                    if similarity >= self.similarity_threshold:
                        # Calculate extendability score
                        extendability = self._calculate_extendability(
                            motif, motif_node, current_graph, frontier_node
                        )

                        # Combined score with higher weight on semantic similarity
                        score = 0.6 * similarity + 0.4 * extendability
                        candidates.append((motif, score, motif_node))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates

    def _calculate_extendability(
        self,
        motif: nx.DiGraph,
        motif_node: str,
        current_graph: nx.DiGraph,
        frontier_node: str,
    ) -> float:
        """
        Calculate how extendable a motif is from the frontier node.

        Returns:
            Float score between 0-1 indicating extendability
        """
        # Count potential new connections
        in_edges = motif.in_degree(motif_node)
        out_edges = motif.out_degree(motif_node)

        current_in = current_graph.in_degree(frontier_node)
        current_out = current_graph.out_degree(frontier_node)

        # Score based on how many new edges could be added
        potential_new_edges = max(0, in_edges - current_in) + max(
            0, out_edges - current_out
        )

        # Normalize score
        max_possible = motif.number_of_edges()
        return potential_new_edges / max_possible if max_possible > 0 else 0

    def _should_merge_nodes(self, label1: str, label2: str) -> bool:
        """
        Determine if two nodes should be merged based on semantic similarity.
        """
        similarity = self.similarity_engine.node_similarity(label1, label2)
        return similarity >= self.node_merge_threshold

    def _find_mergeable_node(self, G: nx.DiGraph, new_label: str) -> Optional[str]:
        """
        Find existing node that can be merged with the new node.
        """
        for node in G.nodes():
            existing_label = G.nodes[node].get("label", "")
            if self._should_merge_nodes(new_label, existing_label):
                return node
        return None

    def _integrate_motif(
        self,
        G: nx.DiGraph,
        motif: nx.DiGraph,
        anchor_node: str,
        covered_nodes: Set[str],
    ) -> None:
        """
        Integrate a motif into the growing graph with node merging.
        """
        # Create mapping for node integration
        node_mapping = {}

        # First, handle the anchor node
        node_mapping[anchor_node] = anchor_node

        # Process each node in the motif
        for node in motif.nodes():
            if node in node_mapping:
                continue

            new_label = motif.nodes[node].get("label", "")

            # Try to find a mergeable node
            mergeable_node = self._find_mergeable_node(G, new_label)

            if mergeable_node:
                # Merge with existing node
                node_mapping[node] = mergeable_node
            else:
                # Add as new node
                G.add_node(node, label=new_label)
                node_mapping[node] = node

        # Add edges using the node mapping
        for u, v in motif.edges():
            mapped_u = node_mapping[u]
            mapped_v = node_mapping[v]
            if not G.has_edge(mapped_u, mapped_v):
                G.add_edge(mapped_u, mapped_v)

        # Update covered nodes
        covered_nodes.update(node_mapping.values())

    def reconstruct_graph(
        self, target_node: str, max_iterations: int = 100, min_score: float = 0.3
    ) -> nx.DiGraph:
        """
        Reconstruct graph based on motifs with demographic consideration
        """
        print("Starting graph reconstruction...")
        G = nx.DiGraph()
        G.add_node(target_node, label=target_node)
        covered_nodes = {target_node}
        frontier = {target_node}

        # 跟踪已使用的 motif 类型
        used_motif_types = set()

        for iteration in range(max_iterations):
            if not frontier:
                break

            print(f"Iteration {iteration}, Frontier size: {len(frontier)}")
            new_frontier = set()

            for node in frontier:
                if node not in G:
                    continue

                candidates = self.find_motif_candidates_reverse(node, covered_nodes, G)

                # 按照 demographic similarity 对候选进行分组
                demo_groups = {}
                for motif, score, motif_node, motif_type, demo_sim in candidates:
                    if demo_sim > 0.6:  # 只考虑 demographic similarity 较高的候选
                        if motif_type not in demo_groups:
                            demo_groups[motif_type] = []
                        demo_groups[motif_type].append(
                            (motif, score, motif_node, demo_sim)
                        )

                # 优先选择未使用的 motif 类型
                selected_candidates = []
                for motif_type, group_candidates in demo_groups.items():
                    if (
                        motif_type not in used_motif_types
                        and len(selected_candidates) < 2
                    ):
                        # 从该类型中选择最高分的候选
                        best_candidate = max(group_candidates, key=lambda x: x[1])
                        selected_candidates.append(best_candidate)
                        used_motif_types.add(motif_type)

                # 如果还需要更多候选，从所有高分候选中选择
                if len(selected_candidates) < 2:
                    remaining_candidates = [
                        c
                        for type_candidates in demo_groups.values()
                        for c in type_candidates
                        if c not in selected_candidates
                    ]
                    remaining_candidates.sort(key=lambda x: x[1], reverse=True)
                    selected_candidates.extend(
                        remaining_candidates[: 2 - len(selected_candidates)]
                    )

                # 集成选中的候选
                for motif, score, motif_node, _ in selected_candidates:
                    if score >= min_score:
                        node_mapping = self._integrate_motif_reverse(
                            G, motif, node, motif_node, covered_nodes
                        )

                        # 更新 frontier
                        for original_node in motif.nodes():
                            if original_node != motif_node:
                                mapped_node = node_mapping.get(original_node)
                                if mapped_node and mapped_node not in covered_nodes:
                                    new_frontier.add(mapped_node)

            frontier = new_frontier
            print(f"Graph size: {len(G.nodes())} nodes, {len(G.edges())} edges")

        # 最终清理
        self._clean_up_graph(G, target_node)
        return G

    def _clean_up_graph(self, G: nx.DiGraph, target_node: str) -> None:
        """
        Clean up the final graph
        """
        # 1. 移除 target_node 的出边
        outgoing = list(G.out_edges(target_node))
        for u, v in outgoing:
            G.remove_edge(u, v)

        # 2. 确保所有节点都有到 target_node 的路径
        for node in G.nodes():
            if node != target_node and not nx.has_path(G, node, target_node):
                G.add_edge(node, target_node, modifier=1.0)

        # 3. 移除重复的边
        edges_to_remove = []
        for u, v in G.edges():
            if G.has_edge(v, u):
                # 保留修饰符值较大的边
                uv_modifier = abs(G.edges[u, v].get("modifier", 1.0))
                vu_modifier = abs(G.edges[v, u].get("modifier", 1.0))
                if uv_modifier < vu_modifier:
                    edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((v, u))

        for edge in edges_to_remove:
            if G.has_edge(*edge):
                G.remove_edge(*edge)

    def find_motif_candidates_reverse(
        self, target_node: str, covered_nodes: Set[str], current_graph: nx.DiGraph
    ) -> List[Tuple[nx.DiGraph, float, str, str, float]]:
        """
        Find candidate motifs where target node is the outcome.
        Now prioritizes motifs based on demographic matching.
        """
        candidates = []
        target_label = current_graph.nodes[target_node].get("label", target_node)

        print(f"Finding candidates for target node: {target_label}")

        if not hasattr(self.motif_library, "motifs") or not self.motif_library.motifs:
            print("Warning: Motif library is empty!")
            return candidates

        # 根据 target demographic 确定优先级
        priority_types = self._get_priority_motif_types()
        print(f"Priority motif types for current demographics: {priority_types}")

        # 遍历 motifs
        for motif in self.motif_library.motifs:
            try:
                motif_type = motif["metadata"]["motif_type"]
                # 计算 motif type 的优先级权重
                type_weight = self._calculate_type_weight(motif_type, priority_types)
                
                # 构建 motif 图
                motif_graph = nx.DiGraph()
                for node in motif["nodes"]:
                    motif_graph.add_node(node, label=motif["node_labels"][node])
                for edge in motif["edges"]:
                    if len(edge) >= 3:
                        motif_graph.add_edge(edge[0], edge[1], **edge[2])
                    else:
                        motif_graph.add_edge(edge[0], edge[1], modifier=1.0)

                motif_graph.graph["motif_metadata"] = motif["metadata"]

                # 找到 motif 中连接到 stance 的节点
                stance_nodes = []
                for node in motif_graph.nodes():
                    if motif_graph.nodes[node].get("label", "") == target_label:
                        stance_nodes.append(node)

                if stance_nodes:
                    # 计算 demographic similarity
                    demo_sim = self._demographic_similarity(motif_graph, self.target_demographic)

                    # 计算整体分数，加入 type_weight
                    semantic_sim = self._calculate_semantic_similarity(motif_graph, current_graph)
                    structural_score = self._calculate_motif_structural_fit(motif_graph, current_graph)
                    coverage_score = self._calculate_coverage_score(motif_graph, covered_nodes)

                    final_score = (
                        0.2 * semantic_sim +
                        0.2 * structural_score +
                        0.2 * coverage_score +
                        0.2 * demo_sim +
                        0.2 * type_weight  # 加入 type_weight
                    )

                    for stance_node in stance_nodes:
                        candidates.append(
                            (motif_graph, final_score, stance_node, motif_type, demo_sim)
                        )

            except Exception as e:
                print(f"Error processing motif: {e}")
                continue

        # 按照分数和类型进行排序
        candidates.sort(key=lambda x: (x[1], priority_types.get(x[3], 0)), reverse=True)
        
        print(f"\nTop candidates by type:")
        for motif_type in set(c[3] for c in candidates):
            type_candidates = [c for c in candidates if c[3] == motif_type]
            if type_candidates:
                print(f"{motif_type}: Score = {type_candidates[0][1]:.3f}")

        return candidates

    def _get_priority_motif_types(self) -> Dict[str, float]:
        """
        Based on target demographics, determine which motif types should be prioritized
        """
        priorities = {}
        
        if not self.target_demographic:
            return priorities

        # 根据人口统计特征设置优先级
        if self.target_demographic.get("householder type") == "Owner-occupied":
            priorities["Economic_Owner"] = 1.0
            priorities["Family"] = 0.8
            priorities["Transportation_Car"] = 0.7
        elif self.target_demographic.get("householder type") == "Renter":
            priorities["Economic_Renter"] = 1.0
            priorities["Career"] = 0.8
            priorities["Transportation_Transit"] = 0.7

        if self.target_demographic.get("has children under 18"):
            priorities["Family"] = max(priorities.get("Family", 0), 0.9)

        if self.target_demographic.get("means of transportation") == "Public Transportation":
            priorities["Transportation_Transit"] = max(priorities.get("Transportation_Transit", 0), 0.9)
        elif self.target_demographic.get("means of transportation") == "Car / Truck / Van":
            priorities["Transportation_Car"] = max(priorities.get("Transportation_Car", 0), 0.9)

        # 收入相关优先级
        income = self.target_demographic.get("income", "")
        if any(high in income for high in [">$200,000", "$150,000"]):
            priorities["Economic_Owner"] = max(priorities.get("Economic_Owner", 0), 0.9)
        elif any(low in income for low in ["$50,000", "$75,000"]):
            priorities["Economic_Renter"] = max(priorities.get("Economic_Renter", 0), 0.9)

        return priorities

    def _calculate_type_weight(self, motif_type: str, priority_types: Dict[str, float]) -> float:
        """
        Calculate weight for a motif type based on demographic priorities
        """
        return priority_types.get(motif_type, 0.1)

    def _calculate_motif_structural_fit(
        self, motif: nx.DiGraph, current_graph: nx.DiGraph
    ) -> float:
        """
        Calculate how well the entire motif structure fits with the current graph
        """
        # 计算节点重叠
        motif_nodes = set(motif.nodes())
        graph_nodes = set(current_graph.nodes())
        overlap_ratio = len(motif_nodes & graph_nodes) / len(motif_nodes)

        # 计算边模式相似度
        motif_edge_count = motif.number_of_edges()
        graph_edge_count = current_graph.number_of_edges()
        edge_ratio = abs(motif_edge_count - graph_edge_count) / max(
            motif_edge_count, graph_edge_count
        )

        # 综合分数
        return 0.7 * (1 - overlap_ratio) + 0.3 * (1 - edge_ratio)

    def _calculate_coverage_score(
        self,
        motif: nx.DiGraph,
        covered_nodes: Set[str],
    ) -> float:
        """
        Calculate coverage score for reverse building.
        """
        # Count nodes in motif that are not covered
        uncovered_nodes = set(motif.nodes()) - covered_nodes
        uncovered_count = len(uncovered_nodes)

        # Normalize score
        max_possible = motif.number_of_nodes()
        return uncovered_count / max_possible if max_possible > 0 else 0

    def _integrate_motif_reverse(
        self,
        G: nx.DiGraph,
        motif: nx.DiGraph,
        target_node: str,
        motif_node: str,
        covered_nodes: Set[str],
    ) -> Dict[str, str]:
        """
        Integrate a motif into the graph while ensuring path connectivity to target node.
        Returns the node mapping dictionary for frontier updates.
        """
        # 创建节点映射
        node_mapping = {}
        node_mapping[motif_node] = target_node

        # 第一步：映射所有节点
        for node in motif.nodes():
            if node in node_mapping:
                    continue

            new_label = motif.nodes[node].get("label", "").replace(" ", "_").lower()

            # 尝试找到可合并的节点
            mergeable = None
            max_similarity = 0

            for existing_node in G.nodes():
                if existing_node == target_node:
                    continue
                existing_label = G.nodes[existing_node].get("label", "").lower()
                similarity = self.similarity_engine.node_similarity(
                    new_label, existing_label
                )

                if (
                    similarity > max_similarity
                    and similarity >= self.node_merge_threshold
                ):
                    max_similarity = similarity
                    mergeable = existing_node

            if mergeable:
                node_mapping[node] = mergeable
            else:
                new_node = f"n{len(G.nodes()) + 1}"
                G.add_node(new_node, label=new_label)
                node_mapping[node] = new_node

        # 第二步：添加边（保持修饰符）
        for u, v, data in motif.edges(data=True):
            mapped_u = node_mapping[u]
            mapped_v = node_mapping[v]

            # 跳过如果是从 target_node 出发的边
            if mapped_u == target_node:
                continue

            # 检查是否已存在相反的边
            if G.has_edge(mapped_v, mapped_u):
                continue

            # 添加边及其修饰符
            modifier = data.get("modifier", 1.0)
            G.add_edge(mapped_u, mapped_v, modifier=modifier)

        # 更新覆盖的节点
        covered_nodes.update(node_mapping.values())

        return node_mapping

    def update_frontier(self, G: nx.DiGraph, covered_nodes: Set[str]) -> Set[str]:
        """
        Update frontier to include only nodes that could have incoming edges.
        """
        frontier = set()

        # Add nodes that have incoming edges but might accept more
        for node in covered_nodes:
            # Only add to frontier if the node isn't upzoning_stance
            if node != "upzoning_stance":
                # Look for nodes that point to current node
                predecessors = set(G.predecessors(node))
                frontier.update(predecessors - covered_nodes)

        return frontier

    def save_as_json(
        self, G: nx.DiGraph, output_dir: str, filename: str = "reconstructed_graph.json"
    ) -> str:
        """
        Save the reconstructed graph as a JSON file.

        Args:
            G: Reconstructed NetworkX DiGraph
            output_dir: Directory to save the file
            filename: Output filename

        Returns:
            Path to saved JSON file
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Create JSON structure
        graph_data = {
            "nodes": {},
            "edges": {},
            "metadata": {
                "description": "Reconstructed causal graph from motifs",
                "node_count": len(G.nodes()),
                "edge_count": len(G.edges()),
            },
        }

        # Add nodes
        for i, node in enumerate(G.nodes()):
            node_id = str(node)
            graph_data["nodes"][node_id] = {
                "id": node_id,
                "label": G.nodes[node].get("label", str(node)),
                "type": "concept",
                "in_degree": G.in_degree(node),
                "out_degree": G.out_degree(node),
            }

        # Add edges with modifiers
        for i, (source, target) in enumerate(G.edges()):
            edge_id = f"e{i}"
            edge_data = G.edges[source, target]
            graph_data["edges"][edge_id] = {
                "id": edge_id,
                "source": str(source),
                "target": str(target),
                "type": "causal",
                "modifier": edge_data.get(
                    "modifier", 1.0
                ),  # Default to positive influence
            }

        # Save to file
        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)

        return output_path

    def save_as_mmd(
        self, G: nx.DiGraph, output_dir: str, filename: str = "reconstructed_graph.mmd"
    ) -> str:
        """
        Save the reconstructed graph as a Mermaid markdown file.
        Includes positive/negative causal relationships and link styling.
        """
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)

        # Create Mermaid markdown content
        md_lines = ["```mermaid"]
        mmd_lines = ["flowchart TD"]

        # Add nodes with labels
        for node in G.nodes():
            label = G.nodes[node].get("label", str(node))
            # Clean label for Mermaid compatibility
            clean_label = label.replace(" ", "_").replace("-", "_")
            mmd_lines.append(f"    {node}[{clean_label}]")

        # Add edges with different styles for positive/negative relationships
        edge_styles = []
        for i, (source, target) in enumerate(G.edges()):
            # For demonstration, we'll alternate between positive and negative relationships
            # In a real implementation, this should be based on edge attributes
            if i % 2 == 0:
                # Positive causal relationship
                mmd_lines.append(f"    {source} --> {target}")
                edge_styles.append("stroke:#00AA00,stroke-width:2px")
            else:
                # Negative causal relationship
                mmd_lines.append(f"    {source} --x {target}")
                edge_styles.append("stroke:#FF0000,stroke-dasharray:3,stroke-width:2px")

        # Add metadata as comments
        mmd_lines.insert(1, "%% Reconstructed Causal Graph")
        mmd_lines.insert(2, f"%% Nodes: {len(G.nodes())}")
        mmd_lines.insert(3, f"%% Edges: {len(G.edges())}")

        # Add link styles
        for i, style in enumerate(edge_styles):
            mmd_lines.append(f"    linkStyle {i} {style}")

        md_lines = md_lines + mmd_lines + ["```"]

        # Save both .mmd and .md files
        with open(output_path, "w") as f:
            f.write("\n".join(mmd_lines))

        # Save markdown version
        md_path = output_path.replace(".mmd", ".md")
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        return output_path


def main(args):
    """
    Example usage of the improved motif-based reconstruction.
    """
    print("\nInitializing reconstruction process...")
    print(f"Using motif library from: {args.motif_library}")

    # 检查文件是否存在
    if not os.path.exists(args.motif_library):
        print(f"Error: Motif library file not found at {args.motif_library}")
        return

    # 直接加载 JSON 文件
    try:
        with open(args.motif_library, "r") as f:
            library_data = json.load(f)
            print(
                f"Successfully loaded motif library with {len(library_data.get('motifs', []))} motifs"
            )
    except Exception as e:
        print(f"Error loading motif library: {e}")
        return

    # 创建 MotifLibrary 实例
    library = MotifLibrary(
        min_motif_size=library_data.get("min_motif_size", 3),
        max_motif_size=library_data.get("max_motif_size", 5),
        min_semantic_similarity=library_data.get("min_semantic_similarity", 0.4),
    )

    # 直接设置 motifs
    library.motifs = library_data.get("motifs", [])

    print(f"\nMotif Library Statistics:")
    print(f"- Number of motifs: {len(library.motifs)}")
    print(f"- Min motif size: {library.min_motif_size}")
    print(f"- Max motif size: {library.max_motif_size}")
    print(f"- Semantic similarity threshold: {library.min_semantic_similarity}")

    # 设置 target demographic
    print("\nTarget Demographic Information:")
    for key, value in args.target_demographic.items():
        print(f"- {key}: {value}")

    # 创建 reconstructor
    reconstructor = MotifBasedReconstructor(
        library,
        similarity_threshold=0.3,
        node_merge_threshold=0.8,
        target_demographic=args.target_demographic,
        demographic_weight=0.3,
    )

    print("\nStarting graph reconstruction...")
    # 重建图
    reconstructed_graph = reconstructor.reconstruct_graph(
        args.seed_node, max_iterations=args.max_iterations, min_score=0.3
    )

    # 分析结果
    print("\nReconstruction Results:")
    print(f"- Total nodes: {len(reconstructed_graph.nodes())}")
    print(f"- Total edges: {len(reconstructed_graph.edges())}")

    # 输出节点信息
    print("\nNode Information:")
    for node in reconstructed_graph.nodes():
        label = reconstructed_graph.nodes[node].get("label", "")
        in_degree = reconstructed_graph.in_degree(node)
        out_degree = reconstructed_graph.out_degree(node)
        print(f"- {node}: {label} (in: {in_degree}, out: {out_degree})")

    # 创建可视化
    print("\nGenerating visualizations...")

    # 保存为 JSON
    json_path = reconstructor.save_as_json(
        reconstructed_graph, args.output_dir, "reconstructed_graph.json"
    )
    print(f"- JSON saved to: {json_path}")

    # 保存为 MMD
    mmd_path = reconstructor.save_as_mmd(
        reconstructed_graph, args.output_dir, "reconstructed_graph.mmd"
    )
    print(f"- MMD saved to: {mmd_path}")


if __name__ == "__main__":
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Motif-based graph reconstruction")

    # 添加参数
    parser.add_argument(
        "--seed_node",
        type=str,
        default="support_for_upzoning",
        help="Starting node for reconstruction",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=10, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(CURRENT_DIR, "graph_reconstruction"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--motif_library",
        type=str,
        default=os.path.join(CURRENT_DIR, "motif_library.json"),
        help="Path to motif library JSON file",
    )

    args = parser.parse_args()

    # 设置示例 target demographic
    # args.target_demographic = {
    #     "age": 28,
    #     "Geo Mobility": "Different house in United States 1 year ago",
    #     "householder type": "Renter",
    #     "Gross rent": "25.0 to 29.9 percent",
    #     "means of transportation": "Public Transportation",
    #     "income": "$50,000-$74,999",
    #     "occupation": "Management, business, science, and arts occupations",
    #     "marital status": "Never Married",
    #     "has children under 18": False,
    #     "children age range": "No Children",
    #     "ZIP code": 94105,
    # }

    args.target_demographic = {
        "age": 45,
        "Geo Mobility": "Same house 1 year ago",
        "householder type": "Owner-occupied",
        "Gross rent": "Not computed",
        "means of transportation": "Car / Truck / Van",
        "income": ">$200,000",
        "occupation": "Sales and office occupations",
        "marital status": "Now Married",
        "has children under 18": True,
        "children age range": "6 to 17 years old",
        "ZIP code": 92103,
    }

    main(args)
