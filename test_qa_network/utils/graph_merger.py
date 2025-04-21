from .llm_interface import LLMInterface

class GraphMerger:
    """Handles node merging operations in the graph"""
    def __init__(self, llm_interface):
        """
        Initialize GraphMerger with LLM interface
        Args:
            llm_interface: Instance of LLMInterface for node analysis
        """
        self.llm = llm_interface
    
    def check_and_merge_nodes(self, nodes, edges, qa_history):
        """
        Check and merge similar nodes using LLM
        Args:
            nodes: Dictionary of nodes
            edges: Dictionary of edges
            qa_history: Dictionary of QA history
        Returns:
            Tuple of (updated_nodes, updated_edges, updated_qa_history)
        """
        if len(nodes) <= 1:
            return nodes, edges, qa_history
            
        # Get merge suggestions from LLM
        merge_groups = self.llm.analyze_nodes_for_merging(nodes)
        
        if not merge_groups:
            return nodes, edges, qa_history
            
        # Copy original data
        new_nodes = nodes.copy()
        new_edges = edges.copy()
        new_qa_history = qa_history.copy()
        
        # Process merges
        id_mapping = self._process_merge_groups(merge_groups, new_nodes)
        new_edges = self._update_edges(new_edges, id_mapping, new_nodes)
        new_qa_history = self._update_qa_history(new_qa_history, id_mapping, new_nodes)
        
        return new_nodes, new_edges, new_qa_history
    
    def _process_merge_groups(self, merge_groups, nodes):
        """
        Process merge groups and return id mapping
        Args:
            merge_groups: List of merge groups from LLM
            nodes: Dictionary of nodes
        Returns:
            Dictionary mapping old node IDs to new node IDs
        """
        id_mapping = {}
        
        for group in merge_groups:
            node_ids = [nid for nid in group["node_ids"] if nid in nodes]
            if len(node_ids) < 2:
                continue
            
            # Keep the first node ID as base and update its data
            base_node_id = node_ids[0]
            nodes_to_merge = node_ids[1:]
            
            # Update base node's label and collect QA sources
            nodes[base_node_id]["label"] = group["merged_label"]
            for node_id in nodes_to_merge:
                if node_id in nodes:
                    nodes[base_node_id]["source_qa"].extend(nodes[node_id]["source_qa"])
                    id_mapping[node_id] = base_node_id
            
            nodes[base_node_id]["source_qa"] = list(set(nodes[base_node_id]["source_qa"]))
            
            print(f"Merging nodes {nodes_to_merge} into {base_node_id}: {group['merged_label']}")
            print(f"Reason: {group['reason']}")
            
            # Remove merged nodes
            for node_id in nodes_to_merge:
                if node_id in nodes:
                    del nodes[node_id]
        
        return id_mapping
    
    def _update_edges(self, edges, id_mapping, nodes):
        """
        Update edges with new node IDs
        Args:
            edges: Dictionary of edges
            id_mapping: Dictionary mapping old node IDs to new node IDs
            nodes: Dictionary of nodes
        Returns:
            Updated edges dictionary
        """
        updated_edges = {}
        
        for edge_id, edge_data in edges.items():
            source = edge_data["source"]
            target = edge_data["target"]
            
            new_source = id_mapping.get(source, source)
            new_target = id_mapping.get(target, target)
            
            if new_source not in nodes or new_target not in nodes:
                continue
                
            if new_source == new_target:
                continue
                
            updated_edges[edge_id] = {
                **edge_data,
                "source": new_source,
                "target": new_target
            }
            
            if edge_id not in nodes[new_source]["outgoing_edges"]:
                nodes[new_source]["outgoing_edges"].append(edge_id)
            if edge_id not in nodes[new_target]["incoming_edges"]:
                nodes[new_target]["incoming_edges"].append(edge_id)
        
        # Clean up edge lists
        for node_id in nodes:
            nodes[node_id]["incoming_edges"] = [
                e for e in nodes[node_id]["incoming_edges"] 
                if e in updated_edges
            ]
            nodes[node_id]["outgoing_edges"] = [
                e for e in nodes[node_id]["outgoing_edges"]
                if e in updated_edges
            ]
        
        return updated_edges
    
    def _update_qa_history(self, qa_history, id_mapping, nodes):
        """
        Update QA history with new node IDs
        Args:
            qa_history: Dictionary of QA history
            id_mapping: Dictionary mapping old node IDs to new node IDs
            nodes: Dictionary of nodes
        Returns:
            Updated QA history dictionary
        """
        for qa_id, qa_data in qa_history.items():
            updated_pairs = []
            for pair in qa_data.get("extracted_pairs", []):
                source = pair.get("source")
                target = pair.get("target")
                
                # Map to new IDs if nodes were merged
                new_source = id_mapping.get(source, source)
                new_target = id_mapping.get(target, target)
                
                # Only keep pairs where both nodes still exist
                if new_source in nodes and new_target in nodes:
                    updated_pairs.append({
                        **pair,
                        "source": new_source,
                        "target": new_target,
                        "merged": bool(source in id_mapping or target in id_mapping)
                    })
            
            qa_history[qa_id]["extracted_pairs"] = updated_pairs
            
            # Add merge info to QA history
            if any(p.get("merged") for p in updated_pairs):
                qa_history[qa_id]["merge_info"] = {
                    "merged_nodes": {old: new for old, new in id_mapping.items()
                                  if old in [p.get("source") for p in qa_data["extracted_pairs"]] or
                                     old in [p.get("target") for p in qa_data["extracted_pairs"]]}
                }
        
        return qa_history 