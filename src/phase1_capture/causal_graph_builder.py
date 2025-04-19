"""
Causal Graph Builder - Extracts causal relationships from QA pairs and builds a directed acyclic graph (DAG).
"""

import os
import json
from pathlib import Path
import re
import openai
from dotenv import load_dotenv
import time
import networkx as nx
from collections import defaultdict

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

class CausalGraphBuilder:
    """
    A class to build causal graphs from question-answer pairs.
    Extracts causal relationships and organizes them into a DAG.
    """
    
    def __init__(self, processed_data_dir, output_dir=None):
        """
        Initialize Causal Graph Builder with directories for processed data and output.
        
        Args:
            processed_data_dir (str): Path to directory containing processed QA pairs
            output_dir (str, optional): Path to directory where causal graphs will be saved
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.processed_data_dir
        
        # Create output directory if it doesn't exist
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)
    
    def _extract_causal_relationships(self, qa_pairs, interview_name):
        """
        Extract causal relationships from QA pairs using LLM.
        
        Args:
            qa_pairs (list): List of QA pair dictionaries
            interview_name (str): Name of the interviewee
            
        Returns:
            list: List of dictionaries containing causal relationships
        """
        # Prepare system message
        system_message = """
        You are an AI specialized in identifying causal relationships from text.
        Your task is to extract cause-effect relationships from question-answer pairs.
        
        For each QA pair:
        1. Identify key concepts or factors mentioned
        2. Determine if there are causal relationships between these concepts
        3. Structure these as "cause" -> "effect" relationships
        4. Assign a confidence level (high, medium, low) to each relationship
        
        Some examples of causal relationships:
        - "Housing prices increased" -> "Teachers can't afford to live in the city"
        - "Growing up in a diverse neighborhood" -> "Developed appreciation for different cultures"
        - "Lack of affordable housing" -> "Essential workers live far away"
        
        Format your response as a valid JSON object with a key 'causal_relationships' that contains an array of objects.
        Each object should have:
        - 'cause': The factor or concept causing an effect
        - 'effect': The resulting effect
        - 'explanation': Brief explanation of this causal relationship
        - 'confidence': Confidence level (high, medium, low)
        - 'source_qa': Reference to the QA pair this was derived from (use QA pair's question)
        - 'stance_implications': How this relationship might inform a stance on housing policy
        
        Be precise about distinguishing correlation from causation, and only extract relationships with clear causal direction.
        """
        
        # Prepare QA pairs for the prompt
        qa_pairs_formatted = json.dumps(qa_pairs, indent=2)
        
        # Prepare prompt for LLM
        prompt = f"""
        Below are question-answer pairs extracted from an interview with {interview_name}. 
        
        Please identify causal relationships in these QA pairs. For each causal relationship:
        1. Clearly identify the cause (factor/action/condition)
        2. Clearly identify the effect (result/outcome/consequence)
        3. Provide a brief explanation of why this is causal
        4. Assess your confidence in this causal link
        5. Note which QA pair this relationship was derived from
        6. Describe how this might inform a stance on housing policy
        
        QA Pairs:
        {qa_pairs_formatted}
        """
        
        try:
            # Call OpenAI API
            print(f"Extracting causal relationships for {interview_name}...")
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            print(f"Received response, length: {len(content)} chars")
            
            # Save the raw response for debugging
            debug_dir = self.output_dir / interview_name / "debug"
            debug_dir.mkdir(exist_ok=True, parents=True)
            with open(debug_dir / "causal_relationships_response.json", "w", encoding="utf-8") as f:
                f.write(content)
            
            try:
                # Parse JSON response
                result = json.loads(content)
                
                # Check if result has 'causal_relationships' key
                if 'causal_relationships' in result:
                    relationships = result['causal_relationships']
                    print(f"Successfully extracted {len(relationships)} causal relationships from {interview_name}")
                    return relationships
                else:
                    # Try to find any key that contains an array of objects with cause/effect fields
                    for key, value in result.items():
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and 'cause' in value[0] and 'effect' in value[0]:
                                print(f"Found causal relationships under key '{key}' instead of 'causal_relationships'")
                                return value
                    
                    # If all else fails
                    print(f"Could not find causal relationships in standard format. Result keys: {list(result.keys())}")
                    return []
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response for {interview_name}: {str(e)}")
                return []
                
        except Exception as e:
            print(f"Error calling OpenAI API for {interview_name}: {str(e)}")
            return []
    
    def _build_graph(self, causal_relationships):
        """
        Build a directed graph from causal relationships.
        
        Args:
            causal_relationships (list): List of dictionaries containing causal relationships
            
        Returns:
            nx.DiGraph: NetworkX directed graph
        """
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges from causal relationships
        for rel in causal_relationships:
            cause = rel['cause']
            effect = rel['effect']
            
            # Add cause node if it doesn't exist
            if not G.has_node(cause):
                G.add_node(cause, type='factor')
            
            # Add effect node if it doesn't exist
            if not G.has_node(effect):
                G.add_node(effect, type='factor')
            
            # Add edge from cause to effect
            G.add_edge(cause, effect, 
                      explanation=rel.get('explanation', ''),
                      confidence=rel.get('confidence', 'medium'),
                      source_qa=rel.get('source_qa', ''),
                      stance_implications=rel.get('stance_implications', ''))
        
        return G
    
    def _connect_to_stance(self, G, interview_name):
        """
        Connect graph nodes to stance nodes based on stance implications.
        
        Args:
            G (nx.DiGraph): NetworkX directed graph
            interview_name (str): Name of the interviewee
            
        Returns:
            nx.DiGraph: Updated graph with stance connections
        """
        # Create stance nodes if they don't exist
        stance_nodes = [
            "Support more housing development",
            "Preserve neighborhood character",
            "Prioritize affordable housing",
            "Support housing for essential workers"
        ]
        
        # Prepare system message for LLM
        system_message = """
        You are an AI specialized in connecting causal relationships to policy stances.
        Your task is to analyze causal relationships and determine how they might inform housing policy stances.
        
        For each causal relationship, determine which stance(s) it might support or oppose, and why.
        
        Format your response as a valid JSON object with a key 'stance_connections' that contains an array of objects.
        Each object should have:
        - 'node': The factor that connects to a stance 
        - 'stance': The stance it connects to
        - 'connection_type': 'supports' or 'opposes'
        - 'explanation': Brief explanation of why this factor supports or opposes the stance
        """
        
        # Extract nodes and edges for the prompt
        nodes = list(G.nodes())
        edges = []
        for u, v, data in G.edges(data=True):
            edges.append({
                'cause': u,
                'effect': v,
                'explanation': data.get('explanation', ''),
                'stance_implications': data.get('stance_implications', '')
            })
        
        # Prepare prompt for LLM
        prompt = f"""
        Below are causal relationships extracted from an interview with {interview_name}.
        
        Please identify which housing policy stances these causal relationships would support or oppose.
        
        Stances to consider:
        - "Support more housing development"
        - "Preserve neighborhood character"
        - "Prioritize affordable housing"  
        - "Support housing for essential workers"
        
        Nodes (factors/concepts):
        {json.dumps(nodes, indent=2)}
        
        Causal Relationships:
        {json.dumps(edges, indent=2)}
        """
        
        try:
            # Call OpenAI API
            print(f"Connecting factors to stances for {interview_name}...")
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            print(f"Received stance connections, length: {len(content)} chars")
            
            # Save the raw response for debugging
            debug_dir = self.output_dir / interview_name / "debug"
            debug_dir.mkdir(exist_ok=True, parents=True)
            with open(debug_dir / "stance_connections_response.json", "w", encoding="utf-8") as f:
                f.write(content)
            
            # Add stance nodes to the graph
            for stance in stance_nodes:
                G.add_node(stance, type='stance')
            
            try:
                # Parse JSON response
                result = json.loads(content)
                
                # Check if result has 'stance_connections' key
                if 'stance_connections' in result:
                    connections = result['stance_connections']
                    
                    # Add edges from factors to stances
                    for conn in connections:
                        node = conn['node']
                        stance = conn['stance']
                        conn_type = conn['connection_type']
                        explanation = conn.get('explanation', '')
                        
                        if node in G.nodes() and stance in G.nodes():
                            G.add_edge(node, stance, 
                                      connection_type=conn_type,
                                      explanation=explanation)
                    
                    print(f"Added {len(connections)} stance connections to the graph")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response for stance connections: {str(e)}")
            
            return G
                
        except Exception as e:
            print(f"Error connecting to stances: {str(e)}")
            return G
    
    def _ensure_dag(self, G):
        """
        Ensure the graph is a DAG by removing cycles if necessary.
        
        Args:
            G (nx.DiGraph): NetworkX directed graph
            
        Returns:
            nx.DiGraph: DAG version of the graph
        """
        # Check if the graph has cycles
        if not nx.is_directed_acyclic_graph(G):
            print("Graph contains cycles, removing edges to create a DAG...")
            
            # Find cycles
            cycles = list(nx.simple_cycles(G))
            print(f"Found {len(cycles)} cycles")
            
            # Remove edges to break cycles (prioritize removing low confidence edges)
            edges_to_remove = []
            for cycle in cycles:
                # Convert cycle to edges
                cycle_edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
                
                # Find the edge with lowest confidence to remove
                lowest_conf = None
                edge_to_remove = None
                
                for u, v in cycle_edges:
                    edge_data = G.get_edge_data(u, v)
                    if edge_data:
                        confidence = edge_data.get('confidence', 'medium')
                        # Assign numeric values to confidence levels
                        conf_value = {'low': 1, 'medium': 2, 'high': 3}.get(confidence, 2)
                        
                        if lowest_conf is None or conf_value < lowest_conf:
                            lowest_conf = conf_value
                            edge_to_remove = (u, v)
                
                if edge_to_remove:
                    edges_to_remove.append(edge_to_remove)
            
            # Remove edges
            for u, v in edges_to_remove:
                G.remove_edge(u, v)
                print(f"Removed edge: {u} -> {v}")
            
            # Check again
            if not nx.is_directed_acyclic_graph(G):
                print("Warning: Graph still contains cycles after attempted removal")
        
        return G
    
    def _generate_mermaid(self, G, interview_name):
        """
        Generate Mermaid diagram code for the graph.
        
        Args:
            G (nx.DiGraph): NetworkX directed graph
            interview_name (str): Name of the interviewee
            
        Returns:
            str: Mermaid diagram code
        """
        # Start the mermaid graph definition
        mermaid = ["graph TD;"]
        
        # Add nodes
        node_ids = {}
        for i, node in enumerate(G.nodes()):
            node_id = f"n{i}"
            node_ids[node] = node_id
            
            # Check node type
            node_type = G.nodes[node].get('type', 'factor')
            
            # Style based on type
            if node_type == 'stance':
                mermaid.append(f'    {node_id}["{node}"]:::stanceNode;')
            else:
                mermaid.append(f'    {node_id}["{node}"]:::factorNode;')
        
        # Add edges
        for u, v, data in G.edges(data=True):
            u_id = node_ids[u]
            v_id = node_ids[v]
            
            # Style based on edge type
            if 'connection_type' in data:
                if data['connection_type'] == 'supports':
                    mermaid.append(f'    {u_id} -->|supports| {v_id};')
                elif data['connection_type'] == 'opposes':
                    mermaid.append(f'    {u_id} -->|opposes| {v_id};')
                else:
                    mermaid.append(f'    {u_id} --> {v_id};')
            else:
                # Style based on confidence
                confidence = data.get('confidence', 'medium')
                if confidence == 'high':
                    mermaid.append(f'    {u_id} ==>|{confidence}| {v_id};')
                elif confidence == 'low':
                    mermaid.append(f'    {u_id} -.->|{confidence}| {v_id};')
                else:
                    mermaid.append(f'    {u_id} -->|{confidence}| {v_id};')
        
        # Add class definitions
        mermaid.append("    classDef stanceNode fill:#f9f,stroke:#333,stroke-width:2px;")
        mermaid.append("    classDef factorNode fill:#bbf,stroke:#333,stroke-width:1px;")
        
        # Add title
        mermaid.append(f"    title[\"Causal Graph: {interview_name}\"]:::title;")
        mermaid.append("    classDef title fill:none,stroke:none;")
        
        return "\n".join(mermaid)
    
    def _save_graph_as_json(self, G, file_path):
        """
        Save the graph as a JSON file.
        
        Args:
            G (nx.DiGraph): NetworkX directed graph
            file_path (Path): Path to save the JSON file
        """
        # Convert graph to dictionary format
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        # Add nodes
        for node, attrs in G.nodes(data=True):
            node_data = {
                'id': node,
                'type': attrs.get('type', 'factor')
            }
            graph_data['nodes'].append(node_data)
        
        # Add edges
        for u, v, attrs in G.edges(data=True):
            edge_data = {
                'source': u,
                'target': v
            }
            # Add all edge attributes
            edge_data.update(attrs)
            graph_data['edges'].append(edge_data)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    def process_interview(self, interview_name):
        """
        Process a single interview to build a causal graph.
        
        Args:
            interview_name (str): Name of the interview to process
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Ensure interviewee name doesn't have the _en_subs.txt suffix
            if interview_name.endswith('_en_subs.txt'):
                interview_name = interview_name.replace('_en_subs.txt', '')
                
            interview_dir = self.processed_data_dir / interview_name
            qa_file = interview_dir / 'qa_pairs.json'
            
            if not qa_file.exists():
                print(f"QA pairs file not found for {interview_name}")
                return False
            
            # Load QA pairs
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_pairs = json.load(f)
            
            if not qa_pairs:
                print(f"No QA pairs found for {interview_name}")
                return False
            
            # Create output directory for this interview
            output_dir = self.output_dir / interview_name
            if not output_dir.exists():
                output_dir.mkdir(parents=True)
            
            # Extract causal relationships
            causal_relationships = self._extract_causal_relationships(qa_pairs, interview_name)
            
            # Save causal relationships to JSON
            relationships_file = output_dir / 'causal_relationships.json'
            with open(relationships_file, 'w', encoding='utf-8') as f:
                json.dump(causal_relationships, f, indent=2, ensure_ascii=False)
            
            # Build graph
            G = self._build_graph(causal_relationships)
            
            # Connect to stance nodes
            G = self._connect_to_stance(G, interview_name)
            
            # Ensure it's a DAG
            G = self._ensure_dag(G)
            
            # Generate mermaid diagram
            mermaid = self._generate_mermaid(G, interview_name)
            mermaid_file = output_dir / 'causal_graph.mmd'
            with open(mermaid_file, 'w', encoding='utf-8') as f:
                f.write(mermaid)
            
            # Save graph as JSON
            graph_file = output_dir / 'causal_graph.json'
            self._save_graph_as_json(G, graph_file)
            
            # Save metadata
            metadata = {
                'interview_name': interview_name,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'num_causal_relationships': len(causal_relationships),
                'num_nodes': G.number_of_nodes(),
                'num_edges': G.number_of_edges(),
                'is_dag': nx.is_directed_acyclic_graph(G)
            }
            
            metadata_file = output_dir / 'causal_graph_metadata.json'
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Successfully built causal graph for {interview_name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            return True
            
        except Exception as e:
            print(f"Error building causal graph for {interview_name}: {str(e)}")
            return False
    
    def process_all_interviews(self):
        """
        Process all interviews to build causal graphs.
        
        Returns:
            dict: Statistics about the processing
        """
        # Get all interview directories
        interview_dirs = [f.name for f in self.processed_data_dir.iterdir() if f.is_dir()]
        
        if not interview_dirs:
            print("No processed interview directories found.")
            return {'total': 0, 'successful': 0, 'failed': 0}
        
        # Process each interview
        successful = 0
        failed = 0
        
        for interview_name in interview_dirs:
            print(f"Building causal graph for {interview_name}...")
            result = self.process_interview(interview_name)
            
            if result:
                successful += 1
            else:
                failed += 1
        
        # Return statistics
        stats = {
            'total': len(interview_dirs),
            'successful': successful,
            'failed': failed
        }
        
        print(f"Causal graph building complete. Total: {stats['total']}, Successful: {stats['successful']}, Failed: {stats['failed']}")
        return stats


if __name__ == "__main__":
    # Define paths
    processed_data_dir = "data/housing_choice_community_interviews/processed"
    
    # Create and run graph builder
    builder = CausalGraphBuilder(processed_data_dir)
    builder.process_all_interviews() 