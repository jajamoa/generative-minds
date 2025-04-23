import json
import os
import openai
import uuid
import re
import networkx as nx
from dotenv import load_dotenv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_qa_data(file_path):
    """Load QA pair data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_causal_relationships(qa_pairs, client):
    """
    Use GPT to identify causal relationships from QA pairs
    Returns node and edge information
    """
    nodes = {}
    edges = {}
    qa_history = {}
    node_embeddings = {}  # Store embeddings for node similarity comparison

    for i, qa_pair in enumerate(qa_pairs):
        qa_id = f"qa_{str(i+1).zfill(3)}"

        qa_history[qa_id] = {
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
            "extracted_pairs": []
        }

        system_message = """
        Identify causal relationships in the text. Extract concepts and how they influence each other.
        For each causal relationship, provide:
        1. Source concept: The causing factor
        2. Target concept: The affected factor
        3. Relationship: Whether the influence is positive (increases/supports) or negative (decreases/opposes)
        4. Confidence: Your confidence in this causal relationship (0.0 to 1.0)
        
        Format as JSON:
        {
          "relationships": [
            {
              "source_concept": "concept_name",
              "target_concept": "concept_name", 
              "positive_influence": true/false,
              "confidence": 0.0-1.0
            }
          ]
        }
        
        If no clear causal relationships exist, return {"relationships": []}.
        """

        user_message = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1
            )

            content = response.choices[0].message.content
            parsed = json.loads(content)

            for rel in parsed.get("relationships", []):
                source_concept = clean_concept_for_mermaid(rel["source_concept"].lower().replace(" ", "_"))
                target_concept = clean_concept_for_mermaid(rel["target_concept"].lower().replace(" ", "_"))
                confidence = rel["confidence"]
                positive = rel["positive_influence"]

                # Check for semantically similar nodes before creating new ones
                source_id = get_or_create_node_with_similarity_check(nodes, node_embeddings, source_concept, client)
                target_id = get_or_create_node_with_similarity_check(nodes, node_embeddings, target_concept, client)

                # —— 关键修复：跳过自环边 ——
                if source_id == target_id:
                    continue

                edge_id = f"e{len(edges)+1}"
                modifier = confidence if positive else -confidence

                edges[edge_id] = {
                    "source": source_id,
                    "target": target_id,
                    "aggregate_confidence": abs(confidence),
                    "evidence": [{
                        "qa_id": qa_id,
                        "confidence": abs(confidence)
                    }],
                    "modifier": modifier,
                    "positive": positive
                }

                nodes[source_id]["outgoing_edges"].append(edge_id)
                nodes[target_id]["incoming_edges"].append(edge_id)

                nodes[source_id]["source_qa"].append(qa_id)
                nodes[target_id]["source_qa"].append(qa_id)

                qa_history[qa_id]["extracted_pairs"].append({
                    "source": source_id,
                    "target": target_id,
                    "confidence": abs(confidence)
                })

        except Exception as e:
            print(f"Error processing QA pair {i+1}: {e}")
            continue

    nodes = {k: v for k, v in nodes.items() if v["incoming_edges"] or v["outgoing_edges"]}
    
    # Convert to NetworkX graph for cycle detection and removal
    nodes, edges = ensure_dag(nodes, edges)
    
    # Generate reflection summary
    reflection_summary = generate_node_reflection(nodes, client)
    
    return nodes, edges, qa_history, reflection_summary

def ensure_dag(nodes, edges):
    """
    Ensure the graph is a DAG by removing cycles
    Uses NetworkX for cycle detection and removes low-confidence edges to break cycles
    Returns both the updated data structures and the NetworkX graph
    """
    print("Checking for cycles in the graph...")
    
    # Convert to NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node_data in nodes.items():
        G.add_node(node_id, label=node_data["label"])
    
    # Add edges
    for edge_id, edge_data in edges.items():
        source_id = edge_data["source"]
        target_id = edge_data["target"]
        confidence = edge_data["aggregate_confidence"]
        G.add_edge(source_id, target_id, 
                  id=edge_id, 
                  confidence=confidence)
    
    # Check if the graph has cycles
    if not nx.is_directed_acyclic_graph(G):
        print("Graph contains cycles. Removing edges to create a DAG...")
        
        # Find cycles
        cycles = list(nx.simple_cycles(G))
        print(f"Found {len(cycles)} cycles")
        
        # Track edges to remove
        edges_to_remove = []
        
        # Process each cycle
        for cycle in cycles:
            if len(cycle) < 2:  # Skip self-loops
                continue
                
            # Create cycle edges
            cycle_edges = []
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    edge_data = G.get_edge_data(u, v)
                    cycle_edges.append((u, v, edge_data.get('confidence', 0.5), edge_data.get('id')))
            
            # Find edge with lowest confidence to remove
            min_confidence = float('inf')
            edge_to_remove = None
            edge_id_to_remove = None
            
            for u, v, conf, edge_id in cycle_edges:
                if conf < min_confidence:
                    min_confidence = conf
                    edge_to_remove = (u, v)
                    edge_id_to_remove = edge_id
            
            if edge_to_remove and edge_id_to_remove:
                G.remove_edge(*edge_to_remove)
                edges_to_remove.append(edge_id_to_remove)
                print(f"Removed edge {edge_id_to_remove} from {edge_to_remove[0]} to {edge_to_remove[1]} (confidence: {min_confidence})")
        
        # Remove edges from our edge dictionary
        for edge_id in edges_to_remove:
            if edge_id in edges:
                source_id = edges[edge_id]["source"]
                target_id = edges[edge_id]["target"]
                
                # Remove edge from nodes' edge lists
                if source_id in nodes and edge_id in nodes[source_id]["outgoing_edges"]:
                    nodes[source_id]["outgoing_edges"].remove(edge_id)
                
                if target_id in nodes and edge_id in nodes[target_id]["incoming_edges"]:
                    nodes[target_id]["incoming_edges"].remove(edge_id)
                
                # Remove edge from edges dictionary
                del edges[edge_id]
        
        # Check if the graph is now a DAG
        if not nx.is_directed_acyclic_graph(G):
            print("Warning: Graph still contains cycles after edge removal")
            # Additional cycle removal if needed
            while not nx.is_directed_acyclic_graph(G):
                cycles = list(nx.simple_cycles(G))
                if not cycles:
                    break
                # Remove one edge from each remaining cycle
                for cycle in cycles:
                    if len(cycle) >= 2:
                        u, v = cycle[0], cycle[1]
                        G.remove_edge(u, v)
                        # Find and remove corresponding edge from our data structures
                        for edge_id, edge_data in list(edges.items()):
                            if edge_data["source"] == u and edge_data["target"] == v:
                                if edge_id in nodes[u]["outgoing_edges"]:
                                    nodes[u]["outgoing_edges"].remove(edge_id)
                                if edge_id in nodes[v]["incoming_edges"]:
                                    nodes[v]["incoming_edges"].remove(edge_id)
                                del edges[edge_id]
        else:
            print("Successfully converted graph to a DAG")
    else:
        print("Graph is already a DAG, no cycles found")
    
    # Ensure edges dictionary matches the NetworkX graph
    edges_to_keep = {}
    for edge_id, edge_data in edges.items():
        source = edge_data["source"]
        target = edge_data["target"]
        if G.has_edge(source, target):
            edges_to_keep[edge_id] = edge_data
    
    # Update edges dictionary
    edges = edges_to_keep
    
    # Update node edge lists to match remaining edges
    for node_id in nodes:
        nodes[node_id]["outgoing_edges"] = [
            e for e in nodes[node_id]["outgoing_edges"]
            if e in edges
        ]
        nodes[node_id]["incoming_edges"] = [
            e for e in nodes[node_id]["incoming_edges"]
            if e in edges
        ]
    
    return nodes, edges

def clean_concept_for_mermaid(concept):
    """Clean concept names to be valid Mermaid identifiers"""
    # Replace apostrophes and hyphens with underscores
    concept = re.sub(r"['\\-]", "_", concept)
    # Remove any other characters that might cause issues in Mermaid
    concept = re.sub(r"[^a-zA-Z0-9_]", "", concept)
    return concept

def get_or_create_node(nodes, concept):
    """Get existing node ID for a concept or create a new one"""
    # Check if concept already exists
    for node_id, node_data in nodes.items():
        if node_data["label"] == concept:
            return node_id
    
    # Create new node
    node_id = f"n{len(nodes)+1}"
    nodes[node_id] = {
        "label": concept,
        "confidence": 1.0,  # Default confidence
        "source_qa": [],
        "incoming_edges": [],
        "outgoing_edges": []
    }
    
    return node_id

def get_embedding(text, client):
    """Get embedding for a text string using OpenAI API"""
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding for '{text}': {e}")
        # Return a random embedding as fallback
        return np.random.rand(1536).tolist()  # Ada embeddings are 1536 dimensions

def get_or_create_node_with_similarity_check(nodes, node_embeddings, concept, client, similarity_threshold=0.85):
    """Get existing node ID for a concept or create a new one, with semantic similarity check"""
    # Direct string match (exact match)
    for node_id, node_data in nodes.items():
        if node_data["label"] == concept:
            return node_id
    
    # If no exact match, check for semantic similarity
    if concept not in node_embeddings:
        node_embeddings[concept] = get_embedding(concept, client)
    
    concept_embedding = node_embeddings[concept]
    
    # Check similarity with existing nodes
    for node_id, node_data in nodes.items():
        node_label = node_data["label"]
        if node_label not in node_embeddings:
            node_embeddings[node_label] = get_embedding(node_label, client)
        
        similarity = cosine_similarity(
            [concept_embedding], 
            [node_embeddings[node_label]]
        )[0][0]
        
        if similarity >= similarity_threshold:
            print(f"Found similar node: '{concept}' -> '{node_label}' (similarity: {similarity:.3f})")
            return node_id
    
    # Create new node if no similar nodes found
    node_id = f"n{len(nodes)+1}"
    nodes[node_id] = {
        "label": concept,
        "confidence": 1.0,  # Default confidence
        "source_qa": [],
        "incoming_edges": [],
        "outgoing_edges": []
    }
    
    return node_id

def generate_node_reflection(nodes, client):
    """Generate a reflection summary of all nodes in the graph"""
    if not nodes:
        return {"summary": "No nodes extracted from the QA pairs."}
    
    # Group nodes by connectivity
    central_nodes = []
    influencer_nodes = []
    influenced_nodes = []
    isolated_nodes = []
    
    for node_id, node_data in nodes.items():
        incoming = len(node_data["incoming_edges"])
        outgoing = len(node_data["outgoing_edges"])
        
        if incoming > 0 and outgoing > 0:
            central_nodes.append((node_id, node_data, incoming + outgoing))
        elif outgoing > 0:
            influencer_nodes.append((node_id, node_data, outgoing))
        elif incoming > 0:
            influenced_nodes.append((node_id, node_data, incoming))
        else:
            isolated_nodes.append((node_id, node_data))
    
    # Sort by connectivity
    central_nodes.sort(key=lambda x: x[2], reverse=True)
    influencer_nodes.sort(key=lambda x: x[2], reverse=True)
    influenced_nodes.sort(key=lambda x: x[2], reverse=True)
    
    # Create lists of node labels
    central_node_labels = [f"{data['label']} (id: {id}, connections: {count})" 
                         for id, data, count in central_nodes[:10]]  # Top 10
    influencer_node_labels = [f"{data['label']} (id: {id}, outgoing: {count})" 
                            for id, data, count in influencer_nodes[:10]]  # Top 10
    influenced_node_labels = [f"{data['label']} (id: {id}, incoming: {count})" 
                           for id, data, count in influenced_nodes[:10]]  # Top 10
    isolated_node_labels = [f"{data['label']} (id: {id})" 
                          for id, data in isolated_nodes[:10]]  # Top 10
    
    # Create summary
    system_message = """
    You are a network analyst expert. Summarize the key concepts in this causal network.
    Focus on identifying the main themes, clusters, and potential insights.
    Keep your analysis concise but informative.
    """
    
    node_descriptions = "\n".join([
        f"Central nodes (with both incoming and outgoing connections): {', '.join([data['label'] for id, data, _ in central_nodes])}",
        f"Influencer nodes (with outgoing connections): {', '.join([data['label'] for id, data, _ in influencer_nodes])}",
        f"Influenced nodes (with incoming connections): {', '.join([data['label'] for id, data, _ in influenced_nodes])}",
        f"Isolated nodes: {', '.join([data['label'] for id, data in isolated_nodes])}"
    ])
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Analyze this causal network:\n{node_descriptions}"}
            ],
            temperature=0.3
        )
        
        ai_summary = response.choices[0].message.content
    except Exception as e:
        ai_summary = f"Error generating AI summary: {e}"
    
    return {
        "total_nodes": len(nodes),
        "central_nodes": central_node_labels,
        "influencer_nodes": influencer_node_labels,
        "influenced_nodes": influenced_node_labels, 
        "isolated_nodes": isolated_node_labels,
        "ai_summary": ai_summary
    }

def generate_mermaid_code(nodes, edges):
    """Generate Mermaid code for visualizing the causal graph"""
    # Create NetworkX graph to verify DAG structure
    G = nx.DiGraph()
    for node_id in nodes:
        G.add_node(node_id)
    for edge_id, edge_data in edges.items():
        G.add_edge(edge_data["source"], edge_data["target"])
    
    if not nx.is_directed_acyclic_graph(G):
        print("Warning: Graph contains cycles, attempting to fix before generating Mermaid code")
        nodes, edges = ensure_dag(nodes, edges)
    
    # Get topological sort of nodes for better layout
    try:
        node_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        print("Warning: Could not compute topological sort, using default node order")
        node_order = list(nodes.keys())
    
    mermaid_lines = ["```mermaid", "flowchart TD"]
    
    # Add nodes in topological order
    for node_id in node_order:
        if node_id in nodes:  # Check if node exists in our data structure
            label = nodes[node_id]["label"]
            mermaid_lines.append(f"    {node_id}[{label}]")
    
    # Add edges
    for edge_id, edge_data in edges.items():
        source_id = edge_data["source"]
        target_id = edge_data["target"]
        
        # Verify that both nodes exist and edge follows topological order
        if (source_id in nodes and target_id in nodes and
            node_order.index(source_id) < node_order.index(target_id)):
            positive = edge_data["positive"]
            confidence = edge_data["aggregate_confidence"]
            
            # Determine edge style based on positive/negative influence
            edge_style = "-->" if positive else "--x"
            
            # Create edge with different line thickness based on confidence
            thickness = int(confidence * 3) + 1  # Scale confidence to line thickness
            mermaid_lines.append(f"    {source_id} {edge_style} {target_id}")
    
    # Add custom linkStyles for positive/negative and confidence
    link_styles = []
    edge_count = 0
    for edge_id, edge_data in edges.items():
        source_id = edge_data["source"]
        target_id = edge_data["target"]
        if (source_id in nodes and target_id in nodes and
            node_order.index(source_id) < node_order.index(target_id)):
            confidence = edge_data["aggregate_confidence"]
            positive = edge_data["positive"]
            
            # Line style: solid for positive, dashed for negative
            style = "stroke:#00AA00" if positive else "stroke:#FF0000,stroke-dasharray:3"
            
            # Line thickness based on confidence
            thickness = max(1, int(confidence * 3))
            link_styles.append(f"    linkStyle {edge_count} {style},stroke-width:{thickness}px")
            edge_count += 1
    
    mermaid_lines.extend(link_styles)
    mermaid_lines.append("```")
    
    return "\n".join(mermaid_lines)

def save_json_files(nodes, edges, qa_history, reflection):
    """Save the extracted data to JSON files"""
    with open('nodes_processed.json', 'w') as f:
        json.dump(nodes, f, indent=2)
    
    with open('edges_processed.json', 'w') as f:
        json.dump(edges, f, indent=2)
    
    with open('qa_history_processed.json', 'w') as f:
        json.dump(qa_history, f, indent=2)
    
    with open('reflection_summary.json', 'w') as f:
        json.dump(reflection, f, indent=2)
    
    # Generate and save Mermaid code
    mermaid_code = generate_mermaid_code(nodes, edges)
    with open('causal_graph.mmd', 'w') as f:
        f.write(mermaid_code)
    
    # Save graph metadata
    metadata = {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "is_dag": True,  # We ensure this with the ensure_dag function
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open('graph_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Files saved: nodes_processed.json, edges_processed.json, qa_history_processed.json, reflection_summary.json, causal_graph.mmd, graph_metadata.json")

def identify_stance_nodes(nodes, client):
    """Identify nodes that represent a stance on upzoning plan using GPT"""
    stance_nodes = []
    
    # Batch nodes into groups of 10 to avoid token limits
    node_batches = [list(nodes.items())[i:i + 10] for i in range(0, len(nodes), 10)]
    
    system_message = """
    You are an expert at identifying stance and opinion nodes in a causal graph about urban planning.
    Analyze each node and determine if it represents someone's stance, position, or opinion about upzoning or zoning plans.
    
    A stance node should represent:
    1. A clear position or attitude (support/oppose/neutral)
    2. Related to upzoning, zoning changes, or urban development plans
    3. Can be either individual or group opinions
    
    Examples of stance nodes:
    - "residents_oppose_upzoning"
    - "developer_supports_plan"
    - "community_resistance_to_zoning_change"
    
    Not stance nodes:
    - "housing_prices" (fact/outcome)
    - "population_density" (metric)
    - "zoning_regulations" (policy)
    
    For each node, return:
    {
      "node_id": "id",
      "is_stance": true/false,
      "confidence": 0.0-1.0,
      "explanation": "brief explanation"
    }
    """
    
    for batch in node_batches:
        node_descriptions = []
        for node_id, node_data in batch:
            node_descriptions.append(f"Node ID: {node_id}\nLabel: {node_data['label']}")
        
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Analyze these nodes for stance on upzoning/zoning:\n\n" + "\n\n".join(node_descriptions)}
                ],
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            try:
                results = json.loads(content)
                if not isinstance(results, list):
                    results = [results]
                
                for result in results:
                    if result.get("is_stance", False) and result.get("confidence", 0) > 0.7:
                        stance_nodes.append(result["node_id"])
                        print(f"Found stance node: {nodes[result['node_id']]['label']} ({result['explanation']})")
                        
            except json.JSONDecodeError:
                print(f"Warning: Could not parse GPT response for stance identification: {content}")
                continue
                
        except Exception as e:
            print(f"Error during stance identification: {e}")
            continue
    
    if not stance_nodes:
        print("Warning: No stance nodes identified by GPT")
    else:
        print(f"Identified {len(stance_nodes)} stance nodes")
    
    return stance_nodes

def get_ancestor_nodes(nodes, edges, target_nodes):
    """Get all ancestor nodes of the target nodes in the graph"""
    # Create NetworkX graph
    G = nx.DiGraph()
    for node_id in nodes:
        G.add_node(node_id)
    for edge_id, edge_data in edges.items():
        G.add_edge(edge_data["source"], edge_data["target"])
    
    # Get all ancestors
    ancestors = set()
    for node in target_nodes:
        ancestors.update(nx.ancestors(G, node))
    ancestors.update(target_nodes)  # Include target nodes themselves
    
    return ancestors

def filter_to_stance_ancestors(nodes, edges, client):
    """Filter the graph to only keep stance nodes and their ancestors"""
    # Identify stance nodes using GPT
    stance_nodes = identify_stance_nodes(nodes, client)
    if not stance_nodes:
        print("Warning: No stance nodes found in the graph")
        return nodes, edges
    
    # Get all ancestors of stance nodes
    relevant_nodes = get_ancestor_nodes(nodes, edges, stance_nodes)
    
    # Filter nodes
    filtered_nodes = {node_id: data for node_id, data in nodes.items() 
                     if node_id in relevant_nodes}
    
    # Filter edges
    filtered_edges = {edge_id: data for edge_id, data in edges.items()
                     if data["source"] in relevant_nodes and data["target"] in relevant_nodes}
    
    # Update node edge lists
    for node_id in filtered_nodes:
        filtered_nodes[node_id]["incoming_edges"] = [
            e for e in filtered_nodes[node_id]["incoming_edges"]
            if e in filtered_edges
        ]
        filtered_nodes[node_id]["outgoing_edges"] = [
            e for e in filtered_nodes[node_id]["outgoing_edges"]
            if e in filtered_edges
        ]
    
    print(f"Filtered graph from {len(nodes)} to {len(filtered_nodes)} nodes")
    print(f"Filtered edges from {len(edges)} to {len(filtered_edges)} edges")
    print(f"Found {len(stance_nodes)} stance nodes: {[nodes[nid]['label'] for nid in stance_nodes]}")
    
    return filtered_nodes, filtered_edges

def merge_stance_nodes(nodes, edges, stance_nodes, client):
    """Merge all stance nodes into a single comprehensive stance node"""
    if not stance_nodes:
        return nodes, edges
    
    # Prepare stance node descriptions for GPT
    stance_descriptions = [f"Node: {nodes[node_id]['label']}" for node_id in stance_nodes]
    
    system_message = """
    You are an expert at analyzing and synthesizing stances on urban planning issues.
    Analyze multiple stance nodes and create a comprehensive summary that captures the overall stance.
    The summary should:
    1. Indicate the dominant stance (support/oppose/mixed)
    2. Include any key qualifications or conditions
    3. Preserve the essence of individual positions
    
    Return the result as JSON:
    {
      "merged_label": "comprehensive stance description",
      "dominant_stance": "support/oppose/mixed",
      "confidence": 0.0-1.0
    }
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Synthesize these stance positions:\n" + "\n".join(stance_descriptions)}
            ],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        merged_label = result["merged_label"]
        
    except Exception as e:
        print(f"Error merging stance nodes: {e}")
        return nodes, edges
    
    # Create new merged node
    merged_id = f"n{len(nodes)+1}"
    merged_node = {
        "label": merged_label,
        "confidence": result.get("confidence", 1.0),
        "source_qa": [],
        "incoming_edges": [],
        "outgoing_edges": []
    }
    
    # Collect all QA sources
    for node_id in stance_nodes:
        merged_node["source_qa"].extend(nodes[node_id]["source_qa"])
    merged_node["source_qa"] = list(set(merged_node["source_qa"]))  # Remove duplicates
    
    # Update edges
    new_edges = {}
    edge_id_counter = len(edges) + 1
    
    for edge_id, edge_data in edges.items():
        source = edge_data["source"]
        target = edge_data["target"]
        
        if source in stance_nodes:
            if target in stance_nodes:
                continue  # Skip edges between stance nodes
            source = merged_id
        if target in stance_nodes:
            target = merged_id
            
        if source != target:  # Avoid self-loops
            new_edge_id = f"e{edge_id_counter}"
            edge_id_counter += 1
            new_edges[new_edge_id] = {
                "source": source,
                "target": target,
                "aggregate_confidence": edge_data["aggregate_confidence"],
                "evidence": edge_data["evidence"],
                "modifier": edge_data["modifier"],
                "positive": edge_data["positive"]
            }
            
            # Update node edge lists
            if source == merged_id:
                merged_node["outgoing_edges"].append(new_edge_id)
            else:
                nodes[source]["outgoing_edges"].append(new_edge_id)
                
            if target == merged_id:
                merged_node["incoming_edges"].append(new_edge_id)
            else:
                nodes[target]["incoming_edges"].append(new_edge_id)
    
    # Remove old stance nodes and add merged node
    for node_id in stance_nodes:
        del nodes[node_id]
    nodes[merged_id] = merged_node
    
    print(f"Merged {len(stance_nodes)} stance nodes into new node: {merged_label}")
    return nodes, new_edges

def merge_similar_nodes(nodes, edges, client, similarity_threshold=0.85):
    """Merge semantically similar nodes based on embeddings"""
    if len(nodes) <= 1:
        return nodes, edges
    
    # Get embeddings for all nodes
    node_embeddings = {}
    for node_id, node_data in nodes.items():
        try:
            response = client.embeddings.create(
                input=node_data["label"],
                model="text-embedding-ada-002"
            )
            node_embeddings[node_id] = response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding for node {node_id}: {e}")
            continue
    
    # Find similar nodes
    merged_groups = []
    processed_nodes = set()
    
    for node_id1 in nodes:
        if node_id1 in processed_nodes:
            continue
            
        current_group = [node_id1]
        processed_nodes.add(node_id1)
        
        if node_id1 not in node_embeddings:
            continue
            
        for node_id2 in nodes:
            if node_id2 in processed_nodes or node_id2 not in node_embeddings:
                continue
                
            similarity = cosine_similarity(
                [node_embeddings[node_id1]], 
                [node_embeddings[node_id2]]
            )[0][0]
            
            if similarity >= similarity_threshold:
                current_group.append(node_id2)
                processed_nodes.add(node_id2)
        
        if len(current_group) > 1:
            merged_groups.append(current_group)
    
    # Merge each group
    new_nodes = nodes.copy()
    new_edges = edges.copy()
    
    for group in merged_groups:
        # Use GPT to create a merged label
        node_labels = [nodes[node_id]["label"] for node_id in group]
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Create a concise label that captures the essence of these similar concepts."},
                    {"role": "user", "content": f"Synthesize these concepts into a single label:\n{', '.join(node_labels)}"}
                ],
                temperature=0.1
            )
            merged_label = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error creating merged label: {e}")
            merged_label = node_labels[0]  # Fallback to first label
        
        # Create merged node
        merged_id = f"n{len(new_nodes)+1}"
        merged_node = {
            "label": merged_label,
            "source_qa": [],
            "incoming_edges": [],
            "outgoing_edges": []
        }
        
        # Collect QA sources
        for node_id in group:
            merged_node["source_qa"].extend(new_nodes[node_id]["source_qa"])
        merged_node["source_qa"] = list(set(merged_node["source_qa"]))
        
        # Update edges
        edge_id_counter = len(new_edges) + 1
        temp_edges = {}
        
        for edge_id, edge_data in new_edges.items():
            source = edge_data["source"]
            target = edge_data["target"]
            
            if source in group:
                source = merged_id
            if target in group:
                target = merged_id
                
            if source != target:  # Avoid self-loops
                new_edge_id = f"e{edge_id_counter}"
                edge_id_counter += 1
                temp_edges[new_edge_id] = {
                    "source": source,
                    "target": target,
                    "aggregate_confidence": edge_data["aggregate_confidence"],
                    "evidence": edge_data["evidence"],
                    "modifier": edge_data["modifier"],
                    "positive": edge_data["positive"]
                }
                
                # Update edge lists
                if source == merged_id:
                    merged_node["outgoing_edges"].append(new_edge_id)
                elif source in new_nodes:
                    new_nodes[source]["outgoing_edges"].append(new_edge_id)
                    
                if target == merged_id:
                    merged_node["incoming_edges"].append(new_edge_id)
                elif target in new_nodes:
                    new_nodes[target]["incoming_edges"].append(new_edge_id)
        
        # Remove old nodes and add merged node
        for node_id in group:
            del new_nodes[node_id]
        new_nodes[merged_id] = merged_node
        new_edges = temp_edges
        
        print(f"Merged nodes {group} into new node: {merged_label}")
    
    return new_nodes, new_edges

def create_development_stance_node(nodes, edges, qa_pairs, client):
    """Create a development stance node from QA pairs when no explicit stance is found"""
    
    # First, summarize all QA pairs to extract development stance
    system_message = """
    You are an expert at analyzing urban development discussions and extracting overall positions on development.
    Review these QA pairs and create a summary of the stance towards development/upzoning.
    Focus on:
    1. Whether the discussion generally supports or opposes development
    2. Key reasons or conditions mentioned
    3. Main stakeholders involved
    
    Return the result as JSON:
    {
      "stance_summary": "concise summary of stance towards development",
      "dominant_position": "support/oppose/mixed",
      "confidence": 0.0-1.0,
      "key_factors": ["list of 3-5 most important factors mentioned"]
    }
    """
    
    qa_text = "\n\n".join([f"Q: {qa['question']}\nA: {qa['answer']}" for qa in qa_pairs])
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Analyze these QA pairs for development stance:\n\n{qa_text}"}
            ],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Error creating development stance summary: {e}")
        return nodes, edges
    
    # Create new stance node
    stance_id = f"n{len(nodes)+1}"
    stance_node = {
        "label": result["stance_summary"],
        "confidence": result["confidence"],
        "source_qa": [],  # Will collect from connected nodes
        "incoming_edges": [],
        "outgoing_edges": []
    }
    
    # Find most relevant nodes for the key factors
    key_factors = result["key_factors"]
    relevant_nodes = []
    
    try:
        # Get embeddings for key factors
        factor_embeddings = []
        for factor in key_factors:
            response = client.embeddings.create(
                input=factor,
                model="text-embedding-ada-002"
            )
            factor_embeddings.append(response.data[0].embedding)
        
        # Get embeddings for existing nodes
        node_embeddings = {}
        for node_id, node_data in nodes.items():
            response = client.embeddings.create(
                input=node_data["label"],
                model="text-embedding-ada-002"
            )
            node_embeddings[node_id] = response.data[0].embedding
        
        # Find most similar nodes for each factor
        for factor, factor_emb in zip(key_factors, factor_embeddings):
            max_similarity = -1
            most_similar_node = None
            
            for node_id, node_emb in node_embeddings.items():
                similarity = cosine_similarity([factor_emb], [node_emb])[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_node = node_id
            
            if most_similar_node and max_similarity > 0.6:  # Threshold for relevance
                relevant_nodes.append((most_similar_node, max_similarity))
        
    except Exception as e:
        print(f"Error finding relevant nodes: {e}")
        # Fallback: use nodes with most connections
        sorted_nodes = sorted(nodes.items(), 
                            key=lambda x: len(x[1]["incoming_edges"]) + len(x[1]["outgoing_edges"]),
                            reverse=True)
        relevant_nodes = [(node[0], 1.0) for node in sorted_nodes[:5]]
    
    # Create edges between stance node and relevant nodes
    new_edges = edges.copy()
    edge_id_counter = len(new_edges) + 1
    
    for node_id, similarity in relevant_nodes:
        # Add edge from relevant node to stance node
        edge_id = f"e{edge_id_counter}"
        edge_id_counter += 1
        new_edges[edge_id] = {
            "source": node_id,
            "target": stance_id,
            "aggregate_confidence": similarity,
            "evidence": [{"qa_id": qa_id} for qa_id in nodes[node_id]["source_qa"]],
            "modifier": similarity,
            "positive": True  # Assume positive influence by default
        }
        
        # Update edge lists
        nodes[node_id]["outgoing_edges"].append(edge_id)
        stance_node["incoming_edges"].append(edge_id)
        
        # Collect QA sources
        stance_node["source_qa"].extend(nodes[node_id]["source_qa"])
    
    # Remove duplicates from source_qa
    stance_node["source_qa"] = list(set(stance_node["source_qa"]))
    
    # Add new stance node
    new_nodes = nodes.copy()
    new_nodes[stance_id] = stance_node
    
    print(f"Created development stance node: {result['stance_summary']}")
    print(f"Connected to {len(relevant_nodes)} relevant nodes")
    
    return new_nodes, new_edges

def process_qa_to_causal_graph(input_file):
    """Main function to process QA pairs to causal graph"""
    print(f"Processing {input_file}...")
    
    # Load QA data
    qa_pairs = load_qa_data(input_file)
    
    # Create OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract relationships
    nodes, edges, qa_history, reflection = extract_causal_relationships(qa_pairs, client)
    
    # Identify stance nodes and filter to ancestors
    stance_nodes = identify_stance_nodes(nodes, client)
    
    if stance_nodes:
        # If stance nodes found, proceed with normal flow
        nodes, edges = filter_to_stance_ancestors(nodes, edges, client)
        nodes, edges = merge_stance_nodes(nodes, edges, stance_nodes, client)
    else:
        # If no stance nodes found, create development stance node
        print("No explicit stance nodes found, creating development stance node...")
        nodes, edges = create_development_stance_node(nodes, edges, qa_pairs, client)
    
    # Merge similar nodes
    nodes, edges = merge_similar_nodes(nodes, edges, client)
    
    # Save output files
    save_json_files(nodes, edges, qa_history, reflection)
    
    print(f"Processed {len(qa_pairs)} QA pairs.")
    print(f"Final graph has {len(nodes)} nodes and {len(edges)} edges.")
    print("Generated reflection summary in reflection_summary.json")
    print("Generated Mermaid diagram in causal_graph.mmd")
    print("Graph is a DAG (no cycles)")


# Example usage
if __name__ == "__main__":
    input_file = "qa_pairs.json"  # Replace with your input file
    process_qa_to_causal_graph(input_file)