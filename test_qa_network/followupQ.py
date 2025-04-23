import json
import networkx as nx
import heapq
import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
# --------- Loading and Graph Construction ---------
def load_json_files(nodes_file, edges_file, qa_history_file):
    with open(nodes_file, 'r') as f:
        nodes = json.load(f)
    with open(edges_file, 'r') as f:
        edges = json.load(f)
    with open(qa_history_file, 'r') as f:
        qa_history = json.load(f)
    return nodes, edges, qa_history

def create_causal_graph(nodes, edges, qa_history):
    G = nx.DiGraph()
    for node_id, node_data in nodes.items():
        G.add_node(node_id, 
                   label=node_data['label'],
                   confidence=node_data['confidence'],
                   source_qa=node_data['source_qa'])
    for edge_id, edge_data in edges.items():
        G.add_edge(edge_data['source'], 
                   edge_data['target'], 
                   id=edge_id,
                   aggregate_confidence=edge_data['aggregate_confidence'],
                   evidence=[ev['qa_id'] for ev in edge_data['evidence']],
                   modifier=edge_data['modifier'])
    return G

# --------- Utilities ---------
def find_low_confidence_edges(G, threshold=0.9):
    low_conf_edges = []
    for u, v, data in G.edges(data=True):
        if data.get('aggregate_confidence', 1.0) < threshold:
            low_conf_edges.append((u, v, data))
    return low_conf_edges

def find_unverified_nodes(G):
    unverified = []
    for node, data in G.nodes(data=True):
        if data.get('confidence', 1.0) < 0.9:
            unverified.append((node, data))
    return unverified

# --------- Follow-up Question Generator ---------
# def gpt_rewrite_prompt(raw_prompt, model="gpt-3.5-turbo"):
#     """
#     Use GPT to rewrite a follow-up prompt to be more natural and human-like
#     """
#     client = openai.OpenAI()  # New way to initialize client

#     system_prompt = "You are an assistant helping a researcher ask clear, human-style follow-up questions based on given context."
#     user_prompt = f"Rewrite the following follow-up prompt to sound more natural, but keep the key information:\n\n{raw_prompt}"

#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=1,
#         max_tokens=250,
#     )

#     rewritten_prompt = response.choices[0].message.content.strip()
#     return rewritten_prompt

# def generate_followup_items(G):
#     """
#     Generate follow-up question items (with priority scores), GPT-rewritten
#     """
#     items = []

#     # Low-confidence edges
#     low_conf_edges = find_low_confidence_edges(G)
#     for u, v, data in low_conf_edges:
#         priority = data['aggregate_confidence']
       
#         raw_prompt = (
#             f"How do you think {G.nodes[u]['label']} might influence {G.nodes[v]['label']}?"
#         )
#         rewritten_prompt = gpt_rewrite_prompt(raw_prompt)
#         items.append((priority, rewritten_prompt))

#     # Unverified nodes
#     unverified_nodes = find_unverified_nodes(G)
#     for node, data in unverified_nodes:
#         priority = data['confidence']
#         raw_prompt = (
#             f"In your view, why might {data['label']} matter for the bigger picture?"
#         )
#         rewritten_prompt = gpt_rewrite_prompt(raw_prompt)
#         items.append((priority, rewritten_prompt))
    
#     return items
def generate_natural_followup_questions(G, model="gpt-3.5-turbo"):
    """
    Generate very simple, casual, daily-language follow-up questions for ordinary people.
    """
    client = openai.OpenAI()
    items = []

    system_prompt = (
        "You're helping a community researcher talk to everyday people. "
        "Take the idea we give you and turn it into a friendly, casual question — "
        "something someone might ask their neighbor, friend, or coworker. "
        "Avoid academic or policy language. Make it simple, clear, and natural."
    )

    low_conf_edges = find_low_confidence_edges(G)
    for u, v, data in low_conf_edges:
        priority = data['aggregate_confidence']
        user_prompt = (
            f"We think there might be a connection between '{G.nodes[u]['label']}' and '{G.nodes[v]['label']}', "
            f"but we want to ask someone in the community about it. "
            f"Can you write a short, natural question — like something you'd say in a casual conversation — "
            f"to help explore this?"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.2,
            max_tokens=100,
        )

        question = response.choices[0].message.content.strip()
        items.append((priority, question))

    unverified_nodes = find_unverified_nodes(G)
    for node, data in unverified_nodes:
        priority = data['confidence']
        user_prompt = (
            f"Here's something that might be important: '{data['label']}'. "
            f"We want to ask everyday people about it. "
            f"Can you write a friendly, casual question to help someone share their thoughts or experience?"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1.5,
            max_tokens=100,
        )

        question = response.choices[0].message.content.strip()
        items.append((priority, question))

    return items


# --------- Queue Manager ---------
class QuestionQueue:
    def __init__(self):
        self.queue = []

    def add_items(self, items):
        for priority, prompt in items:
            heapq.heappush(self.queue, (priority, prompt))  # heap based on priority

    def pop_item(self):
        if self.queue:
            return heapq.heappop(self.queue)
        else:
            return None, None

    def is_empty(self):
        return len(self.queue) == 0

# --------- Main Workflow ---------

def main():
    nodes_file = "nodes_processed.json"
    edges_file = "edges_processed.json"
    qa_history_file = "qa_history_processed.json"

    try:
        nodes, edges, qa_history = load_json_files(nodes_file, edges_file, qa_history_file)
        G = create_causal_graph(nodes, edges, qa_history)

        # Initialize the queue
        q_queue = QuestionQueue()

        # Generate initial follow-up items
        followup_items = generate_natural_followup_questions(G)
        q_queue.add_items(followup_items)

        print("Starting interactive questioning...\n")

        while not q_queue.is_empty():
            priority, question = q_queue.pop_item()
            if question:
                print(f"Priority {priority:.2f}: {question}")
                input("Your response (press Enter to continue)...\n")

                # (可选扩展)：
                # 如果想在回答后根据新的知识图变化，重新添加新问题，可以在这里刷新图和队列。
                # 例子: 更新G，然后q_queue.add_items(generate_followup_items(G))

        print("No more questions! 🎉")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check if the file paths are correct.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print("Please check if the JSON files are properly formatted.")

if __name__ == "__main__":
    main()
