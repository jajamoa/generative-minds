import os
import json
from pathlib import Path
import dashscope
from typing import List
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()
API_KEY = os.getenv("DASHSCOPE_API_KEY")
print("Loaded DASHSCOPE_API_KEY:", API_KEY)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Constants
TRANSCRIPT_DIR = Path("/Users/zhenzemo/generative-minds/src/evaluation/models/m10_1000ppl/data/processed_transcript")
OUTPUT_DIR = Path("/Users/zhenzemo/generative-minds/src/evaluation/models/m10_1000ppl/data/political_expert_reflection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERTS = [
    # "psychologist",
    # "behavioral economist",
    "political scientist",
    # "demographer"
]

SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}

REFLECTION_PROMPT_TEMPLATE = """
Imagine you are a highly trained {expert} with a PhD, observing the following interview transcript of a real person.

Your task is to write down insightful, high-level reflections based on the person's answers in the interview. Your reflections should cover personal traits, values, behaviors, beliefs, goals, and any other relevant social, psychological, political, or economic patterns you notice.

(You should make more than 5 observations and fewer than 10 observations. Choose the number that makes sense given the depth of the interview.)

# Transcript:

# {transcript}

Write your observations and insights below. Number each item.
"""

def load_transcript(file_path: Path) -> tuple[str, str]:
    """Load transcript from a file and return transcript text and agent ID."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        lines = [f"Q: {item['question']}\nA: {item['answer']}" for item in data["transcript"] if item['answer'].strip()]
    return "\n\n".join(lines), file_path.stem  # Using filename as agent ID

def query_reflection(expert: str, transcript_text: str) -> List[str]:
    """Generate expert reflection for a transcript."""
    prompt = REFLECTION_PROMPT_TEMPLATE.format(expert=expert, transcript=transcript_text)
    messages = [
        SYSTEM_MSG,
        {"role": "user", "content": prompt},
    ]
    try:
        response = dashscope.Generation.call(
            api_key=API_KEY,
            model="qwen-plus",
            messages=messages,
            result_format="message"
        )
        content = response['output']['choices'][0]['message']['content'].strip()
        return [line.strip() for line in content.split("\n") if line.strip()]
    except Exception as e:
        logging.error(f"DashScope API error for expert '{expert}': {e}")
        return []

def process_single_transcript(file_path: Path):
    try:
        transcript_text, agent_id = load_transcript(file_path)
        output_path = OUTPUT_DIR / f"{agent_id}.json"

        if output_path.exists():
            logging.info(f"Skipping {agent_id}, output already exists.")
            return

        logging.info(f"Processing transcript: {file_path.name} -> agent ID: {agent_id}")

        expert = EXPERTS[0]  # You currently use only one expert
        logging.info(f"  Generating reflection as '{expert}' for {agent_id}")
        items = query_reflection(expert, transcript_text)

        output = {
            "agent_id": agent_id,
            expert: items  # Directly assign expert key
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)

        logging.info(f"✅ Saved reflection for {agent_id}")

    except Exception as e:
        logging.error(f"Failed to process {file_path.name}: {e}")


def main():
    # Get all transcript files
    transcript_files = list(TRANSCRIPT_DIR.glob("*.json"))
    total_files = len(transcript_files)
    logging.info(f"Found {total_files} transcript files to process")
    
    # Process each file sequentially
    for i, transcript_file in enumerate(transcript_files, 1):
        logging.info(f"Processing file {i}/{total_files}: {transcript_file.name}")
        process_single_transcript(transcript_file)

if __name__ == "__main__":
    main()
