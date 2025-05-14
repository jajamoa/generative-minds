import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import dashscope
from typing import List
from dotenv import load_dotenv

load_dotenv()
print("Loaded DASH_SCOPE_API_KEY:", os.getenv("DASHSCOPE_API_KEY"))

# Constants
TRANSCRIPT_DIR = Path("/Users/zhenzemo/generative-minds/src/evaluation/models/m10_1000ppl/data/processed_transcript")
OUTPUT_DIR = Path("/Users/zhenzemo/generative-minds/src/evaluation/models/m10_1000ppl/data/expert_reflection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPERTS = [
    # "psychologist",
    # "behavioral economist",
    "political scientist"
    # "demographer"
]

SYSTEM_MSG = {"role": "system", "content": "You are a helpful assistant."}

REFLECTION_PROMPT_TEMPLATE = """
Imagine you are a highly trained {expert} with a PhD, observing the following interview transcript of a real person.

Your task is to write down insightful, high-level reflections based on the person's answers in the interview. Your reflections should cover personal traits, values, behaviors, beliefs, goals, and any other relevant social, psychological, political, or economic patterns you notice.

(You should make more than 5 observations and fewer than 10. Choose the number that makes sense given the depth of the interview.)

Transcript:
=====
{transcript}
=====

Write your observations and insights below. Number each item.
"""

def load_transcript(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        data = json.load(f)
    lines = [f"Q: {item['question']}\nA: {item['answer']}" for item in data["transcript"] if item['answer'].strip()]
    return "\n\n".join(lines), data.get("prolific_id", file_path.stem)

def query_reflection(expert: str, transcript_text: str) -> List[str]:
    prompt = REFLECTION_PROMPT_TEMPLATE.format(expert=expert, transcript=transcript_text)
    messages = [
        {"role": "system", "content": SYSTEM_MSG["content"]},
        {"role": "user", "content": prompt},
    ]
    try:
        response = dashscope.Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model="qwen-plus",
            messages=messages,
            result_format="message"
        )
        content = response['output']['choices'][0]['message']['content'].strip()
        return [line.strip(" ") for line in content.split("\n") if line.strip()]
    except Exception as e:
        print(f"DashScope API error: {e}")
        return []
    
def process_single_transcript(file_path: Path):
    transcript_text, agent_id = load_transcript(file_path)
    reflection = {}
    for expert in EXPERTS:
        try:
            items = query_reflection(expert, transcript_text)
            reflection[expert] = items
        except Exception as e:
            print(f"Error processing {agent_id} for expert {expert}: {e}")
            reflection[expert] = []
    output = {"agent_id": agent_id, "reflection": reflection}
    output_path = OUTPUT_DIR / f"{agent_id}.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved reflection for {agent_id}")

def main():
    transcript_files = list(TRANSCRIPT_DIR.glob("*.json"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_single_transcript, f) for f in transcript_files]
        for _ in as_completed(futures):
            pass

if __name__ == "__main__":
    main()