import os
import json
import shutil
from pathlib import Path

def copy_transcripts():
    # Get base directory
    base_dir = Path(__file__).parent.parent
    eval_dir = base_dir.parent.parent
    
    # Define paths
    agents_file = eval_dir / "experiment" / "eval" / "data" / "sf_prolific_survey" / "responses_5.11_with_geo.json"
    source_transcript = base_dir / "data" / "processed_transcript" / "65d53085794e617217d43942.json"
    output_dir = base_dir / "data" / "processed_transcript"
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read agents data
    with open(agents_file, 'r', encoding='utf-8') as f:
        agents = json.load(f)
    
    # Get all agent IDs
    agent_ids = [agent['id'] for agent in agents]
    
    # Copy transcript for each agent ID
    for agent_id in agent_ids:
        target_file = output_dir / f"{agent_id}.json"
        shutil.copy(source_transcript, target_file)
        print(f"Created transcript for agent {agent_id}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total agents processed: {len(agent_ids)}")
    print(f"Total transcripts created: {len(agent_ids)}")
    print(f"Transcripts location: {output_dir}")

if __name__ == "__main__":
    copy_transcripts() 