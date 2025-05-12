"""
- Age: {agent['agent'].get('age')}
- Location: ({agent['coordinates']['lat']}, {agent['coordinates']['lng']})
- Income: {agent['agent'].get('income', '')}
- Education: {agent['agent'].get('education', 'bachelor')}
- Occupation: {agent['agent'].get('occupation', '')}
- Gender: {agent['agent'].get('gender', 'unknown')}
"""

from typing import Dict, Any

dependencies = {
    "Housing Affordability": ["age", "income", "housing tenure"],
    "Neighborhood Aesthetics": ["income", "location"],
    "Infrastructure Strain": ["housing tenure", "location"],
    "Community Benefits": ["location"],
    "Small Business Impact": ["occupation"]
}

def get_prompt_first_layer(agent: Dict[str, Any], proposal: Dict[str, Any]) -> str:
    agent_lat = agent['coordinates']['lat']
    agent_lng = agent['coordinates']['lng']
    nearest_cell = None
    min_distance = float('inf')

    for cell_id, cell in proposal['cells'].items():
        bbox = cell['bbox']
        cell_lat = (bbox['north'] + bbox['south']) / 2
        cell_lng = (bbox['east'] + bbox['west']) / 2
        distance = ((agent_lat - cell_lat) ** 2 + (agent_lng - cell_lng) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            nearest_cell = cell

    def get_prompt_for_dependency(dependency: str) -> str:
        return  f"""
    You are a resident with the following attributes:
    {[f"{key}: {agent.get(key, None)}" for key in dependencies[dependency]]}

    You are considering the following rezoning proposal:
    Proposal Details:
        - Nearest Rezoning Area: A {nearest_cell['category']} zone with height limit changed to {nearest_cell['height_limit']} feet
        - Distance from Resident: {min_distance:.4f} degrees (approximately {min_distance * 111:.1f} km)
        - Default Height Limit: {proposal['height_limits']['default']} feet

    Your thoughts on the impact of this proposal regarding {dependency} are:
    """

    prompts = {
        "Housing Affordability": get_prompt_for_dependency("Housing Affordability"),
        "Neighborhood Aesthetics": get_prompt_for_dependency("Neighborhood Aesthetics"),
        "Infrastructure Strain": get_prompt_for_dependency("Infrastructure Strain"),
        "Community Benefits": get_prompt_for_dependency("Community Benefits"),
        "Small Business Impact": get_prompt_for_dependency("Small Business Impact")
    }

    return prompts

def get_prompt_second_layer(intermediate_thoughts: Dict[str, str]) -> str:
    joined_thoughts = "\n".join([f"- {dependency}: {intermediate_thoughts[dependency]}" for dependency in intermediate_thoughts])
    return f"""
You are a resident and you are considering the following rezoning proposal:
You have some thoughts on the impact of this proposal.
{joined_thoughts}
Now, you need to generate:
1. Opinion (support/oppose/neutral)
2. A brief comment explaining their stance (1-2 sentences)
3. Key themes in the comment (2-3 keywords)
Format: opinion|comment|theme1,theme2,theme3
"""

