import uuid

def run_agent(input_data):
    goal = input_data.get("goal", "Unnamed Goal")
    epic_id = str(uuid.uuid4())[:8]
    return {
        "epic_id": epic_id,
        "summary": f"Epic created for: {goal}"
    }