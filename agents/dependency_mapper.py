import random

def run_agent(input_data):
    tickets = input_data.get("tickets", [])
    dependencies = {ticket: f"BLOCKED-{i}" for i, ticket in enumerate(tickets[:2])}
    return {
        "dependencies": dependencies,
        "summary": f"Identified {len(dependencies)} dependencies"
    }