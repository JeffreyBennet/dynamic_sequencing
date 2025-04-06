def run_agent(input_data):
    completed = input_data.get("completed_tickets", [])
    return {
        "updated_milestones": completed,
        "summary": f"Marked {len(completed)} roadmap items as complete"
    }