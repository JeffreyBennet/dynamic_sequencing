def run_agent(input_data):
    sprint = input_data.get("sprint", "Unknown Sprint")
    return {
        "status": "synced",
        "summary": f"Synced tasks to Jira for {sprint}"
    }