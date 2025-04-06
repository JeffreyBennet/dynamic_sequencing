def run_agent(input_data):
    tickets = input_data.get("tickets", [])
    sprint_name = "Sprint-21"
    return {
        "sprint": sprint_name,
        "tickets": tickets,
        "summary": f"Planned {len(tickets)} tickets into {sprint_name}"
    }