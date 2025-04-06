def run_agent(input_data):
    tickets = input_data.get("tickets", [])
    prioritized = sorted(tickets)
    return {
        "prioritized_tickets": prioritized,
        "summary": f"Prioritized {len(prioritized)} tickets"
    }