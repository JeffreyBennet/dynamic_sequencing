def run_agent(input_data):
    tickets = input_data.get("tickets", [])
    updates = {ticket: "In Progress" for ticket in tickets}
    return {
        "updates": updates,
        "summary": f"Updated statuses for {len(tickets)} tickets"
    }