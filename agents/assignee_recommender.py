def run_agent(input_data):
    tickets = input_data.get("tickets", [])
    team = input_data.get("team", ["alice", "bob", "carol"])
    assignments = {ticket: team[i % len(team)] for i, ticket in enumerate(tickets)}
    return {
        "assignments": assignments,
        "summary": f"Assigned {len(tickets)} tickets to team"
    }