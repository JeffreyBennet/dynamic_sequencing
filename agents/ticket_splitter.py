import uuid

def run_agent(input_data):
    epic_id = input_data.get("epic_id")
    tickets = [f"TICKET-{str(uuid.uuid4())[:8]}" for _ in range(5)]
    return {
        "epic_id": epic_id,
        "tickets": tickets,
        "summary": f"Split epic {epic_id} into {len(tickets)} tickets"
    }