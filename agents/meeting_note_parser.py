def run_agent(input_data):
    notes = input_data.get("notes", "")
    extracted = [f"Task from: {line.strip()}" for line in notes.split(".") if line.strip()]
    return {
        "tasks": extracted,
        "summary": f"Extracted {len(extracted)} tasks from meeting notes"
    }