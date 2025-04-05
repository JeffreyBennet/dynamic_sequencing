from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load tokenizer and model from save directory
model_dir = "./pm-agent-selector"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

# Move to correct device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()


# These must match your label encoding order
label_names = ['assignee_recommender', 'dependency_mapper', 'epic_generator', 'jira_sync_agent', 'meeting_note_parser', 'priority_assessor', 'roadmap_updater', 'sprint_planner', 'status_updater', 'ticket_splitter']

def predict_agent(goal: str, working_memory: str):
    input_text = goal + " | " + working_memory
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        pred_label = label_names[pred_index]
    
    return pred_label, probs[0][pred_index].item()

test_cases = [
    {
        "goal": "Create a plan to launch the new education app",
        "working_memory": "",
        "expected_agent": "epic_generator"
    },
    {
        "goal": "Create a plan to launch the new fintech app",
        "working_memory": "Epic: 'Fintech App Launch'",
        "expected_agent": "ticket_splitter"
    },
    {
        "goal": "Prioritize tasks for sprint 20",
        "working_memory": "Tickets: UI revamp, backend refactor, load testing",
        "expected_agent": "priority_assessor"
    },
    {
        "goal": "Assign sprint tasks based on current load",
        "working_memory": "Parsed tasks: auth bug, UI cleanup, dashboard chart",
        "expected_agent": "assignee_recommender"
    },
    {
        "goal": "Turn our team meeting into Jira action items",
        "working_memory": "Meeting notes: deployment date, QA needed, update copy",
        "expected_agent": "meeting_note_parser"
    },
    {
        "goal": "Sync latest ticket updates with Jira",
        "working_memory": "",
        "expected_agent": "jira_sync_agent"
    },
    {
        "goal": "Organize tasks for Sprint 15",
        "working_memory": "Sprint 15: 10 slots open, 6 tasks available",
        "expected_agent": "sprint_planner"
    },
    {
        "goal": "Update roadmap based on completed tickets",
        "working_memory": "Synced: 10 of 20 tasks complete",
        "expected_agent": "roadmap_updater"
    },
    {
        "goal": "Analyze blocking issues in Sprint 22",
        "working_memory": "Dependency: Backend migration blocks analytics dashboard",
        "expected_agent": "dependency_mapper"
    },
    {
        "goal": "Update ticket statuses after team sync",
        "working_memory": "Meeting outcome: UI done, QA started",
        "expected_agent": "status_updater"
    }
]

for test in test_cases:
    agent, confidence = predict_agent(test["goal"], test["working_memory"])
    print(f"Goal: {test['goal'] } | Working Memory: {test['working_memory']}")
    print(f"Predicted: {agent}, Expected: {test['expected_agent']}, ✅ Correct? {agent == test['expected_agent']}")
    print("---")
