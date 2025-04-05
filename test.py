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

def predict_agent(action_item: str):
    inputs = tokenizer(action_item, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
        pred_label = label_names[pred_index]

    return pred_label, probs[0][pred_index].item()

test_cases = [
    {"action_item": "create a new epic to cover our onboarding workflow", "expected_agent": "epic_generator"},
    {"action_item": "split the 'payments' epic into smaller, trackable tickets", "expected_agent": "ticket_splitter"},
    {"action_item": "review the current sprint backlog and prioritize tasks for delivery", "expected_agent": "priority_assessor"},
    {"action_item": "assign the new bugs and feature requests to the dev team based on bandwidth", "expected_agent": "assignee_recommender"},
    {"action_item": "extract any action items from yesterday’s meeting notes", "expected_agent": "meeting_note_parser"},
    {"action_item": "make sure all updates from this sprint are reflected in Jira", "expected_agent": "jira_sync_agent"},
    {"action_item": "plan out the upcoming sprint and slot in all high-priority tickets", "expected_agent": "sprint_planner"},
    {"action_item": "update the roadmap to reflect completion of the user analytics module", "expected_agent": "roadmap_updater"},
    {"action_item": "analyze sprint 22 to uncover blockers and task dependencies", "expected_agent": "dependency_mapper"},
    {"action_item": "update the status of tickets after this morning’s sync call", "expected_agent": "status_updater"}
]


for test in test_cases:
    agent, confidence = predict_agent(test["action_item"])
    print(f"Action Item: {test['action_item']}")
    print(f"Predicted: {agent}, Expected: {test['expected_agent']}, ✅ Correct? {agent == test['expected_agent']}")
    print("---")

