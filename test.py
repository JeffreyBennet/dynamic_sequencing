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
label_names = [
    "epic_generator",
    "ticket_splitter",
    "priority_assessor",
    "assignee_recommender",
    "sprint_planner",
    "dependency_mapper",
    "status_updater",
    "jira_sync_agent",
    "meeting_note_parser",
    "roadmap_updater"
]

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

goal = "Turn our meeting into Jira tasks"
working_memory = "Meeting notes: discussed UI polish, testing, release date"

agent, confidence = predict_agent(goal, working_memory)
print(f"Predicted agent: {agent} (confidence: {confidence:.2f})")
