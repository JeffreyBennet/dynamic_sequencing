from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai 
import os

# Load model
model_dir = "./pm-agent-selector"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

label_names = [
    'assignee_recommender', 'dependency_mapper', 'epic_generator', 'jira_sync_agent',
    'meeting_note_parser', 'priority_assessor', 'roadmap_updater',
    'sprint_planner', 'status_updater', 'ticket_splitter'
]

def predict_agent(action_item: str):
    inputs = tokenizer(action_item, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_index = torch.argmax(probs, dim=-1).item()
    return label_names[pred_index], probs[0][pred_index].item()

def get_next_action(goal, memory):
    prompt = f"""
You are a workflow orchestrator breaking down a project goal into **strictly atomic tasks**, one at a time.

🛑 Rules:
- Return **only one simple action per response**
- The action must be small enough for a single tool or agent to handle (e.g. "create epic", "prioritize tasks", "sync to Jira")
- ❌ Do NOT combine multiple tasks like "create epic and assign it"
- ❌ Do NOT give overviews like "prepare sprint" — break that down
- ✅ Example of correct atomic task: "create epic for onboarding"
- ✅ Another: "split epic into user stories"
- ✅ Another: "assign backlog tickets to developers"

---

📝 Goal: {goal}

✅ Completed steps:
{memory if memory.strip() else 'None'}

---

💡 What is the **next strictly atomic action** to perform? If everything is done, reply only with: DONE.
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a task manager who decides the next step in a project."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

def simulate_agent_run(agent_name, memory):
    return f"Agent [{agent_name}]: completed task."

def run_dynamic_sequence(goal):
    memory = ""
    step = 1
    steps = []

    while True:
        action_item = get_next_action(goal, memory)
        if action_item.lower().strip() == "done":
            print("🎯 All tasks completed.")
            break

        agent, confidence = predict_agent(action_item)
        memory += f"\nStep {step}: {action_item}\n{simulate_agent_run(agent, memory)}\n"
        steps.append((action_item, agent, confidence))

        print(f"✅ Step {step}:")
        print(f"Action Item: {action_item}")
        print(f"→ Agent: {agent} (conf: {confidence:.2f})")
        print(f"→ Working Memory:\n{memory}")
        print("—" * 40)
        step += 1

    return steps

# Example Goal
goal = (
    "We are launching a new health app next month. I need you to create the necessary epics, "
    "break them into tickets, assign them, prioritize tasks, organize them into a sprint, and sync it all to Jira. "
    "Then update the roadmap and check for blockers."
)

run_dynamic_sequence(goal)
