from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai 
import os
import uuid
import json
from datetime import datetime

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
- Minimize repetition of steps. Look in completed steps and see if what you want to do is already completed.
- You may repeat a task **if it is logically required again** based on project complexity
- The action must be small enough for a single tool or agent to handle (e.g. "create epic", "prioritize tasks", "sync to Jira")
- Assume a ticket completes a task end to end (for example organizing tickets agent with organize and make changes reflect in the jira board, this is one action)
- ❌ Do NOT combine multiple tasks like "create epic and assign it", creating and assigning are two different operations
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

def simulate_agent_run(agent_name, input_data):
    try:
        module = __import__(f"agents.{agent_name}", fromlist=["run_agent"])
        return module.run_agent(input_data)
    except Exception as e:
        return {"error": str(e), "summary": f"Failed to simulate agent [{agent_name}]"}

def log_run(step, action_item, agent, input_data, output_data):
    log = {
        "step": step,
        "timestamp": datetime.utcnow().isoformat(),
        "action_item": action_item,
        "agent": agent,
        "input": input_data,
        "output": output_data
    }
    os.makedirs("agent_logs", exist_ok=True)
    with open(f"agent_logs/step_{step}_{agent}_{uuid.uuid4().hex[:6]}.json", "w") as f:
        json.dump(log, f, indent=2)

def run_dynamic_sequence(goal):
    memory = ""
    step = 1
    steps = []
    context = {}  # shared input context

    while True:
        action_item = get_next_action(goal, memory)
        if action_item.lower().strip() == "done":
            print("🎯 All tasks completed.")
            break

        agent, confidence = predict_agent(action_item)

        input_data = {"goal": goal, "memory": memory, "step": step, "action": action_item}
        output_data = simulate_agent_run(agent, input_data)

        log_run(step, action_item, agent, input_data, output_data)

        if "error" in output_data:
            print(f"❌ Step {step} failed: {output_data['error']}")
            print("🚨 Halting execution due to agent failure.")
            break

        memory += f"\nStep {step}: {action_item}\nAgent [{agent}] → {output_data.get('summary', '...')}\n"
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