from typing import Dict, List, Callable, Set
import uuid
import openai

# === Agent Definitions ===

class Agent:
    def __init__(self, name: str, inputs: List[str], outputs: List[str], fn: Callable[[Dict], Dict]):
        self.name = name
        self.inputs = set(inputs)
        self.outputs = set(outputs)
        self.fn = fn

# === Agent Registry ===

class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register(self, name: str, inputs: List[str], outputs: List[str], fn: Callable[[Dict], Dict]):
        self.agents[name] = Agent(name, inputs, outputs, fn)

    def get_all(self) -> List[Agent]:
        return list(self.agents.values())

# === Simulated Agent Functions ===

def epic_generator(ctx): return {"epic_id": f"epic-{uuid.uuid4().hex[:6]}"}
def ticket_splitter(ctx): return {"ticket_ids": [f"ticket-{i}" for i in range(5)]}
def assignee_recommender(ctx): return {"assignments": {"ticket-0": "Alice", "ticket-1": "Bob"}}
def priority_assessor(ctx): return {"prioritized_tickets": ["ticket-1", "ticket-0", "ticket-2"]}
def sprint_planner(ctx): return {"sprint_id": "sprint-21"}
def jira_sync_agent(ctx): return {"jira_synced": True}
def roadmap_updater(ctx): return {"roadmap_updated": True}
def dependency_mapper(ctx): return {"blockers": []}
def status_updater(ctx): return {"status_updates": {"ticket-1": "In Progress"}}
def meeting_note_parser(ctx): return {"parsed_notes": ["fix auth", "update docs"]}

# === OpenAI Goal Output Extractor ===

def extract_goal_outputs(prompt: str, registry: AgentRegistry) -> Set[str]:
    agents_info = [
        {
            "name": agent.name,
            "inputs": list(agent.inputs),
            "outputs": list(agent.outputs)
        }
        for agent in registry.get_all()
    ]
    agents_description = "\n".join(
        f"- Name: {agent['name']}, Outputs: {agent['outputs']}"
        for agent in agents_info
    )
    formatted_prompt = f"""
Prompt: {prompt}

Break the task down into smaller tasks and identify the final outputs needed to complete the project.
Return a Python list of the needed final outputs needed using exact variable names like in the agent registry.
ONLY RETURN A PYTHON LIST like: ['agent_name1', 'agent_name2']
Here are agents you can use:
{agents_description}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an AI that extracts final outputs required to complete a software project."},
            {"role": "user", "content": f"""Prompt: {formatted_prompt}"""}
        ],
        temperature=0
    )
    result = response.choices[0].message.content.strip()
    print(result)
    try:
        return set(eval(result))
    except Exception:
        return set()

# === Dependency Tracer ===

def get_required_agents(goal_outputs: Set[str], registry: AgentRegistry) -> Set[str]:
    required_agents = set()
    outputs_to_find = set(goal_outputs)

    while outputs_to_find:
        output = outputs_to_find.pop()
        for agent in registry.get_all():
            if output in agent.outputs:
                if agent.name not in required_agents:
                    required_agents.add(agent.name)
                    outputs_to_find.update(agent.inputs)
    return required_agents

# Convert extracted goal agent names to the actual output keys
def get_goal_output_keys(goal_agent_names: Set[str], registry: AgentRegistry) -> Set[str]:
    output_keys = set()
    for agent in registry.get_all():
        if agent.name in goal_agent_names:
            output_keys.update(agent.outputs)
    return output_keys


# === Graph Executor ===

class GraphExecutor:
    def __init__(self, registry: AgentRegistry, goal_outputs: Set[str], required_agents: Set[str]):
        self.registry = registry
        self.goal_outputs = goal_outputs
        self.required_agents = required_agents
        self.context: Dict[str, any] = {}
        self.completed_agents: Set[str] = set()
        self.trace: List[str] = []

    def can_run(self, agent: Agent) -> bool:
        return (
            agent.name not in self.completed_agents and
            agent.name in self.required_agents and
            agent.inputs.issubset(self.context.keys())
        )

    def run(self):
        while not self.goal_outputs.issubset(self.context.keys()):
            progress = False
            for agent in self.registry.get_all():
                if self.can_run(agent):
                    output = agent.fn(self.context)
                    self.context.update(output)
                    self.completed_agents.add(agent.name)
                    self.trace.append(f"✅ Ran [{agent.name}] → {output}")
                    progress = True
                    break
            if not progress:
                self.trace.append("⚠️ No further progress possible — missing dependencies.")
                break
        return self.trace

# === Main Test Runner ===

def main():
    prompt = (
        "We are launching a new health app next month. I need you to create the necessary epics and tickets only "
        "and assign them to the right people. Also, prioritize the tickets and plan the sprint."
    )

    registry = AgentRegistry()
    registry.register("epic_generator", [], ["epic_id"], epic_generator)
    registry.register("ticket_splitter", ["epic_id"], ["ticket_ids"], ticket_splitter)
    registry.register("assignee_recommender", ["ticket_ids"], ["assignments"], assignee_recommender)
    registry.register("priority_assessor", ["ticket_ids"], ["prioritized_tickets"], priority_assessor)
    registry.register("sprint_planner", ["prioritized_tickets"], ["sprint_id"], sprint_planner)
    registry.register("jira_sync_agent", ["sprint_id"], ["jira_synced"], jira_sync_agent)
    registry.register("roadmap_updater", ["jira_synced"], ["roadmap_updated"], roadmap_updater)
    registry.register("dependency_mapper", ["ticket_ids"], ["blockers"], dependency_mapper)
    registry.register("status_updater", ["assignments"], ["status_updates"], status_updater)
    registry.register("meeting_note_parser", [], ["parsed_notes"], meeting_note_parser)

    goal_agent_names = extract_goal_outputs(prompt, registry)
    print(f"🎯 Extracted Goal Outputs: {goal_agent_names}\n")

    goal_output_keys = get_goal_output_keys(goal_agent_names, registry)
    required_agents = get_required_agents(goal_output_keys, registry)

    executor = GraphExecutor(registry, goal_output_keys, required_agents)
    trace = executor.run()

    print("🧠 Execution Trace:\n")
    for line in trace:
        print(line)

if __name__ == "__main__":
    import os
    main()
