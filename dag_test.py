from typing import Dict, List, Callable, Set
import uuid
import pandas as pd

# === Simulated Agent Definitions ===

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

def epic_generator(context): return {"epic_id": f"epic-{uuid.uuid4().hex[:6]}"}
def ticket_splitter(context): return {"ticket_ids": [f"ticket-{i}" for i in range(5)]}
def assignee_recommender(context): return {"assignments": {"ticket-0": "Alice", "ticket-1": "Bob"}}
def priority_assessor(context): return {"prioritized_tickets": ["ticket-1", "ticket-0", "ticket-2"]}
def sprint_planner(context): return {"sprint_id": "sprint-21"}
def jira_sync_agent(context): return {"jira_synced": True}
def roadmap_updater(context): return {"roadmap_updated": True}
def dependency_mapper(context): return {"blockers": []}
def status_updater(context): return {"status_updates": {"ticket-1": "In Progress"}}
def meeting_note_parser(context): return {"parsed_notes": ["fix auth", "update docs"]}

# === Graph Executor ===

class GraphExecutor:
    def __init__(self, registry: AgentRegistry, goal_outputs: Set[str]):
        self.registry = registry
        self.goal_outputs = goal_outputs
        self.context: Dict[str, any] = {}
        self.completed_agents: Set[str] = set()
        self.trace: List[str] = []

    def can_run(self, agent: Agent) -> bool:
        return agent.name not in self.completed_agents and agent.inputs.issubset(self.context.keys())

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

# === Setup & Run Test ===

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

goal_outputs = {"roadmap_updated", "blockers"}  # final desired outputs
executor = GraphExecutor(registry, goal_outputs)
execution_trace = executor.run()

print(execution_trace)