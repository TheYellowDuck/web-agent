"""Minimal, from-scratch web-agent scaffold.

Every component here is intentionally small and explainable: browser control,
an accessibility-tree observation layer, a typed action space, a model-agnostic
LLM client, and a ReAct + reflection loop.
"""

from agent.types import Action, Step, Task, Trajectory

__all__ = ["Action", "Step", "Task", "Trajectory"]
