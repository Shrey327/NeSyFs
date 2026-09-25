from __future__ import annotations

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from nesyfs.models import (
    ActionProposal,
    EntityNames,
    Fact,
    MemoryDelta,
    Prediction,
    Reflection,
)


def _structured(model: ChatOpenAI, schema, prompt: str):
    result = model.with_structured_output(schema).invoke(prompt)
    return result if isinstance(result, schema) else schema.model_validate(result)


class NeSyFSLanguageModel:
    """LLM tasks used by fast reaction, reflection, graph updates, and planning."""

    def __init__(self, model_name: str, api_key: str):
        self.model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
        # create_agent is the reactive policy. Environment execution stays outside
        # the agent so an action can be reflected on before it has side effects.
        self._fast_agent = create_agent(
            model=self.model,
            tools=[],
            system_prompt=(
                "You are the fast-thinking policy in a partially observable environment. "
                "Choose one valid next action using the task, latest observation, and "
                "retrieved graph facts. Do not claim to execute it. Return the action "
                "and a short rationale."
            ),
            response_format=ActionProposal,
            name="nesyfs_fast_policy",
        )

    def propose_action(
        self, task: str, observation: str, facts: list[Fact], feedback: str = ""
    ) -> ActionProposal:
        prompt = (
            f"Task:\n{task}\n\nLatest observation:\n{observation}\n\n"
            f"Relevant belief graph facts:\n{_facts_text(facts)}\n\n"
            f"Reflection feedback on the previous proposal:\n{feedback or 'None'}"
        )
        result = self._fast_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
        value = result.get("structured_response")
        return value if isinstance(value, ActionProposal) else ActionProposal.model_validate(value)

    def reflect(
        self, task: str, observation: str, action: str, facts: list[Fact]
    ) -> Reflection:
        return _structured(
            self.model,
            Reflection,
            "Judge from a third-person perspective whether the proposed action is "
            "consistent with progress toward the task, given the latest observation "
            "and graph facts. Reject unsafe, impossible, or clearly irrelevant actions. "
            "Return aligned and concise corrective feedback.\n\n"
            f"Task: {task}\nObservation: {observation}\nAction: {action}\n"
            f"Graph facts:\n{_facts_text(facts)}",
        )

    def extract_entities(self, text: str) -> EntityNames:
        return _structured(
            self.model,
            EntityNames,
            "Extract concrete named entities from the task and observation that are "
            "useful for retrieving environment state facts. Keep names short and "
            "faithful to the text. Return an empty list if there are none.\n\n" + text,
        )

    def extract_memory(
        self, action: str, observation: str, existing_facts: list[Fact]
    ) -> MemoryDelta:
        return _structured(
            self.model,
            MemoryDelta,
            "Convert the environment action and new observation into concise, atomic "
            "subject-predicate-object facts. Only record explicit or strongly implied "
            "state. If a new fact makes an existing fact about the same property false, "
            "include that existing fact's id in superseded_fact_ids. Do not remove facts "
            "merely because they are absent from this observation.\n\n"
            f"Action: {action}\nObservation: {observation}\n"
            f"Existing facts:\n{_facts_text(existing_facts)}",
        )

    def predict(self, task: str, observation: str, action: str, facts: list[Fact]) -> Prediction:
        return _structured(
            self.model,
            Prediction,
            "Simulate a plausible next environment observation for this candidate "
            "action, using the current partial belief state. Then label whether this "
            "transition makes concrete progress toward the task. Do not assume hidden "
            "facts without support.\n\n"
            f"Task: {task}\nCurrent observation: {observation}\n"
            f"Candidate action: {action}\nGraph facts:\n{_facts_text(facts)}",
        )


def _facts_text(facts: list[Fact]) -> str:
    return "\n".join(
        f"[{fact.id}] ({fact.subject}, {fact.predicate}, {fact.object})"
        for fact in facts
    ) or "No relevant facts retrieved."
