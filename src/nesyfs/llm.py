from __future__ import annotations

from collections import OrderedDict
import json
import re
import time

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from nesyfs.config import MLX_PROVIDERS
from nesyfs.models import (
    ActionProposal,
    EntityNames,
    Fact,
    MemoryDelta,
    Prediction,
    Reflection,
)


# Per-call output caps for local models. Each schema is short JSON; a hard cap stops
# a rambling small model from burning GPU time on tokens that are thrown away.
LOCAL_MAX_TOKENS = {"entities": 128, "reflect": 160, "propose": 192, "predict": 256, "memory": 384}
MEMO_SIZE = 256
_THINK = re.compile(r"\A\s*<think>[\s\S]*?</think>\s*", re.IGNORECASE)


def _structured(model: ChatOpenAI, schema, prompt: str, local: bool = False,
                max_tokens: int | None = None, stats: dict | None = None):
    stats = {} if stats is None else stats
    if local:
        # MLX's OpenAI-compatible server does not constrain JSON-schema output.
        # One correction is allowed; invalid schemas are never silently accepted.
        messages = [
            ("system", "Return only a valid JSON object matching this schema. "
             "Do not include markdown or explanatory text. All required strings must "
             "be nonempty. When there are no items, use an empty list. Schema: "
             + json.dumps(schema.model_json_schema())),
            ("user", prompt),
        ]
        options = {"max_tokens": max_tokens} if max_tokens else {}
        for attempt in range(2):
            result = model.invoke(messages, **options)
            _add_usage(stats, result)
            content = result.content
            if isinstance(content, str):
                # Qwen3 can still emit an empty or full think block; it is never part of the JSON.
                content = _THINK.sub("", content)
                stats["raw"] = content[:2000]
                content = re.sub(r"\A\s*```(?:json)?\s*\n([\s\S]*?)\n```\s*\Z", r"\1", content, flags=re.IGNORECASE)
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                stats["retries"] = stats.get("retries", 0) + 1
                if attempt == 1:
                    raise
                details = [{"location": item["loc"], "type": item["type"]}
                           for item in error.errors(include_input=False, include_url=False)]
                messages.extend([
                    ("assistant", result.content),
                    ("user", "The JSON did not match the schema: " + json.dumps(details[:8]) +
                     ". Return corrected JSON only. Do not invent facts; use an empty array when needed."),
                ])
    result = model.with_structured_output(schema).invoke(prompt)
    return result if isinstance(result, schema) else schema.model_validate(result)


def _add_usage(stats: dict, result) -> None:
    usage = getattr(result, "usage_metadata", None)
    if not isinstance(usage, dict):
        return
    cached = (usage.get("input_token_details") or {}).get("cache_read") or 0
    for key, value in (("prompt_tokens", usage.get("input_tokens")),
                       ("cached_tokens", cached), ("completion_tokens", usage.get("output_tokens"))):
        if isinstance(value, int):
            stats[key] = stats.get(key, 0) + value


class NeSyFSLanguageModel:
    """LLM tasks used by fast reaction, reflection, graph updates, and planning."""

    def __init__(self, model_name: str, api_key: str, provider: str = "openai", on_call=None):
        # on_call(record) receives one dict per model task, for live inspection.
        self.on_call = on_call
        # Exact-prompt memo for local greedy decoding: identical prompts (reflection
        # votes, identical planner particles) give identical output, so regenerate nothing.
        self._memo: OrderedDict = OrderedDict()
        if provider != "openai" and provider not in MLX_PROVIDERS:
            raise ValueError("Unsupported model provider")
        self._local = provider in MLX_PROVIDERS
        if self._local:
            expected_model, base_url, _ = MLX_PROVIDERS[provider]
            if model_name != expected_model:
                raise ValueError("Unexpected local model")
            self.model = ChatOpenAI(
                model=model_name, api_key="local-only", base_url=base_url,
                temperature=0, timeout=120, max_retries=0, max_tokens=512,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        else:
            options = {"reasoning_effort": "none"} if model_name == "gpt-6-luna" else {}
            self.model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0, **options)
        # create_agent is the reactive policy. Environment execution stays outside
        # the agent so an action can be reflected on before it has side effects.
        self._policy_prompt = (
                "You are the fast-thinking policy in a partially observable environment. "
                "Choose one valid next action using the task, latest observation, and "
                "retrieved graph facts. Do not claim to execute it. Return the action "
                "and a short rationale."
        )
        if not self._local:
            self._fast_agent = create_agent(
                model=self.model,
                tools=[],
                system_prompt=self._policy_prompt,
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
        if self._local:
            return self._call("propose", ActionProposal, self._policy_prompt + "\n\n" + prompt)
        def remote(stats):
            result = self._fast_agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            value = result.get("structured_response")
            return value if isinstance(value, ActionProposal) else ActionProposal.model_validate(value)
        return self._call("propose", ActionProposal, prompt, run=remote)

    def _call(self, kind: str, schema, prompt: str, run=None):
        key = (kind, prompt)
        stats: dict = {}
        started = time.perf_counter()
        if self._local and key in self._memo:
            self._memo.move_to_end(key)
            value, memo = self._memo[key], True
        else:
            memo = False
            value = (run(stats) if run else _structured(
                self.model, schema, prompt, local=self._local,
                max_tokens=LOCAL_MAX_TOKENS.get(kind) if self._local else None, stats=stats))
            if self._local:
                self._memo[key] = value
                if len(self._memo) > MEMO_SIZE:
                    self._memo.popitem(last=False)
        if self.on_call is not None:
            self.on_call({"kind": kind, "memo": memo, "ms": round((time.perf_counter() - started) * 1000),
                          "prompt": prompt[-1500:], "output": value.model_dump(), **stats})
        # Callers may mutate results (planner fallback rewrites the action); keep the memo clean.
        return value.model_copy(deep=True)

    def reflect(
        self, task: str, observation: str, action: str, facts: list[Fact]
    ) -> Reflection:
        return self._call(
            "reflect",
            Reflection,
            "Judge from a third-person perspective whether the proposed action is "
            "consistent with progress toward the task, given the latest observation "
            "and graph facts. Reject unsafe, impossible, or clearly irrelevant actions. "
            "Return aligned and concise corrective feedback.\n\n"
            f"Task: {task}\nObservation: {observation}\nAction: {action}\n"
            f"Graph facts:\n{_facts_text(facts)}",
        )

    def extract_entities(self, text: str) -> EntityNames:
        return self._call(
            "entities",
            EntityNames,
            "Extract concrete named entities from the task and observation that are "
            "useful for retrieving environment state facts. Keep names short and "
            "faithful to the text. Return an empty list if there are none.\n\n" + text,
        )

    def extract_memory(
        self, action: str, observation: str, existing_facts: list[Fact]
    ) -> MemoryDelta:
        return self._call(
            "memory",
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
        return self._call(
            "predict",
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
