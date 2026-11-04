from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Fact(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    observed_at: str
    evidence: str = ""


class ExtractedFact(BaseModel):
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str = Field(min_length=1)


class MemoryDelta(BaseModel):
    facts: list[ExtractedFact] = Field(default_factory=list)
    superseded_fact_ids: list[str] = Field(default_factory=list)

    @field_validator("facts", mode="before")
    @classmethod
    def _unary_facts(cls, value):
        # Small models write states as (fridge, closed, ""); keep them as (fridge, is, closed)
        # instead of failing the whole step, and drop facts that still lack a part.
        if not isinstance(value, list):
            return value
        facts = []
        for item in value:
            if isinstance(item, dict) and all(isinstance(item.get(k, ""), str) for k in ("subject", "predicate", "object")):
                subject, predicate, obj = (item.get(k, "").strip() for k in ("subject", "predicate", "object"))
                if subject and predicate and not obj:
                    predicate, obj = "is", predicate
                if not (subject and predicate and obj):
                    continue
                item = {"subject": subject, "predicate": predicate, "object": obj}
            facts.append(item)
        return facts


class EntityNames(BaseModel):
    names: list[str] = Field(default_factory=list)


class ActionProposal(BaseModel):
    action: str = Field(min_length=1)
    rationale: str = ""


class Lesson(BaseModel):
    lesson: str = Field(min_length=1)

    @field_validator("lesson", mode="before")
    @classmethod
    def _nested_lesson(cls, value):
        # Small models sometimes nest it: {"lesson": {"title": ..., "lesson": "..."}}.
        if isinstance(value, dict) and isinstance(value.get("lesson"), str):
            return value["lesson"]
        return value


class Reflection(BaseModel):
    aligned: bool
    feedback: str = ""


class Prediction(BaseModel):
    observation: str
    progress: bool


class StepResult(BaseModel):
    action: str
    observation: str
    planning_used: bool
    reflection_attempts: int
    rationale: str = ""
