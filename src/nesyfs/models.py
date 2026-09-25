from __future__ import annotations

from pydantic import BaseModel, Field


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


class EntityNames(BaseModel):
    names: list[str] = Field(default_factory=list)


class ActionProposal(BaseModel):
    action: str = Field(min_length=1)
    rationale: str = ""


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
