"""Pydantic contracts shared by workflow, local inference and source fallback."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal['vehicle', 'employment']


class Classification(BaseModel):
    domains: list[Domain]
    intents: list[str] = Field(default_factory=list)
    role: str = ''
    stage: str = ''
    needs_clarification: bool = False
    clarification_question: str = ''

class PlannedTask(BaseModel):
    task_id: str = Field(pattern=r'^t[1-6]$')
    domain: Domain
    question: str = Field(min_length=5)
    depends_on: list[str] = Field(default_factory=list)
    facet: str = ''  # Optional focused evidence need; empty for legacy/model plans.

class Plan(BaseModel):
    tasks: list[PlannedTask] = Field(min_length=1, max_length=6)

class Claim(BaseModel):
    text: str
    evidence_ids: list[str]
    supporting_quote: str = Field(min_length=8, description='Exact verbatim excerpt from cited evidence')
    category: Literal['steps', 'where', 'deadline', 'documents', 'insurance', 'cost', 'benefit_amount', 'support', 'eligibility', 'healthcare'] = 'steps'
    origin: Literal['model_generated', 'source_supplement', 'source_only', 'tool_extract', 'unknown'] = 'unknown'
    in_model_prompt: bool | None = None
    claim_id: str = ''  # Assigned from exact claim + quote + evidence in the answer audit.

class Draft(BaseModel):
    claims: list[Claim]
    model_context_evidence_ids: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    disclaimer: str = 'Nem hivatalos, tájékoztató prototípus; ellenőrizd a hivatkozott forrásokat.'

class FastSelection(BaseModel):
    """Tiny schema: Python, not the LLM, extracts verbatim source statements."""
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)

class FacetSelection(BaseModel):
    """One independently inspectable evidence set per information need."""
    evidence_by_need: dict[str, list[str]] = Field(default_factory=dict)
    missing_needs: list[str] = Field(default_factory=list)

class NaturalClaim(BaseModel):
    evidence_id: str
    text: str = Field(min_length=12, max_length=360)
    category: Literal['steps', 'where', 'deadline', 'documents', 'insurance',
                      'cost', 'benefit_amount', 'support', 'eligibility', 'healthcare'] = 'steps'

class NaturalResponse(BaseModel):
    claims: list[NaturalClaim] = Field(default_factory=list, max_length=10)
    disclaimer: str = ''
