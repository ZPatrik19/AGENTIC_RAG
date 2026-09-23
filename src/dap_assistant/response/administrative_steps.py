"""Evidence-backed administrative steps, not inferred legal obligations.

This layer preserves full list/table lines, precise provenance and conditional
language. It does not decide eligibility, legal validity or current rates.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field


class AdministrativeStep(BaseModel):
    step_id: str
    title: str
    description: str
    category: Literal['steps', 'deadline', 'where', 'documents', 'insurance', 'cost', 'benefit_amount', 'support', 'eligibility', 'healthcare']
    domain: str
    role: str = 'general'
    source_evidence_ids: list[str] = Field(min_length=1)
    official_channels: list[str] = Field(default_factory=list)
    conditional: bool = False
    document_version: str | None = None


CHANNELS = ('kormányablak', 'Webes Ügysegéd', 'Digitális Állampolgár',
            'NAV Ügyfélportál', 'Vállalkozói Ügysegéd', 'e-bejelentő',
            'foglalkoztatási osztály', 'e-Papír', 'közműszolgáltató')
ACTION = re.compile(r'bejelent|átír|biztosít|igényel|szükség|kell\b|indíts|jelentkez|'
                    r'foglal|kötni|szerződés|irat|igazolás|díj|illeték|Ft\b|'
                    r'napon belül|munkanapon belül|hitel|támogatás|ügysegéd|kormányablak', re.I)
PRICE = re.compile(r'\b\d[\d\s.,–-]*\s*(?:Ft|forint)\b', re.I)


def categorize(text: str) -> str:
    low = text.casefold()
    if any(t in low for t in ('járadékalap 60', '60 százaléka', 'napi járadék összeg', 'minimálbér napi összege')):
        return 'benefit_amount'
    if any(t in low for t in ('képzési támogatás', 'lakhatási támogatás', 'utazási támogatás', 'vállalkozóvá válás támogatás')):
        return 'support'
    if any(t in low for t in ('jogosult', 'jogszerző idő', '360 nap')) and 'járadék' in low:
        return 'eligibility'
    if 'egészségügyi szolgáltatás' in low or 'tb ellátás' in low:
        return 'healthcare'
    if PRICE.search(text) or any(t in low for t in ('illetéktábla', 'illeték mértéke')):
        return 'cost'
    if 'napon belül' in low or 'munkanapon belül' in low or 'határidő' in low:
        return 'deadline'
    if 'biztosít' in low:
        return 'insurance'
    if any(channel.casefold() in low for channel in CHANNELS):
        return 'where'
    if any(word in low for word in ('okmány', 'igazolás', 'irat', 'szerződés', 'jegyzőkönyv')):
        return 'documents'
    return 'steps'


def extract_steps(evidence: list[dict], *, limit: int = 45) -> list[AdministrativeStep]:
    """Retain verbatim source lines including bare PDF/HTML bullets and table rows."""
    results: list[AdministrativeStep] = []
    seen: set[str] = set()
    for item in evidence:
        eid = str(item.get('evidence_id', ''))
        if not eid or not item.get('source_url', '').startswith('https://'):
            continue
        for line in str(item.get('text', '')).splitlines():
            original = line.strip()
            normalized = original.strip('•* -\t')
            if not 9 <= len(normalized) <= 950 or not ACTION.search(normalized):
                continue
            # Preserve original in description so the audit can verify provenance.
            key = re.sub(r'\s+', ' ', normalized).casefold()
            if key in seen:
                continue
            seen.add(key)
            channels = [channel for channel in CHANNELS if channel.casefold() in normalized.casefold()]
            category = categorize(normalized)
            results.append(AdministrativeStep(
                step_id=f"{eid}:{len(results) + 1}",
                title=normalized[:94], description=normalized,
                category=category, domain=str(item.get('domain', '')),
                role=str(item.get('role', 'general')),
                source_evidence_ids=[eid], official_channels=channels,
                conditional=bool(re.search(r'\bha\b|esetén|amennyiben|feltétel', normalized, re.I)),
                document_version=item.get('document_version'),
            ))
            if len(results) >= limit:
                return results
    return results


def needs_summary(steps: list[AdministrativeStep]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.category] = counts.get(step.category, 0) + 1
    return counts
