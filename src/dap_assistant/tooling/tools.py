"""Read-only deterministic tools. No official transactions or guessed deadlines."""
from __future__ import annotations

from datetime import date, timedelta
import re
from typing import Literal

from pydantic import BaseModel, Field


class DeadlineRule(BaseModel):
    rule_id: str
    evidence_id: str
    anchor_event: str
    days: int = Field(ge=1, le=3650)
    unit: Literal['calendar_day', 'working_day']
    include_start: bool | None = None
    weekend_policy: Literal['no_extension', 'next_working_day'] | None = None
    workday_calendar_verified: bool = False


def calculate_deadline(event_date: date | None, rule: DeadlineRule, verified_evidence_ids: set[str], holidays: set[date] | None = None) -> dict:
    """No exact legal deadline when commencement/extension/workday calendar isn't verified."""
    base = {'tool': 'calculate_deadline', 'rule_id': rule.rule_id, 'source_evidence_id': rule.evidence_id,
            'documented_duration': f'{rule.days} {rule.unit}'}
    if rule.evidence_id not in verified_evidence_ids:
        return {**base, 'status': 'incomplete', 'reason': 'No verified supporting source'}
    if event_date is None:
        return {**base, 'status': 'incomplete', 'reason': 'Event date not supplied'}
    if rule.include_start is None or rule.weekend_policy is None:
        return {**base, 'status': 'incomplete', 'reason': 'Commencement or weekend extension rule unverified'}
    if rule.unit == 'working_day' and (holidays is None or not rule.workday_calendar_verified):
        return {**base, 'status': 'incomplete', 'reason': 'Official working-day calendar not verified'}
    result = event_date
    remaining = rule.days - (1 if rule.include_start else 0)
    if remaining < 0:
        return {**base, 'status': 'incomplete', 'reason': 'Invalid inclusion configuration'}
    while remaining:
        result += timedelta(days=1)
        if rule.unit == 'calendar_day' or (result.weekday() < 5 and result not in (holidays or set())):
            remaining -= 1
    if rule.weekend_policy == 'next_working_day':
        if holidays is None:
            return {**base, 'status': 'incomplete', 'reason': 'Holiday calendar missing for extension'}
        while result.weekday() >= 5 or result in holidays:
            result += timedelta(days=1)
    return {**base, 'status': 'calculated', 'date': result.isoformat()}


def build_document_checklist(evidence: list[dict]) -> dict:
    """Extract literal documentary requirements; no invented items or inferred eligibility."""
    words = ('igazolás', 'okmány', 'szerződés', 'lakcímkártya', 'taJ-kártya', 'adóigazolvány', 'jegyzőkönyv', 'dokumentum', 'forgalmi engedély')
    result: dict[str, dict] = {}
    for item in evidence:
        for line in re.split(r'[\n]+', item['text']):
            line = line.strip(' •-*\t')
            if 4 < len(line) <= 350 and any(word.lower() in line.lower() for word in words):
                key = re.sub(r'\W+', ' ', line.casefold()).strip()
                result.setdefault(key, {'text': line, 'evidence_ids': []})
                if item['evidence_id'] not in result[key]['evidence_ids']:
                    result[key]['evidence_ids'].append(item['evidence_id'])
    return {'tool': 'build_document_checklist', 'status': 'success', 'items': list(result.values())[:25]}


def extract_duration_mentions(evidence: list[dict]) -> list[dict]:
    """Surface sourced durations without guessing their legal start day or extension."""
    found = []
    for item in evidence:
        for match in re.finditer(r'\b(\d{1,3})\s*(munka)?napon?\s+belül\b', item['text'], re.IGNORECASE):
            found.append({'days': int(match.group(1)), 'unit': 'working_day' if match.group(2) else 'calendar_day',
                          'evidence_id': item['evidence_id'], 'text': item['text'][max(0, match.start()-100):match.end()+100]})
    return found[:8]


def requested_tools(question: str) -> tuple[bool, bool]:
    """Whether the user explicitly asks for a document checklist or a deadline.

    Retrieved source text must never trigger unrelated tool calls on its own.
    """
    q = question.casefold()
    checklist = any(term in q for term in (
        'dokument', 'irat', 'papír', 'igazolás', 'teend', 'intéz',
        'bejelent', 'átír', 'elintéz', 'kell', 'indít', 'lépés',
    ))
    deadlines = any(term in q for term in (
        'határid', 'mikor', 'hány nap', 'meddig', 'napom', 'napon belül',
        'számíts', 'milyen dátum',
    ))
    return checklist, deadlines


def wants_native_tool_call(question: str) -> bool:
    """Use expensive native tool selection only for an explicitly requested tool task.

    The cheap local document checklist still runs for broad procedural questions.
    A broad 'mit kell tennem' must not trigger a separate Ollama tool round trip.
    """
    low = question.casefold()
    return bool(re.search(
        r'(?:dokumentum\w*|irat\w*|igazol[aá]s\w*|okm[aá]ny\w*|'
        r'pap[ií]r\w*|iratlista|ellenőrzőlista|hat[aá]rid[oő]\w*|'
        r'mikor\s+j[aá]r\s+le|melyik\s+napig)', low,
    ))


def prepare_unemployment_benefit_calculation(question: str, evidence: list[dict]) -> dict | None:
    """Find source-backed calculation inputs and the official calculator link; do not approximate an individual award."""
    from ..context_engineering.information_needs import question_needs
    if not question_needs(question).benefit_amount:
        return None
    relevant = []
    for item in evidence:
        low = item.get('text', '').casefold()
        if any(term in low for term in ('járadékalap 60', '60 százalék', 'napi járadék összeg',
                                        'minimálbér', 'folyósítható járadék összesen',
                                        'jogosultsági idő')):
            relevant.append(item.get('evidence_id'))
    return {
        'tool': 'prepare_unemployment_benefit_calculation',
        'status': 'needs_user_input',
        'required_inputs': [
            'az álláskeresővé válást megelőző időszak jogviszonyai',
            'a havi járulékalap(ok)',
            'a nyilvántartásba vétel és a kérelem időpontja',
        ],
        'official_calculator_url': 'https://nfsz.munka.hu/tart/jaradek_kalkulator',
        'source_evidence_ids': [eid for eid in relevant if eid][:6],
        'reason': ('A személyre szabott járadékösszeg a járulékalap és a jogosultsági idő '
                   'függvénye; ezek nélkül csak a dokumentált számítási szabály ismertethető.'),
    }
