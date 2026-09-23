"""Extract fallible lexical question cues; source alignment does not establish legal obligations."""
from __future__ import annotations

import re

from .information_needs import question_needs


_GOAL_NAMES = {
    'benefit_amount': 'benefit_amount', 'eligibility': 'eligibility',
    'costs': 'cost', 'deadline': 'deadline', 'documents': 'documents',
    'where': 'where', 'supports': 'supports', 'steps': 'procedure',
}
_TOPIC_NAMES = {'costs': 'cost', 'supports': 'supports', 'steps': 'steps'}


def analyze_question(question: str, *, domain: str = '', role: str = '',
                     stage: str = '') -> dict:
    """Interpret ONLY user phrasing and already-established routing fields.

    This object contains tags, not copied user text. Unknown stages remain
    unknown; dates, benefit eligibility, and implicit user details are NOT inferred.
    """
    needs = question_needs(question)
    requested = list(needs.requested)
    narrow = [key for key in ('benefit_amount', 'eligibility', 'costs', 'deadline',
                              'documents', 'where', 'supports') if key in requested]
    goal = _GOAL_NAMES[narrow[0]] if narrow else 'procedure' if needs.steps else 'information'
    lowered = question.casefold()
    online = any(term in lowered for term in ('online', 'interneten', 'elektronikusan', 'ügyfélkapun'))
    in_person = any(term in lowered for term in ('személyesen', 'személyes', 'kormányablakban', 'hivatalban'))
    preferred_channel = ('both' if online and in_person else 'online' if online else
                         'in_person' if in_person else 'not_specified')
    return {
        'domain': domain if domain in ('vehicle', 'employment') else 'unknown',
        'role': role if domain == 'vehicle' and role in ('buyer', 'seller') else 'not_specified',
        'stage': stage if stage in ('after_event', 'planning') else 'unknown',
        'goal': goal,
        'requested_needs': requested,
        'scope': 'focused' if narrow else 'general',
        'preferred_channel': preferred_channel,
        'method': 'bounded_question_cues_not_semantic_or_legal_verification',
    }


def align_sources(question_context: dict, source_topics: list[str],
                  *, retrieved_evidence_count: int = 0) -> dict:
    """Inspect selected/retrieved topic cues without treating them as entailment.

    General procedure questions may reasonably involve several subjects; absence
    of an unasked-for topic must not be labelled as a missing legal requirement.
    """
    available = list(dict.fromkeys(source_topics))
    requested = [_TOPIC_NAMES.get(name, name) for name in question_context['requested_needs']
                 if name != 'steps']
    matched = [name for name in requested if name in available]
    absent = [name for name in requested if name not in available]
    return {
        'source_topic_hints': available,
        'requested_topic_hints': requested,
        'matched_topic_hints': matched,
        'unmatched_requested_topic_hints': absent,
        'retrieved_evidence_count': int(retrieved_evidence_count),
        'method': 'lexical_topic_overlap_not_claim_support_or_document_completeness',
    }


_AFTER_SALE = re.compile(r'\b(?:eladtam|eladtuk|eladt\w*|eladás\s+után|eladást\s+követően|már\s+eladt\w*|adtam\s+el)\b', re.I)
_AFTER_JOB = re.compile(r'(?:elvesztettem\s+(?:a\s+)?munk|megszűnt\s+(?:a\s+)?munkaviszony|'
                        r'kirúgtak|elbocsátottak|munkanélküli\s+lettem|felmondtak\s+nekem)', re.I)
_PLANNING = re.compile(r'(?:szeretn[ée]m|szeretn[ée]k|tervezem|tervezek|fogom|'
                       r'fogok|készüljek|mielőtt|eladás\s+előtt|felmondanék)', re.I)

def situation(question: str, domain: str, role: str = '', stage: str = '') -> str:
    """A confirmed event in the question wins over generic context from chat."""
    q = question.casefold()
    if domain == 'vehicle' and role == 'seller' and _AFTER_SALE.search(q):
        return 'after_event'
    if domain == 'employment' and _AFTER_JOB.search(q):
        return 'after_event'
    if _PLANNING.search(q):
        return 'planning'
    if stage in ('after_event', 'planning'):
        return stage
    return 'unknown'
