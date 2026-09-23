"""Bounded, deterministic conversation reference resolution.

A previous answer supplies *topic metadata*, never an instruction or a new legal
fact. The current question always wins when it explicitly names a different topic.
"""
import re
from typing import NotRequired, TypedDict


SUPPORTED_DOMAINS = frozenset({'vehicle', 'employment'})


class PreviousTurn(TypedDict):
    domains: list[str]
    role: str
    stage: str
    focus: NotRequired[str]


# Keep the last supported case across a short sequence of off-topic/clarification
# turns. The previous question/answer text is NEVER copied to a new prompt.
# A new explicit life event always overrides this metadata in classify_intent.
def previous_turn_from_messages(messages: list[dict]) -> PreviousTurn | None:
    if not messages or messages[-1].get('role') != 'assistant':
        return None
    # Bound memory to the recent conversation; only assistant report metadata
    # counts, never user-provided assistant-like text or arbitrary old topics.
    for message in reversed(messages[-16:]):
        if message.get('role') != 'assistant':
            continue
        final = (message.get('report') or {}).get('final') or {}
        status = final.get('response_status')
        if status in ('unsupported', 'needs_clarification'):
            continue  # Off-topic turn does not erase the active administrative case.
        if status not in ('complete', 'partial'):
            return None  # A technical failure is not a validated case transition.
        domains = final.get('domains') or []
        if (not domains or not all(domain in SUPPORTED_DOMAINS for domain in domains)
                or not final.get('evidence')):
            return None
        role = final.get('role', '') if domains == ['vehicle'] else ''
        result = {
            'domains': list(dict.fromkeys(domains))[:2],
            'role': role if role in ('buyer', 'seller') else '',
            'stage': final.get('stage', '') if final.get('stage') in ('after_event', 'planning') else '',
        }
        # Optional structured concept, not earlier free-form messages or facts.
        if final.get('focus') in TOPIC_LABELS and TOPIC_DOMAINS[final['focus']] in domains:
            result['focus'] = final['focus']
        return result
    return None


# Only strong references to the preceding answer are inherited. A new explicit
# life event must be recognized first, even if the question contains "és ott".
_REFERENCE = re.compile(
    r'\b(?:ezek\w*|ezekkel|ezekhez|ez\w*|ehhez|enn[eé]l|arr[oó]l|'
    r'azok\w*|ott|ilyenkor|ugyan(?:ez|az|ott)\w*|el[oő]bb\w*|'
    r'kor[aá]bb\w*|hozz[aá]|vel[uü]k)\b', re.IGNORECASE,
)
_CONTINUATION = re.compile(
    r'^\s*(?:és\s+)?(?:hol|hogyan|mikor|mennyi|milyen|melyik|melyek|mi|meddig|'
    r'hova|hov[aá]|merre)\b.*\b(?:int[eé]z\w*|teend[oő]\w*|'
    r'bejelent\w*|dokumentum\w*|hat[aá]rid[oő]\w*|'
    r'irat\w*|ügy\w*|t[aá]mogat[aá]s\w*|k[oö]lts[eé]g\w*|'
    r'd[ií]j\w*|felt[eé]tel\w*|id[oő]pont\w*|'
    r'biztos[ií]t[aá]s\w*|illet[eé]k\w*|korm[aá]nyablak\w*|l[eé]p[eé]s\w*)',
    re.IGNORECASE,
)

# A short follow-up can name an administrative *subtopic* without repeating the
# life event ("mennyi az eredetvizsga?", "mennyibe kerül az átírás?"). Keep
# this allowlist narrow: bare "díj" or "biztosítás" is not a vehicle domain.
_SUBTOPICS = {
    'vehicle_inspection': re.compile(r'\b(?:eredet(?:is[eé]g)?vizsg[aá]\w*|eredetis[eé]gvizsg[aá]lat\w*)\b', re.I),
    'vehicle_transfer': re.compile(r'\b(?:[aá]t[ií]r[aá]s\w*|[aá]t[ií]rat\w*|tulajdonosv[aá]lt[aá]s\w*|vagyonszerz[eé]si\s+illet[eé]k\w*)\b', re.I),
    'vehicle_registration': re.compile(r'\b(?:forgalmi\s+enged[eé]ly\w*|t[oö]rzsk[oö]nyv\w*)\b', re.I),
    'vehicle_history': re.compile(r'\b(?:jszp\b|el[oő][eé]let\w*|kilom[eé]ter[oó]ra\w*|k[aá]rt[oö]rt[eé]net\w*)\b', re.I),
    'vehicle_insurance': re.compile(r'\b(?:kgfb\b|k[oö]telez[oő]\s+g[eé]pj[aá]rm[uű].*biztos[ií]t\w*|d[ií]jnavig[aá]tor\w*)\b', re.I),
    'employment_benefit_amount': re.compile(r'\b(?:[aá]ll[aá]skeres[eé]si\s+j[aá]rad[eé]k\w*|munkan[eé]lk[uü]li\s+(?:seg[eé]ly|j[aá]rad[eé]k)\w*).*?(?:mennyi|[oö]sszeg|p[eé]nz|kalkul[aá]tor)|\b(?:mennyi|[oö]sszeg|p[eé]nz).*?(?:j[aá]rad[eé]k|munkan[eé]lk[uü]li)\b', re.I),
    'employment_supports': re.compile(r'\b(?:t[aá]mogat[aá]s\w*|k[eé]pz[eé]si\s+t[aá]mogat[aá]s\w*|lakhat[aá]si\s+t[aá]mogat[aá]s\w*|utaz[aá]si\s+t[aá]mogat[aá]s\w*)\b', re.I),
    'employment_healthcare': re.compile(r'\b(?:tb\b|eg[eé]szs[eé]g[uü]gyi\s+(?:ell[aá]t[aá]s|szolg[aá]ltat[aá]s|j[aá]rul[eé]k)\w*|taj\w*)\b', re.I),
    'employment_severance': re.compile(r'\b(?:v[eé]gkiel[eé]g[ií]t[eé]s\w*)\b', re.I),
    'employment_unused_leave': re.compile(r'\b(?:ki\s+nem\s+vett\s+szabads[aá]g\w*|szabads[aá]gmegv[aá]lt[aá]s\w*)\b', re.I),
    'employment_termination': re.compile(r'\b(?:felmond[aá]s\w*|k[oö]z[oö]s\s+megegyez[eé]s\w*|munkaviszony\s+megsz[uű]n\w*)\b', re.I),
}
TOPIC_DOMAINS = {
    name: ('vehicle' if name.startswith('vehicle_') else 'employment')
    for name in _SUBTOPICS
}
TOPIC_LABELS = {
    'vehicle_inspection': 'gépjármű eredetiségvizsgálata',
    'vehicle_transfer': 'gépjármű átírása és vagyonszerzési illetéke',
    'vehicle_registration': 'gépjármű forgalmi engedélye és törzskönyve',
    'vehicle_history': 'gépjármű előéletének és kilométeróra-adatainak ellenőrzése',
    'vehicle_insurance': 'kötelező gépjármű-felelősségbiztosítás',
    'employment_benefit_amount': 'álláskeresési járadék összege és folyósítási ideje',
    'employment_supports': 'álláskeresők támogatásai és szolgáltatásai',
    'employment_healthcare': 'egészségügyi szolgáltatásra való jogosultság',
    'employment_severance': 'végkielégítés',
    'employment_unused_leave': 'ki nem vett szabadság elszámolása',
    'employment_termination': 'munkaviszony megszüntetésének jogcímei',
}


def detect_subtopic(question: str) -> str:
    """Canonical supported concept; never extracts personal data or legal facts."""
    return next((name for name, pattern in _SUBTOPICS.items() if pattern.search(question)), '')


def subtopic_domain(question: str) -> str:
    return TOPIC_DOMAINS.get(detect_subtopic(question), '')


def is_price_question(question: str) -> bool:
    """Centralized cost-intent detection, including combined questions and typos."""
    from .context_engineering.information_needs import question_needs
    return question_needs(question).costs


def is_followup_for_case(question: str, previous: PreviousTurn | dict) -> bool:
    """Resolve a short elliptical price question only if a valid topic is active."""
    if is_contextual_followup(question):
        return True
    focus = previous.get('focus', '')
    if focus not in TOPIC_LABELS or not is_price_question(question):
        return False
    # "És az ára?" refers to the last supported administrative subtopic;
    # an unrelated detailed question must not inherit it.
    return bool(
        len(question.split()) <= 7
        and re.search(
            r'\b(?:[aá]ra|d[ií]ja|[oö]sszege|k[oö]lts[eé]ge|ker[uü]l\w*)\b',
            question.casefold(),
        )
        and re.match(
            r'^\s*(?:[eé]s\s+)?(?:mennyi(?:be)?|[aá]ra?|d[ií]ja?|'
            r'k[oö]lts[eé]ge?|az\s+[aá]ra?)\b',
            question.casefold(),
        )
    )


_OUTSIDE_SCOPE = re.compile(
    r'\b(?:lott[oó]\w*|nyer[oő]sz[aá]m\w*|'
    r'id[oő]j[aá]r[aá]s\w*|foci\w*|meccs\w*|'
    r'horoszk[oó]p\w*|recept\w*|vicc\w*|lak[aá]s\w*|ingatlan\w*|v[aá]llalkoz\w*)\b', re.IGNORECASE,
)

_DOMAIN_ANCHORS = {
    'vehicle': {
        'buyer': 'használt autó vásárlása, vevői ügyintézés',
        'seller': 'használt autó eladása, eladói ügyintézés',
        '': 'gépjármű-adásvételi ügyintézés',
    },
    'employment': {'': 'munkaviszony megszűnéséhez kapcsolódó ügyintézés'},
}


def is_outside_scope(question: str) -> bool:
    """Only unambiguous off-topic examples; never classify legal facts with this."""
    return bool(_OUTSIDE_SCOPE.search(question))


def is_contextual_followup(question: str) -> bool:
    """Require an explicit reference or short administrative follow-up phrasing."""
    return bool(_REFERENCE.search(question) or _CONTINUATION.search(question)
                or detect_subtopic(question))


def resolve_followup(question: str, previous: PreviousTurn) -> str:
    """Add a minimal topical search anchor, not a copy of previous user text."""
    anchors = []
    for domain in previous['domains']:
        if domain in _DOMAIN_ANCHORS:
            anchor = _DOMAIN_ANCHORS[domain]
            anchors.append(anchor.get(previous['role'], anchor['']))
    stage = 'az esemény után' if previous['stage'] == 'after_event' else (
        'tervezési szakaszban' if previous['stage'] == 'planning' else '')
    prefix = '; '.join(anchors)
    if stage:
        prefix += f' ({stage})'
    topic = detect_subtopic(question) or previous.get('focus', '')
    if topic in TOPIC_LABELS and TOPIC_DOMAINS[topic] in previous['domains']:
        prefix += f'. Kapcsolódó ügy: {TOPIC_LABELS[topic]}'
    return f'Korábbi ügyintézési téma: {prefix}. Aktuális kérdés: {question}'


def is_routing_only(question: str, previous: PreviousTurn | None, hint: str = '') -> bool:
    """No index/LLM prerequisite for a request that routing will immediately close."""
    from .fast_path import classify_explicit

    if is_outside_scope(question):
        return True
    if classify_explicit(question) is not None or hint in SUPPORTED_DOMAINS:
        return False
    return not bool(previous and is_followup_for_case(question, previous))
