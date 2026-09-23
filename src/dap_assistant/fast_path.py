"""Conservative, auditable fast path for explicit life-event queries.

Ambiguous questions still reach Qwen. These rules do not infer legal facts or
eligibility; they only avoid redundant LLM classification/planning calls.
"""
from __future__ import annotations

import re

from .llm import Classification, Plan, PlannedTask
from .conversation import subtopic_domain
from .context_engineering.information_needs import question_needs

_DOMAIN_PATTERNS = {
    'vehicle': r'\b(?:aut[oó]\w*|g[eé]pj[aá]rm[uű]\w*|kocsi\w*)\b',
    'employment': r'\b(?:munkaviszony\w*|munkahely\w*|mukahely\w*|munk[aá]\w*|[aá]ll[aá]s\w*|j[aá]rad[eé]k\w*|foglalkoztat\w*)\b',
}


def classify_explicit(question: str) -> Classification | None:
    """Return a result only for clearly mentioned supported domains (max two)."""
    text = question.casefold()
    domains = [domain for domain, pattern in _DOMAIN_PATTERNS.items()
               if re.search(pattern, text, flags=re.IGNORECASE)]
    # Administrative subtopics can independently identify a life event even
    # when the user omits "autó" in the follow-up.
    topic_domain = subtopic_domain(text)
    if topic_domain and topic_domain not in domains:
        domains.append(topic_domain)
    if not domains or len(domains) > 2:
        return None
    role = ''
    if 'vehicle' in domains:
        selling = bool(re.search(
            r'\b(?:elad\w*|eladt\w*|elad[oó]\w*|adtam\s+el|'
            r'el\s+(?:szeretn[eé]m|szeretn[eé]k|akarom|fogom)\s+adni)\b', text))
        buying = bool(re.search(
            r'\b(?:meg(?:v[aá]s[aá]r\w*|vett\w*|venn\w*)|v[aá]s[aá]rol\w*|v[aá]s[aá]rl\w*|meg?vett\w*|venn\w*|venni|megvenn\w*|megvenni|v[eé]tel\w*|vettem|vesz\w*)\b', text))
        if selling != buying:
            role = 'seller' if selling else 'buyer'
    stage = 'unknown'
    if any(token in text for token in ('vásároltam', 'megvettem', 'vettem', 'eladtam',
                                        'adtam el', 'elvesztettem', 'munkanélküli lettem',
                                        'megszűnt', 'kaptam meg', 'elindítottam')):
        stage = 'after_event'
    elif any(token in text for token in ('szeretnék', 'szeretném', 'tervezem',
                                        'vennék', 'vásárolni', 'eladni', 'felmondanék')):
        stage = 'planning'
    return Classification(domains=domains, intents=[f'{domain}_information' for domain in domains],
                          role=role, stage=stage)


# User-request-derived facets, not benchmark labels or fabricated legal facts.
# The ordering is stable so branch t1..tN maps predictably to distinct needs.
_COMPLETE_FACETS = {
    'vehicle': (
        ('steps', 'használt autó tulajdonosváltás ügyintézési lépések sorrendje'),
        ('documents', 'gépjármű adásvételi szerződés szükséges dokumentumok okmányok'),
        ('deadline', 'gépjármű átírás határidők kezdő esemény'),
        ('insurance', 'kötelező gépjármű-felelősségbiztosítás tulajdonosváltás'),
        ('costs', 'eredetiségvizsgálat okmánydíjak vagyonszerzési illeték KGFB költségek Ft'),
    ),
    'employment': (
        ('steps', 'álláskeresőként nyilvántartásba vétel ügyintézés'),
        ('documents', 'munkaviszony megszűnése kilépő dokumentumok foglalkoztatási igazolás'),
        ('supports', 'álláskeresési járadék és elérhető ellátások támogatások'),
        ('deadline', 'munkaviszony megszűnése ügyintézés határidők járadék folyósítás'),
        ('healthcare', 'munkaviszony megszűnése egészségügyi jogosultság TB'),
        ('costs', 'munkaviszony megszűnése pénzügyi teendők járulék és járandóságok'),
    ),
}


_SELLER_FACETS = (
    ('steps', 'gépjármű eladás utáni bejelentés ügyintézés lépései'),
    ('documents', 'gépjármű eladás adásvételi szerződés szükséges iratok'),
    ('deadline', 'gépjármű eladó tulajdonosváltás bejelentés határideje'),
    ('insurance', 'gépjármű eladás kötelező biztosítás megszüntetése'),
)


def _comprehensive_plan(question: str, domain: str) -> Plan | None:
    """Decompose explicitly comprehensive requests into domain-specific searches.

    Routing depends on the question, never on benchmark labels."""
    normalized = question.casefold()
    if not (re.search(r'\bteljes\b', normalized)
            and re.search(r'\b(?:terv\w*|ügyintézés\w*|teendő\w*)\b', normalized)):
        return None
    if domain not in _COMPLETE_FACETS:
        return None
    buying = bool(re.search(r'v[aá]s[aá]r|vett|vettem|v[eé]tel', normalized))
    selling = bool(re.search(r'elad|adtam\s+el', normalized))
    seller = domain == 'vehicle' and selling and not buying
    anchor = ('autó eladása után' if seller else
              'használt autó vásárlása után' if domain == 'vehicle' and buying and not selling else
              'gépjármű tulajdonosváltás' if domain == 'vehicle' else
              'munkaviszony megszűnése után')
    facets = (_SELLER_FACETS + ((('costs', 'gépjármű eladás költségek díjak Ft forint'),)
                              if question_needs(question).costs else ())
              if seller else _COMPLETE_FACETS[domain])
    tasks = [
        PlannedTask(task_id=f't{index}', domain=domain,
                    question=f'{anchor}: {description}', facet=facet)
        for index, (facet, description) in enumerate(facets, start=1)
    ]
    return Plan(tasks=tasks)


def plan_explicit(question: str, domains: list[str]) -> Plan | None:
    """Decompose comprehensive requests; preserve a lean path for simple ones."""
    if not domains or len(domains) > 2:
        return None
    if len(domains) == 1:
        detailed = _comprehensive_plan(question, domains[0])
        if detailed is not None:
            return detailed
    needs = question_needs(question)
    # Do not split ordinary procedural+cost requests into six expensive RAG
    # searches. The full-plan path above is explicitly requested by the user.
    if len(domains) == 1 and needs.steps and needs.costs:
        domain = domains[0]
        procedural = re.sub(r'(?:\s+[ée]s\s+)?milyen\s+k[öo]l?ts?[ée]g\w*.*$', '', question, flags=re.I).strip()
        tasks = [
            PlannedTask(task_id='t1', domain=domain,
                        question=f'{procedural or question} – ügyintézési lépések, helyszínek és határidők'),
            PlannedTask(task_id='t2', domain=domain,
                        question=f'{question} – konkrét díjak költségek Ft illeték biztosítás'),
        ]
    else:
        tasks = [PlannedTask(task_id=f't{i}', domain=domain, question=question)
                 for i, domain in enumerate(domains, start=1)]
    return Plan(tasks=tasks)
