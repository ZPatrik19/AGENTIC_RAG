"""Conservative answer-level guards; source retrieval and the index stay unchanged.

These are deterministic filters of *obviously* unrelated pre-sale background, not
semantic relevance labels or independent legal judgments.
"""
from __future__ import annotations

import re

from ..context_engineering.question_analysis import situation


def post_purchase_procedure(question: str, domain: str, role: str = '') -> bool:
    q = question.casefold()
    return bool(domain == 'vehicle' and role != 'seller'
                and any(term in q for term in ('vettem', 'vásároltam', 'megvettem', 'vásárlás után', 'adásvétel után'))
                and any(term in q for term in ('teendő', 'ügyintéz', 'átírás', 'átírat', 'mit kell', 'mi a dolgom'))
                and not any(term in q for term in ('vásárlás előtt', 'vétel előtt', 'szavatosság', 'rejtett hiba', 'átvizsgálás')))


def pre_purchase_background(text: str) -> bool:
    """Filter inspection/warranty facts only; keep *post-sale* origin inspections."""
    low = text.casefold()
    return any(term in low for term in (
        'műszaki állapot', 'szervizkönyv', 'szervizek száml', 'szakértőt fogad',
        'szakértővel átvizsgál', 'személyes megtekintéskor', 'megtekintéskor ellenőriz',
        'kellékszavatosság', 'kárigényed', 'rejtett hibá',
    ))


def low_value_procedural_unit(text: str, question: str, domain: str,
                              role: str = '') -> bool:
    """Filter unrelated procedural navigation without excluding requested legal steps or deadlines."""
    low = ' '.join(str(text).casefold().split())
    q = question.casefold()
    if post_purchase_procedure(question, domain, role):
        if ('ügyintézési határidő' not in q
                and 'ha jogszabály eltérően nem rendelkezik' in low
                and 'ügyintézési határidő' in low):
            return True
        if ('ügyintézési határidő' not in q
                and 'a rendelet hatálya alá tartozó eljárásokban' in low
                and 'ügyintézési határidő' in low):
            return True
        if ('az oldal' in low and 'foglalkozik' in low
                and not any(term in low for term in ('átírat', 'átírás', 'köt', 'bejelent'))):
            return True
    if (domain == 'employment' and any(term in q for term in
            ('elvesztettem a munk', 'megszűnt a munkaviszony', 'munkanélkül'))
            and any(term in q for term in ('teendő', 'ügyintéz', 'mit kell', 'mi a dolgom'))):
        # A broad life-event answer cannot establish the CURRENT validity of
        # a dated authentication workflow from a historical source snippet.
        # Leave explicit questions about AVDH accessible to the retrieval path.
        if 'avdh' in low and 'avdh' not in q:
            return True
        if ('u1' not in q and
                ('bővebben itt tájékozódhat' in low or 'itt tájékozódhat' in low)):
            return True
        if (not any(term in q for term in ('tb kiskönyv', 'oep igazolvány'))
                and ('tb kiskönyv' in low or 'oep igazolvány' in low)
                and not any(term in low for term in ('szükséges', 'kérd', 'visszaad', 'kiad'))):
            return True
    return False


def healthcare_eligibility_unit(text: str) -> bool:
    """Distinguish an actual entitlement/check from a document named after insurance."""
    low = ' '.join(str(text).casefold().split())
    if ('tb kiskönyv' in low or 'oep igazolvány' in low):
        return False
    return any(term in low for term in (
        'amíg kapod az álláskeresési járadékot', 'járadékot kapsz',
        'járadék folyósítása', 'járadékot kap',
        'egészségügyi szolgáltatásra való jogosultság',
        'egészségügyi jogosultság', 'biztosítási jogviszonyod',
    ))


_HEALTHCARE_EFFECT = re.compile(
    r'(?:nem\s+kell.{0,65}(?:tb|járulék)|ingyenes.{0,45}ellátás|'
    r'igénybe\s+vehet.{0,55}(?:egészségügyi|ellátás)|'
    r'megmarad.{0,45}biztosítási\s+jogviszony)', re.I)
_HEALTHCARE_CONDITION = re.compile(
    r'(?:amíg.{0,80}járadék|járadék.{0,65}(?:folyósítás|ideje|kapod|kapja|alatt)|'
    r'(?:ha|amikor).{0,85}járadék|járadékot\s+kap)', re.I)


def missing_healthcare_condition(text: str, supporting_quote: str, category: str,
                                 domain: str = 'employment') -> bool:
    """Require source-backed eligibility conditions before displaying a healthcare claim."""
    if domain != 'employment' or category != 'healthcare':
        return False
    if not _HEALTHCARE_EFFECT.search(text):
        return False
    # An adjacent conditional sentence alone does not prove a *different*
    # asserted benefit. Keep each entitlement effect in the same source unit.
    low_text, low_quote = text.casefold(), supporting_quote.casefold()
    effect_terms = ('járulék', 'ellátás', 'biztosítási jogviszony')
    effect_is_in_quote = any(term in low_text and term in low_quote for term in effect_terms)
    return not (effect_is_in_quote and _HEALTHCARE_CONDITION.search(text)
                and _HEALTHCARE_CONDITION.search(supporting_quote))


_PRE_SALE = (
    'szervizkönyv', 'szervizek száml', 'szervizszáml', 'vevő személyes megtekint',
    'személyes megtekintéskor', 'hirdetést', 'hirdetéshez', 'hirdetést ad',
    'eladás előtt', 'értékesítés előtt', 'szerződést 4 példányban',
    'szerződést négy példányban', '2 tanú jelenlétében', 'két tanú jelenlétében',
)
_PRE_SALE_EXPLICIT = ('szervizkönyv', 'szervizszámla', 'szerviz', 'hirdetés',
                      'tanú', 'példány', 'szerződés megköt', 'szerződés kitölt')


def relevant_unit(text: str, question: str, domain: str, role: str = '',
                  stage: str = '') -> bool:
    """Exclude clearly unrelated pre-sale text; retain ambiguous and post-sale document requirements."""
    if not (domain == 'vehicle' and role == 'seller'
            and situation(question, domain, role, stage) == 'after_event'):
        return True
    q = question.casefold()
    if any(term in q for term in _PRE_SALE_EXPLICIT):
        return True
    low = ' '.join(str(text).casefold().split())
    # A bare legal-document title is not a post-sale obligation. Keep it
    # accessible when the user explicitly requests the legal source itself.
    if re.match(r'^\d{4}\.\s*évi\b.*\btörvény\b', low):
        return any(term in q for term in ('melyik törvény', 'jogszabály', 'törvény címe'))
    if any(term in low for term in _PRE_SALE):
        return False
    # A generic seller's next-steps answer should not include unrelated
    # insurer offer-handling deadlines or a regulation's scope clause. These
    # remain available if the user explicitly asks about those subjects.
    if ('ajánlat elutasítás' in low or 'ajánlattevő üzemben tartó' in low):
        return 'ajánlat' in q and ('elutasít' in q or 'biztosít' in q)
    if ('joghatósága alól mentességet' in low or 'a rendelet hatálya alá tartozó' in low):
        return 'joghatóság' in q or 'rendelet hatály' in q
    if (('kellékszavatosság' in low or
         ('eladás utáni első' in low and 'bizonyítania' in low and 'probléma' in low))
            and not any(term in q for term in ('szavatosság', 'hiba', 'reklamáció'))):
        return False
    if ('magánokirat nem felel meg' in low or 'magánokirat nem felel' in low):
        return 'hibás' in q or 'magánokirat' in q or 'elutasít' in q
    # Do not mix optional pre-sale inspection or authority processing times
    # into a generic seller checklist. Explicit specialist questions still work.
    if 'eredetiségvizsgálat' in low and not any(
            term in q for term in ('eredetiségvizsgálat', 'eredetvizsga')):
        return False
    if ('nyilvántartásba veszi az eladást' in low and 'napon belül' in low
            and not any(term in q for term in ('nyilvántartásba', 'hatóság mennyi', 'mennyi idő'))):
        return False
    if ('az oldal' in low and 'foglalkozik' in low
            and not any(term in q for term in ('oldal', 'tájékoztató'))):
        return False
    return True
