"""Conservative surface-quality checks, not semantic or legal verification.

Keep exact source provenance and report unresolved defects instead of completing
partial legal sentences by guessing their missing words.
"""
from __future__ import annotations

import re
import hashlib
import json
from collections import defaultdict

from .filters import (missing_healthcare_condition, low_value_procedural_unit,
                           healthcare_eligibility_unit)

_WORDS = re.compile(r'[^\W_]+', re.UNICODE)
_DANGLING_END = frozenset({'a', 'az', 'és', 'vagy', 'hogy', 'ha', 'amely', 'mert', 'de',
                          'el', 'kell', 'nem', 'tól', 'től', '2026', 'január'})
_DEICTIC_START = ('ez azt jelenti', 'ezekre ', 'ennek alapján ', 'az előbbiek ',
                 'ezt követően ', 'az alábbiak ', 'a fentiek ')


def claim_identifier(claim: dict) -> str:
    """Identify a source-bound assertion independent of its presentation category or ordering."""
    material = [
        ' '.join(str(claim.get('text', '')).split()),
        ' '.join(str(claim.get('supporting_quote', '')).split()),
        sorted(set(str(i) for i in claim.get('evidence_ids', []))),
    ]
    digest = hashlib.sha256(json.dumps(material, ensure_ascii=False,
                                        separators=(',', ':')).encode('utf-8')).hexdigest()[:20]
    return f'C_{digest}'


_DEPENDENT_OBJECT_END = re.compile(
    r'\b(?:időtartamát|összegét|adatait|időpontját|feltételeit|napjait)\.$', re.I)
_STANDALONE_PREDICATE = re.compile(
    r'\b(?:kell|szükséges|lehet|igényel\w*|kiállít\w*|csatol\w*|'
    r'ellenőriz\w*|tartalmaz\w*|közl\w*|bemutat\w*|megállapít\w*|'
    r'kér\w*|köt\w*|adja|adnak|kap\w*|fizet\w*|jelent\w*|'
    r'benyújt\w*|küld\w*|folyósítják)\b', re.I)


def non_independent_source_fragment(text: str) -> bool:
    """Reject clearly subordinate legal list fragments without an independent predicate."""
    line = ' '.join(str(text).split())
    if (len(line) >= 110 and _DEPENDENT_OBJECT_END.search(line)
            and ('esetén' in line.casefold() or 'amennyiben' in line.casefold())
            and not _STANDALONE_PREDICATE.search(line)):
        return True
    return False


def is_complete_statement(text: str, *, source_unit: bool = False) -> bool:
    """Reject clear trailing fragments, without asserting linguistic correctness.

    Source tables and short headings can lack a full stop; generated *prose*
    cannot. Never add a missing legal ending or make an unsupported paraphrase.
    """
    line = ' '.join(str(text).split()).strip()
    if not line or len(line) < 8:
        return False
    lower = line.casefold()
    # A cropped source frequently ends with an orphan year followed by a dot.
    # It looks sentence-final to the punctuation test, but cannot be used as an
    # independent date-dependent administrative claim.
    if re.search(r'(?:,|\s)(?:19|20)\d{2}\.$', line):
        return False
    if lower.startswith(_DEICTIC_START):
        return False  # antecedent/condition may be in an omitted source unit
    # A date cropped from the start of a chunk can lose its year. Avoid quoting
    # it as an applicable time-based requirement until the original is checked.
    if re.match(r'^(?:január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)\s+\d{1,2}(?:[.-]|\b)', lower):
        return False
    words = _WORDS.findall(lower)
    if not words:
        return False
    if line.rstrip('\"”’)]').endswith(('.', '!', '?', '…')):
        return True
    if words[-1] in _DANGLING_END:
        return False
    if not source_unit:
        return False
    # Standalone list/table cells: allow a bounded noun phrase or priced row;
    # never treat a long unpunctuated legal clause as a complete sentence.
    if len(line) > 95 or len(words) > 12:
        return False
    if re.search(r'\b(?:kell|szükséges|ha|amennyiben|esetén|megkötésétől|belül)\b', lower):
        return False
    return True


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_WORDS.findall(' '.join(text.casefold().split())))


def same_fact(a: str, b: str) -> bool:
    """Only high-confidence exact/contained repetition, not semantic guessing."""
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return False
    if left == right:
        return True
    short, long = sorted((left, right), key=len)
    if len(short) < 7:
        return False
    return any(long[i:i + len(short)] == short for i in range(len(long) - len(short) + 1))


def clean_claims(claims: list[dict], *, question: str = '', domain: str = '',
                 role: str = '') -> tuple[list[dict], dict]:
    """Discard incomplete statements and repeated text, retaining original IDs."""
    kept: list[dict] = []
    fragments: list[str] = []
    duplicates: list[str] = []
    condition_issues: list[str] = []
    low_value: list[str] = []
    for claim in claims:
        text = str(claim.get('text', ''))
        quote = str(claim.get('supporting_quote', ''))
        if (not is_complete_statement(text, source_unit=text.strip() == quote.strip())
                or not is_complete_statement(quote, source_unit=True)
                or non_independent_source_fragment(text)
                or non_independent_source_fragment(quote)):
            fragments.append(text[:120])
            continue
        # Do not trust a model's 'steps' label for an unconditional TB entitlement.
        health_category = ('healthcare' if domain == 'employment' else claim.get('category', ''))
        if missing_healthcare_condition(text, quote, health_category, 'employment'):
            condition_issues.append(text[:120])
            continue
        if question and (low_value_procedural_unit(text, question, domain, role)
                         or low_value_procedural_unit(quote, question, domain, role)):
            low_value.append(text[:120])
            continue
        if any(same_fact(text, prior.get('text', '')) for prior in kept):
            duplicates.append(text[:120])
            continue
        # Relabel only with independently visible claim-and-source evidence.
        # Keep the text and exact quote untouched; never invent a legal fact.
        if domain in ('vehicle', 'employment'):
            category = claim_category(claim, domain)
            claim = {**claim, 'category': category}
        kept.append({**claim, 'claim_id': claim_identifier(claim)})
    return kept, {'incomplete_claims': fragments, 'duplicate_claims': duplicates,
                  'healthcare_condition_issues': condition_issues,
                  'low_value_claims': low_value,
                  'surface_check_only': True}


def render_claim_groups(claims: list[dict], by_id: dict[str, dict], headings: dict[str, str],
                        *, numbered: bool = True, show_provenance: bool = False) -> list[str]:
    """Group once per category, even when Qwen interleaves categories."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for claim in claims:
        grouped[claim.get('category', 'steps')].append(claim)
    lines: list[str] = []
    for category, group in grouped.items():
        lines.extend(['', f"**{headings.get(category, 'További tudnivalók')}**", ''])
        for i, claim in enumerate(group, 1):
            citations = ' '.join(f"[Forrás {eid}](<{by_id[eid]['source_url']}>)"
                                 for eid in claim.get('evidence_ids', []) if eid in by_id)
            provenance = (' *(Qwen megfogalmazása)*' if claim.get('origin') == 'model_generated'
                          else ' *(utólagos forráskiegészítés)*' if claim.get('origin') == 'source_supplement'
                          else ' *(eszközből származó, szó szerinti forrásrészlet)*'
                          if claim.get('origin') == 'tool_extract' else '') if show_provenance else ''
            identifier = claim_identifier(claim)
            lines.append(f"{str(i) + '.' if numbered else '-'} [{identifier}] {claim['text']}{provenance} {citations}".rstrip())
    return lines


# Internal facet IDs must never leak into the Hungarian user-facing answer.
FACET_LABELS_HU = {
    'steps': 'ügyintézési lépések', 'deadline': 'határidők',
    'documents': 'szükséges dokumentumok', 'insurance': 'kötelező biztosítás',
    'where': 'ügyintézés helye', 'costs': 'díjak és illetékek',
    'supports': 'támogatások és szolgáltatások', 'eligibility': 'jogosultsági feltételek',
    'benefit_amount': 'járadék összege és időtartama',
    'healthcare': 'egészségügyi jogosultság feltételei',
}


def missing_facets_hungarian(facets: list[str]) -> str:
    return ', '.join(FACET_LABELS_HU.get(facet, 'további ellenőrzendő információ')
                     for facet in facets)


def source_fallback_notice(reason: str) -> str:
    """Tell users what actually failed, without claiming a timeout for valid JSON."""
    if reason == 'model_output_unusable':
        return ('**Részleges útmutató: a modell válasza elkészült, de nem tartalmazott '
                'elfogadható, teljes és forráshoz köthető állításokat; szó szerinti '
                'forrásrészletek következnek.**')
    return ('**Részleges útmutató: a helyi modellből nem érkezett érvényes, '
            'befejezett válasz; szó szerinti forrásrészletek következnek.**')


# The following are conservative text/quotation cues, not a semantic or legal judge.
_DEADLINE = re.compile(r'\b\d{1,3}\s*(?:munka)?napon\s+belül\b', re.I)
_OFFICE = ('kormányablak', 'foglalkoztatási osztály', 'járási hivatal',
           'vizsgálóállomás', 'e-papír', 'elektronikus ügyintézés',
           'digitális állampolgár mobilalkalmazás')
_DOCS = ('foglalkoztatói igazolá', 'foglalkoztatási igazolá',
         'igazolólap', 'adásvételi szerződ', 'szerződ',
         'forgalmi engedély', 'törzskönyv', 'szükséges dokumentum',
         'tb kiskönyv', 'oep igazolvány', 'nyilvántartási kére')
_DOCUMENT_ACTIONS = ('szükséges', 'igényel', 'csatol', 'bemutat', 'kiállít',
                     'állít ki', 'állítja ki', 'adja ki', 'ad ki', 'aláír',
                     'elkészít', 'magaddal', 'töltsd ki', 'kérd el',
                     'kérhet', 'benyújt')
_PROCEDURE_ONLY = ('ügyfélkapun keresztül', 'e-papír szolgáltatás',
                   'elektronikus ügyintézés', 'ügyintézési felület')


def documented_paperwork(text: str, quote: str, domain: str) -> bool:
    """Match a named document and its action in both the visible claim and exact source quote."""
    named = tuple(term for term in _DOCS if term != 'szükséges dokumentum')
    if not any(term in text and term in quote for term in named):
        return False
    if any(term in text for term in _PROCEDURE_ONLY) and not any(
            action in text and action in quote for action in _DOCUMENT_ACTIONS):
        return False
    if any(action in text and action in quote for action in _DOCUMENT_ACTIONS):
        return True
    # A literal short list item can be an identifiable document; a bare form
    # name in an employment notice does not establish it must be submitted.
    return (domain == 'vehicle' and len(text) < 75
            and not any(t in text for t in ('napon belül', 'megkötésétől számított',
                                            'határidő', 'biztosítás'))
            and text.strip(' .,:;-') == quote.strip(' .,:;-'))
_INSURANCE = ('kötelező gépjármű-felelősségbiztosítás', 'kgfb',
              'felelősségbiztosítás')
_SUPPORTS = ('támogatott nyelvtanfolyam', 'támogatott képzés',
             'képzési támogatás', 'lakhatási támogatás',
             'álláskeresési járadékot kapsz', 'álláskeresési járadékot igényelhetsz',
             'támogatásokat', 'támogatások')


def claim_facets(claim: dict, domain: str = '') -> set[str]:
    """Detect lexical facets only when visible text and its exact quote corroborate them."""
    text = ' '.join(str(claim.get('text', '')).casefold().split())
    quote = ' '.join(str(claim.get('supporting_quote', '')).casefold().split())
    if not text or not quote:
        return set()
    def shared(terms):
        return any(term in text and term in quote for term in terms)
    found: set[str] = set()
    if shared(_INSURANCE) and domain != 'employment':
        found.add('insurance')
    if shared(_OFFICE):
        found.add('where')
    if documented_paperwork(text, quote, domain):
        found.add('documents')
    # A bare year/number and a generic authority processing deadline are not
    # a vehicle-transfer deadline. Require matching napon-belül phrases.
    if (_DEADLINE.search(text) and _DEADLINE.search(quote)
            and not ('ha jogszabály eltérően nem rendelkezik' in text
                     or 'a rendelet hatálya alá tartozó eljárásokban' in text)):
        found.add('deadline')
    if domain == 'employment':
        if shared(_SUPPORTS):
            found.add('supports')
        if (healthcare_eligibility_unit(text) and healthcare_eligibility_unit(quote)
                and not missing_healthcare_condition(text, quote, 'healthcare', domain)):
            found.add('healthcare')
    if re.search(r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\b', text, re.I) and re.search(
            r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\b', quote, re.I):
        found.add('costs')
    if shared(('kérelmez', 'nyilvántartásba vétel', 'átírat', 'átírás',
               'megköt', 'bejelent', 'igényel', 'foglalkoztatási osztály',
               'eredetiségvizsgálat', 'kell kötn', 'kötni kell')):
        found.add('steps')
    # A deadline, insurance obligation, registration channel or application
    # document may itself be a next step; don't demand a separate 'steps' label.
    if found & {'deadline', 'insurance', 'where', 'documents'}:
        found.add('steps')
    return found


def claim_category(claim: dict, domain: str = '') -> str:
    """Correct clearly mismatched headings without rewriting legal language."""
    category = str(claim.get('category', 'steps'))
    facets = claim_facets(claim, domain)
    if domain == 'employment':
        # Health-entitlement text must not be left under 'Teendők', while a
        # document about insurance must not pose as a healthcare entitlement.
        if 'healthcare' in facets:
            return 'healthcare'
        if category == 'healthcare':
            return 'documents' if 'documents' in facets else 'steps'
        if category == 'documents' and 'documents' not in facets:
            return 'where' if 'where' in facets else 'steps'
        if category in ('steps', 'documents') and 'supports' in facets:
            return 'support'
        if category == 'steps' and 'documents' in facets and 'healthcare' not in facets:
            return 'documents'
    if domain == 'vehicle' and category == 'steps' and 'documents' in facets:
        return 'documents'
    if category == 'documents' and 'documents' not in facets:
        return 'where' if 'where' in facets else 'steps'
    if category == 'deadline' and 'deadline' not in facets:
        return 'steps'
    if category == 'insurance' and 'insurance' not in facets:
        return 'steps'
    return category


def structural_facet_coverage(facets: tuple[str, ...], claims: list[dict],
                              *, domain: str = '') -> dict:
    """Claim text AND exact quote lexical proxy, never a gold completeness score."""
    all_found: set[str] = set()
    fact_rows: list[dict] = []
    for i, claim in enumerate(claims, 1):
        verified_facets = claim_facets(claim, domain)
        all_found.update(verified_facets)
        fact_rows.append({'claim_index': i, 'claim_id': claim_identifier(claim),
                          'observed_facets': sorted(verified_facets),
                          'claimed_category': claim.get('category', 'steps'),
                          'evidence_ids': list(claim.get('evidence_ids', []))})
    # Legacy direct calls may not supply a domain; this intentionally does NOT
    # infer healthcare eligibility or employment supports from a category tag.
    covered = [facet for facet in facets if facet in all_found]
    missing = [facet for facet in facets if facet not in all_found]
    return {'requested': list(facets), 'covered': covered, 'missing': missing,
            'coverage': len(covered) / len(facets) if facets else None,
            'claim_evidence': fact_rows,
            'measurement': 'claim_text_and_source_quote_lexical_proxy_not_semantic_or_legal_completeness'}


def claim_provenance_report(claims: list[dict], model_context_evidence_ids: list[str] | None = None) -> dict:
    """Track claim authorship and whether its exact cited unit appeared in the model prompt."""
    ids = list(dict.fromkeys(model_context_evidence_ids or []))
    allowed = set(ids)
    rows = []
    counts = {'model_generated': 0, 'source_supplement': 0,
              'source_only': 0, 'tool_extract': 0, 'unknown': 0}
    for number, claim in enumerate(claims, 1):
        origin = claim.get('origin', 'unknown')
        if origin not in counts:
            origin = 'unknown'
        claim_ids = list(claim.get('evidence_ids', []))
        in_prompt = claim.get('in_model_prompt')
        # Never infer the exact cited quotation's presence from an evidence ID.
        if in_prompt is True and (not claim_ids or not set(claim_ids) <= allowed):
            in_prompt = None
        if in_prompt not in (True, False):
            in_prompt = None
        counts[origin] += 1
        rows.append({'claim_index': number, 'claim_id': claim_identifier(claim),
                     'origin': origin,
                     'evidence_ids': claim_ids, 'category': claim.get('category', 'steps'),
                     'quoted_unit_in_model_prompt': in_prompt})
    return {'model_context_evidence_ids': ids, 'claims': rows,
            'counts': counts, 'measurement': 'authorship_and_exact_prompt_unit_presence_not_entailment'}
