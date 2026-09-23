"""Evidence selection with explicit information needs and bounded, complete source units.

Budget is a conservative *estimate*, not Qwen's tokenizer count. Ollama's
prompt_eval_count is the source of truth for observed token usage.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import re

from .information_needs import question_needs
from .token_budget import estimated_tokens
from ..response.filters import relevant_unit
from ..response.quality import is_complete_statement
from ..response.filters import (post_purchase_procedure, pre_purchase_background,
                           low_value_procedural_unit, healthcare_eligibility_unit)


NEED_LABELS = {
    'steps': 'ügyintézési lépések',
    'deadline': 'határidők és kezdő esemény',
    'documents': 'szükséges okmányok',
    'insurance': 'kötelező biztosítás',
    'where': 'hivatal és online ügyintézés',
    'costs': 'díjak és illetékek',
    'supports': 'támogatások és szolgáltatások',
    'eligibility': 'jogosultsági feltételek',
    'benefit_amount': 'járadék mértéke és időtartama',
    'healthcare': 'egészségügyi jogosultság',
}


def requested_facets(question: str, domain: str, *, role: str = '') -> tuple[str, ...]:
    """Expand broad procedural questions; never equate a cost with a benefit."""
    needs = question_needs(question)
    names = list(needs.requested)
    low = question.casefold()
    if domain == 'vehicle':
        if needs.steps and role != 'seller':
            names.extend(('insurance', 'deadline', 'documents', 'where'))
        elif needs.steps:
            names.extend(('deadline', 'documents', 'where'))
        if needs.costs:
            names.append('costs')
    elif domain == 'employment':
        if needs.steps:
            names.extend(('supports', 'documents', 'where', 'healthcare'))
        if needs.supports or 'járadék' in low:
            names.extend(('eligibility', 'benefit_amount'))
    return tuple(dict.fromkeys(n for n in names if n in NEED_LABELS))


def facet_queries(question: str, domain: str, *, role: str = '', max_queries: int = 12) -> dict[str, str]:
    """Facet-guided Hungarian search; at most eight bounded retrievals per worker."""
    facets = requested_facets(question, domain, role=role)
    if domain == 'vehicle':
        stems = {
            'steps': 'gépjármű tulajdonjog változás ügyintézés lépései',
            'insurance': 'kötelező gépjármű felelősségbiztosítás megkötése tulajdonosváltás',
            'deadline': 'átírás határidő szerződés napja',
            'documents': 'gépjármű átírás szükséges dokumentumok okmányok',
            'where': 'gépjármű átírás kormányablak DÁP online ügyintézés',
            'costs': 'gépjármű átírás illeték eredetiségvizsgálat forgalmi törzskönyv díj Ft',
        }
        if role == 'seller':
            stems.update({'steps': 'autó eladás tulajdonosváltás bejelentés eladó',
                          'deadline': 'eladás bejelentés határidő eladó',
                          'documents': 'autó eladás adásvételi szerződés okmányok',
                          'where': 'tulajdonosváltás bejelentés Webes Ügysegéd kormányablak'})
    else:
        stems = {
            'steps': 'munkaviszony megszűnése álláskeresőként regisztráció teendők',
            'supports': 'álláskeresési járadék képzési támogatás álláskeresők szolgáltatásai',
            'eligibility': 'álláskeresési járadék jogosultsági feltételek jogszerző idő',
            'benefit_amount': 'álláskeresési járadék összeg folyósítás maximum időtartam',
            'documents': 'munkaviszony megszűnése kilépő iratok foglalkoztatási igazolás',
            'where': 'álláskereső regisztráció foglalkoztatási osztály e-Papír',
            'healthcare': 'álláskereső egészségügyi szolgáltatás TB jogosultság',
            'deadline': 'álláskeresési járadék igénylés bejelentés határidő',
            'costs': 'munkaviszony megszűnése költség egészségügyi járulék',
        }
    if len(facets) <= 1:
        return {facets[0] if facets else 'steps': question}
    chosen = list(facets)
    if len(chosen) > max_queries:
        # Report a configuration error rather than silently losing a user need.
        raise ValueError('Configured facet-query limit would discard an information need')
    return {name: f'{question} {stems.get(name, NEED_LABELS[name])}' for name in chosen}


def complete_units(text: str) -> list[str]:
    """Never slice mid-sentence / mid-table-row; preserve line-boundary semantics."""
    units = []
    for paragraph in re.split(r'\n\s*\n', text.strip()):
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        for line in lines:
            if re.search(r'[.!?](?:\s|$)', line) and len(line) > 240:
                units.extend(s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-ZÁÉÍÓÖŐÚÜŰ])', line) if s.strip())
            else:
                units.append(line)
    grouped: list[str] = []
    for unit in units:
        # A list introduction and a following exception qualify the preceding
        # fact. Keep them atomic; do not make a deadline unconditional to fit.
        dependent = re.match(r'^(?:kivéve|azonban|ugyanakkor|ez alól|ebben az esetben|ha|amennyiben|feltéve)\b',
                             unit, re.I)
        list_continuation = (grouped and re.match(r'^(?:[-•*]|\d+[.)])\s', unit)
                             and re.search(r'\n(?:[-•*]|\d+[.)])\s', grouped[-1]))
        if grouped and (dependent or list_continuation or grouped[-1].rstrip().endswith((':', ';'))):
            grouped[-1] += '\n' + unit
        else:
            grouped.append(unit)
    return grouped


@dataclass(frozen=True)
class SelectionContext:
    evidence: list[dict]
    rejected_ids: list[str]
    estimated_input_tokens: int
    token_budget_estimate: int
    facet_coverage: dict[str, list[str]]
    duplicate_aliases: dict[str, str] = field(default_factory=dict)


def assemble_selection_context(evidence: list[dict], query: str, *, domain: str,
                               role: str = '', token_budget: int = 1600) -> SelectionContext:
    """Pack whole source units, reserving space for each attainable information need."""
    if token_budget < 64:
        raise ValueError('token_budget must leave room for evidence metadata')
    facets = requested_facets(query, domain, role=role)
    after_sale = post_purchase_procedure(query, domain, role)
    terms = {term for term in re.findall(r'[^\W_]+', query.casefold(), re.UNICODE)
             if len(term) > 4}
    by_id = {}
    # Overlapping windows in the same versioned document sometimes repeat an
    # entire source unit. Retain its first real evidence ID, never merge across
    # independent documents or versions with separate provenance.
    identical: dict[tuple, str] = {}
    duplicate_aliases: dict[str, str] = {}
    for item in evidence:
        eid = item.get('evidence_id')
        if not eid or not is_relevant_evidence(item, query, domain, role):
            continue
        key = (item.get('document_id'), item.get('document_version'),
               ' '.join(item.get('text', '').split()).casefold())
        if all(key) and key in identical:
            duplicate_aliases[eid] = identical[key]
            previous = by_id[identical[key]]
            previous['facet_matches'] = list(dict.fromkeys(
                [*previous.get('facet_matches', []), *item.get('facet_matches', [])]))
            continue
        by_id[eid] = {**item}
        if all(key):
            identical[key] = eid
    chosen: dict[str, set[int]] = defaultdict(set)
    facet_coverage: dict[str, list[str]] = defaultdict(list)
    used_versioned_units: dict[tuple[str, str, str], str] = {}
    remaining = token_budget

    def matches(item: dict, facet: str) -> bool:
        # Retrieval facet tags are *candidate annotations*, not a semantic gold.
        return facet in item.get('facet_matches', []) and evidence_matches_facet(item, facet) or (
            not item.get('facet_matches') and evidence_matches_facet(item, facet))

    def candidates_for(facet: str):
        for eid, item in by_id.items():
            if not matches(item, facet):
                continue
            units = complete_units(item.get('text', ''))
            markers = FACET_EVIDENCE_MARKERS.get(facet, ())
            ranked = sorted(enumerate(units), key=lambda pair: (
                # Directly actionable, user-relevant evidence first. Do not
                # promote a generic authority deadline or a navigation label.
                -((12 if after_sale and any(t in pair[1].casefold() for t in
                       ('átírat', 'átírás', 'tulajdonosváltáskor', 'azonnal újat',
                        'eredetiségvizsgálat')) else 0)
                  + (12 if domain == 'employment' and facet == 'steps'
                     and any(t in pair[1].casefold() for t in
                             ('nyilvántartásba vétel', 'álláskeresési járadék iránt')) else 0)
                  + (8 if domain == 'employment' and facet == 'healthcare'
                     and ('járadék' in pair[1].casefold() or 'jogosultságot ellenőriz' in pair[1].casefold()) else 0)),
                -sum(m in pair[1].casefold() for m in markers),
                -sum(t in pair[1].casefold() for t in terms), pair[0]))
            for ix, unit in ranked:
                if not is_complete_statement(unit, source_unit=True):
                    continue
                if not relevant_unit(unit, query, domain, role):
                    continue
                if after_sale and pre_purchase_background(unit):
                    continue
                if low_value_procedural_unit(unit, query, domain, role):
                    continue
                if domain == 'employment' and facet == 'healthcare' and not healthcare_eligibility_unit(unit):
                    continue
                if not evidence_matches_facet({'text': unit}, facet):
                    continue
                yield eid, item, ix, unit

    def take(facet: str, eid: str, item: dict, ix: int, unit: str) -> bool:
        nonlocal remaining
        unit_key = (item.get('document_id') or '', item.get('document_version') or '',
                    ' '.join(unit.split()).casefold())
        # Overlap from two windows in ONE indexed document is not two pieces of
        # evidence. Reuse the first exact citation; do not merge across documents.
        if all(unit_key) and unit_key in used_versioned_units:
            original_id = used_versioned_units[unit_key]
            if original_id not in facet_coverage[facet]:
                facet_coverage[facet].append(original_id)
            return True
        if ix in chosen[eid]:
            if eid not in facet_coverage[facet]:
                facet_coverage[facet].append(eid)
            return True
        # Reserve ID/title/source overhead once, not for each repeated facet.
        overhead = (estimated_tokens(eid + item.get('title', '') + item.get('source_url', '')) + 20
                    if not chosen[eid] else 0)
        cost = estimated_tokens(unit) + overhead + 2
        if cost > remaining:
            return False
        chosen[eid].add(ix)
        if all(unit_key):
            used_versioned_units[unit_key] = eid
        remaining -= cost
        if eid not in facet_coverage[facet]:
            facet_coverage[facet].append(eid)
        return True

    # For an already-purchased car, administrative actions take the first budget
    # slots. A pre-sale check cannot consume the limited prompt budget.
    if after_sale:
        facets = tuple(sorted(facets, key=lambda f: (
            {'insurance': 0, 'deadline': 1, 'steps': 2,
             'documents': 3, 'where': 4}.get(f, 5))))
    elif domain == 'employment' and 'steps' in facets:
        # The WORK_001 report spent almost the entire prompt budget on a TB
        # booklet and a navigation-only U1 line. Reserve room for actual
        # registration and conditional healthcare statements first.
        facets = tuple(sorted(facets, key=lambda f: (
            {'steps': 0, 'healthcare': 1, 'supports': 2,
             'where': 3, 'documents': 4}.get(f, 5))))

    # First pass: each facet gets at least one independent, evidence-backed unit
    # if it can fit; a missing one is reported instead of silently dropped.
    for facet in facets:
        for eid, item, ix, unit in candidates_for(facet):
            if take(facet, eid, item, ix, unit):
                break

    # Second pass: additional complete facts, with a per-source cap to avoid
    # first-document domination. Small contexts remain useful at 2048 tokens.
    for facet in facets:
        for eid, item, ix, unit in candidates_for(facet):
            if remaining <= 0 or sum(len(v) for v in chosen.values()) >= 16:
                break
            if len(chosen[eid]) >= 3:
                continue
            take(facet, eid, item, ix, unit)

    packed: list[dict] = []
    rejected = []
    for eid, item in by_id.items():
        selected_indexes = chosen.get(eid)
        if not selected_indexes:
            rejected.append(eid)
            continue
        units = complete_units(item.get('text', ''))
        assigned_facets = [facet for facet, ids in facet_coverage.items() if eid in ids]
        metadata = {key: item[key] for key in (
            'document_id', 'document_version', 'section_path', 'page_number',
            'published_at', 'effective_from', 'effective_to') if item.get(key) is not None}
        packed.append({**metadata, 'evidence_id': eid, 'title': item.get('title', ''),
                       'domain': item.get('domain', domain),
                       'text': '\n'.join(units[i] for i in sorted(selected_indexes)),
                       'source_url': item.get('source_url', ''), 'facets': assigned_facets})
    return SelectionContext(packed, rejected, token_budget - remaining,
                            token_budget, dict(facet_coverage), duplicate_aliases)


def validate_selection(selection: dict[str, list[str]], allowed_ids: set[str],
                       requested: tuple[str, ...]) -> dict[str, list[str]]:
    """Reject invented IDs, facets, duplicate IDs; never trust model output."""
    if any(name not in requested for name in selection):
        raise ValueError('Model returned an unrequested information need')
    result = {}
    for facet in requested:
        ids = selection.get(facet, [])
        if not isinstance(ids, list) or any(not isinstance(x, str) or x not in allowed_ids for x in ids):
            raise ValueError(f'Invalid selected evidence IDs for {facet}')
        result[facet] = list(dict.fromkeys(ids))
    return result


def validate_selection_response(selection: dict[str, list[str]], missing_needs: list[str],
                                allowed_ids: set[str], requested: tuple[str, ...]
                                ) -> dict[str, list[str]]:
    """Reject selections that omit attainable needs without reporting them as missing."""
    result = validate_selection(selection, allowed_ids, requested)
    if len(missing_needs) != len(set(missing_needs)):
        raise ValueError('Duplicate missing information needs')
    missing = set(missing_needs)
    if missing != {name for name in requested if not result[name]}:
        raise ValueError('Inconsistent selected evidence and missing facets')
    return result

FACET_EVIDENCE_MARKERS = {
    'steps': ('átírás', 'átírat', 'bejelent', 'igényl', 'regisztrál', 'szerződés',
              'szükséges', 'nyilvántartásba vétel', 'kérelmez', 'jelentkez'),
    'deadline': ('napon belül', 'munkanapon', 'határidő', 'folyósítási idő'),
    'documents': ('igazolás', 'okmány', 'szerződés', 'törzskönyv', 'irat', 'kilépő papír', 'igazolvány', 'forgalmi'),
    'insurance': ('kötelező gépjármű', 'kgfb', 'felelősségbiztosítás'),
    'where': ('kormányablak', 'webes ügysegéd', 'mobilalkalmazás',
              'foglalkoztatási osztály', 'járási hivatal', 'elektronikus ügyintézés', 'e-papír'),
    'costs': ('ft', 'forint'),
    'supports': ('támogatás', 'járadék', 'képzés', 'tanácsadás'),
    'eligibility': ('jogosult', 'jogszerző', 'feltétel', '360 nap'),
    'benefit_amount': ('járadékalap', '60 százalék', '60%', 'folyósítási idő', '90 nap'),
    'healthcare': ('tb ', 'egészségügyi', 'biztosítási jogviszony'),
}


def evidence_matches_facet(item: dict, facet: str) -> bool:
    """Conservative lexical coverage diagnostic, NOT semantic gold relevance."""
    text = (item.get('text', '') + ' '.join(item.get('section_path', []))).casefold()
    if facet == 'costs':
        return bool(re.search(r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\b', text, re.I))
    return any(marker in text for marker in FACET_EVIDENCE_MARKERS.get(facet, ()))


def is_relevant_evidence(item: dict, question: str, domain: str, role: str = '') -> bool:
    """Exclude clearly different vehicle classes / roles before composing any answer.

    This is a conservative domain guard, not an RRF relevance threshold.
    """
    if item.get('domain') and item['domain'] != domain:
        return False
    if domain != 'vehicle':
        return True
    item_role = item.get('role', 'general')
    if role in ('buyer', 'seller') and item_role not in ('', 'general', role):
        return False
    question_low = question.casefold()
    text = str(item.get('text', '')).casefold()
    title = str(item.get('title', '')).casefold()
    # A MABISZ micromobility FAQ mentions KGFB but does not describe buying a car.
    if not any(word in question_low for word in ('roller', 'mikromobilit', 'pótkocsi')):
        if any(word in title for word in ('roller', 'mikromobilit', 'pótkocsi')):
            return False
        if any(word in text for word in ('mikromobilitási eszköz', 'roller', 'pótkocsi')):
            if not any(word in text for word in ('autóvásárlás', 'személygépkocsi', 'autó átírás')):
                return False
    return True


def deterministic_facet_selection(packed: SelectionContext, requested: tuple[str, ...],
                                  *, max_per_need: int = 2) -> dict[str, list[str]]:
    """Choose ONLY complete, in-budget evidence; never fabricate IDs.

    Called before the one-shot quick Qwen request and as a safe invalid-ID fallback.
    """
    permitted = {item['evidence_id'] for item in packed.evidence}
    result = {}
    for facet in requested:
        result[facet] = [eid for eid in packed.facet_coverage.get(facet, [])
                         if eid in permitted][:max_per_need]
    return result
