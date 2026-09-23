"""Deterministic, evidence-grounded fallback; no network or LLM calls."""
from __future__ import annotations

import re

from .models import Claim, Draft
from ..conversation import detect_subtopic
from ..context_engineering.information_needs import question_needs, evidence_has_price, is_location_only_question
from .administrative_steps import extract_steps
from .quality import is_complete_statement, same_fact, non_independent_source_fragment
from .filters import (post_purchase_procedure, pre_purchase_background,
                           missing_healthcare_condition, low_value_procedural_unit)
from ..response.filters import relevant_unit
from ..context_engineering.evidence_selection import is_relevant_evidence


def select_excerpt(text: str, question: str, max_chars: int) -> str:
    """Find a relevant contiguous original span within a bounded character budget."""
    if len(text) <= max_chars:
        return text
    terms = {term for term in re.findall(r'[^\W_]+', question.casefold(), re.UNICODE)
             if len(term) >= 5}
    stride = max(1, max_chars // 2)
    starts = range(0, len(text), stride)

    def match_score(start: int) -> int:
        window = text[start:start + max_chars].casefold()
        return sum(len(term) for term in terms if term in window)

    start = max(starts, key=match_score) if terms else 0
    return text[start:start + max_chars]

def _select_source_claims(
    candidates: list[tuple[float, str, str, str]],
    *,
    asks_benefit_amount: bool,
    asks_supports: bool,
    asks_eligibility: bool,
    asks_cost: bool,
    mixed_request: bool,
    location_only: bool,
    asks_documents: bool,
    seller_general: bool,
    topic: str,
    location_terms: tuple[str, ...],
) -> list[Claim]:
    """Select a bounded, non-duplicative set of evidence-backed claims."""
    claims: list[Claim] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    per_category: dict[str, int] = {}
    category_caps = (
        {'benefit_amount': 5, 'deadline': 2, 'eligibility': 2, 'where': 1} if asks_benefit_amount else
        {'support': 7, 'eligibility': 2, 'where': 2} if asks_supports else
        {'eligibility': 5, 'deadline': 2, 'where': 2} if asks_eligibility else
        {'cost': 2 if topic else 5} if asks_cost and not mixed_request else
        {'cost': 4, 'deadline': 2, 'where': 2, 'insurance': 2,
         'documents': 2, 'steps': 2} if mixed_request else
        {'where': 4} if location_only else
        {'documents': 5, 'deadline': 2, 'where': 2, 'insurance': 2, 'steps': 2}
        if asks_documents else
        {'deadline': 3, 'where': 2, 'insurance': 2, 'documents': 2, 'steps': 2,
         'support': 4, 'eligibility': 3, 'benefit_amount': 3, 'healthcare': 2}
    )
    if seller_general:
        category_caps = {'deadline': 1, 'where': 2, 'insurance': 1,
                         'documents': 2, 'steps': 2}
    if mixed_request:
        candidates.sort(key=lambda row: (row[3] != 'cost', -row[0]))

    for score, evidence_id, quote, category in candidates:
        if score < 0 or counts.get(evidence_id, 0) >= 3:
            continue
        if len(claims) >= 4 and category == 'steps' and score < 4:
            continue
        normalized = ' '.join(quote.split()).casefold()
        if (normalized in seen or any(same_fact(quote, previous.text) for previous in claims)
                or per_category.get(category, 0) >= category_caps.get(category, 0)
                or (topic and asks_cost and any(
                    claim.category == 'cost' and claim.evidence_ids == [evidence_id]
                    and (claim.text.casefold() in normalized or normalized in claim.text.casefold())
                    for claim in claims))):
            continue
        if location_only and not any(term in normalized for term in location_terms):
            continue
        seen.add(normalized)
        counts[evidence_id] = counts.get(evidence_id, 0) + 1
        per_category[category] = per_category.get(category, 0) + 1
        claims.append(Claim(
            text=quote, evidence_ids=[evidence_id], supporting_quote=quote,
            category=category, origin='source_only', in_model_prompt=None,
        ))
        limit = (7 if seller_general else 16 if mixed_request
                 else (3 if topic else 8) if asks_cost else 6 if location_only else 12)
        if len(claims) >= limit:
            break

    order = ({'deadline': 0, 'steps': 1, 'insurance': 2, 'where': 3, 'documents': 4, 'cost': 5}
             if mixed_request else
             {'benefit_amount': 0, 'eligibility': 1, 'support': 2, 'deadline': 3,
              'healthcare': 4, 'where': 5, 'documents': 6, 'cost': 7, 'insurance': 8, 'steps': 9})
    if location_only:
        order = {'where': 0}
    claims.sort(key=lambda claim: order.get(claim.category, 9))
    return claims


def _source_disclaimer(
    *, asks_cost: bool, mixed_request: bool, asks_benefit_amount: bool, asks_supports: bool,
) -> str:
    """Return the deterministic caveat matching the requested answer facet."""
    if asks_cost and not mixed_request:
        return (
            'Az összeg az idézett dokumentumban szerepel, nem élő díjlekérdezés eredménye. '
            'A ténylegesen fizetendő, aktuális díjat és az esetleges feltételeket '
            'ellenőrizd az eredeti hivatalos oldalon.'
        )
    if asks_benefit_amount:
        return (
            'Az álláskeresési járadék tényleges összegét az egyéni járulékalap és jogosultsági adatok határozzák meg; '
            'a hivatalos NFSZ kalkulátor használható személyre szabott számításhoz.'
        )
    if asks_supports:
        return (
            'A felsorolt támogatások elérhetősége programonként, életkor és terület szerint eltérhet; '
            'az illetékes foglalkoztatási osztály dönt az egyéni jogosultságról.'
        )
    return (
        'Forrásalapú kivonat: az idézett teendők az eredeti tájékoztatókból származnak. '
        'Az ügyintézési feltételeket és az aktuális határidőket ellenőrizd a megjelölt hivatalos oldalon.'
    )


def source_answer(question: str, evidence: list[dict], selected_ids: list[str] | None = None,
                  *, role: str = '', stage: str = '') -> Draft:
    """Produce useful, *verbatim* next steps without asking an LLM to invent facts.

    Prefer complete action/deadline statements from different official chunks.
    One evidence identifier always refers to the exact quoted source text.
    """
    domain = next((item.get('domain') for item in evidence if item.get('domain')), '')
    if not role and domain == 'vehicle':
        q = question.casefold()
        role = ('seller' if any(term in q for term in ('eladtam', 'eladok', 'eladás után'))
                else 'buyer' if any(term in q for term in ('vásároltam', 'vettem', 'megvettem'))
                else '')
    if domain in ('vehicle', 'employment'):
        evidence = [item for item in evidence if is_relevant_evidence(
            item, question, domain,
            role)]
    # Prefer the directly applicable DÁP seller guide if available; laws
    # remain eligible for genuinely missing procedural facts.
    if domain == 'vehicle' and role == 'seller' and \
            any(str(item.get('source_url', '')).startswith('https://dap.gov.hu/') for item in evidence):
        evidence = sorted(evidence, key=lambda item: (
            not str(item.get('source_url', '')).startswith('https://dap.gov.hu/'),
            str(item.get('document_id', ''))))
    by_id = {item['evidence_id']: item for item in evidence}
    structured_steps = extract_steps(evidence)
    # This deterministic fallback is used only for source mode / model errors. Selection must
    # never suppress a verified deadline or an office found in other chunks.
    preferred = set(selected_ids or [])
    question_low = question.casefold()
    words = {w for w in re.findall(r'[^\W_]+', question_low) if len(w) >= 5}
    asks_where = bool(re.search(r'\b(?:hol|hova|hová|merre|melyik\s+oldalon)\b', question_low))
    location_only = is_location_only_question(question)
    asks_documents = bool(re.search(r'dokumentum|irat|papír|okmány', question_low))
    needs = question_needs(question)
    asks_deadline = needs.deadline
    asks_cost = needs.costs
    asks_benefit_amount = needs.benefit_amount
    asks_supports = needs.supports
    asks_eligibility = needs.eligibility
    mixed_request = asks_cost and needs.steps
    topic = detect_subtopic(question)
    topic_terms = {
        'vehicle_inspection': ('eredetiségvizsg', 'eredetvizsg'),
        'vehicle_transfer': ('átírás', 'átírat', 'tulajdonosvált', 'vagyonszerzési illeték'),
        'vehicle_registration': ('forgalmi', 'törzskönyv'),
    }.get(topic, ())
    if topic == 'vehicle_registration':
        # A question about one document must not list the other document's fee.
        topic_terms = tuple(term for term in topic_terms if term in question_low) or topic_terms
    after_sale = post_purchase_procedure(question, domain, 'buyer')
    past_event = any(w in question_low for w in
                     ('vásároltam', 'vettem', 'eladtam', 'megvettem', 'megszűnt', 'kaptam'))
    seller_general = (domain == 'vehicle' and role == 'seller' and past_event
                      and not any((asks_cost, asks_deadline, asks_documents, asks_where)))
    location_terms = ('kormányablak', 'ügysegéd', 'mobilalkalmazás', 'személyesen',
                      'ügyfélszolgálat', 'weboldal', 'honlap', 'portál',
                      'időpont', 'ügyintézési felület')
    action_terms = ('bejelent', 'átír', 'biztosít', 'eredetiségvizsgálat', 'igényel',
                    'kell', 'szükséges', 'indítsd', 'jelentsd', 'töltsd',
                    'közmű', 'lakcím', 'napon belül', 'munkanapon belül')
    # Each candidate retains the exact original sentence for the audit. Do
    # not fabricate a summary of legal rules or silently cut off a sentence.
    candidates: list[tuple[float, str, str, str]] = []
    for index, (evidence_id, item) in enumerate(by_id.items()):
        chunk_low = item.get('text', '').casefold()
        heading_low = ' '.join(item.get('section_path', [])).casefold()
        topic_in_chunk = any(term in chunk_low or term in heading_low for term in topic_terms)
        for line in item.get('text', '').splitlines():
            for part in re.split(r'(?<=[.!?])\s+', line):
                quote = part.strip(' •-\t')
                # A comma-terminated enumeration item is not a standalone
                # instruction, even when a source parser calls it a step.
                if quote.endswith((',', ';', ':')):
                    continue
                is_standalone_price = asks_cost and bool(re.search(
                    r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\s*$', quote, re.I))
                # HTML/PDF bullet points and tabular price rows often lack a final period.
                # Keep them if there is an independently recognizable action or price.
                if (len(quote) < 9 or not is_complete_statement(quote, source_unit=True)
                        or non_independent_source_fragment(quote) or (
                        quote[-1] not in '.!?' and not is_standalone_price
                        and not any(step.description == quote for step in structured_steps))):
                    continue
                low = quote.casefold()
                if after_sale and pre_purchase_background(quote):
                    continue
                if not relevant_unit(quote, question, domain, role, stage):
                    continue
                if low_value_procedural_unit(quote, question, domain):
                    continue
                # Navigation/headline questions from official pages are not factual answer claims.
                if quote.endswith('?') and not re.search(r'\b(?:ft|forint|százalék|nap|kell|jogosult)\b', low):
                    continue
                if asks_cost and (not mixed_request or evidence_has_price(quote)):
                    # No price answer without an actual number AND currency in
                    # the cited sentence. Avoid unrelated fees in the same page.
                    if not re.search(r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\b', low):
                        continue
                    if topic_terms and (not topic_in_chunk or not any(
                            term in low or term in heading_low for term in topic_terms)):
                        continue
                    if topic == 'vehicle_transfer' and not any(
                            term in low for term in ('átírás', 'átírat', 'tulajdonosvált', 'illeték')):
                        continue  # Do not present one document fee as the total transfer cost.
                if location_only and not any(term in low for term in location_terms):
                    continue
                # Explanations of what a certificate proves are background,
                # not an actionable step after an already completed purchase.
                if past_event and ('igazolja, hogy' in low or 'nem lopott' in low):
                    continue
                if asks_benefit_amount and any(term in low for term in (
                        'járadékalap', '60 százalék', '60%', 'napi összeg', 'minimálbér',
                        'folyósítható járadék összesen', 'napi járadék összeg')):
                    category = 'benefit_amount'
                elif asks_supports and any(term in low for term in (
                        'támogatás', 'képzési támogatás', 'lakhatási támogatás',
                        'utazási támogatás', 'vállalkozóvá válás')):
                    category = 'support'
                elif asks_eligibility and any(term in low for term in (
                        'jogosult', 'jogszerző idő', '360 nap', 'feltétel')):
                    category = 'eligibility'
                elif domain == 'employment' and any(term in low for term in (
                        'támogatott nyelvtanfolyam', 'támogatott képzés',
                        'képzési támogatás', 'lakhatási támogatás')):
                    category = 'support'
                elif asks_cost and evidence_has_price(quote):
                    category = 'cost'
                elif 'napon belül' in low or 'munkanapon belül' in low or 'határidő' in low or 'legfeljebb 90 nap' in low:
                    category = 'deadline'
                elif any(term in low for term in location_terms):
                    category = 'where'
                elif 'egészségügyi' in low or 'tb ellátás' in low or (domain == 'employment' and ('biztosítási jogviszony' in low or 'tb járulék' in low)):
                    category = 'healthcare'
                elif 'biztosít' in low:
                    category = 'insurance'
                elif any(term in low for term in ('szerződés', 'okmány', 'igazolvány', 'papír', 'törzskönyv', 'foglalkoztatói igazolás')):
                    category = 'documents'
                else:
                    category = 'steps'
                # For a broad after-sale checklist, generic legal background
                # should not displace actionable DÁP guidance. Specialist
                # questions keep the unrestricted legal-source path.
                if seller_general and category == 'steps' and (
                        not str(item.get('source_url', '')).startswith('https://dap.gov.hu/')
                        or not quote.endswith(('.', '!', '?'))
                        or not any(term in low for term in (
                            'kell', 'jelentsd', 'értesítsd', 'ellenőrizd',
                            'küldd', 'fizesd', 'őrizd', 'bejelent', 'add át',
                            'gondoskodj', 'igényeld'))):
                    continue
                # Honour the model's explicit selection for ordinary background
                # claims. Still include omitted, source-backed obligations that
                # are essential to a useful procedural answer.
                if selected_ids is not None and evidence_id not in preferred and not asks_where and not asks_cost:
                    essential_documents = category == 'documents' and any(
                        token in low for token in
                        ('szükséges', 'kell', 'csatol', 'töltsd', 'igazolvány',
                         'lakcímkártya', 'forgalmi', 'törzskönyv'))
                    if category not in ('deadline', 'where', 'insurance') and not essential_documents:
                        continue
                score = 2.0 * sum(w in low for w in words)
                score += 2.0 * sum(term in low for term in action_terms)
                score += 2.0 if evidence_id in preferred else 0
                score -= index * 0.1
                if domain == 'vehicle' and role == 'seller' and str(item.get('source_url', '')).startswith('https://dap.gov.hu/'):
                    score += 9
                if asks_benefit_amount:
                    score += 20 if category == 'benefit_amount' else 0
                    score += 8 if any(term in low for term in ('60 százalék', '60%', 'napi összeg', 'minimálbér')) else 0
                    score += 4 if '90 nap' in low else 0
                elif asks_supports:
                    score += 18 if category == 'support' else 0
                    score += 5 if any(term in low for term in ('képzési', 'lakhatási', 'utazási', 'vállalkozóvá')) else 0
                elif asks_eligibility:
                    score += 18 if category == 'eligibility' else 0
                    score += 5 if '360 nap' in low or 'jogszerző' in low else 0
                elif location_only:
                    score += 15 if category == 'where' else 0
                    score += 4 if 'bejelent' in low or 'átír' in low else 0
                elif asks_cost and not mixed_request:
                    score += 18 if any(term in low for term in topic_terms) else 0
                    score += 5 if 'díj' in low or 'költség' in low or 'ára' in low else 0
                elif mixed_request:
                    score += 9 if category == 'cost' else 0
                    score += 5 if category in ('deadline', 'insurance', 'where') else 0
                elif asks_documents:
                    score += 8 if category == 'documents' else 0
                elif asks_deadline:
                    score += 8 if category == 'deadline' else 0
                else:
                    score += {'deadline': 7, 'where': 7, 'insurance': 6,
                              'documents': 4, 'steps': 1, 'cost': 1,
                              'benefit_amount': 5, 'support': 5, 'eligibility': 5,
                              'healthcare': 6}.get(category, 0)
                if (past_event and any(w in low for w in ('hirdet', 'megtekint', 'vásárlás előtt'))
                        and not any(w in low for w in ('napon belül', 'megkötésétől', 'átír', 'bejelent'))):
                    score -= 15
                if not asks_cost and ' ft' in low:
                    score -= 5
                if missing_healthcare_condition(quote, quote, category, domain):
                    continue
                candidates.append((score, evidence_id, quote, category))
    # Extracted administrative lines retain list/table structure and source IDs.
    # They supplement sentence splitting rather than inventing a paraphrase.
    for step in structured_steps:
        eid = step.source_evidence_ids[0]
        quote = step.description
        if quote.rstrip().endswith((',', ';', ':')):
            continue
        if (not is_complete_statement(quote, source_unit=True)
                or non_independent_source_fragment(quote)):
            continue
        if after_sale and pre_purchase_background(quote):
            continue
        if not relevant_unit(quote, question, domain, role, stage):
            continue
        if low_value_procedural_unit(quote, question, domain):
            continue
        if missing_healthcare_condition(quote, quote, step.category, domain):
            continue
        if selected_ids is not None and eid not in preferred and not asks_where and not asks_cost:
            essential = (step.category in ('deadline', 'where', 'insurance') or
                         step.category == 'documents' and any(token in quote.casefold() for token in
                             ('szükséges', 'kell', 'csatol', 'töltsd', 'igazolvány',
                              'lakcímkártya', 'forgalmi', 'törzskönyv')))
            if not essential:
                continue
        if seller_general and step.category == 'steps' and (
                not str(by_id[eid].get('source_url', '')).startswith('https://dap.gov.hu/')
                or not quote.endswith(('.', '!', '?'))
                or not any(term in quote.casefold() for term in (
                    'kell', 'jelentsd', 'értesítsd', 'ellenőrizd',
                    'küldd', 'fizesd', 'őrizd', 'bejelent', 'add át',
                    'gondoskodj', 'igényeld'))):
            continue
        if any(c[1] == eid and c[2] == quote for c in candidates):
            continue
        if location_only and not step.official_channels:
            continue
        if asks_cost and not mixed_request and not evidence_has_price(quote):
            continue
        if topic_terms and asks_cost and not mixed_request and not any(
                term in (quote + ' '.join(by_id[eid].get('section_path', []))).casefold() for term in topic_terms):
            continue
        score = 3.0 + 2 * sum(word in quote.casefold() for word in words)
        score += (10 if step.category == 'cost' and asks_cost else 0)
        score += (9 if step.category == 'where' and asks_where else 0)
        score += (6 if step.category in ('deadline', 'insurance') else 0)
        if domain == 'vehicle' and role == 'seller' and str(by_id[eid].get('source_url', '')).startswith('https://dap.gov.hu/'):
            score += 9
        candidates.append((score, eid, quote, step.category))
    candidates.sort(key=lambda row: row[0], reverse=True)

    claims = _select_source_claims(
        candidates,
        asks_benefit_amount=asks_benefit_amount,
        asks_supports=asks_supports,
        asks_eligibility=asks_eligibility,
        asks_cost=asks_cost,
        mixed_request=mixed_request,
        location_only=location_only,
        asks_documents=asks_documents,
        seller_general=seller_general,
        topic=topic,
        location_terms=location_terms,
    )
    return Draft(
        claims=claims,
        disclaimer=_source_disclaimer(
            asks_cost=asks_cost,
            mixed_request=mixed_request,
            asks_benefit_amount=asks_benefit_amount,
            asks_supports=asks_supports,
        ),
    )
