"""Grounding of model claims and separately marked literal source supplements.

This lexical matching is a conservative proxy, not legal entailment.
"""
from __future__ import annotations

import re

from .models import Claim, Draft, NaturalResponse
from .fallback import source_answer
from .quality import (is_complete_statement, same_fact, claim_facets,
                             claim_category, non_independent_source_fragment)
from .filters import missing_healthcare_condition, low_value_procedural_unit
from ..response.filters import relevant_unit
from ..context_engineering.evidence_selection import complete_units


def _attach_source_quotes(response: NaturalResponse, evidence: list[dict],
                          question: str = '', *, role: str = '', stage: str = '') -> Draft:
    """Keep model-written prose but attach source quotes from original evidence.

    Numeric claims are allowed only when their numbers occur in the SAME complete
    source unit. A lexical anchor is required; this is not semantic/legal proof.
    """
    by_id = {item['evidence_id']: item for item in evidence}
    claims = []
    seen = set()
    used_source_units: set[tuple[str, str, str]] = set()
    warnings: list[str] = []
    for proposal in response.claims:
        item = by_id.get(proposal.evidence_id)
        if item is None:
            continue
        text = ' '.join(proposal.text.split())
        if not is_complete_statement(text):
            warnings.append('model_incomplete_statement')
            continue
        domain = item.get('domain', '')
        if not relevant_unit(text, question, domain, role, stage):
            warnings.append('off_topic_for_life_event')
            continue
        if question and low_value_procedural_unit(text, question, domain):
            warnings.append('model_low_value_statement')
            continue
        # Hungarian inflection changes word endings (átírásra / átírást);
        # conservative 5-character anchors are a structural proxy, NOT proof
        # of semantic entailment. Legal/numerical assertions require review.
        terms = {word[:5] for word in re.findall(r'[^\W_]+', text.casefold(), re.UNICODE)
                 if len(word) > 5}
        numeric = set(re.findall(r'\b\d+(?:[.,]\d+)?\b', text))
        source_units = [unit for unit in complete_units(item.get('text', ''))
                        if is_complete_statement(unit, source_unit=True)]
        ranked = sorted(source_units, key=lambda unit: (
            -len(terms & {w[:5] for w in re.findall(r'[^\W_]+', unit.casefold(), re.UNICODE)
                           if len(w) > 5}), len(unit)))
        quote = next((unit for unit in ranked
                      if numeric <= set(re.findall(r'\b\d+(?:[.,]\d+)?\b', unit))
                      and len(terms & {w[:5] for w in re.findall(r'[^\W_]+', unit.casefold(), re.UNICODE)
                                       if len(w) > 5}) >= 2), None)
        # Multiple paraphrases of the same cited fact are not independent
        # evidence. Keep separate categories when a sentence contains different
        # actionable facts (e.g. an insurance rule AND a deadline).
        source_key = (proposal.category, proposal.evidence_id, quote.casefold()) if quote else None
        if quote and question and low_value_procedural_unit(quote, question, domain):
            warnings.append('model_low_value_statement')
            continue
        if quote and not relevant_unit(quote, question, domain, role, stage):
            warnings.append('off_topic_for_life_event')
            continue
        if quote and missing_healthcare_condition(text, quote, proposal.category):
            warnings.append('healthcare_condition_unverified')
            continue
        if not quote or text.casefold() in seen or source_key in used_source_units or any(
                same_fact(text, prior.text) for prior in claims):
            continue
        # Qwen occasionally labels support programmes as generic "steps". If
        # both the claim and its exact quote mention the support, retain the
        # source and use the descriptive category, not the unreliable label.
        category = proposal.category
        if (domain == 'employment' and category == 'steps'
                and any(t in text.casefold() for t in ('támogatott', 'támogatás', 'képzés'))
                and any(t in quote.casefold() for t in ('támogatott', 'támogatás', 'képzés'))):
            category = 'support'
        seen.add(text.casefold())
        used_source_units.add(source_key)
        claims.append(Claim(text=text, evidence_ids=[proposal.evidence_id],
                            supporting_quote=quote, category=category,
                            origin='model_generated', in_model_prompt=True))
    # Free-form disclaimers were an unaudited channel for invented deadlines.
    # Only a fixed conflict signal is accepted, never model-authored factual prose.
    notice = 'Nem hivatalos, forrásalapú tájékoztatás; az egyedi jogosultságot az illetékes szerv állapítja meg.'
    if response.disclaimer.strip() == 'forrásellentmondás':
        warnings.append('model_reported_source_conflict_unverified')
        notice += ' A modell lehetséges forrásellentmondást jelzett; ez külön ellenőrzést igényel.'
    elif response.disclaimer.strip():
        warnings.append('unverified_model_disclaimer_removed')
    return Draft(claims=claims, quality_warnings=warnings, disclaimer=notice)


def supplement_grounded_facts(draft: Draft, question: str, evidence: list[dict],
                             facets: tuple[str, ...], *, max_claims: int = 9,
                             model_prompt_evidence: list[dict] | None = None,
                             role: str = '', stage: str = '') -> Draft:
    """Fill missing information needs with literal, independently cited source facts.

    Preserve model-authored claims first. Supplementary claims are verbatim, never
    attributed to Qwen. This is a coverage aid, NOT proof of legal correctness.
    """
    mapping = {'steps': ('steps',), 'insurance': ('insurance',),
               'deadline': ('deadline',), 'documents': ('documents',),
               'where': ('where',), 'costs': ('cost',),
               'supports': ('support',), 'eligibility': ('eligibility',),
               'benefit_amount': ('benefit_amount',), 'healthcare': ('healthcare',)}
    source = source_answer(question, evidence, role=role, stage=stage)
    domain = next((item.get('domain', '') for item in evidence if item.get('domain')), '')
    claims = list(draft.claims)
    prompt_units = {item['evidence_id']: ' '.join(item.get('text', '').split()).casefold()
                    for item in (model_prompt_evidence or [])}
    seen = {' '.join(claim.supporting_quote.split()).casefold() for claim in claims}
    added = False
    for facet in facets:
        if len(claims) >= max_claims:
            break
        wanted = mapping.get(facet, ())
        if any(facet in claim_facets(claim.model_dump(), domain)
               for claim in claims):
            continue
        for fact in source.claims:
            quote = ' '.join(fact.supporting_quote.split()).casefold()
            # A heading suggested by a retrieval heuristic cannot establish
            # that a named document is actually needed. Inspect the literal
            # claim and source unit with the same rules as the answer audit.
            observed = claim_facets(fact.model_dump(), domain)
            if (facet in observed
                    and relevant_unit(fact.text, question, domain, role, stage)
                    and relevant_unit(fact.supporting_quote, question, domain, role, stage)
                    and not non_independent_source_fragment(fact.text)
                    and quote not in seen and not any(
                    same_fact(fact.text, prior.text) for prior in claims)):
                corrected = claim_category({**fact.model_dump(),
                                            'category': wanted[0]}, domain)
                claims.append(fact.model_copy(update={
                    'category': corrected,
                    'origin': 'source_supplement',
                    'in_model_prompt': quote in prompt_units.get(fact.evidence_ids[0], ''),
                }))
                seen.add(quote)
                added = True
                break
    if added:
        draft = Draft(claims=claims, model_context_evidence_ids=list(draft.model_context_evidence_ids),
                      quality_warnings=list(draft.quality_warnings), disclaimer=(
            (draft.disclaimer.strip() + ' ') if draft.disclaimer.strip() else ''
        ) + 'A válasz utólagos, külön jelölt, szó szerinti forráskivonatokkal is kiegészült.')
    return draft
