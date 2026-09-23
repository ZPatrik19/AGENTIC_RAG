"""RAG node implementations separated from LangGraph topology assembly."""
from __future__ import annotations

import re
from urllib.parse import urlparse
from collections.abc import Callable

from dap_assistant.documents.sources import ALLOWED_HOSTS
from dap_assistant.rag.retrieval import LocalQdrant
from dap_assistant.settings import Settings
from dap_assistant.rag.state import RAGState
from dap_assistant.conversation import detect_subtopic, TOPIC_LABELS, is_price_question
from dap_assistant.context_engineering.information_needs import evidence_has_price
from dap_assistant.context_engineering.evidence_selection import (
    facet_queries, evidence_matches_facet, NEED_LABELS, is_relevant_evidence,
)


def _annotate_retrieval(item: dict, label: str) -> dict:
    entry = {key: item.get(key) for key in (
        'bm25_rank', 'bm25_score', 'dense_rank', 'dense_score', 'rrf_components')}
    return {**item, 'retrieval_traces': [{**entry, 'query_label': label}]}


def _absorb_retrieval(store: dict[str, dict], item: dict, label: str, facet: str | None = None) -> None:
    incoming = _annotate_retrieval(item, label)
    cid = incoming['chunk_id']
    if cid not in store:
        store[cid] = {**incoming, 'facet_matches': [facet] if facet else
                      list(incoming.get('facet_matches', []))}
        return
    old = store[cid]
    previous = list(old.get('retrieval_traces', []))
    if not any(trace['query_label'] == label for trace in previous):
        previous.extend(incoming['retrieval_traces'])
    matched = list(dict.fromkeys([*old.get('facet_matches', []),
                                 *([facet] if facet else incoming.get('facet_matches', []))]))
    if incoming.get('score', 0) > old.get('score', 0):
        store[cid] = {**incoming, 'retrieval_traces': previous, 'facet_matches': matched}
    else:
        old['retrieval_traces'] = previous
        old['facet_matches'] = matched
class RAGNodes:
    """Stateful node set with explicit retrieval/runtime dependencies."""

    def __init__(
        self,
        settings: Settings,
        *,
        dense: LocalQdrant | None,
        telemetry,
        hybrid_search_fn: Callable,
        diversify_results_fn: Callable,
    ) -> None:
        self.settings = settings
        self.dense = dense
        self.telemetry = telemetry
        self._hybrid_search = hybrid_search_fn
        self._diversify_results = diversify_results_fn

    def process_query(self, state: RAGState) -> dict:
        attempt = state.get('search_attempt', 0)
        original = state.get('original_query', state['query'])
        # Retry broader wording, without changing domain or inventing user facts.
        query = original if attempt == 0 else (
            f"{original} hivatalos {', '.join(NEED_LABELS.get(f, f) for f in state.get('missing_facets', []))}"
            if state.get('missing_facets') else f'{original} hivatalos teendők szükséges dokumentumok')
        if state['domain'] == 'employment' and any(word in original.casefold() for word in ('támogat', 'ellátás')):
            # The official benefit is often named differently than the user's wording.
            query += ' álláskeresési járadék'
        if state['domain'] == 'vehicle' and state.get('role') == 'seller':
            # Search the post-sale procedure, not just generic sales advice.
            query += ' tulajdonosváltás bejelentése Webes Ügysegéd kormányablak biztosító határidő'
        elif state['domain'] == 'vehicle' and state.get('role') == 'buyer':
            query += ' gépjármű átírása kormányablak biztosítás eredetiségvizsgálat határidő'
        # Normalize informal Hungarian terminology before BM25 + dense search.
        topic = detect_subtopic(original)
        if topic:
            query += ' ' + TOPIC_LABELS[topic]
        if is_price_question(original):
            # Vehicle fees are not useful query expansion for employment amounts,
            # including severance and unemployment benefits.
            query += (' díja ára költsége Ft forint összeg illeték eredetiségvizsgálat forgalmi törzskönyv'
                      if state['domain'] == 'vehicle' else
                      ' összeg mértéke számítása jogosultsági feltételek')
        return {'query': query, 'original_query': original, 'search_attempt': attempt + 1}

    def retrieve(self, state: RAGState) -> dict:
        """Retrieve candidates, add domain coverage and merge information needs."""
        candidates = self._hybrid_search(
            state['query'], state['domain'], self.settings, limit=16,
            dense=self.dense, role=state.get('role', ''),
            telemetry=self.telemetry, run_id=state.get('run_id', ''),
        )
        candidates = [_annotate_retrieval(item, 'main') for item in candidates]
        candidates = self._coverage_candidates(state, candidates)
        return self._collect_facets(state, candidates, role=state.get('role', ''))

    def _coverage_candidates(self, state: RAGState, candidates: list[dict]) -> list[dict]:
        """Add bounded, domain-specific coverage without a second LLM call."""
        question = state.get('original_query', state['query']).casefold()
        role = state.get('role', '')
        if (state['domain'] == 'vehicle' and role in ('buyer', 'seller')
                and (re.search(r'\b(?:hol|hova|hová|merre)\b', question)
                     or any(word in question for word in ('teend', 'ügyintéz', 'bejelent')))):
            # One cheap, role-filtered coverage lookup. The first search may
            # rank contract boilerplate above the office/deadline paragraph.
            targeted = (
                'eladás után tulajdonosváltás bejelentése 15 napon belül '
                'Webes Ügysegéd személyesen kormányablak biztosítás megszüntetése'
                if role == 'seller' else
                'vásárlás után átírás 15 napon belül kormányablak '
                'kötelező gépjármű felelősségbiztosítás eredetiségvizsgálat'
            )
            supplemental = self._hybrid_search(targeted, state['domain'], self.settings,
                                         limit=12, dense=self.dense, role=role,
                                         telemetry=self.telemetry, run_id=state.get('run_id', ''))
            merged = {item['chunk_id']: item for item in candidates}
            for item in supplemental:
                _absorb_retrieval(merged, item, 'vehicle_coverage')
            candidates = list(merged.values())
        topic = detect_subtopic(question)
        if state['domain'] == 'employment' and topic:
            employment_queries = {
                'employment_benefit_amount': (
                    'álláskeresési járadék összege járadékalap 60 százalék minimálbér napi összeg '
                    'folyósítás 90 nap kalkulátor jogosultsági idő'
                ),
                'employment_supports': (
                    'álláskeresők támogatásai szolgáltatások képzési támogatás lakhatási támogatás '
                    'utazási támogatás vállalkozóvá válás támogatás aktuális programok'
                ),
                'employment_healthcare': (
                    'munkaviszony megszűnés egészségügyi szolgáltatás jogosultság 45 nap '
                    'egészségügyi szolgáltatási járulék biztosítás megszűnés'
                ),
                'employment_severance': (
                    'Munka Törvénykönyve végkielégítés 77 § munkáltatói felmondás összeg szolgálati idő'
                ),
                'employment_unused_leave': (
                    'Munka Törvénykönyve ki nem vett szabadság megváltás munkaviszony megszűnés 125 §'
                ),
                'employment_termination': (
                    'Munka Törvénykönyve felmondás munkáltatói felmondás munkavállalói felmondás '
                    'közös megegyezés felmondási idő 64 65 66 68 69 §'
                ),
            }
            targeted = employment_queries.get(topic)
            if targeted:
                supplemental = self._hybrid_search(targeted, state['domain'], self.settings,
                                             limit=16, dense=self.dense, role=role,
                                             telemetry=self.telemetry, run_id=state.get('run_id', ''))
                merged = {item['chunk_id']: item for item in candidates}
                for item in supplemental:
                    merged.setdefault(item['chunk_id'], item)
                candidates = list(merged.values())
        if state['domain'] == 'employment' and any(token in question for token in ('támogat', 'segély', 'járadék', 'ellátás')):
            # Coverage query for official benefit/support pages. This does not infer eligibility;
            # it only makes sure the dedicated NFSZ sources can compete with broad DÁP chunks.
            targeted = ('álláskeresési járadék feltételek összeg 60 százalék 90 nap NFSZ kalkulátor '
                        'álláskereső támogatások képzés lakhatás utazás szolgáltatások')
            supplemental = self._hybrid_search(targeted, state['domain'], self.settings,
                                         limit=16, dense=self.dense, role=role,
                                         telemetry=self.telemetry, run_id=state.get('run_id', ''))
            merged = {item['chunk_id']: item for item in candidates}
            for item in supplemental:
                _absorb_retrieval(merged, item, 'employment_topic')
            candidates = list(merged.values())
        if state['domain'] == 'vehicle' and topic and is_price_question(question):
            # Focused lexical+dense query: the generic vehicle query frequently
            # ranks duties/deadlines above the paragraph stating the actual fee.
            targeted = f'{TOPIC_LABELS[topic]} díja ára költsége Ft forint'
            supplemental = self._hybrid_search(targeted, state['domain'], self.settings,
                                         limit=12, dense=self.dense, role=role,
                                         telemetry=self.telemetry, run_id=state.get('run_id', ''))
            merged = {item['chunk_id']: item for item in candidates}
            for item in supplemental:
                _absorb_retrieval(merged, item, 'vehicle_fee')
            candidates = list(merged.values())
        # Explicit vehicle-duty queries require the complete, coherent NAV table.
        # RRF otherwise favors generic fee paragraphs, and a partial table must
        # never reach the calculator as though it were a verified rate source.
        if (state['domain'] == 'vehicle' and role != 'seller'
                and is_price_question(question)
                and any(term in question for term in ('illeték', 'névre', 'átírás', 'átírat'))):
            from dap_assistant.rag.retrieval import _chunk_snapshot
            merged = {item['chunk_id']: item for item in candidates}
            for item in _chunk_snapshot(self.settings.data_dir):
                if item.get('document_id') == 'nav-vehicle-duty-2026':
                    _absorb_retrieval(merged, {**item, 'score': 0.0}, 'structured_source_sweep')
            candidates = list(merged.values())
        # Full purchase/sale cost queries demand monetary evidence across more
        # than one section. The common first search favors procedural chunks.
        # A bounded, role-filtered local sweep is deterministic and avoids new
        # network requests or another LLM invocation.
        if is_price_question(question) and not topic:
            from dap_assistant.rag.retrieval import _chunk_snapshot
            from time import perf_counter
            started = perf_counter()
            merged = {item['chunk_id']: item for item in candidates}
            for item in _chunk_snapshot(self.settings.data_dir):
                if (item.get('domain') == state['domain']
                        and (not role or item.get('role', 'general') in ('general', role))
                        and evidence_has_price(item.get('text', ''))):
                    _absorb_retrieval(merged, {**item, 'score': 0.0}, 'structured_source_sweep')
            candidates = list(merged.values())
            if self.telemetry is not None:
                self.telemetry.record(state.get('run_id', ''), 'fee_coverage_lookup', perf_counter() - started)
        return candidates

    def _collect_facets(self, state: RAGState, candidates: list[dict], *, role: str) -> dict:
        """Preserve prior attempts and retry only missing information needs."""
        # A focused planner branch searches its own information need. Legacy
        # single-task plans still use the existing multifacet retrieval logic.
        facet = state.get('facet', '')
        if facet and facet not in NEED_LABELS:
            raise ValueError(f'Unsupported RAG facet: {facet}')
        facets = ({facet: f"{state.get('original_query', state['query'])} {NEED_LABELS[facet]}"}
                  if facet else facet_queries(state.get('original_query', state['query']),
                                              state['domain'], role=role))
        merged: dict[str, dict] = {}
        for item in state.get('candidates', []):
            if item.get('chunk_id'):
                merged[item['chunk_id']] = {**item, 'facet_matches': list(item.get('facet_matches', []))}
        # Retain valid evidence from earlier attempts. A narrower retry must
        # not erase already retrieved insurance, deadline, or document chunks.
        for item in candidates:
            if item['chunk_id'] not in merged:
                merged[item['chunk_id']] = {**item, 'facet_matches': list(item.get('facet_matches', []))}
            else:
                old = merged[item['chunk_id']]
                traces = {trace['query_label']: trace for trace in old.get('retrieval_traces', [])}
                for trace in item.get('retrieval_traces', []):
                    traces.setdefault(trace['query_label'], trace)
                merged[item['chunk_id']] = {**(item if item.get('score', 0) > old.get('score', 0) else old),
                                              'retrieval_traces': list(traces.values()),
                                              'facet_matches': list(dict.fromkeys([*old.get('facet_matches', []),
                                                                                  *item.get('facet_matches', [])]))}
        if len(facets) == 1:
            only = next(iter(facets))
            for item in merged.values():
                if evidence_matches_facet(item, only):
                    item['facet_matches'] = list(dict.fromkeys(item['facet_matches'] + [only]))
        if len(facets) > 1 or state.get('missing_facets'):
            # Retry only missing needs with a reformulated query, not exactly
            # the same search that failed. Default MAX_RAG_ATTEMPTS=2 means
            # one targeted retry, with no unbounded recursion.
            retry_missing = (set(state.get('missing_facets', []))
                             if state.get('search_attempt', 0) > 1 else set())
            for facet, facet_query in facets.items():
                if retry_missing and facet not in retry_missing:
                    continue
                if retry_missing:
                    facet_query += ' ' + state['query']
                additional = self._hybrid_search(facet_query, state['domain'], self.settings,
                                           limit=8, dense=self.dense, role=role,
                                           telemetry=self.telemetry, run_id=state.get('run_id', ''))
                for item in additional:
                    _absorb_retrieval(merged, item, 'facet:' + facet + ':attempt:' +
                           str(state.get('search_attempt', 1)), facet)
        candidates = list(merged.values())
        return {'candidates': candidates, 'requested_facets': list(facets),
                'pre_rerank_chunk_ids': [c['chunk_id'] for c in sorted(candidates, key=lambda c: -c.get('score', 0))]}


    def rerank(self, state: RAGState) -> dict:
        # RRF already ranks candidates. A lightweight domain/heading reranker avoids another model.
        terms = set(state['original_query'].casefold().split())
        topic = detect_subtopic(state['original_query'])
        price_question = is_price_question(state['original_query'])
        def score(c):
            heading = ' '.join(c.get('section_path', [])).casefold()
            content = c.get('text', '').casefold()
            bonus = sum(0.001 for word in terms if len(word) > 3 and word in heading)
            # Additional statutes and tariff pages must not crowd out the
            # primary DÁP buyer/seller procedure for ordinary life-event Q&A.
            if state['domain'] == 'vehicle' and state.get('role') in ('buyer', 'seller'):
                core = 'dap-vehicle-' + state['role']
                if c.get('document_id') == core:
                    bonus += 0.010
            if state['domain'] == 'employment' and c.get('document_id') in (
                    'dap-employment-overview', 'dap-employment-benefit',
                    'nfsz-unemployment-allowance'):
                bonus += 0.006
            if state['domain'] == 'employment':
                topic_id = detect_subtopic(state['original_query'])
                topic_docs = {
                    'employment_benefit_amount': {'nfsz-unemployment-allowance', 'nfsz-benefit-calculator', 'njt-employment-benefits'},
                    'employment_supports': {'nfsz-support-overview', 'dap-employment-overview', 'nfsz-entrepreneurship-support'},
                    'employment_healthcare': {'neak-post-employment-healthcare', 'neak-health-service-contribution', 'nav-health-contribution-2026'},
                    'employment_severance': {'njt-labour-code-termination'},
                    'employment_unused_leave': {'njt-labour-code-termination'},
                    'employment_termination': {'njt-labour-code-termination'},
                }.get(topic_id, set())
                if c.get('document_id') in topic_docs:
                    bonus += 0.055
                # Prefer authoritative, topic-labelled chunks over incidental keyword hits.
                if topic_id and any(term in c.get('topics', []) for term in {
                        'employment_benefit_amount': ('allowance_amount', 'calculator', 'entitlement_period'),
                        'employment_supports': ('support_overview', 'training_support', 'housing_support', 'travel_support'),
                        'employment_healthcare': ('healthcare', 'passive_entitlement', 'contribution'),
                        'employment_severance': ('severance',),
                        'employment_unused_leave': ('unused_leave',),
                        'employment_termination': ('termination', 'notice'),
                    }.get(topic_id, ())):
                    bonus += 0.015
            if any(term in state['original_query'].casefold()
                   for term in ('törvény', 'jogszabály', 'paragrafus', ' §')):
                if c.get('document_id', '').startswith('njt-'):
                    bonus += 0.020
            if state.get('role') == 'seller':
                bonus += 0.006 if 'tulajdonosváltás bejelentése' in heading else 0
                bonus += 0.004 if 'kormányablak' in content and 'ügysegéd' in content else 0
                bonus += 0.003 if '15 napon belül' in content and 'bejelent' in content else 0
                bonus += 0.002 if 'biztosítás megszüntetése' in heading else 0
            if (price_question and c.get('document_id') == 'nav-vehicle-duty-2026'
                    and any(t in state['original_query'].casefold() for t in ('illeték', 'névre', 'átírás', 'átírat'))):
                bonus += 0.065
            if price_question and not topic and evidence_has_price(content):
                bonus += 0.04  # General cost facet: do not lose priced source chunks.
            if state['domain'] == 'vehicle' and topic in (
                    'vehicle_inspection', 'vehicle_transfer', 'vehicle_registration') and price_question:
                topic_terms = {
                    'vehicle_inspection': ('eredetiségvizsg', 'eredetvizsg'),
                    'vehicle_transfer': ('átírás', 'átírat', 'tulajdonosvált'),
                    'vehicle_registration': ('forgalmi', 'törzskönyv'),
                }[topic]
                if topic == 'vehicle_registration':
                    topic_terms = tuple(term for term in topic_terms
                                        if term in state['original_query'].casefold()) or topic_terms
                if (any(term in content for term in topic_terms)
                        and re.search(r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\b', content)):
                    bonus += 0.05
            return c['score'] + bonus
        # Preserve more evidence for a procedural answer: location, deadline,
        # documents and insurance may be in separate chunks of the same page.
        # Preserve the best match for each information need before filling the
        # remaining evidence slots; deduplicate across searches by chunk_id.
        ranked_all = sorted(
            ({**candidate, 'rerank_score': score(candidate)} for candidate in state['candidates']
             if is_relevant_evidence(candidate, state['original_query'], state['domain'], state.get('role', ''))),
            key=lambda item: (-item['rerank_score'], item['chunk_id']),
        )
        topic_query = state['original_query'].casefold()
        if state['domain'] == 'vehicle' and not any(t in topic_query for t in ('gépjárműadó', 'pótkocsi')):
            ranked_all = [c for c in ranked_all if not (
                c.get('document_id', '') == 'nav-vehicle-tax-2026'
                or ('pótkocsi' in ' '.join(c.get('section_path', [])).casefold()))]
        facet_first = []
        selected_ids = set()
        for facet in state.get('requested_facets', []):
            match = next((c for c in ranked_all
                          if facet in c.get('facet_matches', []) and c['chunk_id'] not in selected_ids), None)
            if match:
                facet_first.append(match)
                selected_ids.add(match['chunk_id'])
        ranked = (facet_first + [c for c in self._diversify_results(ranked_all, limit=24)
                                 if c['chunk_id'] not in selected_ids])[:20]
        return {'ranked': ranked, 'ranked_chunk_ids': [c['chunk_id'] for c in ranked],
                'ranked_document_ids': [c['document_id'] for c in ranked]}

    def assess(self, state: RAGState) -> dict:
        accepted = [c for c in state['ranked']
                    if c['domain'] == state['domain'] and urlparse(c.get('source_url', '')).scheme == 'https'
                    and urlparse(c['source_url']).hostname in ALLOWED_HOSTS]
        if state['domain'] == 'vehicle' and is_price_question(state['original_query']):
            topic = detect_subtopic(state['original_query'])
            if topic == 'vehicle_inspection':
                # An unrelated vehicle registration fee does not answer the
                # inspection-price question. Retry, then report insufficient.
                accepted = [c for c in accepted if
                            ('eredetiségvizsg' in c['text'].casefold()
                             or 'eredetvizsg' in c['text'].casefold())
                            and re.search(r'\b\d[\d\s.,–-]*\s*(?:ft|forint)\b', c['text'], re.I)]
        # Facet coverage is a retrieval diagnostic, NOT a relevance judgment:
        # additional semantic verification happens at answer audit / golden eval.
        facet_map = {facet: [c['chunk_id'] for c in accepted
                             if facet in c.get('facet_matches', []) and evidence_matches_facet(c, facet)]
                     for facet in state.get('requested_facets', [])}
        missing = [facet for facet, ids in facet_map.items() if not ids]
        return {
            'evidence': accepted,
            'retrieval_status': ('complete' if accepted and not missing else
                                 'partial' if accepted else 'insufficient'),
            'selected_facet_evidence': facet_map,
            'missing_facets': missing,
            'missing_information': ([f'Nincs találat ehhez az információigényhez: {facet}'
                                     for facet in missing] if accepted else
                                    ['No sufficiently relevant indexed official document was found.']),
        }

    def prepare(self, state: RAGState) -> dict:
        # Keep complete indexed chunks and precise stable provenance for downstream
        # token-budgeted evidence selection, not character-clipped snippets.
        evidence = []
        for c in state.get('evidence', [])[:20]:
            evidence.append({**c, 'evidence_id': 'E_' + c['chunk_id'][:16]})
        return {'evidence': evidence, 'rejected_chunk_ids': [c['chunk_id'] for c in state.get('candidates', [])
                                                              if c['chunk_id'] not in {e['chunk_id'] for e in evidence}]}

    def after_assessment(self, state: RAGState) -> str:
        if state['retrieval_status'] == 'complete' or state['search_attempt'] >= self.settings.max_rag_attempts:
            return 'prepare_context'
        return 'process_query'
