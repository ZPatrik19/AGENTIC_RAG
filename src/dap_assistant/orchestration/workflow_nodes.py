"""Node implementations for the top-level assistant LangGraph."""
from __future__ import annotations

from datetime import date
from time import perf_counter

from langgraph.graph import END
from langgraph.types import Send

from .state import AssistantState, WorkerState
from ..response.audit import audit_answer
from ..llm import LLMError, answer_context, source_answer
from ..conversation import (
    SUPPORTED_DOMAINS, is_contextual_followup, is_followup_for_case, is_outside_scope,
    resolve_followup, detect_subtopic, TOPIC_LABELS, is_price_question,
)
from ..fast_path import classify_explicit, plan_explicit
from ..context_engineering.information_needs import question_needs
from ..response.administrative_steps import extract_steps
from ..tooling.calculators import (
    parse_vehicle_duty_request, calculate_vehicle_acquisition_duty,
    calculate_illustrative_loan, parse_loan_request,
)
from ..settings import Settings
from ..prompt_engineering.answer_prompt import answer_plan
from ..context_engineering.question_analysis import situation, analyze_question
from ..context_engineering.context_builder import prepare_graph_context, generation_diagnostics
from ..tooling.tools import (
    DeadlineRule, build_document_checklist, calculate_deadline, extract_duration_mentions,
    requested_tools, wants_native_tool_call, prepare_unemployment_benefit_calculation,
)


class WorkflowNodes:
    """Main-graph node implementations with explicit runtime dependencies."""

    def __init__(self, settings: Settings, llm, rag_graph, telemetry=None) -> None:
        self.settings = settings
        self.llm = llm
        self.rag_graph = rag_graph
        self.telemetry = telemetry

    def classify_intent(self, state: AssistantState) -> dict:
        question = state['user_question']
        hinted = state.get('user_context', {}).get('domain_hint')
        from ..llm import Classification

        # A previous answer never authorizes answering an explicitly unrelated question.
        # This fast gate makes e.g. lottery questions independent of Ollama/index status.
        if is_outside_scope(question):
            return {'domains': [], 'intents': [], 'role': '', 'stage': '',
                    'classification_status': 'unsupported',
                    'unsupported_reason': 'out_of_scope', 'resolved_question': question,
                    'context_resolution': {'source': 'out_of_scope', 'inherited': False}}

        explicit = classify_explicit(question)
        previous = state.get('previous_turn') or {}
        prior_domains = previous.get('domains') or []
        valid_previous = (prior_domains and len(prior_domains) <= 2
                          and all(d in SUPPORTED_DOMAINS for d in prior_domains))
        source = 'explicit'
        resolved = question
        if explicit is not None:
            # A new, explicit life event always beats both session history and UI hint.
            parsed = (explicit if self.settings.fast_routing or self.settings.answer_mode == 'source'
                      else self.llm.classify(question, run_id=state.get('run_id', '')))
            if not parsed.domains:
                parsed = explicit
            # "Mennyi az eredetvizsga?" is explicitly a vehicle question but
            # doesn't repeat buyer/seller. Retain that role only for the SAME
            # domain; a genuinely new domain always takes precedence.
            if (parsed.domains == ['vehicle'] and not parsed.role
                    and prior_domains == ['vehicle'] and previous.get('role') in ('buyer', 'seller')):
                parsed.role = previous['role']
                if not parsed.stage or parsed.stage == 'unknown':
                    parsed.stage = previous.get('stage', '')
                source = 'explicit_subtopic_with_previous_role'
            if detect_subtopic(question):
                resolved = resolve_followup(question, {
                    'domains': parsed.domains, 'role': parsed.role,
                    'stage': parsed.stage, 'focus': detect_subtopic(question),
                })
        elif valid_previous and is_followup_for_case(question, previous):
            parsed = Classification(domains=list(prior_domains),
                                    intents=[f'{d}_information' for d in prior_domains],
                                    role=previous.get('role', '') if prior_domains == ['vehicle'] else '',
                                    stage=previous.get('stage', ''))
            resolved = resolve_followup(question, previous)
            source = 'previous_turn'
        elif hinted in SUPPORTED_DOMAINS:
            parsed = Classification(domains=[hinted], intents=[f'{hinted}_information'])
            source = 'user_selection'
        else:
            # Never hand an unanchored query to Qwen and accept an invented domain.
            # Contextless pronouns merit clarification, not a guess at a life event.
            is_ambiguous = is_contextual_followup(question)
            return {'domains': [], 'intents': [], 'role': '', 'stage': '',
                    'classification_status': 'needs_clarification' if is_ambiguous else 'unsupported',
                    'clarification_question': 'Melyik élethelyzet ügyeire gondolsz: autóvásárlás/-eladás vagy munkahely elvesztése?',
                    'unsupported_reason': '' if is_ambiguous else 'unknown_topic',
                    'resolved_question': question,
                    'context_resolution': {'source': 'needs_clarification' if is_ambiguous else 'unsupported',
                                           'inherited': False}}
        focus = detect_subtopic(question) or (previous.get('focus', '') if source == 'previous_turn' else '')
        question_analysis = analyze_question(question, domain=(parsed.domains or [''])[0],
                                             role=parsed.role, stage=situation(
                                                 question, (parsed.domains or [''])[0],
                                                 parsed.role, parsed.stage))
        return {'domains': parsed.domains, 'intents': parsed.intents, 'role': parsed.role, 'stage': parsed.stage,
                'question_analysis': question_analysis,
                'focus': focus if focus in TOPIC_LABELS else '',
                'classification_status': 'needs_clarification' if parsed.needs_clarification else 'supported' if parsed.domains else 'unsupported',
                'clarification_question': parsed.clarification_question,
                'resolved_question': resolved,
                'context_resolution': {'source': source,
                                       'inherited': source in ('previous_turn', 'explicit_subtopic_with_previous_role'),
                                       'focus': focus if focus in TOPIC_LABELS else '',
                                       'previous_domains': list(prior_domains) if source == 'previous_turn' else []}}

    def clarify_query(self, state: AssistantState) -> dict:
        return {'final_answer': state.get('clarification_question') or 'Kérlek, pontosítsd az élethelyzetet.',
                'response_status': 'needs_clarification'}

    def plan_tasks(self, state: AssistantState) -> dict:
        if state.get('subtasks'):
            tasks = {k: dict(v) for k, v in state['subtasks'].items()}
        else:
            question = state.get('resolved_question', state['user_question'])
            explicit_plan = plan_explicit(question, state['domains'])
            # An explicit comprehensive request must not lose information needs
            # merely because optional model-based routing returns fewer tasks.
            plan = (explicit_plan if (self.settings.fast_routing or self.settings.answer_mode == 'source'
                                      or (explicit_plan is not None and len(explicit_plan.tasks) > 2))
                    else (self.llm.plan(question, state['domains'], run_id=state.get('run_id', ''))))
            tasks = {t.task_id: {**t.model_dump(), 'status': 'pending', 'role': state.get('role', '') if t.domain == 'vehicle' else ''} for t in plan.tasks}
            if len(tasks) != len(plan.tasks) or len(tasks) > self.settings.max_subtasks:
                raise ValueError(
                    f'Invalid task count or duplicate IDs: {len(tasks)} planned, '
                    f'MAX_SUBTASKS={self.settings.max_subtasks}; increase MAX_SUBTASKS for full plans'
                )
            if any(t['domain'] not in state['domains'] for t in tasks.values()):
                raise ValueError('Planner returned an unexpected domain')
            for task in tasks.values():
                if task['task_id'] in task['depends_on'] or any(dep not in tasks for dep in task['depends_on']):
                    raise ValueError('Invalid task dependency')
            # DAG check (bounded by MAX_SUBTASKS).
            visited: set[str] = set()
            def visit(task_id, ancestors):
                if task_id in ancestors:
                    raise ValueError('Circular task dependencies')
                if task_id in visited:
                    return
                for dep in tasks[task_id]['depends_on']:
                    visit(dep, ancestors | {task_id})
                visited.add(task_id)
            for task_id in tasks:
                visit(task_id, set())
        ready = [k for k, t in tasks.items() if t['status'] == 'pending'
                 and all(tasks[dep]['status'] == 'complete' for dep in t['depends_on'])]
        for k in ready:
            tasks[k]['status'] = 'running'
        return {'subtasks': tasks, 'pending_task_ids': ready,
                'execution_round': state.get('execution_round', 0) + 1}

    def rag_worker(self, state: WorkerState) -> dict:
        task = state['task']
        from contextlib import nullcontext
        span = (self.telemetry.measure(state.get('run_id', ''), 'rag_subgraph')
                if self.telemetry is not None else nullcontext())
        started = perf_counter()
        with span:
            result = self.rag_graph.invoke({'task_id': task['task_id'], 'domain': task['domain'],
                                       'query': task['question'], 'role': task.get('role', ''),
                                       'facet': task.get('facet', ''), 'search_attempt': 0,
                                       'run_id': state.get('run_id', '')},
                                      config={'recursion_limit': 16})
        task_duration_s = perf_counter() - started
        return {'branch_results': {task['task_id']: {
            'task_id': task['task_id'], 'elapsed_s': task_duration_s,
            'retrieval_status': result['retrieval_status'],
            'search_attempt': result['search_attempt'], 'evidence': result.get('evidence', []),
            'missing_information': result.get('missing_information', []),
            'ranked_chunk_ids': result.get('ranked_chunk_ids', []),
            'ranked_document_ids': result.get('ranked_document_ids', []),
            'pre_rerank_chunk_ids': result.get('pre_rerank_chunk_ids', []),
            'requested_facets': result.get('requested_facets', []),
            'selected_facet_evidence': result.get('selected_facet_evidence', {}),
            'rejected_chunk_ids': result.get('rejected_chunk_ids', []),
            'candidate_debug': [
                {**{key: item.get(key) for key in (
                    'chunk_id', 'document_id', 'score', 'bm25_rank', 'bm25_score',
                    'dense_rank', 'dense_score', 'rrf_components', 'facet_matches',
                    'retrieval_traces')},
                 'rerank_score': next((ranked['rerank_score']
                     for ranked in result.get('ranked', [])
                     if ranked['chunk_id'] == item['chunk_id']), None)}
                for item in result.get('candidates', [])[:50]],
        }}}

    def evidence_gate(self, state: AssistantState) -> dict:
        tasks = {k: dict(t) for k, t in state['subtasks'].items()}
        # A global retry counter used to let only the first incomplete
        # parallel branch retry. Keep a per-task budget so each independent
        # worker gets at most one additional attempt, without unbounded loops.
        initial_retry = state.get('retry_count', 0)
        retry = initial_retry
        for k, task in tasks.items():
            if task['status'] != 'running' or k not in state.get('branch_results', {}):
                continue
            result = state['branch_results'][k]
            if result.get('retrieval_status') == 'complete' and result['evidence']:
                task['status'] = 'complete'
            elif task.get('worker_retries', initial_retry) < 1:
                task['status'] = 'pending'
                task['worker_retries'] = task.get('worker_retries', initial_retry) + 1
                retry = max(retry, task['worker_retries'])
            elif result.get('evidence'):
                # A bounded retry may still miss a requested facet. Preserve
                # verified, domain-filtered evidence instead of discarding all
                # useful facts; the answer remains explicitly partial.
                task['status'] = 'partial'
            else:
                task['status'] = 'failed'
        evidence_map = {}
        for k, result in state.get('branch_results', {}).items():
            if tasks[k]['status'] in ('complete', 'partial'):
                for item in result.get('evidence', []):
                    evidence_map[item['evidence_id']] = item
        evidence = list(evidence_map.values())
        ready = [k for k, task in tasks.items() if task['status'] == 'pending'
                 and all(tasks[dep]['status'] == 'complete' for dep in task['depends_on'])]
        # Do not silently retry a failed prerequisite or pretend dependent tasks ran.
        completed = all(task['status'] == 'complete' for task in tasks.values())
        steps = [step.model_dump() for step in extract_steps(evidence)]
        return {'subtasks': tasks, 'evidence': evidence, 'administrative_steps': steps,
                'sources': {e['chunk_id']: {k: e.get(k) for k in ('document_id', 'source_url', 'title', 'domain', 'retrieved_at', 'document_version')} for e in evidence},
                'pending_task_ids': ready, 'retry_count': retry,
                'validation': {'status': 'passed' if completed and evidence else 'partial' if evidence else 'failed',
                               'issues': [k for k, t in tasks.items() if t['status'] != 'complete']}}

    def execute_tools(self, state: AssistantState) -> dict:
        evidence = state.get('evidence', [])
        # Select side-effect-free calculators/checklists from the actual user
        # request, not the enriched retrieval anchor ("ügyintézési téma" would
        # otherwise spuriously request a checklist on every follow-up).
        checklist_needed, deadline_needed = requested_tools(state['user_question'])
        results = {}
        if checklist_needed:
            results['checklist'] = build_document_checklist(evidence)
        if deadline_needed:
            duration_mentions = extract_duration_mentions(evidence)
            event = state.get('user_context', {}).get('event_date')
            try:
                parsed_date = date.fromisoformat(event) if event else None
            except ValueError:
                parsed_date = None
            for index, mention in enumerate(duration_mentions[:3], 1):
                rule = DeadlineRule(rule_id=f'sourced-duration-{mention["evidence_id"]}-{index}',
                                    evidence_id=mention['evidence_id'], anchor_event='unverified',
                                    days=mention['days'], unit=mention['unit'])
                results[f'deadline_{index}'] = calculate_deadline(
                    parsed_date, rule, {e['evidence_id'] for e in evidence})
        # A NAV tariff must be present as a real indexed, versioned source; no
        # live rates, approximate horsepower or exemptions are invented.
        if state.get('domains') == ['vehicle'] and state.get('role') != 'seller':
            request = parse_vehicle_duty_request(
                state.get('resolved_question', state['user_question']),
                date.fromisoformat(state.get('reference_date', date.today().isoformat())).year,
            )
            if request is not None:
                results['vehicle_duty'] = calculate_vehicle_acquisition_duty(request, evidence)
        benefit_calc = prepare_unemployment_benefit_calculation(
            state.get('resolved_question', state['user_question']), evidence)
        if benefit_calc is not None:
            results['unemployment_benefit'] = benefit_calc
        loan = parse_loan_request(state.get('resolved_question', state['user_question']))
        if loan is not None:
            results['illustrative_loan'] = calculate_illustrative_loan(loan)
        native_results, native_trace = {}, {}
        # Genuine model-initiated function calling; the deterministic helpers above
        # do not count as native LLM tool calls. Never trigger Ollama in dummy mode.
        if (self.settings.native_tool_calling_enabled and self.settings.llm_provider == 'ollama'
                and self.settings.answer_mode != 'source' and evidence
                and wants_native_tool_call(state['user_question'])):
            invoke = getattr(self.llm, 'call_native_tools', None)
            if callable(invoke):
                native_results, native_trace = invoke(
                    state['user_question'], evidence, run_id=state.get('run_id', ''))
        return {'tool_results': results, 'native_tool_results': native_results,
                'native_tool_trace': native_trace}

    def engineer_context(self, state: AssistantState) -> dict:
        """Expand evidence only with indexed, same-version section siblings that fill a missing need."""
        original = state.get('evidence', [])
        section_chunks = []
        if original and (self.settings.data_dir / 'processed').exists():
            from dap_assistant.rag.retrieval import _chunk_snapshot
            # The cache is reused by BM25; there is no new model/network call.
            section_chunks = _chunk_snapshot(self.settings.data_dir)
        selected, report = prepare_graph_context(
            original, state.get('resolved_question', state['user_question']),
            (state.get('domains') or [''])[0], state.get('role', ''), self.settings,
            section_chunks=section_chunks)
        if self.telemetry is not None:
            self.telemetry.selection(state.get('run_id', ''), {'graph_context_engineering': report})
        # A source found through the same-version parent section is real indexed
        # evidence and must also be available to the final claim/source audit.
        additional = [item for item in selected if item.get('context_origin') ==
                      'same_version_parent_section_sibling']
        sources = dict(state.get('sources', {}))
        for item in additional:
            sources[item['chunk_id']] = {k: item.get(k) for k in (
                'document_id', 'source_url', 'title', 'domain', 'retrieved_at', 'document_version')}
        return {'context_evidence': selected, 'context_engineering': report,
                'evidence': list(state.get('evidence', [])) + additional, 'sources': sources}

    def generate_answer(self, state: AssistantState) -> dict:
        context = answer_context(
            domains=state.get('domains', []), role=state.get('role', ''),
            stage=state.get('stage', ''), strategy=self.settings.answer_mode,
            reference_date=state.get('reference_date', ''),
            event_date=state.get('user_context', {}).get('event_date'),
            event_date_confirmed=state.get('user_context', {}).get('event_date_confirmed', False),
            event_date_source=state.get('user_context', {}).get('event_date_source', 'default'),
            subtasks=state.get('subtasks', {}),
            focus=state.get('focus', ''),
        )
        context['original_question'] = state['user_question']
        context['resolved_question'] = state.get('resolved_question', state['user_question'])
        context['context_resolution'] = state.get('context_resolution', {})
        context['question_analysis'] = state.get('question_analysis', {})
        context['context_engineering'] = state.get('context_engineering', {})
        context['response_plan'] = answer_plan(
            state['user_question'], (state.get('domains') or [''])[0],
            state.get('role', ''), state.get('stage', ''), state.get('evidence', []))
        # Only the production LangGraph tool node can set this flag. The simple
        # baseline/hybrid comparisons have no model-delivered tool round-trip.
        context['native_tool_round_trip'] = bool(
            state.get('native_tool_trace', {}).get('result_returned_to_model')
            and any(call.get('status') == 'executed' and call.get('returned_to_model')
                    for call in state.get('native_tool_trace', {}).get('calls', [])))

        def with_diagnostics(draft, *, fallback: bool, reason: str, strategy: str) -> dict:
            from ..response.quality import claim_facets
            claims = [claim.model_dump() for claim in draft.claims]
            observed = sorted(set().union(*(claim_facets(c, (state.get('domains') or [''])[0])
                                            for c in claims))) if claims else []
            trace = self.telemetry.snapshot(state.get('run_id', '')) if self.telemetry is not None else {}
            context['context_engineering'] = {**context.get('context_engineering', {}),
                'final_request_budget': trace.get('selection', {}).get('answer_context_budget', {})}
            diagnostics = generation_diagnostics(
                trace, fallback=fallback, reason=reason,
                model_claim_count=sum(c.get('origin') == 'model_generated' for c in claims),
                requested_facets=context.get('context_engineering', {}).get('requested_facets', []),
                draft_facets=observed)
            context['generation_diagnostics'] = diagnostics
            if self.telemetry is not None:
                self.telemetry.selection(state.get('run_id', ''), {'generation_diagnostics': diagnostics})
            return {'answer_context': context, 'answer_draft': draft.model_dump(),
                    'answer_fallback': fallback, 'answer_fallback_reason': reason,
                    'answer_strategy': strategy}
        # A short fee lookup can be answered from an exact, cited source
        # sentence without asking Qwen to reproduce numerical information.
        if is_price_question(state['user_question']) and state.get('focus'):
            draft = source_answer(state.get('resolved_question', state['user_question']),
                                  state.get('evidence', []), role=state.get('role', ''),
                                  stage=state.get('stage', ''))
            if draft.claims:
                return with_diagnostics(draft, fallback=False, reason='', strategy='verified_fee_extract')
        try:
            draft = self.llm.answer(state.get('resolved_question', state['user_question']),
                               state.get('context_evidence') or state.get('evidence', []),
                               list(state.get('tool_results', {}).values()) +
                               list(state.get('native_tool_results', {}).values()),
                               run_id=state.get('run_id', ''), context=context)
            source_only_fallback = 'model_no_grounded_complete_claims' in draft.quality_warnings
            return with_diagnostics(
                draft, fallback=source_only_fallback,
                reason='model_output_unusable' if source_only_fallback else '',
                strategy=('source_fallback_unusable_model' if source_only_fallback else
                          'source' if self.settings.answer_mode == 'source' else
                          'dummy' if self.settings.llm_provider == 'dummy' else self.settings.answer_mode))
        except LLMError as exc:
            # Preserve retrieved evidence and return a clearly labelled VERBATIM
            # source extract without replaying a timed-out local generation.
            draft = source_answer(state.get('resolved_question', state['user_question']),
                                  state.get('evidence', []), role=state.get('role', ''),
                                  stage=state.get('stage', ''))
            draft.disclaimer = (
                'A helyi modell nem készített érvényes, befejezett választ. '
                'A fenti forrásidézetek nem teljes körű vagy személyre szabott jogi útmutatók.'
            )
            return {**with_diagnostics(draft, fallback=True, reason='model_request_failed',
                                       strategy='model_error_source_fallback'),
                    'errors': state.get('errors', []) + [f'llm_generation_unavailable:{type(exc.__cause__).__name__}']}

    def answer_audit(self, state: AssistantState) -> dict:
        return audit_answer(state)

    def safe_response(self, state: AssistantState) -> dict:
        if state.get('final_answer'):
            return {'response_status': 'partial'}
        if not state.get('domains'):
            text = ('A DÁP Life Events Assistant kizárólag autóvásárlási és -eladási, '
                    'munkahely elvesztésével kapcsolatos ügyintézési kérdésekben segít. Az ettől eltérő kérdésre '
                    'nem keresek és nem generálok választ.'
                    if state.get('unsupported_reason') == 'out_of_scope' else
                    'Nem azonosítható a két támogatott ügyintézési élethelyzet egyike. '
                    'Kérdezz autóvásárlásról, autóeladásról vagy munkahely elvesztéséről.')
        else:
            text = 'Nem található elegendő, ellenőrizhető hivatalos bizonyíték a válaszhoz. Ellenőrizd a DÁP aktuális tájékoztatóit: https://dap.gov.hu/'
        return {'final_answer': text, 'response_status': 'unsupported' if not state.get('domains') else 'partial'}

    def after_classify(self, state):
        if state['classification_status'] == 'needs_clarification':
            return 'clarify_query'
        return 'plan_tasks' if state['classification_status'] == 'supported' else 'safe_response'

    def dispatch(self, state):
        ids = state.get('pending_task_ids', [])
        if not ids:
            return 'safe_response'
        return [Send('rag_worker', {'task': state['subtasks'][task_id], 'search_attempt': 0, 'run_id': state.get('run_id', '')}) for task_id in ids]

    def after_gate(self, state):
        if state.get('pending_task_ids'):
            return 'plan_tasks'
        if state.get('validation', {}).get('status') == 'failed':
            return 'safe_response'
        if any(t['status'] in ('pending', 'running') for t in state['subtasks'].values()):
            return 'safe_response'
        return 'engineer_context'

    def after_context(self, state):
        resolved = state.get('resolved_question', state['user_question'])
        try:
            year = date.fromisoformat(state.get('reference_date', date.today().isoformat())).year
            vehicle_duty = (state.get('domains') == ['vehicle'] and state.get('role') != 'seller'
                            and parse_vehicle_duty_request(resolved, year) is not None)
            loan = parse_loan_request(resolved) is not None
        except (ValueError, TypeError):
            vehicle_duty = loan = False
        native_requested = (self.settings.native_tool_calling_enabled
                            and self.settings.llm_provider == 'ollama'
                            and self.settings.answer_mode != 'source'
                            and bool(state.get('evidence'))
                            and wants_native_tool_call(state['user_question']))
        # Benefit calculations must also be routed here when a question asks
        # for an amount but does not contain a generic checklist/deadline cue.
        benefit = question_needs(state['user_question']).benefit_amount
        return 'execute_tools' if (any(requested_tools(state['user_question'])) or vehicle_duty
                                   or benefit or loan or native_requested) else 'generate_answer'

    def after_audit(self, state):
        return END if state.get('final_answer') else 'safe_response'

    def measured(self, name, fn):
        if self.telemetry is None:
            return fn
        def wrapped(state):
            with self.telemetry.measure(state.get('run_id', ''), 'main/' + name):
                return fn(state)
        return wrapped
