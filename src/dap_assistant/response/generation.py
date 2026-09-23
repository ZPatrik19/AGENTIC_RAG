"""Orchestrate evidence selection, one answer inference, and bounded recovery.

The transport stays independent of the RAG and grounding layers.
"""
from __future__ import annotations

import json
import time

from .models import NaturalResponse, FacetSelection, Draft
from .fallback import source_answer
from ..prompt_engineering.prompt_payload import _verified_tool_hints
from .grounding import _attach_source_quotes, supplement_grounded_facts
from ..prompt_engineering.answer_prompt import answer_plan, generation_instruction
from ..context_engineering.context_builder import context_budget
from ..context_engineering.token_budget import fit_answer_payload, serialize, ContextBudgetError, answer_output_budget
from ..context_engineering.evidence_selection import (assemble_selection_context, requested_facets,
                                 validate_selection_response, deterministic_facet_selection)
from ..inference.errors import LLMError, LLMGenerationLengthError


class AnswerGenerationMixin:
    """Runs the answer workflow through the adapter-provided ``_ask`` interface."""

    def _recover_answer(self, *, question: str, role: str, plan: dict, facets: tuple[str, ...],
                        tool_hints: list[dict], selected_evidence: list[dict],
                        budget_plan: dict, run_id: str, reason: str) -> tuple[NaturalResponse, list[dict], list[str]]:
        """Run the single bounded instruct-model recovery allowed by the workflow."""
        model = self.settings.agentic_answer_recovery_model
        if self.telemetry is not None:
            self.telemetry.selection(run_id, {'answer_recovery': {
                'attempted': True, 'reason': reason, 'model': model, 'max_attempts': 1}})
        try:
            recovery_instruction = (generation_instruction(plan) +
                ' A helyreállító válasz rövid JSON legyen. Ha a parafrázis bizonytalan, '
                'másolj teljes, önálló forrásmondatot; ne használj címet vagy töredéket.')
            recovery_payload, recovery_budget = fit_answer_payload(
                {'question': question, 'role': role, 'stage': plan['stage'],
                 'information_needs': list(facets),
                 'verified_tool_excerpts': tool_hints, 'evidence': selected_evidence},
                recovery_instruction, NaturalResponse.model_json_schema(),
                window=budget_plan['requested_num_ctx'],
                output=max(128, min(1024, self.settings.agentic_answer_recovery_num_predict)),
                safety=self.settings.context_safety_tokens)
            if self.telemetry is not None:
                self.telemetry.selection(run_id, {'recovery_context_budget': recovery_budget})
            if not recovery_budget['fits'] or not recovery_payload['evidence']:
                raise LLMError('Recovery evidence does not fit the context window')
            response = self._ask(
                recovery_instruction, serialize(recovery_payload),
                NaturalResponse, run_id=run_id, phase='answer', model_override=model,
                num_predict_override=self.settings.agentic_answer_recovery_num_predict)
            recovered_evidence = recovery_payload['evidence']
            recovered_ids = [item['evidence_id'] for item in recovered_evidence]
            return response, recovered_evidence, recovered_ids
        except LLMError:
            if self.telemetry is not None:
                self.telemetry.selection(run_id, {'answer_recovery': {
                    'attempted': True, 'reason': reason, 'model': model,
                    'max_attempts': 1, 'status': 'failed'}})
            raise

    def answer(self, question: str, evidence: list[dict], tools: list[dict], run_id: str = '',
               context: dict | None = None) -> Draft:
        if self.settings.answer_mode == 'source':
            if self.telemetry is not None:
                self.telemetry.progress(run_id, 'source', 0, 0.0)
            return source_answer(question, evidence, role=(context or {}).get('role', ''),
                                 stage=(context or {}).get('stage', ''))
        # First select evidence per requested information need; *quick* affects
        # final answer length, not how many topics the retriever may cover.
        facets = requested_facets(question, (context or {}).get('life_events', ['vehicle'])[0],
                                  role=(context or {}).get('role', ''))
        # Quick mode must not block on TWO separate Qwen inferences. A compact,
        # deterministic facet selector preserves coverage with no invented IDs.
        fast = self.settings.answer_mode == 'quick' and self.settings.quick_single_pass
        domain = (context or {}).get('life_events', ['vehicle'])[0]
        role = (context or {}).get('role', '')
        stage = (context or {}).get('stage', '')
        budget_plan = context_budget(self.settings, question, domain, role)
        budget = budget_plan['excerpt_budget_estimate']
        assembled = assemble_selection_context(evidence, question,
                     domain=domain, role=role, token_budget=budget)
        selected = deterministic_facet_selection(assembled, facets)
        if self.telemetry is not None:
            self.telemetry.selection(run_id, {'requested_facets': list(facets),
                'available_chunk_ids': [item.get('chunk_id', '') for item in evidence],
                'included_evidence_ids': [item['evidence_id'] for item in assembled.evidence],
                'rejected_evidence_ids': assembled.rejected_ids,
                'estimated_input_tokens': assembled.estimated_input_tokens,
                'token_budget_estimate': assembled.token_budget_estimate,
                'selection_mode': 'deterministic_single_pass' if fast else 'llm',
                'dynamic_context_budget': budget_plan,
                'duplicate_aliases': assembled.duplicate_aliases,
                'selected_source_unit_coverage_proxy': {
                    'covered': [f for f in facets if assembled.facet_coverage.get(f)],
                    'missing': [f for f in facets if not assembled.facet_coverage.get(f)],
                    'measurement': 'lexical_complete_units_not_semantic_entailment'}})
        if self.settings.answer_mode in ('quick', 'detailed'):
            if not fast:
                selection_payload = {
                    'question': question, 'information_needs': list(facets),
                    'available_full_text': assembled.evidence,
                }
                started = time.perf_counter()
                try:
                    chosen = self._ask(
                        'Válaszd ki a bizonyítékokat MINDEN információigényhez. '
                        'Kizárólag available_full_text evidence_id értékei használhatók. '
                        'A hiányzókat missing_needs jelöli.',
                        json.dumps(selection_payload, ensure_ascii=False),
                        FacetSelection, run_id=run_id, phase='selection',
                    )
                    try:
                        selected = validate_selection_response(
                            chosen.evidence_by_need, chosen.missing_needs,
                            {item['evidence_id'] for item in assembled.evidence}, facets)
                    except ValueError as exc:
                        # Invalid IDs are an untrusted model output, not an app crash.
                        if self.telemetry is not None:
                            self.telemetry.selection(run_id, {
                                'selection_validation_error': type(exc).__name__,
                                'recovered_with': 'deterministic_in_budget_sources'})
                        selected = deterministic_facet_selection(assembled, facets)
                except LLMError:
                    # Propagate to the workflow: it marks the source-only fallback
                    # as PARTIAL, not as a successful Qwen answer. Never retry.
                    raise
                finally:
                    if self.telemetry is not None:
                        self.telemetry.record(run_id, 'evidence_selection_llm',
                                              time.perf_counter()-started)
            selected_ids = list(dict.fromkeys(eid for ids in selected.values() for eid in ids))
            if self.telemetry is not None:
                self.telemetry.selection(run_id, {'chosen_by_need': selected,
                    'missing_needs': [name for name in facets if not selected[name]]})
            selected_evidence = [item for item in assembled.evidence if item['evidence_id'] in selected_ids]
            if not selected_evidence:
                return source_answer(question, evidence, role=role, stage=stage)
            plan = answer_plan(question, domain, role, stage, selected_evidence)
            if self.telemetry is not None:
                self.telemetry.selection(run_id, {'response_plan': plan})
            # After a verified native-tool round-trip the source-only supplements
            # and checklist audit handle coverage. Ask the normal answer model for
            # fewer short claims: this avoids spending 768 tokens on 7 verbose ones.
            compact_agentic = bool(fast and (context or {}).get('native_tool_round_trip') is True)
            tool_hints = (_verified_tool_hints(tools, selected_evidence, question,
                                              role, plan['stage']) if compact_agentic else [])
            prompt_payload = (
                {'question': question, 'role': role, 'stage': plan['stage'],
                 'question_analysis': plan['question_analysis'],
                 'response_plan': {'source_topic_hints': plan['source_topic_hints'],
                                   'broad_procedural_question': plan['broad_procedural_question'],
                                   'source_alignment': plan['source_alignment']},
                 'information_needs': list(facets),
                 'missing_evidence_needs': [f for f in facets if not selected.get(f)],
                 'verified_tool_excerpts': tool_hints,
                 'evidence': selected_evidence}
                if compact_agentic else
                {'question': question, 'information_needs': list(facets),
                 'missing_evidence_needs': [f for f in facets if not selected.get(f)],
                 'selected_by_need': selected, 'answer_mode': self.settings.answer_mode,
                 'evidence': selected_evidence,
                 'context': ({k: (context or {}).get(k) for k in
                             ('life_events', 'life_event_labels', 'role', 'stage',
                              'answer_strategy', 'reference_date', 'event_date_confirmed')}
                             if fast else context or {})})
            compact_instruction = generation_instruction(plan)
            output_budget = answer_output_budget(self.settings, len(facets))
            prompt_payload, final_budget = fit_answer_payload(
                prompt_payload, compact_instruction, NaturalResponse.model_json_schema(),
                window=budget_plan['requested_num_ctx'], output=output_budget,
                safety=self.settings.context_safety_tokens)
            selected_evidence = prompt_payload['evidence']
            selected_ids = [item['evidence_id'] for item in selected_evidence]
            if self.telemetry is not None:
                self.telemetry.selection(run_id, {'answer_context_budget': final_budget,
                    'included_evidence_ids': selected_ids})
            if not final_budget['fits'] or not selected_evidence:
                if self.telemetry is not None:
                    self.telemetry.llm_failure(run_id, 'answer', 'ContextBudgetError', 0,
                        diagnostic={'failure_kind': 'input_context_budget', **final_budget})
                raise LLMError('No complete evidence fits the input context budget') from ContextBudgetError()
            # Natural Hungarian composition remains a real Qwen inference, not
            # Python sentence concatenation. Original source text stays attached.
            started = time.perf_counter()
            try:
                recovered = False
                recovery_reason = ''


                try:
                    generated = self._ask(
                        compact_instruction,
                        serialize(prompt_payload),
                        NaturalResponse, run_id=run_id, phase='answer',
                    )
                except LLMError as error:
                    # A second real LLM inference is permitted ONLY for a
                    # completed first response that was truncated at num_predict,
                    # after a genuine model-delivered tool cycle. Never replay
                    # a timeout, partial stream, invalid JSON or schema failure.
                    if not (fast and self.settings.agentic_answer_recovery_enabled
                            and (context or {}).get('native_tool_round_trip') is True
                            and isinstance(error.__cause__, LLMGenerationLengthError)):
                        raise
                    if not self.settings.agentic_answer_recovery_model:
                        raise
                    # Never reuse a truncated output fragment as a source claim.
                    recovery_reason = 'initial_output_token_limit'
                    generated, selected_evidence, selected_ids = self._recover_answer(
                        question=question, role=role, plan=plan, facets=facets,
                        tool_hints=tool_hints, selected_evidence=selected_evidence,
                        budget_plan=budget_plan, run_id=run_id, reason=recovery_reason)
                    recovered = True
                # Check against exactly the source units supplied to Qwen, not
                # an unrelated sentence elsewhere in the same retrieved chunk.
                draft = _attach_source_quotes(generated, selected_evidence, question,
                                              role=role, stage=plan['stage'])
                # A completed, schema-valid response may nevertheless have zero
                # independently verifiable claims (observed in WORK_001). The
                # length-only recovery above cannot handle that case. Permit ONE
                # distinct recovery after genuine native tools, never a retry of
                # a timeout, broken JSON, failed tools or an already retried answer.
                if (not draft.claims and not recovered and compact_agentic
                        and self.settings.agentic_answer_recovery_enabled
                        and self.settings.agentic_answer_recovery_model):
                    recovery_reason = 'initial_no_grounded_claims'
                    try:
                        generated, selected_evidence, selected_ids = self._recover_answer(
                            question=question, role=role, plan=plan, facets=facets,
                            tool_hints=tool_hints, selected_evidence=selected_evidence,
                            budget_plan=budget_plan, run_id=run_id, reason=recovery_reason)
                    except LLMError:
                        # The first model completed but was unusable; preserve
                        # this honest fallback rather than replaying once more.
                        pass
                    else:
                        draft = _attach_source_quotes(generated, selected_evidence, question,
                                                      role=role, stage=plan['stage'])
                        recovered = True
                draft.model_context_evidence_ids = list(selected_ids)
                if final_budget['budget_removed_evidence_ids']:
                    draft.quality_warnings.append('context_budget_evidence_omitted')
                    notice = ('A kontextuskeret miatt nem minden visszakeresett bizonyíték '
                              'került a modellhez; a tájékoztatás részleges.')
                    draft.disclaimer = ' '.join(filter(None, [draft.disclaimer, notice]))
                if recovered and self.telemetry is not None:
                    self.telemetry.selection(run_id, {'answer_recovery': {
                        'attempted': True, 'reason': recovery_reason,
                        'model': self.settings.agentic_answer_recovery_model, 'max_attempts': 1,
                        'status': 'grounded_claims' if draft.claims else 'no_grounded_claims'}})
                if not draft.claims:
                    # A valid JSON envelope is not necessarily a usable cited answer.
                    # Preserve the real source extract and mark it as a partial fallback.
                    fallback = source_answer(question, evidence, selected_ids,
                                             role=role, stage=plan['stage'])
                    fallback.quality_warnings = draft.quality_warnings + ['model_no_grounded_complete_claims']
                    fallback.model_context_evidence_ids = list(selected_ids)
                    return fallback
                return supplement_grounded_facts(draft, question, evidence, facets,
                                                 model_prompt_evidence=selected_evidence,
                                                 role=role, stage=plan['stage'])
            finally:
                if self.telemetry is not None:
                    self.telemetry.record(run_id, 'answer_generation_llm', time.perf_counter()-started)

        raise ValueError('ANSWER_MODE must be quick, detailed, or source')
