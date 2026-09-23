"""Run real RAG node, Agentic workflow and load evaluations without inventing missing reference scores."""
from __future__ import annotations


from datetime import date, datetime, timezone
from hashlib import sha256
from pathlib import Path
from time import perf_counter
import uuid

from .load_testing import _llm_perf as _llm_perf, summarize_load_llm as summarize_load_llm, _component_stats as _component_stats, run_load as run_load
from .dataset import active_dataset, index_versions, load_dataset
from .reporting import _markdown_report as _markdown_report, _run_name_segment as _run_name_segment, list_saved_runs as list_saved_runs, load_saved_run as load_saved_run, render_report_markdown as render_report_markdown, run_report_stem as run_report_stem, save_run as save_run
from .automatic_reference import ensure_auto_reference
from .metrics import average_available
from .functional_metrics import _prompt_context_ids as _prompt_context_ids, _context_chunk_ids as _context_chunk_ids, _full_metrics, _rag_metrics, _FUNCTIONAL_METRIC_NAMES
from ..observability.telemetry import Telemetry
from .runtime_environment import environment
from dap_assistant.rag.dense_resources import acquire_dense, release_dense
from dap_assistant.documents.ingestion import load_chunks
from ..settings import Settings
from dap_assistant.rag.graph_contract import RAG_NODE_ORDER, RAG_SUBFLOWS, STANDALONE_RAG_TARGETS


# Compatibility names used by callers of this evaluation module.
RAG_NODES = RAG_NODE_ORDER
SUBFLOW_TARGETS = RAG_SUBFLOWS
STANDALONE_TARGETS = STANDALONE_RAG_TARGETS


def _ollama_runtime_info(settings: Settings) -> dict:
    """Best-effort runtime verification from local Ollama. No network beyond configured localhost."""
    if settings.llm_provider != 'ollama':
        return {'available': False, 'configured_model': settings.ollama_model, 'active_context_window': None}
    try:
        import httpx
        with httpx.Client(timeout=3, trust_env=False) as client:
            base = settings.ollama_base_url.rstrip('/')
            version = None
            try:
                version = client.get(base + '/api/version').json().get('version')
            except Exception:
                pass
            ps = client.get(base + '/api/ps')
            ps.raise_for_status()
            models = ps.json().get('models', [])
            entry = next((m for m in models if m.get('name') == settings.ollama_model), None)
            return {
                'available': True, 'configured_model': settings.ollama_model, 'ollama_version': version,
                'loaded': entry is not None,
                'active_context_window': (entry or {}).get('context_length'),
                'size_vram_bytes': (entry or {}).get('size_vram'),
            }
    except Exception as exc:
        return {'available': False, 'configured_model': settings.ollama_model,
                'active_context_window': None, 'error': f'{type(exc).__name__}: {exc}'}


def _environment(settings: Settings, chunks: list[dict]) -> dict:
    return environment(settings, chunks)


def _retrieval_configuration(settings: Settings) -> dict:
    return {
        'embedding_provider': settings.embedding_provider,
        'embedding_model': settings.embedding_model,
        'embedding_device': settings.embedding_device,
        'lexical_retrieval': 'BM25',
        'dense_top_k': 12,
        'bm25_top_k': 12,
        'fusion': 'Reciprocal Rank Fusion k=60',
        'rag_candidate_limit': 16,
        'prepared_context_limit': 20,  # Actual upper bound in rag_graph.prepare
        'facet_search_limit_per_query': 8,
    }


def available_targets() -> dict[str, dict[str, tuple[str, ...]]]:
    """UI-facing targets derived from the actual RAG graph node names."""
    return {
        'single_node': dict(STANDALONE_TARGETS),
        'subflow': dict(SUBFLOW_TARGETS),
        'full_workflow': {'agentic/full': ()},
    }


def select_cases(*, topic: str = 'all', question_ids: list[str] | None = None,
                 dataset_path: Path | None = None) -> list[dict]:
    cases = load_dataset(dataset_path or active_dataset())
    if topic not in ('all', 'vehicle', 'employment'):
        raise ValueError('topic must be all, vehicle or employment')
    if topic != 'all':
        cases = [case for case in cases if case['category'] == topic]
    if question_ids:
        lookup = {case['question_id']: case for case in cases}
        missing = [qid for qid in question_ids if qid not in lookup]
        if missing:
            raise ValueError(f'Unknown/non-selected question IDs: {missing}')
        cases = [lookup[qid] for qid in question_ids]
    if not cases:
        raise ValueError('No evaluation cases selected')
    return cases


def _role(case: dict) -> str:
    if case['category'] == 'vehicle':
        return 'buyer'
    return ''


def _rag_input(case: dict, run_id: str, first_node: str) -> dict:
    base = {
        'task_id': case['question_id'],
        'run_id': run_id,
        'domain': case['category'],
        'role': _role(case),
        'query': case['question'],
        'original_query': case['question'],
        'search_attempt': 0,
    }
    # A hybrid_retrieval-only run consumes a checked query fixture directly.
    if first_node == 'hybrid_retrieval':
        base['search_attempt'] = 1
    return base


def _prepare_rag_fixture(case: dict, run_id: str, nodes: tuple[str, ...],
                         settings: Settings, dense) -> dict:
    """Prepare actual upstream graph state outside the measured node/subflow timing."""
    first = nodes[0]
    state = _rag_input(case, run_id, first)
    if first in ('process_query', 'hybrid_retrieval'):
        return state
    from dap_assistant.rag.rag_graph import build_rag_graph
    if first == 'rerank_results':
        prefix = build_rag_graph(settings, dense=dense, telemetry=None,
                                 selected_nodes=('process_query', 'hybrid_retrieval'))
        return prefix.invoke(_rag_input(case, run_id + '-fixture', 'process_query'),
                             config={'recursion_limit': 8})
    if first in ('evaluate_evidence', 'prepare_context'):
        prefix = build_rag_graph(settings, dense=dense, telemetry=None,
                                 selected_nodes=('process_query', 'hybrid_retrieval', 'rerank_results'))
        prepared = prefix.invoke(_rag_input(case, run_id + '-fixture', 'process_query'),
                                 config={'recursion_limit': 8})
        if first == 'prepare_context':
            assessor = build_rag_graph(settings, dense=dense, telemetry=None,
                                       selected_nodes=('evaluate_evidence',))
            prepared = assessor.invoke(prepared, config={'recursion_limit': 4})
        return prepared
    raise ValueError(f'Unsupported RAG fixture start node: {first}')



def run_functional(settings: Settings, *, scope: str = 'full_workflow', target: str = 'agentic/full',
                   topic: str = 'all', question_ids: list[str] | None = None,
                   local_judge: bool = False, progress=None) -> dict:
    chunks = load_chunks(settings.data_dir)
    if not chunks:
        raise RuntimeError('Processed official documents missing; download and index first')
    # Offline, idempotent: no model, embedding or paid API calls.
    ensure_auto_reference(settings.data_dir, chunks=chunks)
    cases = select_cases(topic=topic, question_ids=question_ids)
    versions = index_versions(chunks)
    telemetry = Telemetry()
    dense = acquire_dense(settings)
    rows: list[dict] = []
    try:
        if scope == 'full_workflow':
            from ..workflow import build_workflow, initial_state
            app = build_workflow(settings, dense=dense, telemetry=telemetry)
            for index, case in enumerate(cases, 1):
                run_id = str(uuid.uuid4())
                started = perf_counter()
                output = None
                error = None
                try:
                    state = initial_state(case['question'], reference_date=date.fromisoformat(case['reference_date']))
                    state['run_id'] = run_id
                    output = app.invoke(state, config={'configurable': {'thread_id': run_id}, 'recursion_limit': 30})
                except Exception as exc:  # preserve exact failure type
                    error = f'{type(exc).__name__}: {exc}'
                trace = telemetry.snapshot(run_id)
                if output is None:
                    metrics = {name: None for name in _FUNCTIONAL_METRIC_NAMES}
                    metrics['workflow_success_rate'] = 0.0
                    details = {}
                    answer = ''
                    response_status = 'error'
                else:
                    metrics, details = _full_metrics(case, output, trace, versions, chunks, telemetry, run_id, error)
                    answer = output.get('final_answer', '')
                    response_status = output.get('response_status')
                    if local_judge:
                        from .local_judge import judge_answer
                        judgment = judge_answer(
                            settings,
                            question=case['question'], answer=answer,
                            reference_facts=case.get('expected_facts', []),
                            evidence=output.get('evidence', []),
                            reference_status=details['reference_status'],
                            human_reviewed=case.get('human_reviewed', False),
                        )
                        if judgment:
                            metrics['answer_correctness'] = judgment['correctness']
                            metrics['answer_completeness'] = judgment['completeness']
                            metrics['answer_relevancy'] = judgment['relevancy']
                            metrics['faithfulness'] = judgment['faithfulness']
                            judge_claims = judgment.get('claims', [])
                            if judge_claims:
                                evaluable = [c for c in judge_claims if c.get('status') != 'not_evaluable']
                                if evaluable:
                                    metrics['unsupported_claim_rate'] = sum(c.get('status') == 'unsupported' for c in evaluable) / len(evaluable)
                                    metrics['contradicted_claim_rate'] = sum(c.get('status') == 'contradicted' for c in evaluable) / len(evaluable)
                            details['local_judge'] = judgment
                rows.append({
                    'question_id': case['question_id'], 'topic': case['category'],
                    'question': case['question'], 'difficulty': case['difficulty'],
                    'scope': scope, 'target': target,
                    'success': error is None, 'response_status': response_status,
                    'latency_s': perf_counter() - started, 'error': error,
                    'expected_facts': case.get('expected_facts', []),
                    'expected_subtasks': case.get('expected_subtasks', []),
                    'retrieved_sources': sorted({item.get('document_id') for item in (output or {}).get('evidence', []) if item.get('document_id')}),
                    'final_context': details.get('final_context_chunk_ids', []),
                    'generated_answer': answer,
                    'missing_facts': [],
                    'unsupported_claims': details.get('claim_support', {}).get('counts', {}).get('unsupported'),
                    'citation_validation': metrics.get('citation_accuracy'),
                    'node_execution_trace': trace,
                    'metrics': metrics, 'details': details,
                })
                if progress:
                    progress(index, len(cases))
        else:
            if scope not in ('single_node', 'subflow'):
                raise ValueError('scope must be single_node, subflow or full_workflow')
            mapping = available_targets()[scope]
            if target not in mapping:
                raise ValueError('Invalid target for selected evaluation scope')
            nodes = mapping[target]
            from dap_assistant.rag.rag_graph import build_rag_graph
            app = build_rag_graph(settings, dense=dense, telemetry=telemetry, selected_nodes=nodes)
            for index, case in enumerate(cases, 1):
                run_id = str(uuid.uuid4())
                started = perf_counter()
                output = None
                error = None
                try:
                    state = _prepare_rag_fixture(case, run_id, nodes, settings, dense)
                    state['run_id'] = run_id
                    output = app.invoke(state, config={'recursion_limit': 16})
                except Exception as exc:
                    error = f'{type(exc).__name__}: {exc}'
                trace = telemetry.snapshot(run_id)
                if output is None:
                    metrics = {name: None for name in _FUNCTIONAL_METRIC_NAMES}
                    metrics['workflow_success_rate'] = 0.0
                    details = {}
                else:
                    metrics, details = _rag_metrics(case, output, versions, chunks, nodes=nodes)
                rows.append({
                    'question_id': case['question_id'], 'topic': case['category'],
                    'question': case['question'], 'difficulty': case['difficulty'],
                    'scope': scope, 'target': target,
                    'success': error is None, 'response_status': 'n/a',
                    'latency_s': perf_counter() - started, 'error': error,
                    'expected_facts': case.get('expected_facts', []),
                    'expected_subtasks': case.get('expected_subtasks', []),
                    'retrieved_sources': sorted({item.get('document_id') for item in (output or {}).get('evidence', []) if item.get('document_id')}),
                    'final_context': details.get('final_context_chunk_ids', []),
                    'generated_answer': '', 'missing_facts': [], 'unsupported_claims': None,
                    'citation_validation': None, 'node_execution_trace': trace,
                    'metrics': metrics, 'details': details,
                })
                if progress:
                    progress(index, len(cases))
    finally:
        release_dense(dense)

    summary_metrics = {name: average_available([row['metrics'] for row in rows], name)
                       for name in _FUNCTIONAL_METRIC_NAMES}
    return {
        'kind': 'functional_v4',
        'run_id': str(uuid.uuid4()),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': {
            'scope': scope, 'target': target, 'topic': topic,
            'question_ids': question_ids or [case['question_id'] for case in cases],
            'model': settings.ollama_model, 'context_window': settings.ollama_num_ctx,
            'retrieval_configuration': _retrieval_configuration(settings),
            'node_metric_schema_version': 2 if scope != 'full_workflow' else None,
            'dataset_version': '4.0',
            'dataset_sha256': sha256(active_dataset().read_bytes()).hexdigest(),
            'local_judge': local_judge,
        },
        'execution_environment': _environment(settings, chunks),
        'rows': rows,
        'summary': {
            'metrics': summary_metrics,
            'evaluated_cases': len(rows),
            'successful_runs': sum(row['success'] for row in rows),
            'failed_runs': sum(not row['success'] for row in rows),
        },
    }


# Compatibility for existing CLI scripts and third-party integrations.
