"""Only real source IDs, bounded context, and observed generation termination."""
from dataclasses import replace

from dap_assistant.context_engineering.evidence_selection import assemble_selection_context
from dap_assistant.context_engineering.context_builder import (context_budget, generation_diagnostics,
                                         prepare_graph_context)
from dap_assistant.settings import Settings
from dap_assistant.presentation.workflow_visualization import execution_dot


def fixture_settings():
    return replace(Settings(), answer_mode='quick', ollama_num_ctx=8192,
                   ollama_quick_num_ctx=2048, quick_single_pass=True)


def test_dynamic_budget_respects_requested_ctx_and_question_complexity():
    settings = fixture_settings()
    simple = context_budget(settings, 'Mennyibe kerül az eredetiségvizsgálat?', 'vehicle', 'buyer')
    broad = context_budget(settings, 'Vettem egy használt autót. Milyen teendőim vannak?', 'vehicle', 'buyer')
    assert simple['requested_num_ctx'] == broad['requested_num_ctx'] == 2048
    assert 320 <= simple['excerpt_budget_estimate'] < broad['excerpt_budget_estimate'] <= 574
    assert not simple['num_ctx_modified']
    large = context_budget(replace(settings, ollama_quick_num_ctx=8192),
                           'Vettem egy autót. Milyen teendőim vannak?', 'vehicle', 'buyer')
    assert large['excerpt_budget_estimate'] <= 700


def test_context_dedup_keeps_separate_document_versions_and_original_provenance():
    base = {'document_id': 'buyer', 'document_version': 'v1', 'domain': 'vehicle',
            'role': 'buyer', 'source_url': 'https://example.org/buyer',
            'text': 'Az átírást 15 napon belül intézd el.', 'facet_matches': ['deadline']}
    chunks = [
        {**base, 'chunk_id': 'a' * 32, 'evidence_id': 'E_aaaaaaaaaaaaaaaa'},
        {**base, 'chunk_id': 'b' * 32, 'evidence_id': 'E_bbbbbbbbbbbbbbbb'},
        {**base, 'chunk_id': 'c' * 32, 'evidence_id': 'E_cccccccccccccccc',
         'document_version': 'v2'},
    ]
    selected, report = prepare_graph_context(chunks, 'Vettem egy autót. Milyen teendőim vannak?',
                                             'vehicle', 'buyer', fixture_settings())
    assert len(selected) == 3  # different chunk IDs remain available for audit
    assert report['parent_section_siblings_added'] == []
    packed = assemble_selection_context(chunks, 'Mennyi a határidő az autó átírására?',
                                        domain='vehicle', role='buyer', token_budget=1000)
    assert len(packed.evidence) == 2  # one identical chunk per document version
    assert {x['evidence_id'] for x in packed.evidence} == {
        'E_aaaaaaaaaaaaaaaa', 'E_cccccccccccccccc'}


def test_parent_section_expansion_requires_matching_document_version_and_missing_facet():
    base = {'document_id': 'buyer', 'document_version': 'v1', 'domain': 'vehicle',
            'role': 'buyer', 'source_url': 'https://example.org/buyer',
            'section_text_sha256': 'section-digest', 'section_path': ['Autóvásárlás']}
    initial = {**base, 'chunk_id': 'a' * 32, 'evidence_id': 'E_aaaaaaaaaaaaaaaa',
               'text': 'Az átírást 15 napon belül intézd el.'}
    sibling = {**base, 'chunk_id': 'b' * 32,
               'text': 'Tulajdonosváltáskor új kötelező gépjármű-felelősségbiztosítást kell kötni.'}
    wrong_version = {**sibling, 'chunk_id': 'c' * 32, 'document_version': 'v2'}
    wrong_section = {**sibling, 'chunk_id': 'd' * 32, 'section_text_sha256': 'other-section'}
    selected, report = prepare_graph_context(
        [initial], 'Vettem egy használt autót. Milyen teendőim vannak?',
        'vehicle', 'buyer', fixture_settings(),
        section_chunks=[wrong_version, wrong_section, sibling])
    assert [x['chunk_id'] for x in selected] == ['a' * 32, 'b' * 32]
    assert selected[-1]['evidence_id'] == 'E_bbbbbbbbbbbbbbbb'
    assert report['parent_section_siblings_added'][0]['anchor_chunk_id'] == 'a' * 32
    assert report['parent_section_siblings_added'][0]['document_version'] == 'v1'
    assert report['coverage_measurement'].startswith('lexical_')


def test_no_parent_section_expansion_without_matching_index_or_need():
    initial = {'chunk_id': 'a' * 32, 'evidence_id': 'E_aaaaaaaaaaaaaaaa',
               'text': 'A kérdést személyesen a kormányablakban intézheted.',
               'document_id': 'buyer', 'document_version': 'v1', 'domain': 'vehicle',
               'section_text_sha256': 'parent'}
    sibling = {**initial, 'chunk_id': 'b' * 32, 'text': 'A kérdést személyesen a kormányablakban intézheted.'}
    chosen, report = prepare_graph_context([initial], 'Hol intézhetem a gépjármű átírását?',
                                           'vehicle', 'buyer', fixture_settings(),
                                           section_chunks=[sibling])
    assert len(chosen) == 1
    assert report['parent_section_siblings_added'] == []


def test_generation_diagnostics_distinguish_length_from_naturally_incomplete():
    length = generation_diagnostics({
        'llm_usage': [{'phase': 'answer', 'requested_num_ctx': 2048,
                       'requested_num_predict': 768, 'prompt_eval_count': 950,
                       'eval_count': 768, 'done_reason': 'length'}],
        'llm_failures': [{'phase': 'answer', 'failure_kind': 'output_token_limit'}],
    }, fallback=True, reason='model_request_failed', model_claim_count=0,
        requested_facets=['steps', 'documents'], draft_facets=[])
    assert length['status'] == 'output_token_limit'
    assert length['token_limit_vs_content_gap'] == 'output_limit'
    assert length['eval_count'] == 768
    completed = generation_diagnostics({
        'llm_usage': [{'phase': 'answer', 'requested_num_ctx': 2048,
                       'requested_num_predict': 768, 'prompt_eval_count': 1300,
                       'eval_count': 184, 'done_reason': 'stop'}],
    }, fallback=False, reason='', model_claim_count=2,
        requested_facets=['steps', 'documents'], draft_facets=['steps'])
    assert completed['status'] == 'completed_with_uncovered_facets'
    assert completed['token_limit_vs_content_gap'] == 'content_gap'
    assert completed['uncovered_facets_proxy'] == ['documents']
    assert completed['prompt_near_requested_window'] is False
    unknown = generation_diagnostics({}, fallback=False, reason='', model_claim_count=1,
                                     requested_facets=['steps'], draft_facets=[])
    assert unknown['status'] == 'not_instrumented_or_deterministic_response'
    assert unknown['prompt_eval_count'] is None


def test_graph_view_only_draws_the_context_node_when_executed():
    before = execution_dot({}, [{'node': 'evidence_gate'}, {'node': 'generate_answer'}])
    assert 'engineer_context' not in before
    after = execution_dot({}, [{'node': 'evidence_gate'}, {'node': 'engineer_context'},
                               {'node': 'generate_answer'}])
    assert 'Kontextus és lefedettség' in after
    assert '"engineer_context" -> "generate_answer"' in after


def test_real_main_graph_context_node_feeds_source_audited_generation(monkeypatch, tmp_path):
    """Exercise the real node functions; only LangGraph's constructor is stubbed."""
    import importlib.util
    from pathlib import Path
    import sys
    from types import ModuleType

    from dap_assistant.llm import Claim, Draft
    import dap_assistant
    import dap_assistant.rag.retrieval as retrieval

    graph_pkg = ModuleType('langgraph')
    graph_mod = ModuleType('langgraph.graph')
    checkpoint_pkg = ModuleType('langgraph.checkpoint')
    memory_pkg = ModuleType('langgraph.checkpoint.memory')
    types_mod = ModuleType('langgraph.types')

    class Graph:
        def __init__(self, *_args):
            self.nodes = {}
            self.routes = {}
        def add_node(self, name, fn):
            self.nodes[name] = fn
        def add_edge(self, *_args):
            pass
        def add_conditional_edges(self, name, fn):
            self.routes[name] = fn
        def compile(self, **_kwargs):
            return self

    graph_mod.StateGraph = Graph
    graph_mod.START, graph_mod.END = 'START', 'END'
    memory_pkg.InMemorySaver = type('FakeMemory', (), {})
    types_mod.Send = type('FakeSend', (), {'__init__': lambda self, *_args: None})
    initial = {'document_id': 'buyer', 'document_version': 'v1',
               'source_url': 'https://example.org/buyer', 'title': 'Test',
               'domain': 'vehicle', 'role': 'buyer', 'section_text_sha256': 'section',
               'chunk_id': 'a' * 32, 'evidence_id': 'E_aaaaaaaaaaaaaaaa',
               'text': 'Az átírást 15 napon belül intézd el.'}
    sibling = {**initial, 'chunk_id': 'b' * 32,
               'text': 'Tulajdonosváltáskor új kötelező gépjármű-felelősségbiztosítást kell kötni.'}

    class FakeLLM:
        def answer(self, question, evidence, tools, *, context, **kwargs):
            assert context['context_engineering']['parent_section_siblings_added']
            assert any(e['evidence_id'] == 'E_bbbbbbbbbbbbbbbb' for e in evidence)
            return Draft(claims=[Claim(
                text=sibling['text'], supporting_quote=sibling['text'],
                evidence_ids=['E_bbbbbbbbbbbbbbbb'], category='insurance',
                origin='model_generated')],
                model_context_evidence_ids=['E_bbbbbbbbbbbbbbbb'])

    with monkeypatch.context() as patch:
        for name, module in (('langgraph', graph_pkg), ('langgraph.graph', graph_mod),
                             ('langgraph.types', types_mod),
                             ('langgraph.checkpoint', checkpoint_pkg),
                             ('langgraph.checkpoint.memory', memory_pkg)):
            patch.setitem(sys.modules, name, module)
        path = Path(dap_assistant.__file__).with_name('workflow.py')
        spec = importlib.util.spec_from_file_location('dap_assistant._workflow_p615_nodes', path)
        workflow = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(workflow)
        patch.setattr(workflow, 'get_llm', lambda *_args, **_kwargs: FakeLLM())
        patch.setattr(workflow, 'build_rag_graph', lambda *_args, **_kwargs: None)
        patch.setattr(retrieval, '_chunk_snapshot', lambda *_args: [sibling])
        (tmp_path / 'processed').mkdir()
        graph = workflow.build_workflow(replace(fixture_settings(), data_dir=tmp_path,
                                                llm_provider='dummy'))
        state = workflow.initial_state('Vettem egy használt autót. Milyen teendőim vannak?')
        state.update(domains=['vehicle'], role='buyer', evidence=[initial],
                     validation={'status': 'passed'})
        assert graph.routes['evidence_gate'](state) == 'engineer_context'
        state.update(graph.nodes['engineer_context'](state))
        assert state['evidence'][-1]['document_version'] == 'v1'
        state.update(graph.nodes['generate_answer'](state))
        state.update(graph.nodes['answer_audit'](state))
        assert any(c['evidence_ids'] == ['E_bbbbbbbbbbbbbbbb']
                   for c in state['answer_draft']['claims'])
        assert 'kötelező gépjármű-felelősségbiztosítást' in state['final_answer']
        assert state['answer_context']['generation_diagnostics']['status'] == (
            'not_instrumented_or_deterministic_response')
