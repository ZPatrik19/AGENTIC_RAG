"""Answer audit is pure application logic; no Ollama or LangGraph is required."""
from dap_assistant.response.audit import audit_answer
from dap_assistant.orchestration.state import merge_dict, AssistantState


def test_missing_evidence_fails_closed_and_retains_partial_status():
    state: AssistantState = {
        'user_question': 'Eladtam az autómat. Milyen teendőim vannak?',
        'domains': ['vehicle'], 'role': 'seller', 'stage': 'after_event',
        'answer_draft': {'claims': [], 'quality_warnings': []},
        'evidence': [], 'tool_results': {},
    }
    out = audit_answer(state)
    assert out['answer_validation']['status'] == 'failed'
    assert out['response_status'] == 'partial'
    assert not out['answer_draft']['claims']
    assert not out['final_answer']


def test_parallel_branch_reducer_keeps_both_independent_results():
    assert merge_dict({'t1': {'chunk_id': 'a'}}, {'t2': {'chunk_id': 'b'}}) == {
        't1': {'chunk_id': 'a'}, 't2': {'chunk_id': 'b'},
    }
