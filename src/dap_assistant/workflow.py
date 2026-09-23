"""Top-level LangGraph topology and initial assistant state."""
from __future__ import annotations

from datetime import date

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .orchestration.state import AssistantState, merge_dict as merge_dict
from .orchestration.workflow_nodes import WorkflowNodes
from .llm import get_llm
from .rag.rag_graph import build_rag_graph
from .settings import Settings


def build_workflow(settings: Settings, dense=None, checkpointer=None, telemetry=None):
    """Compile the top-level LangGraph; node behavior lives in ``WorkflowNodes``."""
    nodes = WorkflowNodes(
        settings,
        get_llm(settings, telemetry=telemetry),
        build_rag_graph(settings, dense, telemetry=telemetry),
        telemetry,
    )
    graph = StateGraph(AssistantState)
    for name, fn in (
        ('classify_intent', nodes.classify_intent), ('clarify_query', nodes.clarify_query),
        ('plan_tasks', nodes.plan_tasks), ('rag_worker', nodes.rag_worker),
        ('evidence_gate', nodes.evidence_gate), ('engineer_context', nodes.engineer_context),
        ('execute_tools', nodes.execute_tools),
        ('generate_answer', nodes.generate_answer), ('answer_audit', nodes.answer_audit),
        ('safe_response', nodes.safe_response),
    ):
        graph.add_node(name, nodes.measured(name, fn))
    graph.add_edge(START, 'classify_intent')
    graph.add_conditional_edges('classify_intent', nodes.after_classify)
    graph.add_edge('clarify_query', END)
    graph.add_conditional_edges('plan_tasks', nodes.dispatch)
    graph.add_edge('rag_worker', 'evidence_gate')
    graph.add_conditional_edges('evidence_gate', nodes.after_gate)
    graph.add_conditional_edges('engineer_context', nodes.after_context)
    graph.add_edge('execute_tools', 'generate_answer')
    graph.add_edge('generate_answer', 'answer_audit')
    graph.add_conditional_edges('answer_audit', nodes.after_audit)
    graph.add_edge('safe_response', END)
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def initial_state(question: str, summary: str = '', domain_hint: str = '', event_date: date | None = None,
                  reference_date: date | None = None, previous_turn: dict | None = None) -> AssistantState:
    from datetime import datetime
    from zoneinfo import ZoneInfo
    today = reference_date or datetime.now(ZoneInfo('Europe/Budapest')).date()
    context = {'domain_hint': domain_hint, 'event_date': today.isoformat(),
               'event_date_source': 'default', 'event_date_confirmed': False}
    if event_date:
        context['event_date'] = event_date.isoformat()
        context['event_date_source'] = 'user_selected'
        context['event_date_confirmed'] = True
    elif 'tegnap' in question.casefold():
        from datetime import timedelta
        context['event_date'] = (today - timedelta(days=1)).isoformat()
        context['event_date_source'] = 'inferred_from_question'
    return {'user_question': question, 'resolved_question': question,
            'previous_turn': previous_turn or {}, 'run_id': '', 'conversation_summary': summary,
            'reference_date': today.isoformat(),
            'user_context': context, 'branch_results': {}, 'tool_results': {}, 'retry_count': 0, 'execution_round': 0,
            'subtasks': {}, 'pending_task_ids': [], 'evidence': [], 'administrative_steps': [], 'errors': [], 'sources': {},
            'native_tool_results': {}, 'native_tool_trace': {},
            'final_answer': '', 'response_status': 'pending', 'answer_strategy': 'pending'}
