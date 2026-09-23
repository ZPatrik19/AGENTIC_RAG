"""Independent 5-node LangGraph RAG subgraph with bounded query rewrite."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from dap_assistant.rag.retrieval import LocalQdrant, diversify_results, hybrid_search
from dap_assistant.settings import Settings
from dap_assistant.rag.graph_contract import RAG_NODE_ORDER
from dap_assistant.rag.state import RAGState  # Public re-export for existing imports
from dap_assistant.rag.nodes import RAGNodes




def validate_rag_selection(selected_nodes: tuple[str, ...] | None) -> tuple[str, ...]:
    """Only contiguous real nodes: no invented control edges or graph reordering."""
    if selected_nodes is None:
        return RAG_NODE_ORDER
    nodes = tuple(selected_nodes)
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError('Select at least one unique RAG node')
    if any(node not in RAG_NODE_ORDER for node in nodes):
        raise ValueError('Unknown RAG node; select an existing graph node')
    start = RAG_NODE_ORDER.index(nodes[0])
    if nodes != RAG_NODE_ORDER[start:start + len(nodes)]:
        raise ValueError('Only contiguous nodes in actual RAG graph order are supported')
    if 'evaluate_evidence' in nodes and len(nodes) > 1 and nodes != RAG_NODE_ORDER:
        raise ValueError('Evidence evaluation has a retry edge; use the entire RAG graph or its node alone')
    return nodes


def required_rag_fixture_keys(first_node: str) -> tuple[str, ...]:
    return {
        'process_query': ('query', 'domain'),
        'hybrid_retrieval': ('query', 'domain', 'original_query'),
        'rerank_results': ('original_query', 'domain', 'candidates'),
        'evaluate_evidence': ('original_query', 'domain', 'ranked'),
        'prepare_context': ('evidence',),
    }[first_node]


def validate_rag_fixture(nodes: tuple[str, ...], state: dict) -> None:
    if any(key not in state for key in required_rag_fixture_keys(nodes[0])):
        raise ValueError('Missing real, pre-recorded node input; capture the upstream output first')
    for key in ('candidates', 'ranked', 'evidence'):
        if key in required_rag_fixture_keys(nodes[0]):
            if not isinstance(state[key], list):
                raise ValueError(f'{key} must be a recorded list')
            if any(not item.get('chunk_id') or not item.get('document_id')
                   or not item.get('document_version') for item in state[key]):
                raise ValueError(f'{key} must contain versioned real document chunks')


def build_rag_graph(settings: Settings, dense: LocalQdrant | None = None, telemetry=None,
                    selected_nodes: tuple[str, ...] | None = None):
    """Compile the selected real RAG path; node behavior lives in ``RAGNodes``."""
    selection = validate_rag_selection(selected_nodes)
    nodes = RAGNodes(
        settings, dense=dense, telemetry=telemetry,
        hybrid_search_fn=hybrid_search, diversify_results_fn=diversify_results,
    )

    def measured(name, func):
        if telemetry is None:
            return func

        def wrapped(state):
            with telemetry.measure(state.get('run_id', ''), 'rag/' + name):
                return func(state)
        return wrapped

    functions = {
        'process_query': nodes.process_query,
        'hybrid_retrieval': nodes.retrieve,
        'rerank_results': nodes.rerank,
        'evaluate_evidence': nodes.assess,
        'prepare_context': nodes.prepare,
    }
    graph = StateGraph(RAGState)
    for name in selection:
        graph.add_node(name, measured(name, functions[name]))
    graph.add_edge(START, selection[0])
    if selection == RAG_NODE_ORDER:
        for left, right in zip(selection[:3], selection[1:4]):
            graph.add_edge(left, right)
        graph.add_conditional_edges('evaluate_evidence', nodes.after_assessment)
        graph.add_edge('prepare_context', END)
    else:
        for left, right in zip(selection, selection[1:]):
            graph.add_edge(left, right)
        graph.add_edge(selection[-1], END)
    return graph.compile()
