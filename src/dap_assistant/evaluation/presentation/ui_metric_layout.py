"""Pure presentation rules for node-aware evaluation metrics.

Never attribute an upstream fixture's ranking to a downstream isolated node.
This module reads saved report fields only: no LLM, retrieval or index access.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Any

from dap_assistant.rag.graph_contract import RAG_NODE_ORDER, RAG_SUBFLOWS


@dataclass(frozen=True)
class MetricSection:
    title: str
    keys: tuple[str, ...]
    description: str


RETRIEVAL = (
    'retrieval_recall_at_5', 'retrieval_precision_at_5', 'retrieval_mrr',
)
CONTEXT = ('context_recall', 'context_precision', 'context_coverage')
CITATIONS = ('citation_accuracy', 'citation_coverage', 'abstention_accuracy')
AGENTIC = (
    'tool_selection_accuracy', 'subtask_coverage',
    'tool_call_efficiency', 'workflow_success_rate',
)

def selected_rag_nodes(scope: str, target: str) -> tuple[str, ...]:
    """Return only nodes explicitly measured by the chosen evaluation target."""
    if scope == 'single_node' and target.startswith('rag/'):
        node = target.partition('/')[2]
        return (node,) if node in RAG_NODE_ORDER else ()
    if scope == 'subflow':
        return RAG_SUBFLOWS.get(target, ())
    return ()


def metric_sections(scope: str, target: str) -> tuple[MetricSection, ...]:
    """Show scores only for stages executed by the measured target.

    Full workflow metrics are grouped by responsibility, not erroneously
    presented as per-node measurements. Node/subflow scores are isolated.
    """
    if scope == 'full_workflow':
        return (
            MetricSection('Keresés és rangsorolás', RETRIEVAL,
                          'RAG · hybrid_retrieval → rerank_results; SILVER/proxy referencia alapján.'),
            MetricSection('Bizonyíték és kontextus', CONTEXT,
                          'RAG · evaluate_evidence → prepare_context; teljes workflow esetén a tényleges válaszkontextus.'),
            MetricSection('Hivatkozások és válaszkezelés', CITATIONS,
                          'Válaszgenerálás / válaszellenőrzés; technikai hivatkozás-ellenőrzés, nem szemantikai tényellenőrzés.'),
            MetricSection('Agentic irányítás', AGENTIC,
                          'Fő gráf · eszközválasztás, részfeladat-felismerés és technikai végrehajtás.'),
        )
    nodes = selected_rag_nodes(scope, target)
    if not nodes:
        return ()
    sections: list[MetricSection] = []
    if 'hybrid_retrieval' in nodes or 'rerank_results' in nodes:
        label = 'Hibrid keresés' if 'rerank_results' not in nodes else (
            'Újrarangsorolás' if 'hybrid_retrieval' not in nodes else 'Keresés és újrarangsorolás'
        )
        sections.append(MetricSection(
            label, RETRIEVAL,
            'A mért node/kiválasztott részfolyamat kimeneti rangsora; a korábbi node-ok csak bemeneti fixture-k.',
        ))
    if 'evaluate_evidence' in nodes or 'prepare_context' in nodes:
        final_context = 'prepare_context' in nodes
        sections.append(MetricSection(
            'Előkészített bizonyíték / kontextus' if final_context else 'Bizonyítékválogatás',
            CONTEXT,
            ('A RAG által előkészített bizonyítéklista, nem az LLM-be ténylegesen betöltött prompt.'
             if final_context else
             'Az evaluate_evidence által elfogadott bizonyítéklista; nem a generált válasz vagy az LLM-prompt.'),
        ))
    return tuple(sections)


def node_latency_rows(rows: list[dict[str, Any]], nodes: tuple[str, ...]) -> list[dict[str, Any]]:
    """Extract measured rag/<node> spans, excluding fixture/wall-clock time.

    None (N/A) means no recorded span; an absent span must not become 0 seconds.
    """
    result: list[dict[str, Any]] = []
    for node in nodes:
        name = 'rag/' + node
        durations = []
        for row in rows:
            trace = row.get('node_execution_trace') or {}
            for span in trace.get('spans') or []:
                if span.get('name') != name:
                    continue
                duration = span.get('duration_s')
                if isinstance(duration, (float, int)) and not isinstance(duration, bool) and duration >= 0:
                    durations.append(float(duration))
        result.append({
            'node': name,
            'calls': len(durations),
            'mean_s': mean(durations) if durations else None,
        })
    return result


def execution_success(rows: list[dict[str, Any]]) -> float | None:
    """Successful benchmark execution, not task/answer completion."""
    if not rows:
        return None
    return sum(row.get('success') is True for row in rows) / len(rows)
