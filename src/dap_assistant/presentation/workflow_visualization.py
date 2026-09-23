"""Safe, pure workflow view from actual LangGraph updates (no model reasoning)."""
from __future__ import annotations

import re

from dap_assistant.rag.graph_contract import RAG_NODE_ORDER


def _safe_label(value: object) -> str:
    # Escape DOT special characters; labels are display-only metadata.
    return re.sub(r'[\r\n]+', ' ', str(value))[:95].replace('\\', '\\\\').replace('"', '\\"')


def execution_dot(final: dict, events: list[dict]) -> str:
    """Render only nodes confirmed by actual streamed events and recorded tasks."""
    executed = {e.get('node') for e in events}
    if not executed:
        return 'digraph Workflow { label="Még nincs végrehajtási esemény"; }'
    rows = ['digraph Workflow {', 'rankdir=TB;',
            'graph [bgcolor="transparent", pad="0.25", nodesep="0.4", ranksep="0.55"];',
            'node [shape=box, style="rounded,filled", fillcolor="#e0f2fe", color="#2563eb", fontname="Arial"];',
            'edge [color="#64748b"];']
    names = ('classify_intent', 'clarify_query', 'plan_tasks', 'rag_worker',
             'evidence_gate', 'engineer_context', 'execute_tools', 'generate_answer',
             'answer_audit', 'safe_response')
    labels = {'classify_intent': 'Élethelyzet felismerése', 'clarify_query': 'Pontosítás',
              'plan_tasks': 'Részfeladatok tervezése', 'rag_worker': 'RAG-keresés',
              'evidence_gate': 'Források összevezetése',
              'engineer_context': 'Kontextus és lefedettség', 'execute_tools': 'Eszközök',
              'generate_answer': 'Válasz összeállítása', 'answer_audit': 'Válasz auditja',
              'safe_response': 'Biztonságos lezárás'}
    for name in names:
        if name == 'rag_worker' and final.get('branch_results'):
            continue  # Render each executed worker separately instead.
        if name in executed:
            rows.append(f'"{name}" [label="{labels[name]}"];')
    if 'classify_intent' in executed:
        rows.append('"START" [shape=oval,label="START"]; "START" -> "classify_intent";')
    if 'plan_tasks' in executed:
        if 'classify_intent' in executed:
            rows.append('"classify_intent" -> "plan_tasks";')
        tasks = final.get('subtasks') or {}
        branches = final.get('branch_results') or {}
        for task_id, task in tasks.items():
            # Prefix with an ordinal so IDs with punctuation never collide in DOT.
            identifier = 'task_' + re.sub(r'[^a-zA-Z0-9_]', '_', str(task_id))
            state = str(task.get('status') or 'pending')
            facet = _safe_label(task.get('facet') or 'általános keresés')
            domain = _safe_label(task.get('domain', ''))
            color = '#15803d' if state == 'complete' else '#b45309' if state == 'failed' else '#64748b'
            rows.append(f'"{identifier}" [fillcolor="#f1f5f9",color="{color}",'
                        f'label="{_safe_label(task_id)} · {facet}\\n{domain} · {_safe_label(state)}"];')
            rows.append(f'"plan_tasks" -> "{identifier}";')
            for dependency in task.get('depends_on') or []:
                if dependency not in tasks:
                    continue
                dependency_id = 'task_' + re.sub(
                    r'[^a-zA-Z0-9_]', '_', str(dependency))
                rows.append(f'"{dependency_id}" -> "{identifier}" '
                            '[style=dashed,label="függőség"];')
            # A planned task is not the same thing as an executed RAG branch.
            if task_id not in branches or 'rag_worker' not in executed:
                continue
            branch = branches[task_id]
            worker_id = 'rag_' + re.sub(r'[^a-zA-Z0-9_]', '_', str(task_id))
            attempts = branch.get('search_attempt')
            attempts_label = str(attempts) if isinstance(attempts, int) else '?'
            retrieval = _safe_label(branch.get('retrieval_status', 'ismeretlen'))
            count = len(branch.get('evidence') or [])
            rows.append(f'"{worker_id}" [fillcolor="#dcfce7",color="#15803d",'
                        f'label="RAG worker · {_safe_label(task_id)}\\n'
                        f'{retrieval} · {attempts_label} kör · {count} chunk"];')
            rows.append(f'"{identifier}" -> "{worker_id}";')
            if 'evidence_gate' in executed:
                rows.append(f'"{worker_id}" -> "evidence_gate";')
    if 'rag_worker' in executed and not (final.get('branch_results') or {}):
        # Older/incomplete reports may have a worker event without branch outputs.
        rows.append('"rag_worker" [label="RAG worker · nincs ágeredmény"];')
        if 'evidence_gate' in executed:
            rows.append('"rag_worker" -> "evidence_gate";')
    sequence = ['rag_worker', 'evidence_gate', 'engineer_context', 'execute_tools',
                'generate_answer', 'answer_audit']
    for left, right in zip(sequence, sequence[1:]):
        if left == 'rag_worker' and final.get('branch_results'):
            continue  # Each completed branch already joins evidence_gate.
        if left in executed and right in executed:
            rows.append(f'"{left}" -> "{right}";')
    if 'engineer_context' in executed and 'generate_answer' in executed and 'execute_tools' not in executed:
        rows.append('"engineer_context" -> "generate_answer";')
    for name in ('clarify_query', 'safe_response'):
        if name in executed and 'classify_intent' in executed:
            rows.append(f'"classify_intent" -> "{name}" [style=dashed];')
    for tool_name, result in final.get('tool_results', {}).items():
        if 'execute_tools' not in executed:
            break
        identifier = 'tool_' + re.sub(r'[^a-zA-Z0-9_]', '_', str(tool_name))
        rows.append(f'"{identifier}" [fillcolor="#fef3c7",color="#b45309",label="{_safe_label(result.get("tool", tool_name))}\\n{_safe_label(result.get("status", ""))}"];')
        rows.append(f'"execute_tools" -> "{identifier}" [style=dotted];')
    # Only draw native tool nodes when Ollama actually selected them.
    native = final.get('native_tool_trace') or {}
    if 'execute_tools' in executed and native.get('calls'):
        rows.append('"native_select" [fillcolor="#dbeafe",color="#2563eb",label="Ollama · natív toolválasztás"];')
        rows.append('"execute_tools" -> "native_select" [style=dotted];')
        for index, call in enumerate(native['calls'], 1):
            identifier = f'native_tool_{index}'
            rows.append(f'"{identifier}" [fillcolor="#fef3c7",color="#b45309",'
                        f'label="{_safe_label(call.get("tool_name"))}\\n{_safe_label(call.get("status"))}"];')
            rows.append(f'"native_select" -> "{identifier}" [style=dotted];')
            if call.get('returned_to_model'):
                rows.append(f'"{identifier}" -> "native_feedback" [style=dotted];')
        if native.get('result_returned_to_model'):
            rows.append('"native_feedback" [fillcolor="#dcfce7",color="#15803d",'
                        'label="Ollama · role=tool eredmény visszaadva"];')
            if 'generate_answer' in executed:
                rows.append('"native_feedback" -> "generate_answer" [style=dotted];')
    if 'answer_audit' in executed:
        rows.append('"END" [shape=oval,label="END"]; "answer_audit" -> "END";')
    elif 'safe_response' in executed:
        rows.append('"END" [shape=oval,label="END"]; "safe_response" -> "END";')
    rows.append('}')
    return '\n'.join(rows)


def rag_subgraph_dot(task_id: str, branch: dict) -> str:
    """Show the real RAG topology, not fabricated per-node runtime events.

    A branch result confirms a RAG invocation completed, but does not record
    the exact internal node events. Dashed retry is a possible conditional edge.
    """
    if not isinstance(branch, dict) or not branch:
        raise ValueError('A completed RAG branch is required for the diagram')
    labels = {
        'process_query': 'Keresőkérdés előkészítése',
        'hybrid_retrieval': 'BM25 + vektoros keresés',
        'rerank_results': 'Újrarangsorolás',
        'evaluate_evidence': 'Bizonyítékok vizsgálata',
        'prepare_context': 'Kontextus előkészítése',
    }
    attempts = branch.get('search_attempt')
    attempts_label = str(attempts) if isinstance(attempts, int) else '?'
    rows = [
        'digraph RAG {', 'rankdir=LR;',
        'graph [bgcolor="transparent",pad="0.15",nodesep="0.32",ranksep="0.42"];',
        'node [shape=box,style="rounded,filled",fillcolor="#e0f2fe",'
        'color="#2563eb",fontname="Arial",fontsize=11];',
        'edge [color="#64748b"];',
        f'label="RAG algráf szerkezete · {_safe_label(task_id)} '
        f'({attempts_label} keresési kör)";',
    ]
    for node in RAG_NODE_ORDER:
        rows.append(f'"{node}" [label="{labels[node]}"];')
    for first, second in zip(RAG_NODE_ORDER, RAG_NODE_ORDER[1:]):
        rows.append(f'"{first}" -> "{second}";')
    rows.append('"evaluate_evidence" -> "process_query" '
                '[style=dashed,color="#b45309",label="ha kevés a bizonyíték"];')
    rows.append('}')
    return '\n'.join(rows)
