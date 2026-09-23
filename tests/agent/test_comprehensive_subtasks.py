"""Regression tests for independently searchable, user-derived work packages."""
from dataclasses import replace

import pytest
from pydantic import ValidationError

from dap_assistant.fast_path import classify_explicit, plan_explicit
from dap_assistant.llm import Plan, PlannedTask
from dap_assistant.evaluation.metrics import task_coverage
from dap_assistant.settings import Settings


VEHICLE = (
    'Készíts teljes, időrendi ügyintézési tervet egy használt autó megvásárlása után, '
    'a dokumentumokkal, határidőkkel és költségekkel együtt!'
)
EMPLOYMENT = (
    'Készíts teljes ügyintézési tervet a munkaviszonyom megszűnése után, '
    'a szükséges dokumentumokkal, ellátásokkal, határidőkkel és pénzügyi teendőkkel együtt!'
)


@pytest.mark.parametrize('question,domain,facets', [
    (VEHICLE, 'vehicle', ('steps', 'documents', 'deadline', 'insurance', 'costs')),
    (EMPLOYMENT, 'employment', ('steps', 'documents', 'supports', 'deadline', 'healthcare', 'costs')),
])
def test_full_plan_has_independent_focused_tasks(question, domain, facets):
    classification = classify_explicit(question)
    assert classification is not None and classification.domains == [domain]
    planned = plan_explicit(question, classification.domains)
    assert len(planned.tasks) == len(facets)
    assert [task.task_id for task in planned.tasks] == [f't{i}' for i in range(1, len(facets) + 1)]
    assert [task.facet for task in planned.tasks] == list(facets)
    assert len({task.question for task in planned.tasks}) == len(facets)
    assert all(task.domain == domain and not task.depends_on for task in planned.tasks)
    assert all(len(task.question) >= 5 for task in planned.tasks)
    # The planner has no gold JSON as input: it must work on a new user wording.
    assert plan_explicit(question, [domain]).model_dump() == planned.model_dump()


def test_full_purchase_detects_buyer_role_in_compound_megvasarlas():
    assert classify_explicit(VEHICLE).role == 'buyer'


def test_seller_full_plan_does_not_invent_buyer_duty():
    q = 'Készíts teljes ügyintézési tervet: eladtam az autómat, dokumentumok és határidők!'
    planned = plan_explicit(q, ['vehicle'])
    assert [task.facet for task in planned.tasks] == [
        'steps', 'documents', 'deadline', 'insurance',
    ]
    assert all('vagyonszerzési illeték' not in task.question for task in planned.tasks)


def test_simple_and_mixed_domain_questions_keep_small_plan():
    assert len(plan_explicit('Milyen költségei vannak az autó átírásának?', ['vehicle']).tasks) == 1
    assert len(plan_explicit('Vettem autót, milyen teendők és költségek vannak?', ['vehicle']).tasks) == 2
    assert len(plan_explicit('Eladtam az autómat, megszűnt a munkaviszonyom.',
                             ['vehicle', 'employment']).tasks) == 2


def test_six_tasks_are_validated_but_seven_are_rejected():
    tasks = [PlannedTask(task_id=f't{i}', domain='employment',
                         question=f'{i} külön kérdés') for i in range(1, 7)]
    assert len(Plan(tasks=tasks).tasks) == 6
    with pytest.raises(ValidationError):
        Plan(tasks=tasks + [PlannedTask(task_id='t7', domain='employment', question='hetedik')])
    assert replace(Settings(), max_subtasks=6).max_subtasks == 6


def test_full_workflow_executes_one_rag_worker_per_facet(tmp_path):
    pytest.importorskip('langgraph')
    from dap_assistant.workflow import build_workflow, initial_state

    settings = replace(Settings(), data_dir=tmp_path, llm_provider='dummy',
                       embedding_provider='dummy', answer_mode='source', max_subtasks=6)
    # Empty local corpus: proves execution and honest partial response, not retrieval accuracy.
    for question, expected in ((VEHICLE, 5), (EMPLOYMENT, 6)):
        graph = build_workflow(settings)
        result = graph.invoke(initial_state(question),
                              config={'configurable': {'thread_id': f'facets-{expected}'},
                                      'recursion_limit': 50})
        assert len(result['subtasks']) == expected
        assert set(result['branch_results']) == set(result['subtasks'])
        assert all('ranked_chunk_ids' in branch for branch in result['branch_results'].values())
        assert result['validation']['status'] != 'passed'
        assert not result['pending_task_ids']
        assert all(task['worker_retries'] == 1 for task in result['subtasks'].values())
        assert all(task['status'] == 'failed' for task in result['subtasks'].values())


def test_subtask_coverage_requires_distinct_semantic_facets_not_only_task_count():
    expected = [
        {'domain': 'employment', 'label': label, 'keywords': [], 'requires_dependencies': False}
        for label in ('registration', 'documents', 'benefits', 'deadline', 'health', 'financial')
    ]
    planned = plan_explicit(EMPLOYMENT, ['employment']).tasks
    actual = {task.task_id: task.model_dump() for task in planned}
    assert task_coverage(actual, expected) == 1.0
    misleading = {name: {**value, 'facet': 'steps'} for name, value in actual.items()}
    assert task_coverage(misleading, expected) == pytest.approx(1 / 6)
