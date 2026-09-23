"""Sidebar example questions must be scoped, deterministic, and one-shot."""
from dap_assistant.presentation.ui_presentation import example_questions_for_domain, submitted_question


def test_auto_domain_shows_both_supported_topics():
    groups = example_questions_for_domain('')
    assert [label for label, _ in groups] == [
        'Autóvásárlás és -eladás', 'Munkahely elvesztése',
    ]
    assert all(len(questions) == 3 for _, questions in groups)
    assert len({q for _, questions in groups for q in questions}) == 6


def test_manual_domain_shows_only_its_own_examples():
    vehicle = example_questions_for_domain('vehicle')
    employment = example_questions_for_domain('employment')
    assert len(vehicle) == len(employment) == 1
    assert 'autó' in vehicle[0][1][0].lower()
    assert 'munkámat' in employment[0][1][0].lower()
    assert example_questions_for_domain('unsupported') == ()


def test_suggestion_uses_the_same_input_without_overriding_typed_question():
    assert submitted_question(None, 'Mintakérdés') == 'Mintakérdés'
    assert submitted_question('', 'Mintakérdés') == 'Mintakérdés'
    assert submitted_question('Saját kérdés', 'Mintakérdés') == 'Saját kérdés'
    assert submitted_question(None, None) is None
