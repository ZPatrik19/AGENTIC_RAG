"""Routing contract checked even on machines without optional LangGraph installed."""
import pytest

from dap_assistant.conversation import SUPPORTED_DOMAINS, is_outside_scope
from dap_assistant.fast_path import classify_explicit


def test_supported_domain_contract():
    assert SUPPORTED_DOMAINS == {'vehicle', 'employment'}
    assert classify_explicit('Autót vettem, elvesztettem a munkámat.').domains == [
        'vehicle', 'employment']


@pytest.mark.parametrize('question', [
    'Megvettem a lakást, mit tegyek?',
    'Egyéni vállalkozást szeretnék indítani.',
])
def test_removed_domains_are_not_routed_to_supported_retrieval(question):
    assert is_outside_scope(question)
