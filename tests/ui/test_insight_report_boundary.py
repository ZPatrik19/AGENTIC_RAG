"""UI report code has one home and chat orchestration imports it explicitly."""
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / 'src' / 'dap_assistant'


def test_report_has_one_implementation_and_chat_keeps_navigation():
    report = (PACKAGE / 'presentation' / 'insight_report.py').read_text(encoding='utf-8')
    chat = (PACKAGE / 'ui.py').read_text(encoding='utf-8')
    assert report.count('def render_report(') == 1
    assert 'def render_report(' not in chat
    assert 'from dap_assistant.presentation.insight_report import' in chat
    assert 'render_report(report,' in chat
    assert 'stream_workflow(' in chat
