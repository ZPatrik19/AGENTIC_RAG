"""The documented environment must support the production subtask contract."""
from pathlib import Path


def test_example_subtask_limit_covers_the_six_branch_benchmark():
    root = Path(__file__).resolve().parents[2]
    values = dict(line.split('=', 1) for line in (root / '.env.example').read_text(
        encoding='utf-8').splitlines() if line and not line.lstrip().startswith('#') and '=' in line)
    assert values['MAX_SUBTASKS'] == '6'
    assert values['LLM_PROVIDER'] == 'ollama'
    assert values['ANSWER_MODE'] == 'detailed'
