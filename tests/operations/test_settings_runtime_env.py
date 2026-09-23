"""Settings must honor runtime provider overrides after module import."""
from dataclasses import replace

from dap_assistant.settings import Settings


def test_provider_environment_is_read_per_instance(monkeypatch):
    # Importing Settings before monkeypatching is the regression scenario.
    monkeypatch.setenv('LLM_PROVIDER', 'dummy')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'dummy')
    dummy = Settings()
    assert dummy.llm_provider == 'dummy'
    assert dummy.embedding_provider == 'dummy'
    # Dataclasses.replace must preserve explicitly configured providers.
    assert replace(dummy, answer_mode='quick').embedding_provider == 'dummy'
    monkeypatch.setenv('LLM_PROVIDER', 'ollama')
    monkeypatch.setenv('EMBEDDING_PROVIDER', 'sentence_transformers')
    online = Settings()
    assert online.llm_provider == 'ollama'
    assert online.embedding_provider == 'sentence_transformers'


def test_runtime_defaults_match_current_local_profile(monkeypatch):
    for name in (
        'ANSWER_MODE', 'OLLAMA_READ_TIMEOUT_S', 'OLLAMA_TOTAL_TIMEOUT_S',
        'OLLAMA_QUICK_NUM_CTX', 'OLLAMA_ANSWER_NUM_PREDICT', 'OLLAMA_TEMPERATURE',
        'NATIVE_TOOL_MODEL', 'AGENTIC_ANSWER_RECOVERY_MODEL',
    ):
        monkeypatch.delenv(name, raising=False)
    settings = Settings()
    assert settings.answer_mode == 'detailed'
    assert settings.ollama_read_timeout_s == 180
    assert settings.ollama_total_timeout_s == 480
    assert settings.ollama_quick_num_ctx == 3072
    assert settings.ollama_answer_num_predict == 900
    assert settings.ollama_temperature == 0.1
    assert settings.native_tool_model == 'qwen3:4b'
    assert settings.agentic_answer_recovery_model == 'qwen3:4b'
