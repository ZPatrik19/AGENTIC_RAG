"""Explicit runtime configuration, shared by CLI and Streamlit."""
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
import math
import os

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env', override=False)


def _configured_answer_mode() -> str:
    """Read the public answer strategy from the environment."""
    mode = os.getenv('ANSWER_MODE', 'detailed').strip().lower()
    if mode not in ('quick', 'detailed'):
        raise ValueError('ANSWER_MODE must be quick or detailed')
    return mode


def _env_bool(name: str, default: str) -> bool:
    """Read a boolean at instance creation, not when the module is imported."""
    value = os.getenv(name, default).strip().lower()
    if value in ('1', 'true', 'yes', 'on'):
        return True
    if value in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'{name} must be true/false (received {value!r})')


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    return int(value) if value else None


def _env_path(name: str, default: str) -> Path:
    """Resolve a relative data directory against the repository root."""
    return ROOT / os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the active environment at instantiation time.

    Values supplied via ``dataclasses.replace`` remain explicit. A new Settings
    instance sees subsequent environment changes (including UI/test overrides).
    """

    root: Path = ROOT
    data_dir: Path = field(default_factory=lambda: _env_path('DATA_DIR', 'data'))
    llm_provider: str = field(default_factory=lambda: os.getenv('LLM_PROVIDER', 'ollama'))
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    )
    ollama_model: str = field(default_factory=lambda: os.getenv('OLLAMA_MODEL', 'qwen3:4b'))
    embedding_provider: str = field(
        default_factory=lambda: os.getenv('EMBEDDING_PROVIDER', 'sentence_transformers')
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv('EMBEDDING_MODEL', 'intfloat/multilingual-e5-small')
    )
    embedding_device: str = field(default_factory=lambda: os.getenv('EMBEDDING_DEVICE', 'cpu'))
    fast_routing: bool = field(default_factory=lambda: _env_bool('FAST_ROUTING', 'true'))
    ollama_read_timeout_s: float = field(
        default_factory=lambda: _env_float('OLLAMA_READ_TIMEOUT_S', 180)
    )
    ollama_total_timeout_s: float = field(
        default_factory=lambda: _env_float('OLLAMA_TOTAL_TIMEOUT_S', 480)
    )
    answer_mode: str = field(default_factory=_configured_answer_mode)
    ollama_num_predict: int = field(default_factory=lambda: _env_int('OLLAMA_NUM_PREDICT', 180))
    quick_single_pass: bool = field(default_factory=lambda: _env_bool('QUICK_SINGLE_PASS', 'true'))
    ollama_quick_num_ctx: int = field(
        default_factory=lambda: _env_int('OLLAMA_QUICK_NUM_CTX', 3072)
    )
    ollama_quick_num_predict: int = field(
        default_factory=lambda: _env_int('OLLAMA_QUICK_NUM_PREDICT', 384)
    )
    ollama_selection_num_predict: int = field(
        default_factory=lambda: _env_int('OLLAMA_SELECTION_NUM_PREDICT', 512)
    )
    ollama_answer_num_predict: int = field(
        default_factory=lambda: _env_int('OLLAMA_ANSWER_NUM_PREDICT', 900)
    )
    ollama_num_ctx: int = field(default_factory=lambda: _env_int('OLLAMA_NUM_CTX', 8192))
    ollama_keep_alive: str = field(default_factory=lambda: os.getenv('OLLAMA_KEEP_ALIVE', '15m'))
    ollama_think: bool = field(default_factory=lambda: _env_bool('OLLAMA_THINK', 'false'))
    ollama_temperature: float = field(default_factory=lambda: _env_float('OLLAMA_TEMPERATURE', 0.1))
    # None preserves the installed model's sampling defaults.
    ollama_top_p: float | None = field(default_factory=lambda: _env_optional_float('OLLAMA_TOP_P'))
    ollama_top_k: int | None = field(default_factory=lambda: _env_optional_int('OLLAMA_TOP_K'))
    context_safety_tokens: int = field(
        default_factory=lambda: _env_int('CONTEXT_SAFETY_TOKENS', 128)
    )
    answer_adaptive_output: bool = field(
        default_factory=lambda: _env_bool('ANSWER_ADAPTIVE_OUTPUT', 'true')
    )
    native_tool_calling_enabled: bool = field(
        default_factory=lambda: _env_bool('NATIVE_TOOL_CALLING_ENABLED', 'true')
    )
    native_tool_call_limit: int = 2  # Hard upper bound, not controlled by prompts.
    native_tool_model: str = field(
        default_factory=lambda: os.getenv('NATIVE_TOOL_MODEL', 'qwen3:4b').strip()
    )
    native_tool_num_ctx: int = field(default_factory=lambda: _env_int('NATIVE_TOOL_NUM_CTX', 3072))
    native_tool_num_predict: int = field(
        default_factory=lambda: _env_int('NATIVE_TOOL_NUM_PREDICT', 512)
    )
    native_tool_read_timeout_s: float = field(
        default_factory=lambda: _env_float('NATIVE_TOOL_READ_TIMEOUT_S', 180)
    )
    native_tool_total_timeout_s: float = field(
        default_factory=lambda: _env_float('NATIVE_TOOL_TOTAL_TIMEOUT_S', 300)
    )
    native_tool_stream: bool = field(
        default_factory=lambda: _env_bool('NATIVE_TOOL_STREAM', 'true')
    )
    agentic_answer_recovery_enabled: bool = field(
        default_factory=lambda: _env_bool('AGENTIC_ANSWER_RECOVERY_ENABLED', 'true')
    )
    agentic_answer_recovery_model: str = field(
        default_factory=lambda: os.getenv(
            'AGENTIC_ANSWER_RECOVERY_MODEL',
            os.getenv('NATIVE_TOOL_MODEL', 'qwen3:4b'),
        ).strip()
    )
    agentic_answer_recovery_num_predict: int = field(
        default_factory=lambda: _env_int('AGENTIC_ANSWER_RECOVERY_NUM_PREDICT', 640)
    )
    max_subtasks: int = field(default_factory=lambda: _env_int('MAX_SUBTASKS', 6))
    max_rag_attempts: int = field(default_factory=lambda: _env_int('MAX_RAG_ATTEMPTS', 2))

    @property
    def manifest(self) -> Path:
        return self.root / 'config' / 'document_sources.yaml'


    def __post_init__(self) -> None:
        """Reject invalid settings before an index, model, or graph is opened.

        ``dataclasses.replace`` also invokes this validator, so UI overrides and
        CLI overrides follow the same contract. It does not contact Ollama.
        """
        if self.llm_provider not in ('ollama', 'dummy'):
            raise ValueError('LLM_PROVIDER must be ollama or dummy')
        if self.embedding_provider not in ('sentence_transformers', 'dummy'):
            raise ValueError('EMBEDDING_PROVIDER must be sentence_transformers or dummy')
        # ``source`` is an internal deterministic strategy used by evaluation and
        # offline tests. Public environment configuration accepts only quick/detailed.
        if self.answer_mode not in ('quick', 'detailed', 'source'):
            raise ValueError('ANSWER_MODE must be quick, detailed, or internal source')
        if not self.ollama_model.strip():
            raise ValueError('OLLAMA_MODEL cannot be empty')
        try:
            url = urlsplit(self.ollama_base_url)
            port = url.port  # Forces validation of an explicitly supplied port.
        except ValueError as exc:
            raise ValueError('OLLAMA_BASE_URL must be a valid HTTP(S) URL') from exc
        if (url.scheme not in ('http', 'https') or not url.hostname
                or url.username or url.password or url.query or url.fragment
                or (port is not None and port <= 0)):
            raise ValueError('OLLAMA_BASE_URL must be an HTTP(S) URL without credentials or query')
        for name in (
            'ollama_num_ctx', 'ollama_quick_num_ctx', 'native_tool_num_ctx',
            'ollama_num_predict', 'ollama_quick_num_predict', 'ollama_selection_num_predict',
            'ollama_answer_num_predict', 'native_tool_num_predict',
            'agentic_answer_recovery_num_predict', 'context_safety_tokens',
            'native_tool_call_limit',
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name.upper()} must be a positive integer')
        if not 1 <= self.max_subtasks <= 6:
            raise ValueError('MAX_SUBTASKS must be between 1 and 6 (Plan schema limit)')
        if not 1 <= self.max_rag_attempts <= 10:
            raise ValueError('MAX_RAG_ATTEMPTS must be between 1 and 10')
        for name in (
            'ollama_read_timeout_s', 'ollama_total_timeout_s',
            'native_tool_read_timeout_s', 'native_tool_total_timeout_s',
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name.upper()} must be a finite positive number')
        if not math.isfinite(self.ollama_temperature) or not 0 <= self.ollama_temperature <= 2:
            raise ValueError('OLLAMA_TEMPERATURE must be between 0 and 2')
        if self.ollama_top_p is not None and (
            not math.isfinite(self.ollama_top_p) or not 0 < self.ollama_top_p <= 1
        ):
            raise ValueError('OLLAMA_TOP_P must be greater than 0 and at most 1')
        if self.ollama_top_k is not None and self.ollama_top_k < 1:
            raise ValueError('OLLAMA_TOP_K must be a positive integer or empty')
