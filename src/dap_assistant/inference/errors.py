"""Stable, side-effect-free errors shared by the transport and answer pipeline."""


class LLMError(RuntimeError):
    """A local model request could not produce an accepted result."""


class LLMGenerationLengthError(LLMError):
    """Ollama reported done_reason=length; partial JSON is never an answer."""
