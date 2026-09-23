"""Shared, privacy-conscious execution environment fingerprint for evaluation.

Neither benchmark orchestrator owns this metadata schema. This module reads
only the configured local Ollama endpoints and the explicitly supplied corpus.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import platform
import subprocess

from .dataset import index_versions
from ..settings import Settings


def environment(settings: Settings, chunks: list[dict]) -> dict:
    """No full environment variables, network addresses, prompts or secrets in exported metadata."""
    import httpx
    try:
        import psutil
        cpu = psutil.cpu_count(logical=False)
        ram = psutil.virtual_memory().total
    except ImportError:
        cpu, ram = None, None
    try:
        gpu_info = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=2, check=False)
        gpu = gpu_info.stdout.strip() if gpu_info.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        gpu = None
    versions = index_versions(chunks)
    index_fingerprint = {'documents': sorted(versions.items()),
                         'chunks': sorted((c['chunk_id'], sha256(c['text'].encode('utf-8')).hexdigest()) for c in chunks),
                         'embedding_model': settings.embedding_model,
                         'embedding_provider': settings.embedding_provider}
    index_hash = sha256(json.dumps(index_fingerprint, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    model = {'name': settings.ollama_model, 'provider': settings.llm_provider, 'digest': None, 'ollama_version': None}
    if settings.llm_provider == 'ollama':
        try:
            with httpx.Client(timeout=3, trust_env=False) as client:
                base = settings.ollama_base_url.rstrip('/')
                data = client.get(base + '/api/tags').json()
                entry = next((m for m in data.get('models', []) if m.get('name') == settings.ollama_model), None)
                model['digest'] = entry.get('digest') if entry else None
                response = client.get(base + '/api/version')
                response.raise_for_status()
                model['ollama_version'] = response.json().get('version')
        except (httpx.HTTPError, ValueError, OSError):
            pass
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'os': platform.platform(),
            'cpu': platform.processor() or None, 'cpu_cores': cpu, 'ram_bytes': ram, 'gpu': gpu,
            'model': model, 'embedding_model': settings.embedding_model,
            'embedding_provider': settings.embedding_provider,
            'runtime_configuration': {
                'fast_routing': settings.fast_routing,
                'ollama_read_timeout_s': settings.ollama_read_timeout_s,
                'ollama_total_timeout_s': settings.ollama_total_timeout_s,
                'answer_mode': settings.answer_mode,
                'ollama_num_predict': settings.ollama_num_predict,
                'ollama_num_ctx': settings.ollama_num_ctx,
                'quick_single_pass': settings.quick_single_pass,
                'quick_requested_num_ctx': (min(settings.ollama_num_ctx, settings.ollama_quick_num_ctx)
                                             if settings.answer_mode == 'quick' and settings.quick_single_pass
                                             else None),
                'quick_requested_num_predict': (min(settings.ollama_answer_num_predict, settings.ollama_quick_num_predict)
                                                 if settings.answer_mode == 'quick' and settings.quick_single_pass
                                                 else None),
                'ollama_keep_alive': settings.ollama_keep_alive,
                'native_tool_calling_enabled': settings.native_tool_calling_enabled,
                'native_tool_call_limit': settings.native_tool_call_limit,
            },
            'index_sha256': index_hash, 'document_versions': versions,
            'document_count': len(versions), 'chunk_count': len(chunks)}
