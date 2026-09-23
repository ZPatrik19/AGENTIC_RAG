"""Read-only health check for the actual host Ollama (no model download / inference)."""
from __future__ import annotations

import json
import sys
from urllib.parse import urlparse

import httpx

from dap_assistant.settings import Settings


def inspect_ollama(settings: Settings, transport: httpx.BaseTransport | None = None) -> dict:
    base = settings.ollama_base_url.rstrip('/')
    parsed = urlparse(base)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        raise ValueError('OLLAMA_BASE_URL must be an absolute HTTP(S) address')
    with httpx.Client(transport=transport, timeout=5, trust_env=False) as client:
        tags = client.get(f'{base}/api/tags')
        tags.raise_for_status()
        running = client.get(f'{base}/api/ps')
        running.raise_for_status()
    model_names = [item.get('name') for item in tags.json().get('models', [])]
    active = next((item for item in running.json().get('models', [])
                   if item.get('name') == settings.ollama_model or item.get('model') == settings.ollama_model), None)
    result = {'ollama_online': True, 'model': settings.ollama_model,
              'model_installed': settings.ollama_model in model_names,
              'model_loaded': active is not None,
              'quick_requested_num_ctx': min(settings.ollama_num_ctx, settings.ollama_quick_num_ctx),
              'quick_requested_num_predict': min(settings.ollama_answer_num_predict,
                                                settings.ollama_quick_num_predict),
              'configured_num_ctx': settings.ollama_num_ctx,
              'quick_single_pass': settings.quick_single_pass}
    if active:
        for key in ('size', 'size_vram', 'context_length'):
            result[key] = active.get(key)
        size, vram = active.get('size'), active.get('size_vram')
        if isinstance(size, (int, float)) and size > 0 and isinstance(vram, (int, float)):
            result['reported_vram_fraction'] = round(vram / size, 3)
    return result


def main() -> int:
    try:
        report = inspect_ollama(Settings())
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        print(f'[ERROR] Ollama status unavailable: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['model_installed']:
        print('[WARN] Model is not installed; no automatic pull was performed.')
    elif not report['model_loaded']:
        print('[INFO] Model not currently loaded. First request may include a cold load.')
    elif report.get('reported_vram_fraction', 1) < 0.9:
        print('[INFO] Ollama reports partial GPU residency. CPU offloading may increase latency.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
