"""Thread-safe per-run monotonic spans; never collect prompt or personal text."""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import RLock
from time import perf_counter


@dataclass
class Span:
    name: str
    start_s: float
    end_s: float
    duration_s: float


class Telemetry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._spans: dict[str, list[Span]] = defaultdict(list)
        self._llm: dict[str, list[dict]] = defaultdict(list)
        self._llm_attempts: dict[str, list[dict]] = defaultdict(list)
        self._llm_successes: dict[str, list[dict]] = defaultdict(list)
        self._llm_failures: dict[str, list[dict]] = defaultdict(list)
        self._progress: dict[str, dict] = {}
        self._prompts: dict[str, list[dict]] = defaultdict(list)
        self._selection: dict[str, dict] = defaultdict(dict)
        self._native_tools: dict[str, dict] = {}

    @contextmanager
    def measure(self, run_id: str, component: str):
        start = perf_counter()
        try:
            yield
        finally:
            end = perf_counter()
            self.record(run_id, component, end - start, start=start, end=end)

    def record(self, run_id: str, component: str, duration: float, *, start: float | None = None, end: float | None = None) -> None:
        finish = perf_counter() if end is None else end
        begin = finish - duration if start is None else start
        with self._lock:
            self._spans[run_id].append(Span(component, begin, finish, duration))

    def llm_usage(self, run_id: str, payload: dict, *, phase: str = "unspecified") -> None:
        keys = ('model', 'total_duration', 'load_duration', 'prompt_eval_count',
                'prompt_eval_duration', 'eval_count', 'eval_duration', 'ttft_s',
                'requested_num_ctx', 'requested_num_predict', 'done_reason',
                'done', 'observed_content_bytes', 'observed_content_chars',
                'streamed_frames', 'usage_source', 'observed_thinking_chars')
        with self._lock:
            self._llm[run_id].append({**{k: payload.get(k) for k in keys}, "phase": phase})

    def llm_attempt(self, run_id: str, phase: str) -> None:
        """Record attempted model requests even when Ollama never returns usage."""
        with self._lock:
            self._llm_attempts[run_id].append({'phase': phase})

    def llm_success(self, run_id: str, phase: str) -> None:
        """A structured response counts as successful only after schema validation."""
        with self._lock:
            self._llm_successes[run_id].append({'phase': phase})

    def llm_failure(self, run_id: str, phase: str, kind: str, elapsed_s: float,
                    *, diagnostic: dict | None = None) -> None:
        """Technical failure metadata only; never store model output or source text."""
        with self._lock:
            self._llm_failures[run_id].append({
                'phase': phase, 'type': kind, 'elapsed_s': elapsed_s,
                **(diagnostic or {}),
            })

    def progress(self, run_id: str, phase: str, received_bytes: int, elapsed_s: float) -> None:
        """Technical streaming status; never store unvalidated model output."""
        with self._lock:
            self._progress[run_id] = {
                'phase': phase, 'received_bytes': received_bytes,
                'elapsed_s': round(elapsed_s, 2),
            }

    def prompt(self, run_id: str, phase: str, request: dict) -> None:
        """Ephemeral opt-in export from UI, not persisted in benchmark logs."""
        with self._lock:
            self._progress[run_id] = {'phase': phase, 'received_bytes': 0, 'elapsed_s': 0.0}
            self._prompts[run_id].append({
                'phase': phase, 'model': request.get('model'),
                'think': request.get('think'), 'options': dict(request.get('options', {})),
                'messages': [dict(m) for m in request.get('messages', [])],
                'schema': request.get('format'),
            })

    def native_tool_trace(self, run_id: str, trace: dict) -> None:
        """Persist names, bounded validated arguments and statuses; never source text."""
        with self._lock:
            self._native_tools[run_id] = {**trace, 'calls': [dict(c) for c in trace.get('calls', [])]}

    def selection(self, run_id: str, data: dict) -> None:
        """Store only evidence IDs / facet names; no document or personal text."""
        with self._lock:
            self._selection[run_id].update(data)

    def live(self, run_id: str) -> dict:
        with self._lock:
            return dict(self._progress.get(run_id, {}))

    def prompt_preview(self, run_id: str) -> list[dict]:
        with self._lock:
            return [dict(item) for item in self._prompts.get(run_id, [])]

    def snapshot(self, run_id: str) -> dict:
        with self._lock:
            spans = [asdict(s) for s in self._spans.get(run_id, [])]
            usage = [dict(u) for u in self._llm.get(run_id, [])]
            attempts = [dict(u) for u in self._llm_attempts.get(run_id, [])]
            successes = [dict(u) for u in self._llm_successes.get(run_id, [])]
            failures = [dict(u) for u in self._llm_failures.get(run_id, [])]
            selection = dict(self._selection.get(run_id, {}))
        return {'spans': spans, 'llm_usage': usage, 'llm_attempts': attempts,
                'llm_successes': successes, 'llm_failures': failures,
                'selection': selection,
                'native_tool_trace': dict(self._native_tools.get(run_id, {}))}


@dataclass
class ResourceSample:
    timestamp_s: float
    cpu_percent: float | None
    process_rss_bytes: int | None
    system_ram_percent: float | None
    gpu_percent: float | None
    vram_used_mib: float | None
    vram_total_mib: float | None


class ResourceSampler:
    """Best-effort CPU/RAM/GPU sampler for measured load-test windows only."""
    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval_s = max(0.2, interval_s)
        self.samples: list[ResourceSample] = []
        self._stop = None
        self._thread = None

    @staticmethod
    def _gpu() -> tuple[float | None, float | None, float | None]:
        import subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=1.5, check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                return None, None, None
            first = result.stdout.strip().splitlines()[0]
            gpu, used, total = [float(part.strip()) for part in first.split(',')[:3]]
            return gpu, used, total
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return None, None, None

    def start(self) -> None:
        from threading import Event, Thread
        from time import perf_counter
        try:
            import psutil
        except ImportError:  # pragma: no cover - dependency is in project runtime
            psutil = None
        self.samples = []
        self._stop = Event()
        process = psutil.Process() if psutil is not None else None
        if process is not None:
            process.cpu_percent(None)

        def loop() -> None:
            while not self._stop.is_set():
                cpu = rss = ram = None
                if process is not None:
                    try:
                        cpu = process.cpu_percent(None)
                        rss = process.memory_info().rss
                        ram = psutil.virtual_memory().percent
                    except (OSError, psutil.Error):
                        pass
                gpu, used, total = self._gpu()
                self.samples.append(ResourceSample(
                    timestamp_s=perf_counter(), cpu_percent=cpu, process_rss_bytes=rss,
                    system_ram_percent=ram, gpu_percent=gpu,
                    vram_used_mib=used, vram_total_mib=total,
                ))
                self._stop.wait(self.interval_s)

        self._thread = Thread(target=loop, name='evaluation-resource-sampler', daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        rows = [asdict(sample) for sample in self.samples]
        def avg(key):
            values = [row[key] for row in rows if row[key] is not None]
            return sum(values) / len(values) if values else None
        def peak(key):
            values = [row[key] for row in rows if row[key] is not None]
            return max(values) if values else None
        return {
            'samples': rows,
            'summary': {
                'sample_count': len(rows),
                'cpu_percent_mean': avg('cpu_percent'),
                'cpu_percent_peak': peak('cpu_percent'),
                'process_rss_peak_bytes': peak('process_rss_bytes'),
                'system_ram_percent_mean': avg('system_ram_percent'),
                'system_ram_percent_peak': peak('system_ram_percent'),
                'gpu_percent_mean': avg('gpu_percent'),
                'gpu_percent_peak': peak('gpu_percent'),
                'vram_used_mib_mean': avg('vram_used_mib'),
                'vram_used_mib_peak': peak('vram_used_mib'),
                'vram_total_mib': peak('vram_total_mib'),
            },
        }
