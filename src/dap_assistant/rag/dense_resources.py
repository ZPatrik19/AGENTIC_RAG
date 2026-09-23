"""Reference-count one embedded Qdrant client per process to avoid conflicting index locks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from dap_assistant.settings import Settings


@dataclass
class _Entry:
    client: object
    embedding_model: str
    references: int = 1


_LOCK = RLock()
_CLIENTS: dict[Path, _Entry] = {}


def acquire_dense(settings: Settings):
    """Lease the process-wide client for this index; caller must release it."""
    if settings.embedding_provider != 'sentence_transformers':
        return None
    path = (settings.data_dir / 'vectorstore' / 'qdrant').resolve()
    if not path.is_dir():
        return None
    with _LOCK:
        existing = _CLIENTS.get(path)
        if existing is not None:
            if existing.embedding_model != settings.embedding_model:
                raise RuntimeError(
                    'A Qdrant-indexet már más embeddingmodell használja. '
                    'Állítsd le a Streamlitet az embeddingmodell vagy az index módosítása előtt.'
                )
            existing.references += 1
            return existing.client
        # Lazy import: dummy/lexical-only tests do not need the heavy dependencies.
        from dap_assistant.rag.retrieval import LocalEmbeddings, LocalQdrant

        try:
            client = LocalQdrant(settings, LocalEmbeddings(settings))
        except RuntimeError as exc:
            # Embedded Qdrant owns an exclusive filesystem lock.  A second
            # process cannot share it even though clients within this process
            # use the reference-counted registry above.  Do not delete the
            # index or its lock file: the other process may still be writing.
            if 'already accessed by another instance of qdrant client' not in str(exc).casefold():
                raise
            raise RuntimeError(
                f'A helyi Qdrant-indexet egy másik folyamat használja: {path}. '
                'Állítsd le a Streamlit chatbotot és az értékelési oldalt, valamint '
                'az esetlegesen futó másik benchmarkot (a megfelelő terminálban Ctrl+C), '
                'majd futtasd újra a parancsot. Ha nem találod a folyamatot, '
                'PowerShellben listázd a Python-folyamatokat: '
                'Get-CimInstance Win32_Process | Where-Object { $_.Name -match '
                "'python|streamlit' } | Select-Object ProcessId,Name,CommandLine. "
                'Ne töröld a data/vectorstore/qdrant könyvtárat vagy a lock fájlt. '
                'Egyidejű, külön folyamatokból történő hozzáféréshez Qdrant szerver szükséges.'
            ) from exc
        _CLIENTS[path] = _Entry(client, settings.embedding_model)
        return client


def release_dense(client) -> None:
    """Release a lease; close the underlying Qdrant only after its final user."""
    if client is None:
        return
    with _LOCK:
        for path, entry in list(_CLIENTS.items()):
            if entry.client is client:
                entry.references -= 1
                if entry.references == 0:
                    del _CLIENTS[path]
                    entry.client.close()
                return
        raise ValueError('The Qdrant client was not acquired from the shared registry')
