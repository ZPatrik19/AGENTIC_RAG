"""Start Streamlit in the calling terminal and open the browser after readiness.

Separate the browser watcher from the server so launcher errors stay visible.
This script has no dependency other than Python's standard library.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PORT = 8501


def main(port: int = PORT) -> int:
    log_dir = ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    url = f'http://localhost:{port}'
    opener = [sys.executable, str(ROOT / 'scripts' / 'open_ui.py'), url]
    server = [
        sys.executable, '-m', 'streamlit', 'run',
        str(ROOT / 'src' / 'dap_assistant' / 'Chatbot.py'),
        '--server.address', '127.0.0.1',
        '--server.port', str(port),
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.fileWatcherType', 'none',
    ]
    try:
        with (log_dir / 'browser-launch.log').open('w', encoding='utf-8') as log:
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            subprocess.Popen(opener, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=flags)
        print('[INFO] Streamlit is starting. Server output follows below.', flush=True)
        return subprocess.call(server, cwd=ROOT)
    except OSError as exc:
        print(f'[ERROR] Could not start Streamlit: {exc}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('\n[INFO] Streamlit interrupted by user.')
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
