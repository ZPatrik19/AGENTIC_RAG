"""Wait for Streamlit health before opening a browser; no fixed startup delay."""
import sys
import time
import urllib.request
import webbrowser

url = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8501'
for _ in range(80):
    try:
        with urllib.request.urlopen(url + '/_stcore/health', timeout=1) as response:
            if response.status == 200:
                webbrowser.open(url)
                print('UI ready:', url)
                raise SystemExit(0)
    except (OSError, TimeoutError):
        time.sleep(0.5)
print('Streamlit did not report healthy startup. Check its terminal output.', file=sys.stderr)
raise SystemExit(1)
