# Telepítés és üzemeltetés

## Lokális indítás

Windows 10 / PowerShell: `SETUP.bat`, majd `RUN.bat`. Linux/macOS: `bash setup.sh`, majd `bash run.sh`. A közös `scripts/start_streamlit.py` csak az elérhető Streamlit HTTP-felületnél nyit böngészőt. A projekt `pyproject.toml` fájlja Python **3.12–3.14** verziókat enged meg.

A SETUP a meglévő `.env`-et megőrzi; ha hiányzik, az `.env.example` mintából készül. Az indítók nem állítják át önkényesen az Ollama kontextusát, kimeneti tokenkeretét vagy időkorlátját. A shell környezeti változói elsőbbséget élvezhetnek a `.env`-hez képest; a Streamlitben megadott futási beállítások egyes értékeket felülírhatnak. A tényleges modellkérés paramétereit a **Betekintés → Tokenek és fejlesztői adatok** nézetben ellenőrizd.

A mellékelt `.env.example` kezdeti értékei: `LLM_PROVIDER=ollama`, `ANSWER_MODE=detailed`, `MAX_SUBTASKS=6`, `OLLAMA_NUM_CTX=8192`, `OLLAMA_ANSWER_NUM_PREDICT=900`, `OLLAMA_READ_TIMEOUT_S=180`, `OLLAMA_TOTAL_TIMEOUT_S=480`, `EMBEDDING_DEVICE=cpu`. Ezek **kért keretek**, nem igazolt GPU-elhelyezési vagy tényleges tokenszámok. Ellenőrzés: `ollama ps`; hibadiagnosztika: `python scripts/check_local_ollama.py`.

Az első teljes dokumentum/index előkészítéshez: `python scripts/download_documents.py --index`. A helyi, fájlrendszeres Qdrant-indexet egyszerre csak egy folyamat nyissa meg; futó Streamlit mellett ne indíts független, ugyanazt az indexet író benchmarkot. Index- vagy chunkváltozásnál a SILVER referenciát újra kell előállítani: `python scripts/prepare_golden_review.py --force`.

## Docker

```powershell
docker compose build
docker compose run --rm assistant python -m dap_assistant.cli index
docker compose up -d
```

A Docker saját indexkötetet használ; nem örökli automatikusan a Windowsos indexet. Az Ollama a hoston futhat, ilyenkor a Compose `OLLAMA_DOCKER_URL` beállítása a konténerből elérhető hostcímre mutat. A `docker compose down -v` törli a hozzá tartozó köteteket. A Docker build és a host–Ollama kapcsolat helyi ellenőrzést igényel.

## Tesztelés és ellenőrzés

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe scripts\audit_project.py --root .
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

A statikus audit az importidőben látható AST-függőségeket vizsgálja: **nem** dead-code-elemzés és **nem** bizonyítja, hogy nincsenek futás közben fellépő importciklusok. A kihagyott LangGraph-/Qdrant-/Streamlit-teszteket külön kell értékelni. A modell nélküli tesztek nem igazolják a Qwen válaszminőségét vagy a Docker futtatását. Titkos `.env`-et, lokális indexet és személyes logot ne tölts nyilvános repóba.


## `.env` és SETUP viselkedése

A `.env.example` az ajánlott helyi profilt tartalmazza. A `scripts/config/sync_env.py` a hiányzó kulcsokat hozzáadja, az ismert régi alapértékeket **csak egy kifejezetten korábbi verzióval jelölt** `.env` esetén migrálja, majd eltávolítja az elavult `ENV_CONFIG_VERSION` jelölőt és a nem használt `ANSWER_EVIDENCE_LIMIT` / `ANSWER_EXCERPT_CHARS` kulcsokat. Verziójelölő nélküli fájlban a 240 másodperces timeout is lehet tudatos felhasználói beállítás: azt nem írja át. Módosítás előtt `.env.pre-setup.bak` mentést készít; a következő SETUP már nem migrálja újra az egyszer feldolgozott állományt.

Ha Ollama ideiglenesen nem érhető el és Windows alatt dummy móddal folytatod a setupot, az csak az adott folyamatra érvényes; a setup többé nem írja át tartósan `LLM_PROVIDER=dummy` értékre a `.env` fájlt.

A Chatbot UI a `OLLAMA_READ_TIMEOUT_S` és `OLLAMA_TOTAL_TIMEOUT_S` értékeket kijelzi, de nem írja felül őket. Így az `.env` a timeoutok egyetlen runtime forrása.
