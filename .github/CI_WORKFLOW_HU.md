# CitizenFlow AI – GitHub Actions CI

A `.github/workflows/ci.yml` **egyetlen workflow**, több `needs` függőséggel összekapcsolt jobbal. A GitHub Actions felületén a jobok gráfként jelennek meg.

```mermaid
flowchart TD
    trigger[Push main / Pull request / Manual] --> pre[00 Preflight]
    pre --> repo[01 Repository audit]
    pre --> lint[02 Ruff critical lint]
    pre --> tests[03 Offline pytest<br/>Python 3.12 / 3.13 / 3.14]
    pre --> security[05 Dependency security]
    repo --> graph[04 Real LangGraph + mocked Ollama]
    lint --> graph
    tests --> graph
    lint --> pkg[06 Python package build]
    tests --> pkg
    graph --> docker[07 Docker + Streamlit smoke<br/>manual opt-in]
    pkg --> docker
    security --> docker
    pre --> gate[08 Required CI gate]
    repo --> gate
    lint --> gate
    tests --> gate
    graph --> gate
    pkg --> gate
    security --> gate
    docker -. optional manual job .-> gate
```

## Mikor fut?

- `push` a `main` ágra: valamennyi **kötelező** ellenőrzés.
- `pull_request` a `main` ágra: ugyanazok az ellenőrzések, Docker-build nélkül.
- `workflow_dispatch`: manuális futtatás. A `run_docker` opció bekapcsolásával az image-build, a dummy CLI és a Streamlit egészségellenőrzése is lefut. A Docker-ág CPU-n nagy méretű PyTorch-függőségeket telepíthet, ezért nem indul minden PR-nál.

A `final-gate` csak akkor sikeres, ha **minden kötelező job sikeres**. Egy kötelező job `skipped`, `cancelled` vagy `failure` státusza nem számít sikernek. Az opcionális Docker csak akkor hagyható ki, ha nem kérték; kézi bekapcsolás esetén sikeresen le kell futnia. A branch protection szabálynál a **`08 · Required CI gate`** checket lehet kötelezővé tenni.

## Mi számít ténylegesen teszteltnek?

- `offline-tests`: a teljes `tests/` suite, Python 3.12 / 3.13 / 3.14 alatt. Nincs fizetős API, külső modell- vagy dokumentumletöltés. Egyes opcionális modulok tesztjei *skipped* állapotúak lehetnek, ha a hozzájuk tartozó `rag` extra (Qdrant, sentence-transformers) nincs telepítve. **Ez nem teljes Qdrant-integráció.**
- `graph-integration`: a valódi LangGraph fő gráf és RAG-algráf, szintetikus adatokkal, dummy LLM-mel és `httpx.MockTransport`-tal szimulált Ollama API-val. Nem igényel futó Ollamát.
- `repository-quality`: a projekt saját AST auditja; az importciklus, a presentation-határok és parse-hibák ellenőrzésére. Nem teljes statikus programanalízis.
- `code-quality`: Ruff **kritikus** `E4,E7,E9,F` szabályok; nem teljes formázás- vagy mypy-gate.
- `dependency-security`: ismert telepített csomag-sérülékenységek (`pip-audit`) és tracked credential fájlnevek. A fájlnév-vizsgálat **nem helyettesíti** a tartalomalapú titokkeresést.
- `docker-smoke`: image build, dummy CLI és `/_stcore/health` HTTP válasz; nem bizonyítja a RAG index vagy a helyi Qwen modell válaszminőségét.

A GitHub által hostolt runner Python-függőségekhez és sérülékenység-adatokhoz hálózatot használ, de a CI alatt **nem hív hivatalos közigazgatási forrást, Hugging Face modellt vagy Ollamát**. A `data/raw`, `data/processed`, `data/vectorstore` és az automatikusan generált SILVER referencia helyi tartalmak; a CI nem generál mesterséges helyettesítő hivatalos benchmark-eredményt.

## Használat

A workflow a projekt gyökeréhez képest `.github/workflows/ci.yml` helyen legyen. GitHub → Actions → *CitizenFlow AI · Quality → Graph → Package → Docker* → **Run workflow**; opcionálisan jelöld be a `run_docker` kapcsolót. Docker nélkül a többi ellenőrzés így is teljes értékű offline CI-t alkot.

Élesebb ellenőrzéshez külön, saját környezetben szükséges: `pip install -e '.[rag,dev]'`, a tényleges dokumentumkorpusz/index, az Ollama + Qwen3 4B futtatása, valamint valódi terhelésmérési riport. Ezeket nem állítja automatikusan elvégzettnek a CI.
