# Rendszerarchitektúra – DÁP Élethelyzet-asszisztens

A projekt helyi, magyar nyelvű Agentic RAG prototípus. A forráskód jelenleg két élethelyzetet támogat: autóvásárlás/-eladás és munkahely elvesztése. A rendszer fő részei az offline dokumentum-előkészítés, a LangGraph-alapú online kérdésfeldolgozás, a dedikált RAG algráf, a Prompt/Context Engineering réteg, a válaszgenerálás, a feltételes eszközök és a külön értékelési rendszer.

A Streamlit **Teljes architektúra** oldal a `docs/architecture/full_workflow.svg` statikus rendszertervet mutatja. A konkrét kérés tényleges node-jai és futásidői a chatbot **Betekintés → Folyamat** nézetében láthatók.

## 1. Forráskód rétegei

```text
src/dap_assistant/
├── documents/             # források, letöltés, ingest, korpusz/index metaadatok
├── rag/                   # BM25+dense retrieval, RAG node-logika és dedikált StateGraph
├── orchestration/         # fő LangGraph állapot, node-logika és runtime assembly
├── prompt_engineering/    # végső prompt és prompt payload
├── context_engineering/   # kérdéselemzés, információigények, evidence/context budget
├── response/              # generálás, grounding, audit, fallback, diagnosztika
├── inference/             # Ollama HTTP/streaming, timeout, request/response I/O
├── tooling/               # kalkulátorok, tool routing, natív tool calling
├── observability/         # telemetria
├── evaluation/            # funkcionális/retrieval/load mérés
├── presentation/          # Streamlit megjelenítési segédlogika
├── pages/                 # Streamlit oldalak
├── workflow.py            # fő Agentic LangGraph topology/assembly
├── llm.py                 # stabil LLM facade
└── settings.py            # .env-alapú konfiguráció
```

A struktúra szándékosan külön emeli ki a **Prompt Engineering** és **Context Engineering** részeket, hogy a válaszminőség fejlesztésénél azonnal látható legyen, melyik réteghez kell nyúlni.

## 2. Offline dokumentumfolyamat

`config/document_sources.yaml` → `documents/download.py` → `documents/ingestion.py` → `data/processed/` → `rag/retrieval.py` → opcionális helyi Qdrant-index (`data/vectorstore/`).

- `documents/sources.py`: forrásmanifest és engedélyezett hostok.
- `documents/download.py`: HTML/PDF-letöltés és forrásmetaadatok.
- `documents/ingestion.py`: parse, tisztítás, chunkolás, stabil provenance.
- `documents/index_metadata.py`, `corpus_status.py`: korpusz- és indexállapot.
- `rag/retrieval.py`, `rag/graph_contract.py`, `dense_resources.py`: BM25, embedding, RRF és helyi Qdrant-erőforrás.

## 3. Fő Agentic LangGraph

A fő gráfot a `workflow.py:build_workflow()` építi, a node-ok implementációja az `orchestration/workflow_nodes.py` modulban található. Az állapotszerződés az `orchestration/state.py` modulban van.

| Node | Feladat |
|---|---|
| `classify_intent` | élethelyzet, szerep, kérési cél |
| `clarify_query` | kétértelmű kérés pontosítása |
| `plan_tasks` | részfeladatok és függőségek |
| `rag_worker` | részfeladatonként önálló RAG algráf |
| `evidence_gate` | ágak összevezetése, retry/stop döntés |
| `engineer_context` | bizonyítékok kontextussá szervezése |
| `execute_tools` | feltételes kalkulátor- és toolhívások |
| `generate_answer` | strukturált, forrásalapú modellválasz |
| `answer_audit` | állítás–forrás ellenőrzés, végső szöveg |
| `safe_response` | támogatáson kívüli/hiányos eset kezelése |

A részfeladatok `Send` objektumokkal önálló worker-futtatást kaphatnak. A köztes eredmények a LangGraph állapotban maradnak; az aktuális checkpointer `InMemorySaver`, ezért nem tartós adatbázis.

## 4. Dedikált RAG algráf

A `rag/rag_graph.py` csak a külön `StateGraph` topológiáját állítja össze; a node-logika a `rag/nodes.py`, az állapot pedig a `rag/state.py` modulban van. A retrieval node külön kezeli az alapkeresést, a doménspecifikus lefedettségi kereséseket és a facet/újrapróbálkozás összevezetését; a rangsorolási eredet (query label, BM25/dense rank) nem keveredik össze:

`process_query` → `hybrid_retrieval` → `rerank_results` → `evaluate_evidence` → `prepare_context`

Hiányos bizonyíték esetén az algráf a `MAX_RAG_ATTEMPTS` korlátig újrafogalmazhatja a keresést. A fő workflow a lefordított RAG gráfot a workerből hívja.

## 5. Prompt Engineering

A promptokhoz tartozó kód egy helyen található:

- `prompt_engineering/answer_prompt.py`: választerv és rendszerutasítás (`generation_instruction`).
- `prompt_engineering/prompt_payload.py`: a modellnek küldött strukturált prompt-kontextus és ellenőrzött tool-hintek.

Ha a modell **stílusán, részletességén, prioritásain vagy utasításain** akarsz változtatni, ezt a könyvtárat nézd először.

## 6. Context Engineering

A kontextus felépítése külön réteg:

- `context_engineering/question_analysis.py`: kérdés, szerep, élethelyzet és igények elemzése.
- `context_engineering/information_needs.py`: információigények/facetek felismerése.
- `context_engineering/evidence_selection.py`: releváns forrásrészletek kiválasztása és facet-lefedettség.
- `context_engineering/context_builder.py`: a gráf által átadott bizonyítékok rendezése a válaszhoz.
- `context_engineering/token_budget.py`: prompt- és kimeneti keret becslése, túlméretes payload visszavágása.

Ha a válasz **nem kap elég jó forrást, túl sok/kevés chunk kerül a promptba, vagy hiányzik egy információigény**, ezt a réteget vizsgáld.

## 7. Válaszgenerálás és grounding

- `response/generation.py`: bizonyítékválasztás és végső generálás vezérlése.
- `response/grounding.py`: állítások forráshoz kötése és kiegészítése.
- `response/audit.py`: végső állítás–forrás audit és Markdown-válasz összeállítása.
- `response/fallback.py`: forrásalapú tartalékválasz sikertelen modellkérésnél.
- `response/quality.py`, `filters.py`: minőségi és relevanciaszabályok.
- `response/diagnostics.py`: token- és generálási diagnosztika.

Az `inference/request.py`, `ollama_transport.py` és `response_io.py` kezeli a tényleges Ollama-kommunikációt. A `llm.py` stabil facade a workflow és az értékelés számára.

## 8. Tools, UI és értékelés

A `tooling/` tartalmazza a kalkulátorokat, a Python toolokat, a natív Ollama tool callingot és a tooleredmények válaszba integrálását. A natív tool-protokoll és a bounded HTTP/streaming transport külön modulban van (`native_tool_calling.py`, `native_tool_transport.py`). Ezek kérdésfüggően futnak; nem külön kalkulátorpanelként működnek.

A `presentation/` és `pages/` kizárólag megjelenítési felelősséget kap. A `src/dap_assistant/evaluation/` a mérőlogika, a gyökérbeli `evaluation/` a benchmark/referenciaadatok helye. Az automatikus SILVER referencia nem emberileg jóváhagyott gold; részletes szabályok: [Értékelés és benchmark](ERTEKELES_ES_BENCHMARK_HU.md).

## 9. Üzemeltetési határok

A konfigurációt a `settings.py` az `.env`-ből állítja elő. A helyi Qdrant fájlindexet ne nyissa meg egymástól független több folyamat egyszerre. Indítás, Docker és diagnosztika: [Üzemeltetés](UZEMELTETES_HU.md).

---

# Fejlesztői útmutató

## 10. Fejlesztői gyors térkép

| Ha ezt akarod javítani | Elsőként ezt nézd | Utána ezt |
|---|---|---|
| Rendszerprompt, válasz részletessége | `prompt_engineering/answer_prompt.py` | `response/generation.py` |
| Modellnek küldött JSON/payload | `prompt_engineering/prompt_payload.py` | `inference/request.py` |
| Kérdésből felismert információigények | `context_engineering/information_needs.py` | `question_analysis.py` |
| Rossz/hiányos evidence selection | `context_engineering/evidence_selection.py` | `rag/retrieval.py` |
| Túl nagy/kicsi prompt-kontextus | `context_engineering/context_builder.py` | `token_budget.py` |
| Hibás vagy részleges végső válasz | `response/audit.py` | `response/grounding.py`, `fallback.py` |
| Timeout / streaming / Ollama hiba | `inference/ollama_transport.py` | `response_io.py`, `.env` |
| Retrieval minőség | `rag/retrieval.py`, `rag/nodes.py` | `rag/rag_graph.py` |
| Részfeladatok/routing | `orchestration/workflow_nodes.py` | `workflow.py`, `orchestration/state.py` |
| Kalkulátor vagy tool | `tooling/` | `orchestration/workflow_nodes.py:execute_tools` |
| Streamlit megjelenítés | `presentation/`, `pages/` | `ui.py` |
| Recall/MRR/load test | `evaluation/` | `scripts/evaluate.py`, `benchmark.py` |

## 11. Clean Code döntések

- Egy könyvtár egy jól azonosítható felelősségi területet képvisel.
- A Prompt Engineering és Context Engineering külön top-level csomag, mert a válaszminőség fejlesztésekor ezek a leggyakrabban módosított AI-specifikus rétegek.
- A `response/` nem retrieval- vagy promptlogikát tartalmaz, hanem a generált kimenet kezelését.
- A `presentation/` nem tartalmaz üzleti vagy RAG-logikát.
- A `documents/` nem függ az értékelési rétegtől.
- A `llm.py` facade marad, hogy a workflow-nak ne kelljen ismernie az Ollama transport részleteit.

## 12. Bontási elvek

A graph topology, node-logika és transport külön rétegben marad. Fájlhossz önmagában nem indokol új absztrakciót; bontás akkor indokolt, ha külön felelősség, külön életciklus vagy önállóan tesztelhető szerződés jelenik meg.

## 13. Statikus projektellenőrzés

```powershell
.\.venv\Scripts\python.exe scripts\audit_project.py --root .
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

A `scripts/audit_project.py` importidőben látható függőségeket, parse-hibákat, nagy modulokat és pontos AST-egyezéseket jelez. Nem dead-code elemző, nem kódlefedettségi riport és nem bizonyítja a késői importok futásidejű biztonságát.

## 14. Referenciaadatok

A gyökérbeli `evaluation/` adatkészletek nem alkalmazáskód. A `golden_auto_v4.json` automatikus SILVER proxy; nem szabad emberileg ellenőrzött goldként bemutatni. Korpusz- vagy chunkverzió változásakor újra kell generálni, és a hiányos esetek referenciafüggő metrikái N/A-k maradnak.

---

# P0–P2 tisztítás és ellenőrzési határok

A kiindulópont a korábban javított `cleancoded_v12_fixed.zip`. Ez a jegyzék a **mostani körben ténylegesen ellenőrzött és módosított** részeket választja el a lokális infrastruktúrát igénylő feladatoktól.

| Auditpont | Megvalósítás / bizonyíték |
|---|---|
| P0 – `TaskRecord.status` | `pending/running/complete/partial/failed`; `test_cleanup_contracts.py` ellenőrzi az öt deklarált állapotot. |
| P0 – Docker és `.env.example` | Az összes közös alkalmazás-defaultot teszt ellenőrzi. A konténer Ollama-címe szándékosan `host.docker.internal`, nem `localhost`. |
| P0 – evaluation runner | A régi `runner.py` hiányzik; aktív útvonal: `professional.py`, `functional_metrics.py`, `load_testing.py`, `reporting.py`. |
| P1 – assisted reference | Az assisted kód és két assisted JSON nincs a csomagban. A SILVER 17/20, nem humán gold; három hiányos eset továbbra is hiányos. |
| P1 – történeti elnevezések | Verziószámos tesztfájlok és a fennmaradt `test_v10_*` függvénynevek helyett funkcionális nevek vannak. |
| P1 – modultisztítás | Prezentációs `runtime_view.py`, RAG `graph_contract.py`, kanonikus `observability/telemetry.py`; megszűnt az evaluation telemetria-shim. |
| P1 – `.env` átállás | A régi `ENV_CONFIG_VERSION`, `ANSWER_EVIDENCE_LIMIT` és `ANSWER_EXCERPT_CHARS` kulcsot a SETUP eltávolítja (az utóbbi két beállításnak nem volt runtime-fogyasztója). Csak kifejezetten régi verzióval jelölt fájl migrál alapértéket; az egyedi értékek nem íródnak át. Az eredeti mentés `.env.pre-setup.bak`. |
| P2 – gráfépítés | `workflow.build_workflow()` és `rag_graph.build_rag_graph()` csak topológiát állít össze, explicit node függőségekkel. |
| P2 – RAG retrieval | `RAGNodes.retrieve()` külön kezeli az alapkeresést, `_coverage_candidates()` a doménlefedettséget, `_collect_facets()` a faceteket/ismétlést. Külön tesztek ellenőrzik a rangsorolási eredetet és a retry állapot megőrzését. |
| P2 – natív tool | Az Ollama-szállítás külön modul, a modell által visszaadott hívások validálása és végrehajtása `_execute_model_calls()` helperben történik; a visszautasított hívásokról is teljes eszközüzenet készül. |
| P2 – docstringek | 43 több soros, ismétlődő leírás rövidült; az ellenőrzési szabályokat a kód és a regressziós tesztek megtartják. |

## Ellenőrzés

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe scripts\audit_project.py --root .
.\.venv\Scripts\python.exe -m ruff check src scripts tests
```

A tesztek jelentős része offline/mocked ellenőrzés. A kihagyott, opcionális LangGraph, Streamlit és Qdrant integrációkat, valamint a Windows/Docker/Ollama helyi indítást **külön** kell lefuttatni az alkalmazás saját runtime környezetében. Ebben a csomagban nincs új, valódi Qwen3-terhelési mérés vagy bizonyított bottleneck-következtetés. A Ruff futtatása attól függ, hogy a fejlesztői függőségek telepítve vannak-e.
