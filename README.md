#   – Agentic RAG Chatbot

A ** ** egy magyar nyelvű, LangGraph-alapú **Agentic RAG chatbot prototípus**, amely hétköznapi magyar közigazgatási élethelyzetekben segít eligazodni hivatalos, nyilvánosan elérhető források alapján.

A projekt Pythonban készült. A fő agentic workflow-t és a moduláris RAG algráfot LangGraph kezeli, a generatív modell lokálisan Ollamán keresztül fut, a felhasználói felület pedig Streamlit.

A projekt célja nem egy általános chatbot létrehozása, hanem egy reprodukálható AI Engineering prototípus bemutatása, amely:

* több hivatalos forrásból képes bizonyítékot visszakeresni;
* összetett kérdéseket részfeladatokra bont;
* explicit LangGraph state-et használ;
* conditional routing alapján dönt a következő lépésről;
* több RAG worker futását képes koordinálni;
* retrieval és nem-retrieval toolokat is használ;
* korlátozott retry és fallback logikát alkalmaz;
* evidence-alapú, forráshivatkozásokkal ellátott választ állít elő;
* külön méri a retrieval, az agentic workflow és a teljesítmény főbb komponenseit;
* fizetős LLM API nélkül, helyi környezetben is futtatható.

---

# 1. Problémameghatározás és cél

## Miért releváns a probléma?

A magyarországi közigazgatási ügyintézés során az állampolgárok gyakran olyan élethelyzetekkel találkoznak, amelyek több egymáshoz kapcsolódó adminisztratív feladatot és hatósági eljárást igényelnek.

Az információ általában elérhető, azonban több különböző hivatalos weboldalon, dokumentumban és jogszabályban található meg.

Használt gépjármű vásárlásakor például külön kell tájékozódni:

* az adásvételi szerződésről;
* a kötelező gépjármű-felelősségbiztosításról;
* az eredetiségvizsgálatról;
* a tulajdonosváltozás bejegyzéséről;
* a kapcsolódó illetékekről;
* az ügyintézés határidejéről.

Munkaviszony megszűnésekor többek között releváns lehet:

* az álláskeresőként történő nyilvántartásba vétel;
* az álláskeresési járadék;
* az egészségügyi szolgáltatásra való jogosultság;
* egyes támogatási lehetőségek;
* munkaviszony-megszüntetéshez kapcsolódó határidők és dokumentumok.

A szükséges információ többek között a következő forrásokból származik:

* Digitális Állampolgárság Program;
* NAV;
* Nemzeti Jogszabálytár;
* Nemzeti Foglalkoztatási Szolgálat;
* NEAK;
* MABISZ;
* Magyarország.hu.

**A megoldandó probléma tehát nem az információ teljes hiánya, hanem annak töredezettsége, összetettsége és az adott élethelyzetre történő alkalmazás nehézsége.**

## Milyen felhasználói igényt elégít ki?

A rendszer azoknak az állampolgároknak készült, akik egy élethelyzethez kapcsolódó ügyintézésről gyorsan, közérthetően és ellenőrizhető hivatalos források alapján szeretnének tájékozódni.

A felhasználónak nem kell előre ismernie:

* a hivatalos ügytípus pontos nevét;
* a releváns hatóságokat;
* a kapcsolódó jogszabályokat;
* az ügyintézés pontos sorrendjét.

Elegendő természetes nyelven megfogalmaznia például:

> Tegnap vettem egy használt autót. Mit kell most elintéznem?

A rendszer feladata:

```text
felhasználói kérdés
        ↓
élethelyzet felismerése
        ↓
részfeladatokra bontás
        ↓
hivatalos források keresése
        ↓
eszközök használata
        ↓
bizonyítékok összekapcsolása
        ↓
forrásolt válasz
```

A chatbot tájékoztató prototípus. Nem végez hatósági ügyintézést, és nem helyettesít személyre szabott jogi tanácsadást.

---

# 2. Miért Agentic RAG?

Egy hagyományos Retrieval-Augmented Generation rendszer megfelelő lehet egy szűk kérdés esetén:

> Milyen dokumentumok szükségesek egy használt autó átírásához?

Egy teljes élethelyzet azonban több különálló információigényt tartalmazhat.

Például szükség lehet:

* dokumentumok azonosítására;
* több hivatalos forrás keresésére;
* határidők értelmezésére;
* számítás elvégzésére;
* feltételek és kivételek ellenőrzésére;
* kapcsolódó ügyek felismerésére;
* a teendők megfelelő sorrendbe rendezésére.

Egyetlen retrieval lépés nem feltétlenül biztosít elegendő bizonyítékot minden részfeladathoz.

Ezért a rendszer **Agentic RAG** megközelítést használ.

A LangGraph workflow képes:

```text
recognize
    ↓
plan
    ↓
dispatch
    ↓
retrieve
    ↓
evaluate
    ↓
use tools
    ↓
generate
    ↓
audit
```

A hozzáadott értéket nem önmagában az „agent” elnevezés adja, hanem:

* az explicit state management;
* a részfeladatokra bontás;
* a conditional routing;
* a párhuzamos RAG worker dispatch;
* az eszközválasztás;
* a korlátozott újrakeresés;
* a context engineering;
* az answer audit;
* a kezelhető fallback.

Ennek ára a hagyományos RAG-hoz képest:

* nagyobb implementációs komplexitás;
* több végrehajtási lépés;
* potenciálisan több LLM-hívás;
* magasabb latency;
* komplexebb hibakeresés.

---

# 3. Agentic architektúra

A rendszer két egymástól elkülönített LangGraph szintet használ:

```text
Main Agentic Graph
        │
        └── RAG Subgraph
```

Ez separation of concerns szempontból elkülöníti:

```text
agentic orchestration
        ≠
retrieval implementation
```

## 3.1. Fő LangGraph workflow

A gráf topológiája:

```text
src/dap_assistant/workflow.py
```

A node-ok implementációja:

```text
src/dap_assistant/orchestration/workflow_nodes.py
```

A fő workflow **10 valós LangGraph node-ot** tartalmaz:

```text
1. classify_intent
2. clarify_query
3. plan_tasks
4. rag_worker
5. evidence_gate
6. engineer_context
7. execute_tools
8. generate_answer
9. answer_audit
10. safe_response
```

A fő végrehajtási út:

```text
START
  ↓
classify_intent
  ↓
plan_tasks
  ↓
rag_worker
  ↓
evidence_gate
  ↓
engineer_context
  ↓
execute_tools ─────┐
  ↓                │
generate_answer ◄──┘
  ↓
answer_audit
  ↓
END
```

Nem minden kérdés járja végig ugyanazt az útvonalat.

---

# 4. Conditional routing

A workflow több ponton dinamikusan választ következő node-ot.

## Intent routing

A `classify_intent` után:

```text
supported
    → plan_tasks

needs_clarification
    → clarify_query

unsupported
    → safe_response
```

## Task dispatch

A `plan_tasks` a végrehajtható részfeladatokat `Send` segítségével RAG workereknek adja át:

```text
plan_tasks
    ↓
rag_worker(task_1)

rag_worker(task_2)

rag_worker(task_n)
```

A taskok függőségei explicit módon kezeltek, és a planner DAG-validációt is végez.

## Evidence routing

Az `evidence_gate` ellenőrzi az egyes worker eredményeket.

Lehetséges kimenetek:

```text
evidence megfelelő
    → engineer_context

részfeladat újrapróbálható
    → plan_tasks

nincs használható evidence
    → safe_response
```

A retry korlátozott, így nem alakulhat ki korlátlan agent loop.

## Tool routing

Az `engineer_context` után csak akkor fut `execute_tools`, ha a kérdés ténylegesen igényel:

* checklistet;
* határidő-számítást;
* jármű-vagyonszerzési illetékszámítást;
* álláskeresési járadékhoz kapcsolódó számítást;
* hitelszámítást;
* opcionális natív model tool-callingot.

Ellenkező esetben:

```text
engineer_context
    → generate_answer
```

## Answer audit routing

```text
answer_audit
    ├── érvényes végső válasz → END
    └── nincs biztonságosan használható válasz → safe_response
```

---

# 5. Explicit state management

A köztes eredményeket egy explicit `AssistantState` kezeli:

```text
src/dap_assistant/orchestration/state.py
```

A state többek között a következő információkat tartalmazza:

```text
user_question
resolved_question
domains
intents
role
stage

subtasks
pending_task_ids
branch_results

evidence
context_evidence
sources

tool_results
native_tool_results

validation
answer_draft
answer_context

retry_count
execution_round

final_answer
response_status
errors
```

A RAG workerek eredményei `task_id` alapján kerülnek vissza a közös state-be.

A párhuzamos branch eredményekhez reducer tartozik, így az egymástól független worker eredmények kontrolláltan összevonhatók.

Ez különösen fontos agentic környezetben, mert a state teszi ellenőrizhetővé:

* melyik task futott;
* milyen evidence-et talált;
* volt-e retry;
* milyen tool futott;
* milyen válaszstratégia került alkalmazásra;
* történt-e fallback.

---

# 6. Dedikált RAG subgraph

A retrieval külön LangGraph subgraphként működik:

```text
src/dap_assistant/rag/rag_graph.py
```

A RAG subgraph szintén **5 valódi node-ot** tartalmaz:

```text
process_query
      ↓
hybrid_retrieval
      ↓
rerank_results
      ↓
evaluate_evidence
      ↓
prepare_context
```

Az `evaluate_evidence` conditional routingot használ.

Elégtelen evidence esetén korlátozott újrakeresés indítható.

A subgraph önállóan is tesztelhető és benchmarkolható, így a retrieval teljesítménye elkülöníthető a teljes agentic workflow viselkedésétől.

---

# 7. Retrieval és Context Engineering

A retrieval pipeline hybrid megközelítést használ:

```text
BM25
  +
dense retrieval
  ↓
Reciprocal Rank Fusion
  ↓
reranking
  ↓
deduplication
  ↓
evidence selection
```

Konfigurált retrieval komponensek:

```text
Lexical retrieval: BM25
Dense embeddings: intfloat/multilingual-e5-small
Vector store: QdrantLocal
Fusion: Reciprocal Rank Fusion, k=60
Dense top-k: 12
BM25 top-k: 12
Candidate limit: 16
```

Az RRF score rangfúziós pontszám, **nem relevanciavalószínűség**.

## Context Engineering

A modell nem automatikusan az összes megtalált chunkot kapja meg.

A Context Engineering külön modul:

```text
src/dap_assistant/context_engineering/

├── question_analysis.py
├── information_needs.py
├── evidence_selection.py
├── context_builder.py
└── token_budget.py
```

A pipeline:

```text
retrieved chunks
      ↓
deduplication
      ↓
information needs
      ↓
evidence selection
      ↓
section-aware expansion
      ↓
token budget
      ↓
generation context
```

A context csak az indexelt és verziózott evidence-ekből épül.

---

# 8. Toolok

A workflow nem kizárólag dokumentum-visszakeresést használ.

A projekt több determinisztikus, nem-retrieval toolt is tartalmaz.

Példák:

### Jármű-vagyonszerzési illeték kalkulátor

A tool csak akkor végez számítást, ha a szükséges NAV tarifa valódi, indexelt evidence-ként rendelkezésre áll.

### Határidő-számítás

A rendszer a visszakeresett dokumentumokban található időtartamokat használhatja határidő kiszámítására.

### Dokumentum-checklist

A retrieval evidence alapján összeállítható ügyintézési dokumentumlista.

### Álláskeresési járadékhoz kapcsolódó kalkuláció

A workflow külön eszközt használhat, ha a kérdés összeget vagy jogosultsági számítást igényel.

### Opcionális native tool calling

Ollama provider esetén a modell által kezdeményezett natív function calling is támogatott.

A determinisztikus helper toolok és a natív LLM tool-calling egymástól elkülönülnek.

---

# 9. Adatforrások és skálázható ingestion

A források deklaratívan szerepelnek:

```text
config/document_sources.yaml
```

Egy forrás többek között a következő metadata mezőket tartalmazhatja:

```text
id
title
url
domain
type
destination
required
role
topics
authority
priority
legal_sections
```

Ez lehetővé teszi új dokumentumok hozzáadását anélkül, hogy a retrieval vagy az agentic workflow logikáját át kellene írni.

A fő források:

* DÁP;
* NAV;
* Nemzeti Jogszabálytár;
* NFSZ;
* NEAK;
* MABISZ;
* Magyarország.hu.

A benchmarkolt corpus snapshot:

```text
24 dokumentum
212 chunk
```

## Dokumentumfeldolgozás

```text
Download
   ↓
Parse
   ↓
Clean
   ↓
Chunk
   ↓
Embed
   ↓
Index
```

## Idempotens letöltés

A downloader:

```text
src/dap_assistant/documents/download.py
```

A dokumentumokhoz SHA-256 hash készül.

A rendszer támogat:

* ETag alapú cache validationt;
* Last-Modified alapú cache validationt;
* SHA-256 tartalomellenőrzést;
* atomikus fájlírást;
* korlátozott retry-t;
* HTTP timeoutot;
* allowlistelt HTTPS hostokat;
* robots.txt ellenőrzést;
* letöltési audit logot.

Ha egy dokumentum tartalma nem változott, a meglévő változat újra felhasználható.

A chunk metadata többek között tartalmazza:

```text
document_id
document_version
source_url
domain
section
chunk_id
```

Ez lehetővé teszi az evaluation eredmények korpuszverzióhoz kötését.

---

# 10. Modellválasztás és trade-offok

## Generatív modell

```text
Ollama
└── qwen3:4b
```

## Embedding modell

```text
intfloat/multilingual-e5-small
```

A generatív modell kiválasztásánál elsődleges követelmény volt:

* fizetős API nélkül működjön;
* lokálisan futtatható legyen;
* korlátozott CPU/GPU környezetben is használható maradjon;
* támogassa a magyar nyelvet;
* megfelelő legyen instruction following feladatokra;
* használható legyen strukturált outputtal;
* illeszkedjen agentic/tool-calling feladatokhoz;
* Ollamán keresztül egyszerűen integrálható legyen.

## Alternatíva

Hasonló kategóriájú alternatíva például:

```text
Microsoft Phi-4-mini-instruct
```

A projektben végül Qwen3 4B került kiválasztásra, mert a prototípus szempontjából kedvező kompromisszumot jelentett:

```text
lokális futtatás
+
multilingual használat
+
agentic képességek
+
relatíve kis modellméret
+
Ollama integráció
```

A választás nem jelenti azt, hogy a Qwen3 4B minden hasonló modellnél bizonyítottan jobb.

A projektben nem készült teljes, azonos környezetű:

```text
Qwen3 4B
vs.
Phi-4-mini-instruct
```

benchmark.

Ezért a modellválasztás jelenleg **mérnöki trade-off**, nem univerzális teljesítményállítás.

Fejlesztési/CI célra dummy provider is használható:

```dotenv
LLM_PROVIDER=dummy
```

A dummy provider nem használható valódi LLM válaszminőség bizonyítására.

---

# 11. Modell- és rendszerkorlátok

## Modellkorlát

A Qwen3 4B relatíve kis modell.

Előnye:

* helyben futtatható;
* nincs fizetős inference API;
* kisebb hardverigényű a nagyobb modellekhez képest.

Korlátja:

* komplex magyar közigazgatási szöveg esetén előfordulhat értelmezési hiba;
* hosszú context növelheti az inference költségét;
* strukturált output validációt igényelhet;
* lokális futtatásnál timeout vagy részleges válasz előfordulhat;
* alacsony VRAM esetén CPU offloading növelheti a latency-t.

A workflow ezért source-only fallbackot is támogat arra az esetre, ha a generatív modell nem készít használható választ.

## Kontextuskorlát

Példa konfiguráció:

```dotenv
OLLAMA_NUM_CTX=8192
```

Ez konfigurációs célérték, nem garantálja, hogy minden hardveren optimális.

Nagyobb context esetén nőhet:

* a memóriaigény;
* a KV-cache;
* a prompt feldolgozási idő;
* a teljes inference latency.

Ezért külön evidence selection és token budget réteg működik.

## Adatforrás-korlát

A hivatalos weboldalak:

* változhatnak;
* elavulhatnak;
* struktúrát válthatnak;
* eltérő részletességűek lehetnek.

A dokumentum hash és verziókezelés a reprodukálhatóságot segíti, de önmagában nem bizonyítja, hogy egy szabály jelenleg is hatályos.

## Funkcionális scope

A jelenlegi prototípus két fő domainre fókuszál:

```text
vehicle
employment
```

Azaz:

* gépjármű vásárlás / eladás;
* munkaviszony megszűnése / álláskeresési ügyintézés.

Más élethelyzetek jelenleg nem tekinthetők teljesen támogatott use case-nek.

---

# 12. Streamlit UI

A felhasználói felület:

```text
Streamlit
```

Fő nézetek:

```text
Chatbot
Értékelés és teljesítmény
Teljes architektúra
```

A chatbot felület nem csak a végső választ jeleníti meg.

Megtekinthető többek között:

* az aktuális élethelyzet;
* a visszakeresett hivatalos dokumentumok;
* a retrieved evidence chunkok;
* az agentic workflow főbb lépései;
* a technikai tokenmérések;
* a modellkérések;
* a végső generation prompt;
* az Ollama futási adatok;
* a fallback állapot.

A workflow egy chat submit során egyszer indul el.

Streamlit rerun nem indít automatikusan új fizetős vagy lokális inference-hívást.

A QdrantLocal index és a workflow `cache_resource` használatával újrafelhasználható.

---

# 13. Értékelési módszertan

A projekt külön értékeli:

```text
1. retrieval / RAG node
2. RAG subflow
3. teljes Agentic workflow
4. load scenario
```

A benchmark dataset:

```text
evaluation/golden_v4.json
```

20 kérdést tartalmaz:

```text
10 vehicle
10 employment
```

A retrievalhez automatikusan előállított SILVER referencia is használható.

Fontos:

**a SILVER referencia nem ember által validált golden dataset.**

Ezért a retrieval metrikák proxy értékként értelmezendők.

---

# 14. Funkcionális értékelés – mért eredmények

## 14.1. RAG query-to-rerank subflow

Mérés:

```text
scope: subflow
target: rag/query_to_rerank
questions: 20
model configuration: qwen3:4b
context: 8192
```

A retrieval SILVER referencia 17 kérdésnél volt értékelhető.

| Metrika               | Eredmény |
| --------------------- | -------: |
| Successful executions |  20 / 20 |
| Retrieval Recall@5    |   0.4363 |
| Retrieval Precision@5 |   0.2118 |
| Retrieval MRR         |   0.7004 |

A teljes riport:

[`reports/2026-09-22_22-41-47_funkcionalis_meres_subflow_rag_query_to_rerank.md`](reports/2026-09-22_22-41-47_funkcionalis_meres_subflow_rag_query_to_rerank.md)

### Értelmezés

Az MRR alapján a releváns evidence gyakran előkelő helyen jelenik meg, ugyanakkor a Recall@5 azt mutatja, hogy az automatikus SILVER referencia alapján a releváns chunkok jelentős része nem mindig kerül be az első öt találat közé.

Ez indokolja:

* a hybrid retrieval további finomhangolását;
* a query/facet expansion vizsgálatát;
* a chunking A/B tesztelését;
* a reranking további optimalizálását.

---

## 14.2. Teljes Agentic workflow

A teljes workflow benchmark külön futásban **11 kérdést** értékelt.

Ez a futás önmagában megfelel a feladatban előírt 10–20 kérdéses mini evaluation tartománynak.

| Metrika                     | Eredmény |
| --------------------------- | -------: |
| Evaluated cases             |       11 |
| Successful executions       |  11 / 11 |
| Domain Accuracy             |    1.000 |
| Intent Accuracy             |    1.000 |
| Task Decomposition Accuracy |    1.000 |
| Retrieval Recall@5          |   0.4896 |
| Retrieval Precision@5       |   0.2375 |
| Retrieval MRR               |   0.6587 |
| Source Recall@5             |    1.000 |
| Source Precision@5          |   0.4750 |
| Context Recall              |   0.5417 |
| Context Precision           |   0.4974 |
| Context Coverage            |   0.7500 |
| Citation Integrity Proxy    |    1.000 |
| Citation Accuracy           |    1.000 |
| Citation Coverage           |    1.000 |
| Abstention Accuracy         |    1.000 |
| Tool Selection Accuracy     |   0.4242 |
| Subtask Coverage            |    1.000 |
| Tool Call Efficiency        |   0.3160 |
| Workflow Success Rate       |    1.000 |

A riport:

[`reports/runs/2026-09-22_23-41-39_funkcionalis_meres_full_workflow/report.md`](reports/runs/2026-09-22_23-41-39_funkcionalis_meres_full_workflow/report.md)

A részletes gépi eredmények:

[`reports/runs/2026-09-22_23-41-39_funkcionalis_meres_full_workflow/result.json`](reports/runs/2026-09-22_23-41-39_funkcionalis_meres_full_workflow/result.json)

### Következtetés

Az evaluation alapján az intent/domain felismerés és a task decomposition a vizsgált eseteken stabilan működött.

A retrieval és context metrikák ugyanakkor további optimalizációs lehetőséget mutatnak.

Különösen fejlesztendő terület:

```text
Tool Selection Accuracy = 0.4242
Tool Call Efficiency    = 0.3160
```

Ez arra utal, hogy az agentic workflow-ban az eszközválasztás és az indokolatlan vagy nem optimális tool-hívások kezelése fontosabb fejlesztési terület, mint maga a domain classification.

A `Workflow Success Rate = 1.0` nem értelmezhető automatikusan 100%-os válaszpontosságként.

Ebben a metrikában a siker azt jelenti, hogy a workflow technikailag végrehajtódott. Egy futás ettől még eredményezhet részleges vagy fallback választ.

Hasonlóan a citation metrikák technikai érvényessége sem jelent automatikusan emberileg validált jogi helyességet.

---

# 15. Terheléses teszt – mért eredmények

A load scenario a RAG retrieval/rerank subflow-ra futott.

Konfiguráció:

```text
scope: subflow
target: rag/query_to_rerank

request count: 50
concurrency: 1
warmup: 1
timeout: 60 s
```

Fontos:

**ez retrieval-only load scenario, nem teljes Qwen3 Agentic workflow load test.**

A generatív modell ebben a mérésben nem futott.

## Eredmények

| Metrika             |     Eredmény |
| ------------------- | -----------: |
| Requests            |           50 |
| Successful requests |           50 |
| Failed requests     |            0 |
| Error rate          |           0% |
| Timeout rate        |           0% |
| Mean latency        |     0.1832 s |
| P50                 |     0.2050 s |
| P95                 |     0.3397 s |
| P99                 |     0.3814 s |
| Maximum             |     0.3895 s |
| Throughput          | 5.44 query/s |

Teljes riport:

[`reports/2026-09-22_22-42-56_terheleses_teszt_subflow_rag_query_to_rerank.md`](reports/2026-09-22_22-42-56_terheleses_teszt_subflow_rag_query_to_rerank.md)

---

# 16. Mért bottleneck-elemzés

A load test komponensenkénti telemetry adatokat is tartalmaz.

Főbb átlagok:

| Komponens              |       Mean |
| ---------------------- | ---------: |
| `rag/process_query`    | ~0.00009 s |
| BM25 retrieval         | ~0.00232 s |
| Dense retrieval        | ~0.00600 s |
| `rag/hybrid_retrieval` | ~0.17903 s |
| `rag/rerank_results`   | ~0.00143 s |

A teljes query-to-rerank mérés átlagos ideje:

```text
~0.18317 s
```

A mérés alapján a vizsgált RAG subflow fő bottleneckje:

```text
hybrid retrieval
```

A `rag/hybrid_retrieval` a teljes mért subflow latency túlnyomó részét adta.

A futás során:

```text
220 BM25 retrieval invocation
220 dense retrieval invocation
```

történt 50 request mellett.

Ez azt mutatja, hogy nem kizárólag egyetlen vector lookup költsége fontos: a query/facet keresési stratégia több retrieval műveletet indíthat egy felhasználói kérdéshez.

## Fontos korlátozás

Ebből a mérésből **nem következik**, hogy a teljes Agentic RAG alkalmazás legnagyobb bottleneckje is a retrieval.

A load scenario nem tartalmazta a teljes:

```text
planning
+
tool calling
+
Qwen generation
+
answer audit
```

pipeline-t.

Ezért a korrekt következtetés:

> A mért `rag/query_to_rerank` subflow domináns komponense a hybrid retrieval volt.

A teljes agentic rendszer fő bottleneckjének meghatározásához külön full-workflow load scenario szükséges.

---

# 17. Konkrét optimalizálási irányok

## 17.1. Retrieval expansion csökkentése

A 50 request során végrehajtott nagyszámú retrieval invocation alapján érdemes vizsgálni:

```text
facet search count
query expansion
duplicate search paths
```

A cél:

```text
kevesebb retrieval call
+
változatlan vagy jobb Recall@5
```

## 17.2. Dense candidate pool optimalizálása

A következő konfigurációk azonos benchmarkon A/B tesztelhetők:

```text
dense_top_k
bm25_top_k
rag_candidate_limit
facet_search_limit_per_query
```

A cél nem egyszerűen a latency csökkentése, hanem a retrieval quality és latency együttes optimalizálása.

## 17.3. Tool selection javítása

A full workflow mérés egyik leggyengébb területe:

```text
Tool Selection Accuracy = 0.4242
Tool Call Efficiency    = 0.3160
```

Ezért érdemes:

* szigorúbb deterministic tool-routing szabályokat használni;
* a tool input feltételeket pontosítani;
* a tool szükségességét explicit információigényhez kötni;
* felesleges tool callokat elkerülni.

## 17.4. Context csökkentése

A végső LLM-hívás csak a szükséges evidence-et kapja.

Optimalizálható:

```text
evidence count
excerpt size
token budget
section expansion
```

Ez csökkentheti:

* a prompt token mennyiséget;
* a memóriaigényt;
* a generation latency-t.

---

# 18. Teljesítményértelmezés

A teljes rendszer latency-je több komponensből áll:

```text
routing
+
planning
+
retrieval
+
reranking
+
context engineering
+
tools
+
LLM generation
+
answer audit
```

A retrieval load test csak ennek egy részét vizsgálta.

A teljes workflow esetén lokális Qwen inference és esetleges CPU offloading lényegesen nagyobb válaszidőt okozhat.

Ezért a következő performance iteration indokolt:

```text
50 requests
full_workflow
Qwen3 4B enabled
```

és külön mérendő:

```text
TTFT
generation speed
prompt tokens
generated tokens
per-node latency
CPU
RAM
GPU
VRAM
fallback rate
timeout rate
```

---

# 19. Reprodukálhatóság

A projekt reprodukálhatóságát a következő elemek támogatják:

```text
.env.example
pyproject.toml
config/document_sources.yaml
Dockerfile
docker-compose.yml
SETUP.bat
RUN.bat
setup.sh
run.sh
GitHub Actions CI
versioned evaluation dataset
versioned benchmark reports
```

## Előfeltételek

Lokális futtatáshoz:

```text
Git
Python 3.12–3.14
Ollama
qwen3:4b
```

Dockeres futtatáshoz:

```text
Docker Desktop / Docker Engine
Docker Compose
hoston futó Ollama
qwen3:4b
```

A projekt nem tartalmazza:

* az Ollama modellek bináris fájljait;
* API-kulcsokat;
* `.env` secret fájlt;
* lokális vector store adatokat.

---

# 20. Lokális futtatás

## Ollama

Model letöltése:

```powershell
ollama pull qwen3:4b
```

Ollama indítása:

```powershell
ollama serve
```

## Windows

```powershell
.\SETUP.bat
.\RUN.bat
```

## Linux / macOS

```bash
chmod +x setup.sh run.sh
./setup.sh
./run.sh
```

Példa konfiguráció:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b

EMBEDDING_MODEL=intfloat/multilingual-e5-small
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_DEVICE=cpu

MAX_SUBTASKS=6
MAX_RAG_ATTEMPTS=2

FAST_ROUTING=true
ANSWER_MODE=detailed

OLLAMA_NUM_CTX=8192
OLLAMA_ANSWER_NUM_PREDICT=900
OLLAMA_READ_TIMEOUT_S=180
OLLAMA_TOTAL_TIMEOUT_S=480
```

---

# 21. Docker

A Docker image az alkalmazást és a lokális RAG komponenseket tartalmazza.

Az Ollama a host gépen fut.

A Qdrant ebben a projektben:

```text
QdrantLocal / embedded
```

tehát nincs külön Qdrant server container.

Architektúra:

```text
Windows / Linux host
│
├── Ollama
│   └── qwen3:4b
│
└── Docker
    └──  
        ├── Streamlit
        ├── LangGraph
        ├── RAG
        ├── sentence-transformers
        └── QdrantLocal
```

## Docker konfiguráció ellenőrzése

```powershell
docker compose config --quiet
```

## Image build

```powershell
docker compose build assistant
```

## Image ellenőrzése

```powershell
docker images dap-life-events-assistant
```

## Docker → Ollama kapcsolat

```powershell
docker compose run --rm --no-deps assistant python scripts/check_local_ollama.py
```

## Corpus és index újraépítése

```powershell
docker compose run --rm --no-deps assistant python -m dap_assistant.cli rebuild
```

## Index ellenőrzése

```powershell
docker compose run --rm --no-deps assistant python -m dap_assistant.cli status --verify-qdrant --strict
```

## Alkalmazás indítása

```powershell
docker compose up -d assistant
```

## Konténer állapot

```powershell
docker compose ps
```

## Logok

```powershell
docker compose logs -f assistant
```

## Leállítás

```powershell
docker compose down
```

Streamlit:

```text
http://localhost:8501
```

---

# 22. Tesztelés és CI

## Pytest

```powershell
python -m pytest -q
```

## Ruff

A CI kritikus lint gate:

```powershell
python -m ruff check --select E4,E7,E9,F src scripts tests
```

## Projekt audit

```powershell
python scripts/audit_project.py
```

A GitHub Actions pipeline többek között ellenőrzi:

```text
repository contract
architecture/import boundaries
Python compilation
Ruff critical lint
pytest
Python 3.12 / 3.13 / 3.14
LangGraph integration
dependency audit
package build
optional Docker smoke
```

A CI alapértelmezés szerint nem igényel fizetős LLM API-t.

---

# 23. Evaluation futtatása

A projekt gyökeréből:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_golden_review.py --force
```

Teljes workflow evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate.py --scope full_workflow --topic all
```

Load test:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark.py --count 50 --concurrency 1
```

RAG konfigurációk összehasonlítása:

```powershell
.\.venv\Scripts\python.exe scripts\compare_rag_modes.py --topic all
```

A generált riportok a:

```text
reports/
```

mappában találhatók.

---

# 24. Projektstruktúra

```text
.
├── .github/
│   └── workflows/
├── config/
│   └── document_sources.yaml
├── data/
├── docs/
├── evaluation/
│   └── golden_v4.json
├── reports/
│   └── runs/
├── scripts/
├── src/
│   └── dap_assistant/
│       ├── context_engineering/
│       ├── documents/
│       ├── evaluation/
│       ├── inference/
│       ├── orchestration/
│       ├── pages/
│       ├── presentation/
│       ├── prompt_engineering/
│       ├── rag/
│       ├── response/
│       ├── tooling/
│       ├── Chatbot.py
│       ├── llm.py
│       ├── settings.py
│       ├── ui.py
│       └── workflow.py
├── tests/
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── SETUP.bat
├── RUN.bat
├── setup.sh
└── run.sh
```

---

# 25. Fő tervezési döntések

A projekt főbb mérnöki döntései:

| Döntés                   | Indok                                                             |
| ------------------------ | ----------------------------------------------------------------- |
| LangGraph                | Explicit state, conditional routing és kontrollált agent workflow |
| Külön RAG subgraph       | Moduláris retrieval és külön benchmarkolhatóság                   |
| BM25 + dense retrieval   | Lexikális és szemantikus keresés kombinálása                      |
| RRF                      | Eltérő retrieval rangsorok stabil egyesítése                      |
| QdrantLocal              | Külső vector DB service nélkül futtatható prototípus              |
| Qwen3 4B + Ollama        | Helyi, fizetős API nélküli inference                              |
| Bounded retry            | Végtelen agent loop elkerülése                                    |
| Determinisztikus toolok  | Számítási feladatok kiszervezése az LLM-ből                       |
| Explicit source fallback | Modellhiba esetén ellenőrizhető evidence megőrzése                |
| Versioned corpus         | Reprodukálható evaluation                                         |
| Streamlit                | Gyors, átlátható prototípus UI                                    |
| Docker                   | Reprodukálható futási környezet                                   |

---

# 26. Jelenlegi következtetések

A projekt jelenlegi mérései alapján:

**1. Az agentic routing alapjai stabilak a vizsgált benchmarkon.**

A domain, intent és task decomposition metrikák a 11 kérdéses full-workflow futásban 1.0 értéket értek el.

**2. A retrieval működőképes, de nem tekinthető lezárt problémának.**

A SILVER-alapú Recall@5 körülbelül 0.44–0.49 között alakult a vizsgált futásokban, ezért további retrieval tuning indokolt.

**3. A source-level coverage erősebb, mint a chunk-level retrieval.**

A full-workflow mérés Source Recall@5 értéke 1.0 volt, miközben a chunk-szintű Recall@5 alacsonyabb.

Ez arra utal, hogy a rendszer gyakran megtalálja a megfelelő dokumentumot, de nem minden releváns chunk kerül optimálisan a top találatok közé.

**4. Az agentic eszközválasztás további fejlesztést igényel.**

A Tool Selection Accuracy és Tool Call Efficiency a jelenlegi full-workflow benchmark egyik leggyengébb része.

**5. A mért RAG subflow bottleneck a hybrid retrieval.**

A retrieval-only 50 requestes load testben a `rag/hybrid_retrieval` dominálta a teljes query-to-rerank időt.

**6. A teljes agentic rendszer bottleneckje még nem bizonyított.**

Ehhez külön teljes workflow load teszt szükséges valós Qwen3 inference-szel.

---

# 27. További fejlesztési irányok

A következő iterációkban indokolt:

```text
full-workflow 50 request load test
        ↓
per-node latency analysis

tool-selection A/B test
        ↓
routing precision javítása

retrieval configuration A/B
        ↓
Recall@5 / MRR / latency

chunking A/B
        ↓
context coverage

Qwen3 4B vs alternative local model
        ↓
answer quality / latency / resource usage
```

További fontos irány egy emberileg validált golden benchmark kialakítása.

Ez lehetővé tenné, hogy a jelenlegi SILVER proxy mellett megbízhatóbban mérhető legyen:

```text
Answer Correctness
Answer Completeness
Faithfulness
Citation Accuracy
```

A projekt célja végső soron nem a lehető legkomplexebb Agentic RAG pipeline létrehozása, hanem egy olyan rendszer kialakítása, amelynek:

```text
retrieval quality
agentic behavior
answer quality
latency
resource usage
```

külön-külön mérhető, reprodukálható és fejleszthető.
