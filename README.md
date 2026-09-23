# Agentic RAG Chatbot

Az AI egy magyar nyelvű Agentic RAG chatbot prototípus, amely hétköznapi közigazgatási élethelyzetekben segít eligazodni hivatalos források alapján.

A projekt Pythonban készült, a workflow-t LangGraph kezeli, a válaszokat helyi Ollama modell generálja, a felhasználói felület pedig Streamlit.

A projekt elsődleges célja nem egy általános chatbot létrehozása, hanem egy olyan reprodukálható Agentic RAG prototípus bemutatása, amely:

* több hivatalos forrást képes egy kérdéshez összekapcsolni;
* összetett kérdéseket részfeladatokra bont;
* retrieval és nem-retrieval eszközöket is képes használni;
* ellenőrizhető forrásokra támaszkodó választ generál;
* mérhető retrieval-, workflow- és teljesítménymetrikákat biztosít;
* helyi, fizetős API nélküli környezetben is futtatható.

---

## 1. Problémameghatározás és indoklás

### Miért releváns a probléma?

A magyarországi közigazgatási ügyintézés során az állampolgárok gyakran olyan élethelyzetekkel találkoznak, amelyek több egymáshoz kapcsolódó adminisztratív feladatot és hatósági eljárást igényelnek.

Az információ általában rendelkezésre áll, azonban több hivatalos weboldalon, dokumentumban és jogszabályban található meg.

Használt gépjármű vásárlásakor például külön kell tájékozódni:

* az adásvételi szerződésről;
* a kötelező gépjármű-felelősségbiztosításról;
* az eredetiségvizsgálatról;
* a tulajdonosváltozás bejegyzéséről;
* a kapcsolódó illetékekről.

Munkaviszony megszűnésekor pedig többek között:

* az álláskeresőként történő nyilvántartásba vétel;
* az álláskeresési járadék;
* az egészségügyi szolgáltatásra való jogosultság;
* egyes támogatási lehetőségek

válhatnak relevánssá.

A szükséges információ többek között a Digitális Állampolgárság Program, a NAV, a Nemzeti Jogszabálytár, a Nemzeti Foglalkoztatási Szolgálat, a NEAK és a MABISZ oldalain található.

**A megoldandó probléma tehát nem az információ hiánya, hanem annak töredezettsége, összetettsége és az adott élethelyzetre történő alkalmazás nehézsége.**

### Milyen felhasználói igényt elégít ki?

A rendszer azoknak az állampolgároknak készül, akik egy élethelyzethez kapcsolódó ügyintézésről gyorsan, közérthetően és ellenőrizhető hivatalos források alapján szeretnének tájékozódni.

A felhasználónak nem kell előre ismernie:

* az ügytípus pontos nevét;
* a releváns hatóságokat;
* a jogszabályokat;
* az ügyintézés sorrendjét.

Például elegendő természetes nyelven megfogalmaznia:

> Tegnap vettem egy használt autót. Mit kell most elintéznem?

A rendszer feladata a releváns részfeladatok azonosítása, a megfelelő bizonyítékok visszakeresése és egy követhető, forráshivatkozásokkal ellátott válasz összeállítása.

A chatbot tájékoztatást ad, nem végez hatósági ügyintézést, és nem helyettesít személyre szabott jogi tanácsadást.

---

## 2. Miért Agentic RAG?

Egy hagyományos RAG rendszer jól működhet egy szűk, pontosan megfogalmazott kérdésnél:

> Milyen dokumentumok szükségesek egy használt autó átírásához?

Egy teljes élethelyzet azonban több egymással összefüggő információigényt tartalmazhat:

* dokumentumok azonosítása;
* határidők ellenőrzése;
* különböző hivatalos források összevetése;
* számítások elvégzése;
* feltételek és kivételek azonosítása;
* a teendők megfelelő sorrendbe rendezése.

Egyetlen retrieval lépés nem feltétlenül biztosít elegendő bizonyítékot minden részfeladathoz.

A rendszer ezért LangGraph-alapú Agentic RAG megközelítést használ.

A workflow:

```text
User question
     ↓
Intent / domain recognition
     ↓
Task decomposition
     ↓
RAG worker(s)
     ↓
Tool execution
     ↓
Evidence aggregation
     ↓
Answer generation
     ↓
Answer / citation audit
```

A hozzáadott értéket nem önmagában az „agent” elnevezés adja, hanem:

* a részfeladatok kontrollált koordinációja;
* az explicit állapotkezelés;
* a feltételes routing;
* az eszközhasználat;
* a korlátozott újrakeresés;
* az evidence-alapú válaszgenerálás;
* az eredmény utólagos ellenőrzése.

Ennek ára a hagyományos RAG-hoz képest a nagyobb implementációs komplexitás, több potenciális modellhívás és magasabb latency.

---

Modellválasztás és alternatívák

A helyi generatív modell kiválasztásánál elsődleges szempont volt, hogy a modell fizetős API nélkül, Ollamán keresztül, korlátozott helyi CPU/GPU-erőforrás mellett is használható legyen. Emellett fontos volt a magyar nyelv támogatása, az instruction following, a strukturált válaszadás és az agentic workflow-khoz szükséges tool-calling képesség.

A hasonló méretű, lokálisan futtatható modellek között több reális alternatíva is elérhető. Ilyen például a Microsoft Phi-4-mini-instruct, amely szintén kis erőforrásigényű, MIT licencű modell, és hosszú kontextus kezelésére is alkalmas.

A projektben végül a Qwen3 4B került kiválasztásra, mert a prototípus követelményeihez több szempontból jól illeszkedik:

4 milliárd paraméteres méret: a nagyobb, 7B–14B vagy annál nagyobb modellekhez képest reálisabb helyi futtatást tesz lehetővé korlátozott RAM/VRAM mellett;
multilingual támogatás: a Qwen3 család 119 nyelvet és dialektust támogat, amelyek között a magyar is szerepel;
agentic képességek: a modellcsaládot tool calling és agent-alapú feladatok támogatására is optimalizálták, ami közvetlenül releváns a LangGraph workflow szempontjából;
thinking / non-thinking működés: ugyanaz a modell reasoning-orientált és gyorsabb, közvetlen válaszmódban is használható. Ebben a projektben a think=false konfiguráció csökkenti a generálási overheadet;
helyi futtatás: Ollamán keresztül egyszerűen integrálható a Python alkalmazásba, ezért nincs szükség külső fizetős inference API-ra;
Apache 2.0 licenc: a modell permisszív nyílt licenc alatt érhető el.

A választás ugyanakkor nem azt jelenti, hogy a Qwen3 4B bizonyítottan jobb minden hasonló méretű modellnél. A projektben nem készült teljes, azonos hardveren és azonos benchmark-adatkészleten végrehajtott LLM-összehasonlítás a Qwen3 4B és például a Phi-4-mini-instruct között. A modellválasztás ezért elsősorban mérnöki kompromisszum: a helyi futtathatóság, a magyar nyelvi lefedettség, az agentic képességek és az erőforrásigény egyensúlya alapján történt.

Egy későbbi A/B értékelésben érdemes ugyanazon kérdéskészleten összehasonlítani például:

Qwen3 4B
vs.
Phi-4-mini-instruct

és mérni:

Answer Completeness
Faithfulness
Citation Accuracy
Tool Selection Accuracy
TTFT
Generation Speed
RAM / VRAM usage

# 4. Teljesítmény és bottleneck-elemzés

A rendszer teljesítményét nem kizárólag az LLM inference határozza meg.

A teljes válaszidő több komponensből áll:

```text
Question
   ↓
Routing / planning
   ↓
Retrieval
   ↓
Reranking
   ↓
Evidence selection
   ↓
Optional tools
   ↓
LLM generation
   ↓
Audit
```

## Fő potenciális bottleneckek

### 1. LLM inference

A lokálisan futó Qwen3 4B várhatóan a teljes pipeline egyik legdrágább komponense.

Korlátozott GPU-memória esetén a modell részben CPU-n futhat, ami jelentősen növelheti a latency-t.

Vizsgálandó:

```text
TTFT
tokens / second
generation latency
VRAM utilization
CPU utilization
```

### 2. Többszörös LLM-hívás

Agentic workflow esetén egyetlen felhasználói kérdés több modellhívást is kiválthat:

```text
routing
+ planning
+ tool selection
+ answer generation
+ validation
```

Ez javíthatja az összetett feladatok kezelését, de egyszerű kérdéseknél felesleges latency-t okozhat.

Ezért létezik:

```text
FAST_ROUTING
QUICK_SINGLE_PASS
quick / detailed answer mode
```

### 3. Nagy context

Túl sok retrieval eredmény növeli:

* a prompt token számát;
* a context feldolgozási időt;
* a memóriaigényt;
* a releváns információ „felhígulásának” kockázatát.

Ezért szükséges:

```text
retrieval
   ↓
deduplication
   ↓
reranking
   ↓
evidence selection
   ↓
token budgeting
```

### 4. Retrieval és reranking

A hybrid retrieval több komponenst használ:

```text
BM25
+
Dense retrieval
+
RRF
+
Reranking
```

Ez jobb retrieval minőséget adhat, de több számítási lépést jelent.

Ezért külön érdemes mérni:

```text
retrieval latency
reranking latency
Recall@K
MRR
context coverage
```

### 5. Retry / re-search

Elégtelen evidence esetén a rendszer új retrieval kört indíthat.

Ez növelheti a coverage-et, de:

```text
több retrieval
+
több reranking
+
esetleg több LLM-hívás
=
magasabb latency
```

Ezért a retry-k száma korlátozott:

```dotenv
MAX_RAG_ATTEMPTS=2
```

## Fontos megkülönböztetés

A fenti pontok architekturálisan azonosított potenciális bottleneckek.

Ezeket nem szabad mért bottleneckként bemutatni addig, amíg konkrét benchmark eredmény nem támasztja alá őket.

A load test célja pontosan annak meghatározása, hogy a konkrét hardverkörnyezetben melyik komponens dominálja a teljes válaszidőt.

---

# 5. Értékelési módszertan

A projekt 20 kérdéses értékelő készletet tartalmaz.

A kiértékelés három szinten futtatható:

```text
1. Retrieval / RAG node
2. Teljes RAG subgraph
3. Teljes Agentic workflow
```

## Retrieval metrikák

A fő retrieval metrikák:

* Recall@5;
* Precision@5;
* MRR;
* Context Recall;
* Context Precision;
* Context Coverage.

A retrieval célja nem egyszerűen az, hogy magas similarity score keletkezzen, hanem hogy a végső válaszhoz szükséges bizonyítékok ténylegesen bekerüljenek a contextbe.

## Agentic workflow metrikák

Workflow szinten többek között:

* Tool Selection Accuracy;
* Subtask Coverage;
* Workflow Success Rate;
* retry rate;
* fallback rate

vizsgálható.

Ez különösen fontos, mert egy Agentic RAG rendszer hibája nem feltétlenül retrieval-hiba.

A probléma lehet például:

```text
rossz domain routing
rossz task decomposition
rossz tool selection
jó retrieval + rossz answer generation
jó answer + hibás citation
```

## Referenciaadat

A retrieval kiértékeléshez forrásverzióhoz kötött SILVER referencia is használható.

Ennek előnye:

* reprodukálható;
* automatikusan frissíthető;
* alkalmas retrieval konfigurációk összehasonlítására.

Korlátja:

* nem teljes értékű emberileg annotált golden dataset.

Ezért hosszabb távon manuálisan validált benchmark készlet kialakítása indokolt.

---

# 6. Terheléses teszt

A projekt 50–200 kéréses load teszt futtatását támogatja.

Mérhető:

### Latency

```text
mean
P50
P95
P99
min
max
```

### Throughput

```text
requests / second
completed requests
```

### Stabilitás

```text
error rate
timeout rate
retry rate
```

### Erőforrás

Ha elérhető:

```text
CPU
RAM
GPU
VRAM
```

### LLM

```text
TTFT
generation speed
prompt tokens
answer tokens
context utilization
```

A cél nem csupán annak megállapítása, hogy „gyors-e” a rendszer, hanem annak feltárása, hogy a latency mely komponensekből áll össze.

---

# 7. Konkrét optimalizálási irányok

## Context csökkentése

A végső LLM-hívás csak a szükséges, deduplikált és információigény szerint kiválasztott evidence-et kapja meg.

Ez csökkentheti:

```text
prompt tokens
KV-cache
memory usage
generation latency
```

## Redundáns LLM-hívások csökkentése

Egyszerű kérdéseknél a quick stratégia elkerülhet olyan planning vagy selection hívásokat, amelyek nem adnak lényegi hozzáadott értéket.

## Retrieval optimalizálás

A következő konfigurációk A/B tesztelhetők:

```text
BM25 only
Dense only
Hybrid
Hybrid + reranker
```

Mérendő:

```text
Recall@K
MRR
Context Coverage
Latency
```

## Chunking

Összehasonlítható:

```text
fixed-size
sentence based
section based
parent-child
overlap / no overlap
```

Nem csak retrieval pontosságot, hanem context méretet és latency-t is érdemes mérni.

---

# 8. Agentic architektúra

A rendszer két LangGraph szintből áll.

## Fő Agentic workflow

Topológia:

```text
src/dap_assistant/workflow.py
```

Node implementáció:

```text
src/dap_assistant/orchestration/workflow_nodes.py
```

Fő feladata:

```text
Question
   ↓
Domain / intent analysis
   ↓
Task planning
   ↓
RAG workers
   ↓
Result aggregation
   ↓
Tool execution
   ↓
Answer generation
   ↓
Audit
```

Az állapot LangGraph state-en keresztül halad a node-ok között.

---

## RAG subgraph

Topológia:

```text
src/dap_assistant/rag/rag_graph.py
```

Node implementáció:

```text
src/dap_assistant/rag/nodes.py
```

Pipeline:

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

Elégtelen evidence esetén korlátozott újrakeresés indítható.

---

# 9. Retrieval és Context Engineering

A retrieval hybrid megközelítést használ:

```text
BM25
+
dense embedding retrieval
+
rank fusion
+
reranking
+
deduplication
+
evidence selection
```

A modell nem automatikusan minden megtalált chunkot kap meg.

A Context Engineering külön réteg:

```text
src/dap_assistant/context_engineering/

├── question_analysis.py
├── information_needs.py
├── evidence_selection.py
├── context_builder.py
└── token_budget.py
```

A Prompt Engineering külön modul:

```text
src/dap_assistant/prompt_engineering/

├── answer_prompt.py
└── prompt_payload.py
```

Ez separation of concerns szempontból különválasztja:

```text
retrieval
context construction
prompt construction
generation
```

---

# 10. Tool-ok

A workflow nem kizárólag dokumentumkeresést használ.

Példák:

* gépjármű-vagyonszerzési illeték kalkulátor;
* határidő-számítás;
* dokumentum- és ügyintézési checklist.

A tool execution a LangGraph workflow része.

---

# 11. Adatforrások és dokumentumfeldolgozás

A rendszer hivatalos, nyilvánosan elérhető forrásokat használ, többek között:

* DÁP;
* Nemzeti Jogszabálytár;
* NAV;
* MABISZ;
* Nemzeti Foglalkoztatási Szolgálat;
* NEAK.

A dokumentumfeldolgozás:

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

A chunk metadata többek között tartalmazhat:

```text
source URL
document ID
domain
section
version
hash
```

---

# 12. Modellválasztás

A generatív modell:

```text
Ollama
└── qwen3:4b
```

Embedding modell:

```text
intfloat/multilingual-e5-small
```

A Qwen3 4B választás fő indoka a helyi futtathatóság és a fizetős API-k elkerülése.

A rendszer támogatja:

```dotenv
LLM_PROVIDER=ollama
```

valamint fejlesztési és CI célra:

```dotenv
LLM_PROVIDER=dummy
```

A dummy provider azonban nem használható a valódi LLM válaszminőségének bizonyítására.

---

# 13. Streamlit UI

A felület fő nézetei:

* chatbot;
* retrieved source-ok;
* LangGraph workflow;
* technikai modelladatok;
* evaluation;
* performance;
* architecture.

A Chatbot oldalon:

```text
quick
detailed
```

válaszstratégia választható.

---

# 14. Reprodukálhatóság

A projekt célja, hogy lokálisan és Dockerben is reprodukálhatóan elindítható legyen.

Fontos konfigurációk:

```text
.env.example
pyproject.toml
Dockerfile
docker-compose.yml
config/
```

A repository nem tartalmaz API-kulcsokat vagy lokális modelleket.

---

# 15. Projektstruktúra

```text
.
├── config/
├── data/
├── docs/
├── evaluation/
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
│       ├── llm.py
│       ├── settings.py
│       ├── ui.py
│       └── workflow.py
├── tests/
├── Dockerfile
├── docker-compose.yml
├── SETUP.bat
├── RUN.bat
├── setup.sh
└── run.sh
```

---

# 16. Lokális futtatás

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

A konfiguráció:

```text
.env.example
```

Példa:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:4b

EMBEDDING_MODEL=intfloat/multilingual-e5-small
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

# 17. Docker

A Docker image csak az alkalmazást, a RAG-komponenseket és a Streamlit UI-t tartalmazza.

Az Ollama a host gépen fut.

Architektúra:

```text
Windows host
│
├── Ollama
│   └── qwen3:4b
│
└── Docker
    └── Agentic RAG application
        ├── Streamlit
        ├── LangGraph
        ├── retrieval
        ├── embeddings
        └── QdrantLocal
```

## Docker workflow

```powershell
# Docker Compose konfiguráció ellenőrzése
docker compose config --quiet


# Docker image elkészítése
docker compose build assistant


# Elkészült image ellenőrzése
docker images dap-life-events-assistant


# Docker -> host Ollama kapcsolat ellenőrzése
docker compose run --rm --no-deps assistant python scripts/check_local_ollama.py


# Dokumentumok feldolgozása és Qdrant index létrehozása
docker compose run --rm --no-deps assistant python -m dap_assistant.cli rebuild


# Corpus és Qdrant index ellenőrzése
docker compose run --rm --no-deps assistant python -m dap_assistant.cli status --verify-qdrant --strict


# Streamlit alkalmazás indítása háttérben
docker compose up -d assistant


# Konténer állapotának ellenőrzése
docker compose ps


# Alkalmazás logjai
docker compose logs -f assistant
```

Leállítás:

```powershell
docker compose down
```

Streamlit:

```text
http://localhost:8501
```

---

# 18. Tesztelés

```bash
pytest
```

Ruff:

```bash
ruff check .
ruff format .
```

A CI és a tesztek alapértelmezés szerint nem igényelnek fizetős LLM API-t.

---

# 19. Dokumentáció

További részletes dokumentáció:

```text
docs/ARCHITEKTURA_ES_FEJLESZTES_HU.md
docs/ERTEKELES_ES_BENCHMARK_HU.md
docs/UZEMELTETES_HU.md
docs/architecture/
```

---

# 20. További fejlesztési irányok

A rendszer további iterációiban érdemes kontrollált A/B tesztekkel vizsgálni:

```text
embedding model
retrieval strategy
reranking
chunking
prompt strategy
context budget
LLM model
```

A változtatásokat ugyanazon benchmark kérdéshalmazon kell összehasonlítani.

A cél nem automatikusan a komplexebb pipeline, hanem annak mérése, hogy egy adott változtatás javítja-e:

```text
retrieval quality
answer coverage
faithfulness
latency
resource usage
```

A projekt célja végső soron egy olyan Agentic RAG architektúra bemutatása, amelyben a retrieval, az agentic működés, a válaszminőség és a teljesítmény külön-külön is mérhető és elemezhető.
