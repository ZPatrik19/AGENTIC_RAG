# Értékelés és referenciaadatok

A projekt külön értékeli a **funkcionális működést**, a **RAG-retrievalt/kontextust**, valamint a **terhelést**. Az értékelés nem a chatbot élő válaszának része. A mérőmotor a `src/dap_assistant/evaluation/` alatt van; a gyökérbeli `evaluation/` az adatkészleteket tárolja.

## Megőrzött adatállományok

| Fájl | Szerep |
|---|---|
| `golden_v4.json` | Az eredeti 10 AUTO + 10 WORK benchmarkkérdés és a várható információigények. |
| `golden_auto_v4.json` | Korpuszverzióhoz kötött, **gépi SILVER** chunkjelöltek; nem emberi gold. |

A csomagban szereplő SILVER JSON-nál **17/20** eset `automatic_proxy_pinned`; `AUTO_006`, `AUTO_009` és `AUTO_010` hiányos. Ez **belső gépi konzisztencia**, nem független szemantikai vagy jogi ellenőrzés. Az eredeti benchmark és a SILVER nem helyettesíthető egymással. A felhasználó által külön előállított `golden_reviewed_v4.json` emberi ellenőrzés nélkül nem tekinthető goldnak.

## Metrikák és értelmezés

- A Recall@5, Precision@5, MRR és a kontextusmutatók **csak érvényes, aktuális és az adott feladatra leképezett referencia** mellett számíthatók. A SILVER-alapú eredmény **proxy**.
- Több RAG-ág esetében a keresési rangsorokat részfeladatonként kell értékelni; az összefűzött, végső bizonyítéklista nem azonos a retriever top 5 találatával.
- A hivatkozás technikai érvényessége nem bizonyítja a válasz tartalmi helyességét. Az Answer Correctness, Answer Completeness és LLM Faithfulness nem standard benchmark-metrika emberileg ellenőrzött answer-quality referencia nélkül. Az opcionális lokális judge/semantic review eredménye csak kísérleti diagnosztika, nem human gold.
- TTFT, tokenhasználat és kontextuskihasználtság csak tényleges mérési adat esetén értelmezhető. Hiányzó megfigyelés **N/A**, nem nulla. A warm-up kérések nem számítanak bele a terhelési átlagokba.

## Futtatás

A virtuális környezet Pythonjával, a projekt gyökeréből:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_golden_review.py --force
.\.venv\Scripts\python.exe scripts\evaluate.py --scope full_workflow --topic all
.\.venv\Scripts\python.exe scripts\benchmark.py --count 50 --concurrency 1
.\.venv\Scripts\python.exe scripts\compare_rag_modes.py --topic all
```

A referencia-előkészítő hiányos SILVER esetén nem feltétlenül ad 0 kilépési kódot; olvasd el a **kérdésenkénti** hiánylistát. Forrás- vagy chunkváltozás után az aktuális korpuszhoz kell újragenerálni. A `reports/runs/` alatt mentett eredeti riportok nem számolódnak át automatikusan.

A `scripts/diagnostics/audit_corpus_and_golden.py` korpusz- és referenciadiagnosztika; `--with-dense` esetén valódi embedding/Qdrant szükséges. A `scripts/diagnostics/benchmark_answer_quality.py` rögzített bizonyítékokon segíthet két generálási futás összehasonlításában; az eredmény nem emberi válaszminőség-címke. A helyi index egyszerre több folyamatból történő megnyitását kerülni kell.

## Evaluation végrehajtási út

A projekt egyetlen aktív futtatási útvonalat használ: `professional.py` → `functional_metrics.py` / `load_testing.py` → `reporting.py`. A korábbi `runner.py` és az assisted-reference előkészítő réteg kivezetésre került.
