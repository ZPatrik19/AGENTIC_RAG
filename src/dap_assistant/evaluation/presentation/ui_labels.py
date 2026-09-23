"""Hungarian evaluation presentation; serialized metrics retain stable English keys."""
from __future__ import annotations

METRIC_LABELS = {
    'domain_accuracy': 'Élethelyzet-felismerés pontossága',
    'intent_accuracy': 'Szándékfelismerés pontossága',
    'task_decomposition_accuracy': 'Részfeladatokra bontás pontossága',
    'source_recall_at_5': 'Forrás-visszakeresés (Recall@5)',
    'retrieval_recall_at_5': 'Chunk-visszakeresés (Recall@5)',
    'retrieval_mrr': 'Első releváns találat (MRR)',
    'answer_correctness': 'Válasz helyessége (ellenőrzött referencia)',
    'faithfulness': 'Forráshűség (helyi LLM-bíráló)',
    'reference_fact_surface_coverage': 'Referenciaállítások szöveges lefedettsége',
    'faithfulness_quote_proxy': 'Idézetegyezési közelítő mutató',
    'citation_validity': 'Forráshivatkozások érvényessége',
    'tool_calling_accuracy': 'Eszközválasztás pontossága',
    'task_completion_rate': 'Részfeladatok teljesítési aránya',
    'request_facet_coverage_proxy': 'A kért témák strukturális lefedettsége (közelítő)',
}

COMPONENT_LABELS = {
    'llm_inference': 'LLM-következtetés',
    'embedding_query': 'Lekérdezés beágyazása',
    'dense_retrieval': 'Vektoros visszakeresés',
    'bm25_retrieval': 'Kulcsszavas visszakeresés',
    'retrieval_fusion': 'Vektoros és BM25 találatok összevonása',
    'fee_coverage_lookup': 'Díjakat tartalmazó forrásrészletek célzott keresése',
    'rag/rerank_results': 'RAG: találatok újrarangsorolása',
    'main/answer_audit': 'Fő gráf: válaszellenőrzés',
    'rag_subgraph': 'Teljes RAG algráf',
    'main/classify_intent': 'Fő gráf: élethelyzet felismerése',
    'main/plan_tasks': 'Fő gráf: részfeladatok tervezése',
    'main/rag_worker': 'Fő gráf: dokumentumkeresés',
    'main/evidence_gate': 'Fő gráf: bizonyítékok összesítése',
    'main/execute_tools': 'Fő gráf: eszközök',
    'main/generate_answer': 'Fő gráf: válaszgenerálás',
    'rag/process_query': 'RAG: kérdés előkészítése',
    'rag/hybrid_retrieval': 'RAG: hibrid keresés',
    'rag/evaluate_evidence': 'RAG: bizonyítékértékelés',
    'rag/prepare_context': 'RAG: kontextus előkészítése',
}

OPTIMIZATION_LABELS = {
    'llm_inference': 'Ha az LLM lassú, csökkentsd a felesleges modellhívásokat és a kontextus hosszát; mérj újra ugyanazokon a kérdéseken.',
    'embedding_query': 'Tartsd memóriában az embeddingmodellt, és vizsgáld meg a lekérdezések gyorsítótárazását.',
    'dense_retrieval': 'Vizsgáld meg a Qdrant metaadatszűrőit és az index méretét.',
    'bm25_retrieval': 'Használj tartós fordított indexet a teljes korpusz ismételt feldolgozása helyett.',
    'rag/rerank_results': 'Csökkentsd az újrarangsorolandó találatok számát; ellenőrizd a Recall@K változását.',
    'main/answer_audit': 'Csoportosítsd az ellenőrzéseket, és kerüld az ismételt modellhívásokat.',
}


def metric_label(key: str) -> str:
    return METRIC_LABELS.get(key, key)


def component_label(key: str) -> str:
    return COMPONENT_LABELS.get(key, key)


def error_message(exc: Exception) -> str:
    """User-facing explanation, without hiding the original technical exception."""
    message = str(exc)
    if 'already accessed by another instance of Qdrant client' in message or 'already accessed by another instance' in message:
        return ('A helyi Qdrant-indexet egy másik folyamat is használja. Állítsd le a külön futó '
                'indexelőt, benchmarkot vagy második Streamlit-példányt, majd indítsd újra az alkalmazást. '
                'A chat és az értékelés egy Streamlit-folyamaton belül közös klienst használ.')
    if 'Embedding model/index mismatch' in message or 'Embedding dimension mismatch' in message:
        return 'Az embeddingmodell nem egyezik az indexével. Állítsd le a Streamlitet, majd építsd újra az indexet.'
    if 'Ollama' in message or 'ollama' in message:
        return 'A helyi Ollama vagy a qwen3:4b modell nem elérhető. Ellenőrizd, hogy az Ollama fut-e, és a modell telepítve van-e.'
    if 'Processed official documents missing' in message or 'Dense index missing' in message:
        return 'Nincs kész dokumentumindex. Futtasd: .\\.venv\\Scripts\\python.exe scripts\\download_documents.py --index'
    if 'verziómetaadat' in message or 'index_meta.json' in message:
        return ('A korábbi Qdrant-index verziómetaadata hiányzik. Az első benchmark az aktív '
                'helyi embeddingmodellel újraindexeli a meglévő dokumentumokat, '
                'és helyreállítja a metaadatot. Ne indíts közben külön indexelőfolyamatot.')
    if 'Az index és a feldolgozott dokumentumok eltérnek' in message:
        return ('A feldolgozott dokumentumok vagy az embeddingmodell eltérnek az indextől. '
                'Állítsd le a Streamlitet, és építsd újra az indexet a CLI segítségével.')
    if 'dummy' in message.lower():
        return 'A valódi terheléses méréshez Ollama és valódi embeddingek szükségesek. A dummy mód csak funkcionális próbához használható.'
    return 'A mérés nem fejeződött be. A technikai részletek lent megtekinthetők.'


def optimization_for(component: str) -> str:
    return OPTIMIZATION_LABELS.get(component, f'Vizsgáld meg ezt a komponenst: {component}; mérj összehasonlítható körülmények között.')


def localized_row(row: dict) -> dict:
    """Readable table labels without changing stored/exported report schemas."""
    failure_causes = row.get('failure_causes') or []
    if isinstance(failure_causes, list):
        failure_causes = ', '.join(str(value) for value in failure_causes)
    result = {
        'Kérdésazonosító': row.get('question_id', ''),
        'Sikeres': row.get('success', False),
        'Válaszidő (s)': row.get('latency_s'),
        'Válasz állapota': row.get('response_status', ''),
        'Hiba': row.get('error') or '',
        'Lehetséges hibaok': failure_causes,
    }
    for key, value in row.get('metrics', {}).items():
        result[metric_label(key)] = value
    return result
