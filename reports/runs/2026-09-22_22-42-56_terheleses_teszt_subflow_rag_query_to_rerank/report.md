# Agentic RAG evaluation report

- Run ID: `5e71402e-831a-4f19-84bc-6dc2eccf6363`
- Timestamp: 2026-09-22T20:42:53.839029+00:00
- Kind: load_v4
- Model: `qwen3:4b`
- Context window: 8192

## Configuration

```json
{
  "scope": "subflow",
  "target": "rag/query_to_rerank",
  "topic": "all",
  "request_count": 50,
  "concurrency": 1,
  "timeout_s": 60.0,
  "seed": 42,
  "warmup": 1,
  "timeout_semantics": "latency_SLA_and_generation_timeouts_not_whole_workflow_cancellation",
  "model": "qwen3:4b",
  "context_window": 8192,
  "configured_context_window": 8192,
  "model_runtime": {
    "available": false,
    "configured_model": "qwen3:4b",
    "active_context_window": null,
    "reason": "retrieval_only_target"
  },
  "retrieval_configuration": {
    "embedding_provider": "sentence_transformers",
    "embedding_model": "intfloat/multilingual-e5-small",
    "embedding_device": "cpu",
    "lexical_retrieval": "BM25",
    "dense_top_k": 12,
    "bm25_top_k": 12,
    "fusion": "Reciprocal Rank Fusion k=60",
    "rag_candidate_limit": 16,
    "prepared_context_limit": 20,
    "facet_search_limit_per_query": 8
  },
  "dataset_version": "4.0",
  "dataset_sha256": "a1f8ae0ac383fb1248250ff0c5df076a3f54ad43be15f88a9f0632a8edb177ff",
  "question_distribution": {
    "AUTO_001": 4,
    "AUTO_002": 2,
    "AUTO_003": 4,
    "AUTO_004": 4,
    "AUTO_005": 2,
    "AUTO_006": 1,
    "AUTO_007": 3,
    "AUTO_008": 4,
    "AUTO_009": 4,
    "AUTO_010": 1,
    "WORK_001": 2,
    "WORK_002": 2,
    "WORK_003": 2,
    "WORK_004": 3,
    "WORK_005": 2,
    "WORK_006": 0,
    "WORK_007": 1,
    "WORK_008": 5,
    "WORK_009": 2,
    "WORK_010": 2
  }
}
```

## Summary

```json
{
  "count": 50,
  "mean_s": 0.18317451800161508,
  "min_s": 0.035233400005381554,
  "max_s": 0.38954229999217205,
  "p50_s": 0.20496350001485553,
  "p95_s": 0.3396548600067035,
  "p99_s": 0.3813923749979585,
  "throughput_qps": 5.44081351393028,
  "success_count": 50,
  "failure_count": 0,
  "error_rate": 0.0,
  "success_semantics": "workflow_returned_without_exception_not_answer_correctness",
  "response_status_counts": {
    "None": 0
  },
  "answer_fallback_count": 0,
  "generation_timeout_count": 0,
  "sla_exceeded_count": 0,
  "llm_failure_count": 0,
  "timeout_rate": 0.0,
  "component_stats": {
    "rag/process_query": {
      "invocations": 50,
      "mean_s": 8.958399936091154e-05,
      "p50_s": 8.045000140555203e-05,
      "p95_s": 0.0001664799943682737,
      "max_s": 0.00019409999367780983,
      "total_work_s": 0.004479199968045577
    },
    "bm25_retrieval": {
      "invocations": 220,
      "mean_s": 0.0023237922742158513,
      "p50_s": 0.0021630999981425703,
      "p95_s": 0.004273784988617988,
      "max_s": 0.00627020001411438,
      "total_work_s": 0.5112343003274873
    },
    "embedding_query": {
      "invocations": 220,
      "mean_s": 3.905454121360724e-06,
      "p50_s": 2.9999937396496534e-06,
      "p95_s": 1.072999439202248e-05,
      "max_s": 2.4699984351173043e-05,
      "total_work_s": 0.0008591999066993594
    },
    "dense_retrieval": {
      "invocations": 220,
      "mean_s": 0.005996040000305088,
      "p50_s": 0.004632049996871501,
      "p95_s": 0.008358244986447966,
      "max_s": 0.11165139998774976,
      "total_work_s": 1.3191288000671193
    },
    "retrieval_fusion": {
      "invocations": 220,
      "mean_s": 6.512909101068296e-05,
      "p50_s": 5.204998888075352e-05,
      "p95_s": 0.00014223000616766512,
      "max_s": 0.0002609000075608492,
      "total_work_s": 0.014328400022350252
    },
    "rag/hybrid_retrieval": {
      "invocations": 50,
      "mean_s": 0.17903179599961733,
      "p50_s": 0.20129899999301415,
      "p95_s": 0.3348133950014016,
      "max_s": 0.3861687000025995,
      "total_work_s": 8.951589799980866
    },
    "rag/rerank_results": {
      "invocations": 50,
      "mean_s": 0.0014295599999604746,
      "p50_s": 0.0011625500046648085,
      "p95_s": 0.002719124982831999,
      "max_s": 0.006002499983878806,
      "total_work_s": 0.07147799999802373
    },
    "fee_coverage_lookup": {
      "invocations": 3,
      "mean_s": 0.03363173333733963,
      "p50_s": 0.03256099999998696,
      "p95_s": 0.03737393000337761,
      "max_s": 0.03790870000375435,
      "total_work_s": 0.1008952000120189
    }
  },
  "warmup": [
    {
      "question_id": "AUTO_004",
      "latency_s": 0.03804489999311045,
      "success": true,
      "error": null
    }
  ],
  "resource_summary": {
    "sample_count": 15,
    "cpu_percent_mean": 93.72666666666667,
    "cpu_percent_peak": 105.1,
    "process_rss_peak_bytes": 1466331136,
    "system_ram_percent_mean": 82.24,
    "system_ram_percent_peak": 82.5,
    "gpu_percent_mean": 29.533333333333335,
    "gpu_percent_peak": 36.0,
    "vram_used_mib_mean": 227.93333333333334,
    "vram_used_mib_peak": 232.0,
    "vram_total_mib": 4096.0
  }
}
```

## Methodology note

N/A metrics are never converted to zero. Retrieval/context scores on the automatic SILVER dataset are heuristic proxy estimates, NOT human-verified gold. Local LLM judge scores are fallible estimates.
