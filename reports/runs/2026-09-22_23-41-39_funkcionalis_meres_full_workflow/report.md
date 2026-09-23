# Agentic RAG evaluation report

- Run ID: `fa7340e1-dbf9-493b-89d9-ef76620069c0`
- Timestamp: 2026-09-22T21:41:36.661930+00:00
- Kind: functional_v4
- Model: `qwen3:4b`
- Context window: 8192

## Configuration

```json
{
  "scope": "full_workflow",
  "target": "agentic/full",
  "topic": "all",
  "question_ids": [
    "AUTO_001",
    "AUTO_002",
    "AUTO_003",
    "AUTO_004",
    "AUTO_005",
    "AUTO_006",
    "AUTO_007",
    "AUTO_008",
    "AUTO_009",
    "AUTO_010",
    "WORK_010"
  ],
  "model": "qwen3:4b",
  "context_window": 8192,
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
  "node_metric_schema_version": null,
  "dataset_version": "4.0",
  "dataset_sha256": "a1f8ae0ac383fb1248250ff0c5df076a3f54ad43be15f88a9f0632a8edb177ff"
}
```

## Summary

```json
{
  "metrics": {
    "domain_accuracy": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "intent_accuracy": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "task_decomposition_accuracy": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "retrieval_recall_at_5": {
      "value": 0.4895833333333333,
      "evaluated": 8,
      "total": 11
    },
    "retrieval_precision_at_5": {
      "value": 0.23750000000000002,
      "evaluated": 8,
      "total": 11
    },
    "retrieval_mrr": {
      "value": 0.6586805555555556,
      "evaluated": 8,
      "total": 11
    },
    "source_recall_at_5": {
      "value": 1.0,
      "evaluated": 8,
      "total": 11
    },
    "source_precision_at_5": {
      "value": 0.47500000000000003,
      "evaluated": 8,
      "total": 11
    },
    "context_recall": {
      "value": 0.5416666666666666,
      "evaluated": 8,
      "total": 11
    },
    "context_precision": {
      "value": 0.4974140535798626,
      "evaluated": 8,
      "total": 11
    },
    "context_coverage": {
      "value": 0.75,
      "evaluated": 8,
      "total": 11
    },
    "citation_integrity_proxy": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "semantic_review_coverage": {
      "value": 0.0,
      "evaluated": 11,
      "total": 11
    },
    "citation_accuracy": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "contradicted_claim_rate": {
      "value": null,
      "evaluated": 0,
      "total": 11
    },
    "citation_coverage": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "abstention_accuracy": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "tool_selection_accuracy": {
      "value": 0.42424242424242425,
      "evaluated": 11,
      "total": 11
    },
    "subtask_coverage": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    },
    "tool_call_efficiency": {
      "value": 0.31601731601731603,
      "evaluated": 11,
      "total": 11
    },
    "workflow_success_rate": {
      "value": 1.0,
      "evaluated": 11,
      "total": 11
    }
  },
  "evaluated_cases": 11,
  "successful_runs": 11,
  "failed_runs": 0
}
```

## Methodology note

N/A metrics are never converted to zero. Retrieval/context scores on the automatic SILVER dataset are heuristic proxy estimates, NOT human-verified gold. Local LLM judge scores are fallible estimates.
