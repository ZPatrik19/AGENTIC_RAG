# Agentic RAG evaluation report

- Run ID: `e6ffad0a-58da-4a50-b008-388cb275a656`
- Timestamp: 2026-09-22T20:41:45.135459+00:00
- Kind: functional_v4
- Model: `qwen3:4b`
- Context window: 8192

## Configuration

```json
{
  "scope": "subflow",
  "target": "rag/query_to_rerank",
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
    "WORK_001",
    "WORK_002",
    "WORK_003",
    "WORK_004",
    "WORK_005",
    "WORK_006",
    "WORK_007",
    "WORK_008",
    "WORK_009",
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
  "node_metric_schema_version": 2,
  "dataset_version": "4.0",
  "dataset_sha256": "a1f8ae0ac383fb1248250ff0c5df076a3f54ad43be15f88a9f0632a8edb177ff"
}
```

## Summary

```json
{
  "metrics": {
    "domain_accuracy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "intent_accuracy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "task_decomposition_accuracy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "retrieval_recall_at_5": {
      "value": 0.4362745098039216,
      "evaluated": 17,
      "total": 20
    },
    "retrieval_precision_at_5": {
      "value": 0.21176470588235294,
      "evaluated": 17,
      "total": 20
    },
    "retrieval_mrr": {
      "value": 0.700445632798574,
      "evaluated": 17,
      "total": 20
    },
    "source_recall_at_5": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "source_precision_at_5": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "context_recall": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "context_precision": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "context_coverage": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "citation_integrity_proxy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "semantic_review_coverage": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "citation_accuracy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "contradicted_claim_rate": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "citation_coverage": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "abstention_accuracy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "tool_selection_accuracy": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "subtask_coverage": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "tool_call_efficiency": {
      "value": null,
      "evaluated": 0,
      "total": 20
    },
    "workflow_success_rate": {
      "value": 1.0,
      "evaluated": 20,
      "total": 20
    }
  },
  "evaluated_cases": 20,
  "successful_runs": 20,
  "failed_runs": 0
}
```

## Methodology note

N/A metrics are never converted to zero. Retrieval/context scores on the automatic SILVER dataset are heuristic proxy estimates, NOT human-verified gold. Local LLM judge scores are fallible estimates.
