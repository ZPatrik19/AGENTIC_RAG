"""Conversation continuity and source coverage: offline, deterministic regressions."""
from __future__ import annotations

from dataclasses import replace

import pytest

from dap_assistant.conversation import is_routing_only, previous_turn_from_messages, resolve_followup
from dap_assistant.fast_path import classify_explicit
from dap_assistant.llm import source_answer
from dap_assistant.settings import Settings


SELLER_EVIDENCE = [
    {
        "evidence_id": "E_NOTICE", "domain": "vehicle", "role": "seller",
        "document_id": "fixture-seller", "chunk_id": "seller-notice",
        "text": (
            "Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.\n"
            "A bejelentést online, a Webes Ügysegéden vagy személyesen, "
            "bármelyik kormányablakban megteheted."
        ),
    },
    {
        "evidence_id": "E_INSURANCE", "domain": "vehicle", "role": "seller",
        "document_id": "fixture-seller", "chunk_id": "seller-insurance",
        "text": "Tulajdonosváltáskor a biztosítás megszűnik, de az eladást jelezned kell a biztosítód felé.",
    },
    {
        "evidence_id": "E_DOCS", "domain": "vehicle", "role": "seller",
        "document_id": "fixture-seller", "chunk_id": "seller-docs",
        "text": (
            "A személyes bejelentéshez keresd fel bármelyik kormányablakot, vagy foglalj időpontot.\n"
            "Szükséges okmányok: érvényes személyi igazolvány, lakcímkártya és adásvételi szerződés."
        ),
    },
]


def assistant(status: str, domains: list[str] | None = None, role: str = "seller") -> dict:
    return {"role": "assistant", "content": "response", "report": {"final": {
        "response_status": status,
        "domains": ["vehicle"] if domains is None else domains,
        "role": role, "stage": "after_event",
        "evidence": SELLER_EVIDENCE if status in ("complete", "partial") else [],
    }}}


def test_off_topic_turn_does_not_erase_previous_administrative_case():
    history = [
        {"role": "user", "content": "Eladtam az autómat. Milyen teendőim vannak?"},
        assistant("complete"),
        {"role": "user", "content": "Milyen idő van ma?"},
        assistant("unsupported", domains=[]),
    ]
    previous = previous_turn_from_messages(history)
    assert previous == {"domains": ["vehicle"], "role": "seller", "stage": "after_event"}
    followup = "Ezt melyik oldalon tudom bejelenteni, vagy a kormányablakban is lehet?"
    assert not is_routing_only(followup, previous)
    resolved = resolve_followup(followup, previous)
    assert "autó eladása" in resolved
    assert "munkahely" not in resolved
    # A new, explicitly named life event still overrides the preserved case.
    assert classify_explicit("Most elvesztettem a munkámat.").domains == ["employment"]


def test_context_does_not_survive_new_chat_or_indefinite_unrelated_turns():
    assert previous_turn_from_messages([]) is None
    history = [assistant("complete")]
    history.extend(assistant("unsupported", domains=[]) for _ in range(18))
    assert previous_turn_from_messages(history) is None
    assert previous_turn_from_messages([assistant("complete"), {"role": "assistant", "content": "error"}]) is None


def test_seller_general_answer_covers_deadline_office_documents_and_insurer():
    # Simulates quick Qwen selecting only the document list. Deterministic
    # source coverage must still recover essential obligations from other IDs.
    draft = source_answer("Eladtam az autómat, milyen teendőim vannak?", SELLER_EVIDENCE,
                          selected_ids=["E_DOCS"])
    categories = {claim.category for claim in draft.claims}
    assert {"deadline", "where", "documents", "insurance"} <= categories
    text = " ".join(claim.text for claim in draft.claims)
    assert "15 napon belül" in text
    assert "Webes Ügysegéden" in text
    assert "kormányablakban" in text
    assert "biztosítód" in text
    for claim in draft.claims:
        assert claim.supporting_quote in next(
            e["text"] for e in SELLER_EVIDENCE if e["evidence_id"] == claim.evidence_ids[0]
        )


def test_seller_location_followup_is_not_generic_contract_advice():
    draft = source_answer("Ezt melyik oldalon tudom bejelenteni, vagy a kormányablakban is?",
                          SELLER_EVIDENCE, selected_ids=["E_DOCS"])
    assert draft.claims
    assert all(claim.category == "where" for claim in draft.claims)
    assert any("Webes Ügysegéden" in claim.text for claim in draft.claims)
    assert any("kormányablak" in claim.text for claim in draft.claims)
    assert not any("Szükséges okmányok" in claim.text for claim in draft.claims)


def test_location_without_evidence_is_not_invented():
    draft = source_answer("Hol intézhetem ezt?", [
        {"evidence_id": "E_PREP", "text": "A gépjármű műszaki állapotát ellenőrizni kell."},
    ])
    assert draft.claims == []


def test_graph_recovers_after_off_topic_and_returns_cited_office(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    import json
    from dap_assistant import workflow

    path = tmp_path / "processed" / "vehicle" / "seller.json"
    path.parent.mkdir(parents=True)
    rows = [{**item, "source_url": "https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-adok-el",
             "title": "Synthetic seller fixture", "retrieved_at": "2026-09-20T00:00:00Z",
             "document_version": "fixture", "section_path": ["Bejelentés"], "page_number": None,
             "score": .02} for item in SELLER_EVIDENCE]
    path.write_text(json.dumps({"chunks": rows}, ensure_ascii=False), encoding="utf-8")

    class NoOllama:
        def classify(self, *_a, **_kw):
            raise AssertionError("Should use fast classification")

        def plan(self, *_a, **_kw):
            raise AssertionError("Should use fast planning")

        def answer(self, question, evidence, tools, **_kw):
            return source_answer(question, evidence, selected_ids=["E_DOCS"])

    monkeypatch.setattr(workflow, "get_llm", lambda *_a, **_kw: NoOllama())
    settings = replace(Settings(), data_dir=tmp_path, llm_provider="dummy",
                       embedding_provider="dummy", answer_mode="source")
    graph = workflow.build_workflow(settings)
    previous = previous_turn_from_messages([assistant("complete"), assistant("unsupported", domains=[])])
    result = graph.invoke(
        workflow.initial_state("Ezt melyik oldalon tudom bejelenteni, vagy a kormányablakban is?",
                               previous_turn=previous),
        config={"configurable": {"thread_id": "after-offtopic"}, "recursion_limit": 30},
    )
    assert result["domains"] == ["vehicle"]
    assert result["role"] == "seller"
    assert result["context_resolution"]["inherited"] is True
    assert "kormányablak" in result["final_answer"].casefold()
    assert "Webes Ügysegéd" in result["final_answer"]
    assert "Forrás" in result["final_answer"]
