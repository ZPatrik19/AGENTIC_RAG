"""Regression tests for price follow-ups. Fixtures are synthetic, not legal references."""
from __future__ import annotations

from dataclasses import replace
import json

import pytest

from dap_assistant.conversation import (
    detect_subtopic, is_routing_only, is_followup_for_case,
    previous_turn_from_messages, resolve_followup,
    is_price_question,
)
from dap_assistant.fast_path import classify_explicit
from dap_assistant.llm import source_answer
from dap_assistant.rag.retrieval import hybrid_search
from dap_assistant.settings import Settings


PRICE_EVIDENCE = {
    "evidence_id": "E_INSPECTION_PRICE",
    "document_id": "fixture-buyer",
    "chunk_id": "fixture-buyer-price",
    "source_url": "https://dap.gov.hu/eletesemenyek/autot-veszek-vagy-adok-el/autot-veszek",
    "domain": "vehicle", "role": "buyer",
    "section_path": ["Eredetiségvizsgálat"],
    "text": ("Az eredetiségvizsgálat díja 17 000 - 20 000 Ft. "
             "A díj a gépjármű típusától és hengerűrtartalmától függ."),
}
OTHER_FEE = {
    "evidence_id": "E_REGISTRATION", "document_id": "fixture-buyer",
    "chunk_id": "fixture-buyer-registration", "domain": "vehicle", "role": "buyer",
    "section_path": ["Okmányok"],
    "text": "A forgalmi engedély díja 6 000 Ft. A törzskönyv díja 6 000 Ft.",
}


def _assistant(status="complete", *, focus=""):
    return {"role": "assistant", "content": "Fixture", "report": {"final": {
        "response_status": status,
        "domains": ["vehicle"] if status == "complete" else [],
        "role": "buyer", "stage": "after_event", "focus": focus,
        "evidence": [PRICE_EVIDENCE] if status == "complete" else [],
    }}}


@pytest.mark.parametrize("question", [
    "Mennyi az összege az eredetvizsgának?",
    "Mennyibe kerül az eredetiségvizsgálat?",
    "Az eredetvizsga díja mennyi?",
    "Mennyibe kerül az átírás?",
    "Mennyibe kerül a törzskönyv?",
])
def test_fee_subtopics_are_in_scope_without_the_word_auto(question):
    result = classify_explicit(question)
    assert result is not None and result.domains == ["vehicle"]
    assert is_routing_only(question, None) is False
    assert is_price_question(question)


def test_off_topic_interruption_preserves_vehicle_case_and_price_followup():
    messages = [
        {"role": "user", "content": "Vásároltam egy autót"},
        _assistant(),
        {"role": "user", "content": "Milyen idő van ma?"},
        _assistant("unsupported"),
    ]
    previous = previous_turn_from_messages(messages)
    assert previous == {"domains": ["vehicle"], "role": "buyer", "stage": "after_event"}
    assert "eredetiségvizsgálata" in resolve_followup(
        "Mennyi az összege az eredetvizsgának?", previous,
    )


def test_supported_new_life_event_still_wins_over_old_vehicle_case():
    assert classify_explicit("Milyen dokumentum kell az álláskeresési járadékhoz?").domains == ["employment"]
    assert detect_subtopic("Mennyibe kerül az eredetvizsga?") == "vehicle_inspection"
    assert detect_subtopic("Milyen idő lesz holnap?") == ""


def test_verbatim_inspection_fee_beats_unrelated_registration_fee():
    result = source_answer("Mennyi az összege az eredetvizsgának?",
                           [OTHER_FEE, PRICE_EVIDENCE], selected_ids=["E_REGISTRATION"])
    assert len(result.claims) == 1
    claim = result.claims[0]
    assert claim.category == "cost"
    assert claim.evidence_ids == ["E_INSPECTION_PRICE"]
    assert "17 000 - 20 000 Ft" in claim.text
    assert claim.supporting_quote in PRICE_EVIDENCE["text"]


def test_absent_or_unrelated_fee_is_not_invented():
    assert not source_answer("Mennyi az eredetvizsga díja?", [OTHER_FEE]).claims
    assert not source_answer("Mennyi az eredetvizsga díja?", [
        {**PRICE_EVIDENCE, "text": "Az eredetiségvizsgálat díja az autó típusától függ."},
    ]).claims


def test_recent_supported_subtopic_retained_as_structured_focus():
    previous = previous_turn_from_messages([_assistant(focus="vehicle_inspection")])
    assert previous["focus"] == "vehicle_inspection"
    assert "eredetiségvizsgálata" in resolve_followup("És az ára?", previous)
    assert is_followup_for_case("És az ára?", previous)
    assert is_followup_for_case("Mennyibe kerül?", previous)
    assert not is_routing_only("És az ára?", previous)
    assert not is_followup_for_case("Mennyi a világ népessége?", previous)


def test_registration_fee_only_returns_requested_document_and_handles_no_period():
    registration = {"evidence_id": "E_REG", "text":
                    "Forgalmi engedély díja: 6 000 Ft\nTörzskönyv díja: 6 000 Ft"}
    draft = source_answer("Mennyibe kerül a törzskönyv?", [registration])
    assert [item.text for item in draft.claims] == ["Törzskönyv díja: 6 000 Ft"]


def test_time_question_does_not_trigger_fee_extraction():
    assert not is_price_question("Mennyi időm van az átírásra?")


def test_existing_processed_chunks_support_fee_search_without_download(tmp_path):
    path = tmp_path / "processed" / "vehicle" / "fixture-buyer.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"chunks": [
        {**PRICE_EVIDENCE, "role": "buyer"},
        {**OTHER_FEE, "role": "buyer"},
        {**OTHER_FEE, "chunk_id": "seller-irrelevant", "role": "seller",
         "text": "Az autó eladása után a jármű biztosítását be kell jelenteni."},
    ]}, ensure_ascii=False), encoding="utf-8")
    settings = replace(Settings(), data_dir=tmp_path, embedding_provider="dummy")
    hits = hybrid_search("gépjármű eredetiségvizsgálata díja Ft", "vehicle",
                         settings, dense=None, role="buyer", limit=5)
    assert hits and hits[0]["chunk_id"] == PRICE_EVIDENCE["chunk_id"]
    assert all(hit["role"] == "buyer" for hit in hits)


def test_graph_fee_answer_survives_unrelated_question_without_ollama(tmp_path, monkeypatch):
    pytest.importorskip("langgraph")
    from dap_assistant import workflow

    path = tmp_path / "processed" / "vehicle" / "dap-vehicle-buyer.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"chunks": [
        {**PRICE_EVIDENCE, "title": "Synthetic fixture", "retrieved_at": "2026-09-20T00:00:00Z",
         "document_version": "fixture", "page_number": None, "score": .02},
        {**OTHER_FEE, "source_url": PRICE_EVIDENCE["source_url"], "title": "Synthetic fixture",
         "retrieved_at": "2026-09-20T00:00:00Z", "document_version": "fixture",
         "page_number": None, "score": .02},
    ]}, ensure_ascii=False), encoding="utf-8")

    class NoOllama:
        def classify(self, *_a, **_kw):
            raise AssertionError("Explicit subtopic should not call Qwen")

        def plan(self, *_a, **_kw):
            raise AssertionError("Fast planning should not call Qwen")

        def answer(self, *_a, **_kw):
            raise AssertionError("Verbatim fee should bypass Qwen")

    monkeypatch.setattr(workflow, "get_llm", lambda *_a, **_kw: NoOllama())
    settings = replace(Settings(), data_dir=tmp_path, llm_provider="dummy",
                       embedding_provider="dummy", answer_mode="quick")
    graph = workflow.build_workflow(settings)
    previous = previous_turn_from_messages([_assistant(), _assistant("unsupported")])
    result = graph.invoke(
        workflow.initial_state("Mennyi az összege az eredetvizsgának?", previous_turn=previous),
        config={"configurable": {"thread_id": "fee-followup"}, "recursion_limit": 30},
    )
    assert result["domains"] == ["vehicle"]
    assert result["role"] == "buyer"
    assert result["focus"] == "vehicle_inspection"
    assert result["answer_strategy"] == "verified_fee_extract"
    assert "17 000 - 20 000 Ft" in result["final_answer"]
    assert "6 000 Ft" not in result["final_answer"]
    assert result["answer_validation"]["status"] == "passed"
