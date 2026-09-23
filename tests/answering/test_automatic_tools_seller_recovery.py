"""Offline regression tests for automatic tool routing and seller fallback."""
from __future__ import annotations

from dap_assistant.response.filters import relevant_unit
from dap_assistant.llm import source_answer
from dap_assistant.presentation.runtime_view import progress_milestones, grouped_progress_milestones
from dap_assistant.tooling.tools import wants_native_tool_call, prepare_unemployment_benefit_calculation
from dap_assistant.tooling.calculators import parse_vehicle_duty_request


SELLER_Q = "Eladtam az autómat mit kell tennem?"


def evidence(evidence_id: str, text: str, source: str) -> dict:
    return {"evidence_id": evidence_id, "chunk_id": evidence_id,
            "domain": "vehicle", "role": "seller", "document_id": evidence_id,
            "source_url": source, "text": text}


def test_seller_generic_question_does_not_trigger_expensive_native_tool():
    assert not wants_native_tool_call(SELLER_Q)
    assert wants_native_tool_call("Milyen iratokat kérnek a bejelentéshez?")
    assert wants_native_tool_call("Mikor jár le a határidő?")
    assert not wants_native_tool_call("Elvesztettem a munkámat, mi a teendőm?")


def test_irrelevant_insurer_and_regulation_text_not_in_seller_source_answer():
    snippets = [
        evidence("E_dap_deadline", "Az eladás után 15 napon belül be kell jelentened a tulajdonosváltást.",
                 "https://dap.gov.hu/seller"),
        evidence("E_dap_channel", "A bejelentést online, a Webes Ügysegéden vagy személyesen, bármelyik kormányablakban megteheted.",
                 "https://dap.gov.hu/seller"),
        evidence("E_dap_insurer", "Az eladást jelezned kell a biztosítód felé.",
                 "https://dap.gov.hu/seller"),
        evidence("E_njt_offer", "A biztosító az ajánlat elutasításáról annak beérkezésétől számított tizenöt napon belül értesíti az ajánlattevő üzemben tartót.",
                 "https://njt.jog.gov.hu/example"),
        evidence("E_njt_scope", "Ezt a rendeletet a Magyarország joghatósága alól mentességet élvező személyekre és szervezetekre is alkalmazni kell.",
                 "https://njt.jog.gov.hu/example"),
    ]
    draft = source_answer(SELLER_Q, snippets, role="seller", stage="after_event")
    body = " ".join(claim.text for claim in draft.claims)
    assert "tulajdonosváltást" in body
    assert "Webes Ügysegéden" in body
    assert "biztosítód" in body
    assert "ajánlat elutasításáról" not in body
    assert "joghatósága alól" not in body
    assert all(c.supporting_quote in next(e["text"] for e in snippets if e["evidence_id"] == c.evidence_ids[0])
               for c in draft.claims)


def test_requested_legal_edge_case_is_not_globally_censored():
    statement = "A biztosító az ajánlat elutasításáról az ajánlattevő üzemben tartót értesíti."
    assert not relevant_unit(statement, SELLER_Q, "vehicle", "seller", "after_event")
    assert relevant_unit(statement, "Eladtam az autóm, milyen biztosítási ajánlatot utasítottak el?",
                         "vehicle", "seller", "after_event")
    assert not relevant_unit("1990. évi XCIII. törvény az illetékekről", SELLER_Q,
                             "vehicle", "seller", "after_event")
    assert relevant_unit("1990. évi XCIII. törvény az illetékekről",
                         "Eladtam az autómat, melyik törvény szabályozza az illetéket?",
                         "vehicle", "seller", "after_event")


def test_progress_labels_do_not_claim_zero_second_work():
    from dap_assistant.presentation.runtime_view import duration_label
    assert duration_label(0.0000003) == "< 0,001 s"
    assert progress_milestones([{"node": "classify_intent"}])[0].startswith(
        "✓ **Élethelyzet azonosítása**")


def test_real_index_seller_source_extract_excludes_obvious_background():
    """Use supplied local corpus if present, otherwise the synthetic test above covers it."""
    from pathlib import Path
    import pytest
    from dap_assistant.documents.ingestion import load_chunks
    data_dir = Path(__file__).resolve().parents[2] / 'data'
    if not (data_dir / 'processed' / 'vehicle' / 'dap-vehicle-seller.json').exists():
        pytest.skip('Processed seller corpus is optional in the patch package')
    snippets = [{**item, 'evidence_id': 'E_' + item['chunk_id'][:16]}
                for item in load_chunks(data_dir) if item.get('domain') == 'vehicle'
                and item.get('role') in ('', 'general', 'seller')]
    result = source_answer(SELLER_Q, snippets, role='seller', stage='after_event')
    reply = ' '.join(c.text.casefold() for c in result.claims)
    assert 'bejelent' in reply and len(result.claims) <= 7
    for unrelated in ('ajánlat elutasításáról', 'joghatósága alól',
                      'eredetiségvizsgálatot végeztetni', 'az oldal magyar magánszemélyek',
                      '1990. évi xciii. törvény', 'a vevő nem mást szeretne üzembentartóként',
                      'kellékszavatosság fogalma', 'az eladás utáni első 6 hónapban', 'tulajdonosváltás bejelentése', 'az oldalon szereplő tudnivalók'):
        assert unrelated not in reply


def test_vehicle_duty_request_is_parsed_from_chat_not_sidebar():
    request = parse_vehicle_duty_request(
        "Vettem egy 2015-ös, 155 kW-os autót. Mennyi a vagyonszerzési illeték?",
        2026,
    )
    assert request is not None
    assert request.manufacturing_year == 2015
    assert request.registered_kw == 155
    assert parse_vehicle_duty_request(SELLER_Q, 2026) is None


def test_employment_benefit_routes_to_official_calculator_without_guessing():
    result = prepare_unemployment_benefit_calculation(
        "Mennyi álláskeresési járadékot kapok?", []
    )
    assert result is not None
    assert result["status"] == "needs_user_input"
    assert result["official_calculator_url"].startswith("https://nfsz.munka.hu/")
    assert "amount_huf" not in result


def test_chat_ui_has_no_manual_duty_calculator():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[2] / "src" / "dap_assistant" / "ui.py").read_text(
        encoding="utf-8"
    )
    assert "Autós illetékkalkulátor (opcionális)" not in source
    assert "vehicle_calculation" not in source
    assert "calculate_vehicle_acquisition_duty" not in source


def test_grouped_ui_progress_contains_only_completed_stages():
    rows = grouped_progress_milestones([
        {'node': 'classify_intent'}, {'node': 'plan_tasks'},
        {'node': 'hybrid_retrieval'},
    ])
    assert len(rows) == 3
    assert rows[0].startswith('**A kérés értelmezése**')
    assert 'Élethelyzet és kérési cél felismerése' in rows[0]
    assert rows[1].startswith('**Részfeladatok megtervezése**')
    assert rows[2].startswith('**Dokumentumok felkutatása**')
    assert 'Válasz előállítása' not in '\n'.join(rows)


def test_chat_strips_old_and_new_technical_provenance_labels():
    from dap_assistant.presentation.runtime_view import public_answer_text
    assert 'eszközből' not in public_answer_text(
        'Szerződés. *(eszközből származó, szó szerinti forrásrészlet)*'
    )
    assert 'eszközből' not in public_answer_text(
        'Szerződés. *(eszközből ellenőrzött, szó szerinti forráskivonat)*'
    )
