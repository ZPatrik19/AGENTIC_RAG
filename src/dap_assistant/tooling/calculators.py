"""Read-only calculations. Published rates must be present in indexed official evidence.

Illustrative finance math is explicitly separate from legal fees and bank offers.
No eligibility, exemption, tax assessment, or official vehicle registry access.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re

from pydantic import BaseModel, Field


NAV_VEHICLE_2026 = 'nav-vehicle-duty-2026'
# Rates transcribed from NAV's 2026 dated public table. They are NOT usable
# until the corresponding downloaded source text has passed the row checks.
RATES_2026 = (
    (40, (550, 450, 300)),
    (80, (750, 550, 450)),
    (120, (850, 750, 550)),
    (float('inf'), (950, 850, 750)),
)
ROW_SIGNATURES = (
    (r'0\s*[–—-]\s*40', (550, 450, 300)),
    (r'41\s*[–—-]\s*80', (750, 550, 450)),
    (r'81\s*[–—-]\s*120', (850, 750, 550)),
    (r'120\s*(?:kW\s*)?(?:felett|<|>)', (950, 850, 750)),
)


class VehicleDutyInput(BaseModel):
    manufacturing_year: int = Field(ge=1900, le=2026)
    registered_kw: int | None = Field(default=None, gt=0, le=2000)
    horsepower: int | None = Field(default=None, gt=0, le=3000)
    assessment_year: int = 2026


class LoanInput(BaseModel):
    principal_huf: int = Field(gt=0, le=10_000_000_000)
    annual_interest_percent: Decimal = Field(ge=0, le=100)
    months: int = Field(gt=0, le=600)


def _matching_nav_evidence(evidence: list[dict]) -> dict | None:
    """Require exact table row/rate correspondence, not just a NAV-looking URL."""
    for item in evidence:
        if item.get('document_id') != NAV_VEHICLE_2026:
            continue
        if not str(item.get('source_url', '')).startswith('https://nav.gov.hu/'):
            continue
        text = item.get('text', '')
        if ('2026' not in text
                or not re.search(r'gépjármű|vagyonszerzési', text + str(item.get('title', '')), re.I)):
            continue
        lines = [' '.join(line.split()) for line in text.splitlines()]
        matched = True
        for row_pattern, rates in ROW_SIGNATURES:
            if not any(re.search(row_pattern, line, re.I) and
                       all(re.search(rf'(?<!\d){rate}(?!\d)', line) for rate in rates)
                       for line in lines):
                matched = False
                break
        if matched:
            return item
    return None


def calculate_vehicle_acquisition_duty(request: VehicleDutyInput, evidence: list[dict]) -> dict:
    """2026 NAV vehicle acquisition duty; not an exemption/entitlement decision."""
    base = {'tool': 'calculate_vehicle_acquisition_duty', 'status': 'incomplete',
            'assumptions': ['A gépjármű Magyarországon, 2026-ban illetékköteles.',
                            'Nem vizsgáltuk az illetékmentességeket és kedvezményeket.']}
    if request.assessment_year != 2026:
        return {**base, 'reason': 'Csak a 2026-os, ellenőrzött díjtábla használható.'}
    if request.manufacturing_year > request.assessment_year:
        return {**base, 'reason': 'A gyártási év nem lehet későbbi a vizsgált évnél.'}
    if request.registered_kw is None:
        estimate = (round(request.horsepower * 0.73549875)
                    if request.horsepower is not None else None)
        return {**base, 'reason': 'A forgalmi engedélyben szereplő kW-teljesítmény szükséges.',
                'approximate_kw_from_hp': estimate, 'approximation_only': estimate is not None}
    matched = _matching_nav_evidence(evidence)
    if matched is None:
        return {**base, 'reason': 'A teljes 2026-os NAV-díjtábla nincs ellenőrzött, indexelt forrásban.'}
    age = request.assessment_year - request.manufacturing_year
    age_index = 0 if age <= 3 else 1 if age <= 8 else 2
    rate = next(rates[age_index] for upper, rates in RATES_2026
                if request.registered_kw <= upper)
    return {**base, 'status': 'calculated', 'amount_huf': request.registered_kw * rate,
            'rate_huf_per_kw': rate, 'registered_kw': request.registered_kw,
            'manufacturing_year': request.manufacturing_year,
            'assessment_year': request.assessment_year,
            'source_evidence_ids': [matched['evidence_id']],
            'document_id': NAV_VEHICLE_2026, 'document_version': matched.get('document_version')}


def calculate_illustrative_loan(request: LoanInput) -> dict:
    """Standard fixed-rate annuity only: no THM, fees, bank underwriting or subsidy."""
    principal = Decimal(request.principal_huf)
    monthly_rate = request.annual_interest_percent / Decimal(1200)
    if monthly_rate == 0:
        payment = principal / request.months
    else:
        factor = (Decimal(1) + monthly_rate) ** request.months
        payment = principal * monthly_rate * factor / (factor - Decimal(1))
    rounded = payment.quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    total = (payment * request.months).quantize(Decimal('1'), rounding=ROUND_HALF_UP)
    return {'tool': 'calculate_illustrative_loan', 'status': 'calculated',
            'monthly_payment_huf': int(rounded), 'total_repayment_huf': int(total),
            'total_interest_huf': int(total - principal),
            'assumptions': ['Végig fix névleges éves kamat, havi annuitás.',
                            'Nem THM vagy banki ajánlat; díjak, támogatások és jogosultság nélkül.',
                            'A kamatot és a futamidőt a felhasználó adta meg.']}


def parse_vehicle_duty_request(question: str, assessment_year: int) -> VehicleDutyInput | None:
    q = question.casefold()
    if not ('vehicle' in q or any(s in q for s in
                                 ('autó', 'gépkocsi', 'gépjármű', 'átírás', 'átírat', 'névre írás', 'névre irás', 'névreírás',
                                  'vagyonszerzési illeték'))):
        return None
    if not any(term in q for term in ('illeték', 'átírás', 'átírat', 'névre írás', 'névre irás', 'névreírás')):
        return None
    match = re.search(r'\b(19\d{2}|20\d{2})\s*[-.]?\s*(?:es|ös|as|évjárat|gyártású)?', q)
    kw = re.search(r'\b(\d{2,4})\s*k\s*w\b', q)
    hp = re.search(r'\b(\d{2,4})\s*(?:lóerő\w*|le\b)', q)
    if not match:
        return None
    return VehicleDutyInput(manufacturing_year=int(match.group(1)),
                            registered_kw=int(kw.group(1)) if kw else None,
                            horsepower=int(hp.group(1)) if hp else None,
                            assessment_year=assessment_year)


def parse_loan_request(question: str) -> LoanInput | None:
    """Read ONLY explicitly provided principal, nominal rate and term; no bank data."""
    q = question.casefold().replace('\u00a0', ' ')
    if not any(term in q for term in ('hitel', 'törleszt', 'kölcsön')):
        return None
    principal = re.search(r'\b(\d+(?:[.,]\d+)?)\s*(?:millió|milli[oó])\s*(?:ft|forint)?', q)
    huf = re.search(r'\b(\d{1,3}(?:[\s.]\d{3})+|\d{5,10})\s*(?:ft|forint)\b', q)
    if principal:
        amount = int(Decimal(principal.group(1).replace(',', '.')) * 1_000_000)
    elif huf:
        amount = int(re.sub(r'[\s.]', '', huf.group(1)))
    else:
        return None
    rate = re.search(r'\b(\d{1,2}(?:[.,]\d+)?)\s*%', q)
    term_year = re.search(r'\b(\d{1,2})\s*(?:év|évre|éves|évra)\b', q)
    term_month = re.search(r'\b(\d{1,3})\s*(?:hónap|hóra)\w*', q)
    if not (rate and (term_year or term_month)):
        return None
    return LoanInput(principal_huf=amount,
                     annual_interest_percent=Decimal(rate.group(1).replace(',', '.')),
                     months=(int(term_year.group(1)) * 12 if term_year else int(term_month.group(1))))
