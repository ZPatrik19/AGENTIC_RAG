"""Deterministic, user-question-derived information needs (not legal facts)."""
from __future__ import annotations

import re
from dataclasses import dataclass

_COST = re.compile(
    r"\b(?:mennyi(?:be)?|[áa]ra?|[öo]sszeg\w*|d[íi]j\w*|"
    r"k[öo]l?ts?[ée]g\w*|k[öo]lts[ée]g\w*|illet[ée]k\w*|"
    r"fizet\w*|ker[üu]l\w*)\b", re.IGNORECASE,
)
_EXPLICIT_CHARGES = re.compile(
    r"\b(?:k[öo]lts[ée]g\w*|k[öo]l?ts?[ée]g\w*|illet[ée]k\w*|d[íi]j\w*|fizet\w*)\b", re.I,
)
_DEADLINE = re.compile(r"\b(?:mikor|meddig|hat[aá]rid[oő]\w*|h[aá]ny\s+nap\w*|napom)\b", re.I)
_STEPS = re.compile(r"\b(?:teend[oő]\w*|l[eé]p[eé]s\w*|int[eé]z\w*|"
                    r"bejelent\w*|elint[eé]z\w*|"
                    r"sorrend\w*|folyamat\w*)\b", re.I)
_BENEFIT = re.compile(
    r"\b(?:[aá]ll[aá]skeres[eé]si\s+j[aá]rad[eé]k\w*|munkan[eé]lk[uü]li\s+(?:seg[eé]ly|j[aá]rad[eé]k)\w*|"
    r"ell[aá]t[aá]s\w*|t[aá]mogat[aá]s\w*)\b", re.I,
)
_BENEFIT_AMOUNT = re.compile(
    r"(?:\b(?:mennyi|mekkora|[oö]sszeg\w*|p[eé]nz\w*|kalkul[aá]tor\w*)\b.*"
    r"\b(?:j[aá]rad[eé]k\w*|seg[eé]ly\w*|ell[aá]t[aá]s\w*|t[aá]mogat[aá]s\w*)\b)|"
    r"(?:\b(?:j[aá]rad[eé]k\w*|seg[eé]ly\w*|ell[aá]t[aá]s\w*)\b.*"
    r"\b(?:mennyi|mekkora|[oö]sszeg\w*|p[eé]nz\w*)\b)", re.I,
)
_SUPPORTS = re.compile(r"\b(?:milyen\s+t[aá]mogat[aá]s\w*|t[aá]mogat[aá]sok\w*|mire\s+vagyok\s+jogosult)\b", re.I)
_ELIGIBILITY = re.compile(r"\b(?:jogosult\w*|felt[eé]tel\w*|j[aá]rhat\w*|kaphat\w*)\b", re.I)
_DOCUMENTS = re.compile(r"\b(?:dokumentum\w*|irat\w*|pap[ií]r\w*|okm[aá]ny\w*|igazol[aá]s\w*)\b", re.I)
_WHERE = re.compile(r"\b(?:hol|hova|hov[aá]|melyik\s+(?:oldal\w*|hivatal\w*)|korm[aá]nyablak\w*|foglalkoztat[aá]si\s+oszt[aá]ly\w*)\b", re.I)


@dataclass(frozen=True)
class QuestionNeeds:
    steps: bool
    costs: bool
    deadline: bool
    benefit_amount: bool = False
    supports: bool = False
    eligibility: bool = False
    documents: bool = False
    where: bool = False

    @property
    def requested(self) -> tuple[str, ...]:
        fields = ('steps', 'costs', 'deadline', 'benefit_amount', 'supports',
                  'eligibility', 'documents', 'where')
        return tuple(name for name in fields if getattr(self, name))


def question_needs(question: str) -> QuestionNeeds:
    """Recognize combined requests and keep payable fees separate from benefits."""
    benefit_amount = bool(_BENEFIT_AMOUNT.search(question))
    supports = bool(_SUPPORTS.search(question))
    eligibility = bool(_ELIGIBILITY.search(question)) and bool(_BENEFIT.search(question))
    # The benefit amount and a separate payable charge may both be requested.
    # "járadék összege" alone does not request a payable cost.
    costs = (bool(_COST.search(question)) and not benefit_amount
             or bool(_EXPLICIT_CHARGES.search(question))) and not bool(re.search(
        r"\b(?:mennyi\s+(?:id[oő]|nap|h[eé]t)|h[aá]ny\s+nap)\w*", question, re.I,
    ))
    deadline = bool(_DEADLINE.search(question))
    documents = bool(_DOCUMENTS.search(question))
    where = bool(_WHERE.search(question))
    explicit = costs or deadline or benefit_amount or supports or eligibility or documents or where
    steps = bool(_STEPS.search(question)) or not explicit
    return QuestionNeeds(
        steps=steps, costs=costs, deadline=deadline,
        benefit_amount=benefit_amount, supports=supports, eligibility=eligibility,
        documents=documents, where=where,
    )


def is_location_only_question(question: str) -> bool:
    """Whether the user asks solely for an office/channel, not a combined procedure.

    ``hol`` in a combined ``mit kell ... hol és meddig`` question must not
    suppress evidence about obligations, deadlines or documents.
    """
    needs = question_needs(question)
    if not needs.where or any((needs.costs, needs.deadline, needs.documents,
                               needs.benefit_amount, needs.supports, needs.eligibility)):
        return False
    return not bool(re.search(
        r"\b(?:mit\s+kell|teend[oő]\w*|l[eé]p[eé]s\w*|"
        r"milyen\s+(?:ügyintézési\s+)?teend[oő]\w*)", question, re.I,
    ))


def evidence_has_price(text: str) -> bool:
    return bool(re.search(r"\b\d[\d\s.,–-]*\s*(?:Ft|forint)\b", text, re.I))


def answer_facets(question: str, claims: list[dict]) -> dict[str, object]:
    """Mechanical coverage proxy, never semantic/legal correctness."""
    needs = question_needs(question)
    requested = needs.requested
    covered: list[str] = []
    categories = {c.get('category') for c in claims}
    if needs.steps and categories & {'steps', 'deadline', 'where', 'documents', 'insurance', 'support', 'eligibility'}:
        covered.append('steps')
    if needs.costs and any(c.get('category') == 'cost' and evidence_has_price(c.get('supporting_quote', '')) for c in claims):
        covered.append('costs')
    if needs.deadline and 'deadline' in categories:
        covered.append('deadline')
    if needs.benefit_amount and 'benefit_amount' in categories:
        covered.append('benefit_amount')
    if needs.supports and 'support' in categories:
        covered.append('supports')
    if needs.eligibility and 'eligibility' in categories:
        covered.append('eligibility')
    if needs.documents and 'documents' in categories:
        covered.append('documents')
    if needs.where and 'where' in categories:
        covered.append('where')
    missing = [facet for facet in requested if facet not in covered]
    return {
        'requested': list(requested), 'covered': covered, 'missing': missing,
        'coverage': len(covered) / len(requested) if requested else None,
        'measurement': 'structural_coverage_proxy_not_semantic_correctness',
    }
