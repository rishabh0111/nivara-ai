"""The shape of an eval question, a Real-phrasing case, and a retrieval label
(ticket 09)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from nivara_ai.retrieval.scenarios import ScenarioCategory, ScenarioTopic

#: `"generated"` for the ~400 ordinary cases that are
#: permitted to write directly. `"assistant-drafted-pending-review"`
#: for the sensitive slice as it comes out of the generator — decision 42
#: forbids the assistant from writing this slice itself, so a row with this
#: source is a candidate for a human to review before it counts as anything.
#: `"human-reviewed"` for a row Rishabh Sharma has read in full and approved:
#: the assistant drafted the text, a named human is the one vouching for it.
EvalQuestionSource = Literal["generated", "assistant-drafted-pending-review", "human-reviewed"]


class EvalQuestion(BaseModel):
    id: str
    scenario_id: str
    category: ScenarioCategory
    topic: ScenarioTopic
    text: str
    source: EvalQuestionSource
    #: `"local"` for the generated ordinary set, mirroring `Document.generated_by`
    #: (ticket 08). `"assistant-draft"` for the sensitive slice — deliberately not
    #: `"local"`, so the two provenances can never be confused by a reader who
    #: only skims the field.
    generated_by: str
    prompt_version: str


class RealPhrasingCase(BaseModel):
    """One Meridian Ticket's opening customer message, extracted rather than
    generated (decision 20) — see `nivara_ai.eval.real_phrasing`.

    `source` is always `"real"`; the `Literal` (rather than a plain `str`,
    defaulted) is what makes "this text came from a real Ticket, not a
    template" a property of the type, the same way `EvalQuestionSource`
    does for the generated and drafted sets above.
    """

    id: str
    ticket_id: str
    subject: str
    text: str
    source: Literal["real"] = "real"


class RetrievalLabel(BaseModel):
    """One pairing between an `EvalQuestion` and a Corpus `Chunk`, proposed or
    adjudicated.

    `propose_labels` only ever writes `"proposed"` — nothing in
    `nivara_ai` writes `"adjudicated"`. That value is reserved for a human
    reviewer to write by directly editing the committed file after reviewing
    the proposal (decision 43); no function anywhere in this codebase
    produces it. For this dataset specifically, "adjudicated" records
    approval of the coarse, document-level proposal methodology as adequate
    — every chunk of a question's source document proposed as a candidate —
    not a claim that each pairing was individually re-derived chunk by
    chunk; see `eval/README.md`.
    """

    question_id: str
    chunk_id: str
    status: Literal["proposed", "adjudicated"]
