"""The shape of a Traffic case, a Traffic Turn, and a failure label (ticket 15).

Traffic is synthetic customer-side Conversations driven against the compose
API so their Traces can be read for Error analysis. A `TrafficCase` is one
question to drive; a `TrafficTurn` is what came back and is committed as the
evidence a reviewer re-reads; a `FailureLabel` is one reading of a Turn, its
`note` a concrete single-Turn description rather than a bucket label
(decision 37).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from nivara_ai.retrieval.scenarios import ScenarioCategory, ScenarioTopic
from nivara_ai.turn.trace import Trace

#: Which input set a Traffic case was drawn from. Carried on every Turn so
#: Error analysis reads per set — a failure rate on the sensitive slice is a
#: different fact from one on the generated ordinary set, and reporting per
#: category rather than as an average is decision 45.
TrafficSet = Literal["generated-ordinary", "sensitive", "real-phrasing"]


class TrafficCase(BaseModel):
    """One customer question to drive as a Conversation.

    Derived from an `EvalQuestion` or a `RealPhrasingCase`, never authored
    here — so Traffic reads the same questions the eval harness scores, and
    the taxonomy this produces is about the traffic the harness later
    measures rather than a separate set that happened to be convenient.
    """

    id: str
    set: TrafficSet
    category: ScenarioCategory
    #: `None` for the Real-phrasing slice — a real Ticket's opening message
    #: carries no Scenario topic, because it was not generated from one.
    topic: ScenarioTopic | None
    subject: str
    text: str


class TrafficTurn(BaseModel):
    """One driven Turn: the case that produced it, the customer-visible Answer
    if there was one, and the Trace the endpoint returned.

    Committed to `traffic/turns.jsonl` as the evidence behind the taxonomy —
    a reviewer re-reads these, the same way they re-read `corpus/` and
    `eval/`, rather than taking "we read the traces" on trust. The
    Conversation id and the outcome live on the embedded `trace`; only what
    the Trace does not carry is repeated here.
    """

    case_id: str
    set: TrafficSet
    category: ScenarioCategory
    #: The Answer this service posted, or `None` for an escalation or a
    #: deferral — the Trace records tool calls and steps but not the posted
    #: text.
    answer: str | None
    trace: Trace
    #: The run's own timestamp, UTC — `traffic/README.md` states its date, and
    #: two-tier CI (ticket 18) stamps the age of what it ran against.
    recorded_at: datetime


#: `"drafted"` for a label read off the data but not yet checked;
#: `"adjudicated"` once Rishabh Sharma has read it. Every row in the committed
#: `traffic/labels.jsonl` is `"adjudicated"` (a test asserts it); a fresh
#: label defaults to `"drafted"` and only a hand review moves it, exactly as
#: `nivara_ai.eval.models.RetrievalLabel.status` reserves `"adjudicated"`.
LabelStatus = Literal["drafted", "adjudicated"]


class FailureLabel(BaseModel):
    """One reading of a Traffic Turn.

    `category` is a slug that appears as a heading in `traffic/taxonomy.md`,
    or `"none"` for a Turn that did the right thing. `note` is the concrete,
    single-Turn description of what that Turn did — decision 37 asks that each
    failure be described concretely rather than bucketed on sight, and the
    note is where that description lives.

    `status` records that the label was verified by hand, not just read off
    the Trace — the same draft-then-review path `eval/sensitive.jsonl` and
    `eval/retrieval_labels.jsonl` follow (the assistant may draft from the
    data, but a measured finding is verified by hand). The committed
    set is fully `"adjudicated"`; see `traffic/README.md`.

    Keyed on `case_id` rather than the Trace's `turn_id`: `turn_id` is a
    fresh uuid every run, so a label keyed on it would break the moment
    Traffic is regenerated, whereas a case is the same question every time.
    """

    case_id: str
    category: str
    note: str
    status: LabelStatus = "drafted"
