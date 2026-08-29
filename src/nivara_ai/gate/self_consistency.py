"""Self-consistency across samples — the Gate's one expensive signal.

It runs **only** when the Free signals land a Turn in the Uncertain band
(`nivara_ai.gate.combine.GateModel.place`), because it costs a multiple of model
calls against a requests-per-day ceiling (decision 30). What fraction of Turns
reach the band, and so pay this, is a reported number
(`eval/gate_calibration.md`).

The question put to each sample is the *decision*, not a confidence: answer the
customer, or hand the Conversation to a person. Sampling the same decision `K`
times at a non-zero temperature and reading the spread is a measurement of the
model's stability on this input — it is not the model's opinion of its own
certainty, which decision 32 bars from the Gate because a self-report is both
poorly calibrated and exactly what an injected instruction would target.

- A clear majority to answer → the Gate answers.
- A clear majority to escalate → the Gate escalates.
- A genuine split → the model cannot settle the question, so the Gate asks one
  clarifying Turn (or, if it already has, escalates).

Every sample is one call through the single model seam (`ModelClient`), with a
`recording_id` of ``{prefix}/sample-{i}``, so a Record run captures them and the
harness replays them with no provider key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from nivara_ai.model.client import ModelClient
from nivara_ai.model.errors import ModelProviderError
from nivara_ai.model.types import ModelRequest

Verdict = Literal["answer", "escalate", "split"]

#: The share of valid votes one side needs for the Gate to follow it. Below
#: this the samples are a split and the Turn is genuinely ambiguous.
AGREEMENT = 0.8


@dataclass(frozen=True)
class SelfConsistency:
    samples: int
    answer_count: int
    escalate_count: int
    #: Samples that returned no usable `post_reply`/`escalate` call, or errored.
    invalid_count: int
    verdict: Verdict

    def as_dict(self) -> dict:
        return {
            "samples": self.samples,
            "answer_count": self.answer_count,
            "escalate_count": self.escalate_count,
            "invalid_count": self.invalid_count,
            "verdict": self.verdict,
        }


def _classify_sample(
    response_tool_names: list[str],
) -> Literal["answer", "escalate", "invalid"]:
    if "post_reply" in response_tool_names and "escalate" not in response_tool_names:
        return "answer"
    if "escalate" in response_tool_names and "post_reply" not in response_tool_names:
        return "escalate"
    return "invalid"


def run_self_consistency(
    client: ModelClient,
    *,
    system: str,
    thread: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    provider: str,
    model: str,
    prompt_version: str,
    recording_id_prefix: str,
    samples: int,
    temperature: float,
) -> SelfConsistency:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system}, *thread]

    answer = escalate = invalid = 0
    for index in range(samples):
        request = ModelRequest(
            recording_id=f"{recording_id_prefix}/sample-{index}",
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            messages=messages,
            tools=tools,
            temperature=temperature,
        )
        try:
            response = client.complete(request)
        except ModelProviderError:
            invalid += 1
            continue

        classified = _classify_sample([call.name for call in response.tool_calls])
        if classified == "answer":
            answer += 1
        elif classified == "escalate":
            escalate += 1
        else:
            invalid += 1

    return SelfConsistency(
        samples=samples,
        answer_count=answer,
        escalate_count=escalate,
        invalid_count=invalid,
        verdict=_verdict(answer, escalate),
    )


def _verdict(answer: int, escalate: int) -> Verdict:
    valid = answer + escalate
    if valid == 0:
        # Nothing usable came back — treat it as the model failing to answer,
        # which is a Turn for a person (user story 10).
        return "escalate"
    if answer / valid >= AGREEMENT:
        return "answer"
    if escalate / valid >= AGREEMENT:
        return "escalate"
    return "split"
