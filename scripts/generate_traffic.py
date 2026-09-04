#!/usr/bin/env python
"""Generate customer-side Traffic against the compose API and keep the Traces
(ticket 15).

This is a deliberate, quota-spending run, like `scripts/record_turn.py` — it
drives a few hundred real Turns against a live provider so their Traces can
be read by hand. It is checkpointed: `traffic/turns.jsonl` is written as the
run goes, and re-running skips every case already in it, so a run stopped by
a provider's daily cap resumes for free.

    NIVARA_MODEL_TRANSPORT=live \
    NIVARA_MODEL_BASE_URL=https://api.groq.com/openai/v1 \
    NIVARA_MODEL_API_KEY=... \
    NIVARA_MODEL_NAME=openai/gpt-oss-120b \
    NIVARA_ASSISTANT_TOKEN=nvk_live_... \
    python scripts/generate_traffic.py

Requires the compose stack up with the Corpus indexed. Reading the resulting
Traces and open-coding `traffic/taxonomy.md` from them is by hand and is not
this script.

    python scripts/generate_traffic.py --ordinary 150 --sensitive 100 --real 50

Flags override the default sample; `--seed` changes which Conversations are
drawn; `--pace` is the minimum seconds between model calls (free tiers meter
by the minute).
"""

from __future__ import annotations

import argparse
import sys
import time

from nivara_ai.config import settings
from nivara_ai.eval import load_questions, load_reviewed_sensitive_questions
from nivara_ai.eval.real_phrasing import load_real_phrasing_cases
from nivara_ai.model.client import ModelClient
from nivara_ai.model.errors import ModelRateLimited
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.transport import Transport
from nivara_ai.model.types import ModelRequest, ModelResponse
from nivara_ai.traffic import DEFAULT_SAMPLE, DEFAULT_SEED, run_traffic, select_cases
from nivara_ai.traffic.generate import DEFAULT_TURNS_PATH
from nivara_ai.turn.service import TurnRunner


class _ThoughtSignatureRelay:
    """Echoes a provider's opaque per-tool-call reasoning state back on the
    assistant turn that follows it.

    Gemini 3.x refuses a multi-Step request whose earlier `tool_calls` do not
    carry the `extra_content` thought-signature it emitted — a hard `400`, not
    a degradation. `nivara_ai.turn.loop` reconstructs the assistant turn from
    `ToolCall` (id, name, arguments) alone, so the signature would be lost
    between Steps. This is a run-time workaround for one provider, not a
    provider chain (ticket 21): it stays here in the script, keeps the
    signature opaque, and remembers each one by tool-call id to re-attach when
    that id reappears in a later request's messages. Live-only: nothing here
    touches a Recording.
    """

    def __init__(self, inner: Transport) -> None:
        self._inner = inner
        self._by_call_id: dict[str, dict] = {}

    def complete(self, request: ModelRequest) -> ModelResponse:
        for message in request.messages:
            if message.get("role") != "assistant":
                continue
            for call in message.get("tool_calls") or []:
                extra = self._by_call_id.get(call.get("id"))
                if extra is not None and "extra_content" not in call:
                    call["extra_content"] = extra

        response = self._inner.complete(request)

        for call in (response.raw.get("choices") or [{}])[0].get("message", {}).get("tool_calls") or []:
            if "extra_content" in call and "id" in call:
                self._by_call_id[call["id"]] = call["extra_content"]
        return response


class _PacedTransport:
    """A live provider with a floor on call spacing and a wait-and-retry on
    `429`. The agent loop turns any provider error into an escalation
    (`nivara_ai.turn.loop`), which is correct on the request path but would
    fill a Traffic run with rate-limit escalations that say nothing about the
    service — so the pacing lives here, in the run, not in the loop."""

    def __init__(self, inner: Transport, *, pace_s: float, max_retries: int = 5) -> None:
        self._inner = inner
        self._pace_s = pace_s
        self._max_retries = max_retries
        self._last_call = 0.0

    def complete(self, request: ModelRequest) -> ModelResponse:
        for attempt in range(self._max_retries + 1):
            wait = self._pace_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                response = self._inner.complete(request)
                self._last_call = time.monotonic()
                return response
            except ModelRateLimited as limited:
                self._last_call = time.monotonic()
                if attempt == self._max_retries:
                    raise
                cool_off = limited.retry_after or (self._pace_s * (attempt + 2))
                print(f"  rate limited, waiting {cool_off:.0f}s", file=sys.stderr)
                time.sleep(cool_off)
        raise AssertionError("unreachable")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive customer-side Traffic against the compose API and "
        "checkpoint the Traces to traffic/turns.jsonl (ticket 15). See the "
        "module docstring for the environment it needs."
    )
    parser.add_argument("--ordinary", type=int, default=DEFAULT_SAMPLE["generated-ordinary"])
    parser.add_argument("--sensitive", type=int, default=DEFAULT_SAMPLE["sensitive"])
    parser.add_argument("--real", type=int, default=DEFAULT_SAMPLE["real-phrasing"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pace", type=float, default=2.5, help="min seconds between model calls")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if settings.model_transport != "live" or not settings.model_api_key:
        print("set NIVARA_MODEL_TRANSPORT=live and the NIVARA_MODEL_* provider vars", file=sys.stderr)
        return 2
    if not settings.assistant_token:
        print("set NIVARA_ASSISTANT_TOKEN — Traffic Turns write replies and Notes", file=sys.stderr)
        return 2

    cases = select_cases(
        questions=load_questions() + load_reviewed_sensitive_questions(),
        real_phrasing=load_real_phrasing_cases(),
        sample={
            "generated-ordinary": args.ordinary,
            "sensitive": args.sensitive,
            "real-phrasing": args.real,
        },
        seed=args.seed,
    )

    transport = _PacedTransport(
        _ThoughtSignatureRelay(
            LiveTransport(base_url=settings.model_base_url, api_key=settings.model_api_key)
        ),
        pace_s=args.pace,
    )

    def runner_factory() -> TurnRunner:
        # `disable_gate=True`: the committed `traffic/turns.jsonl` and the
        # taxonomy built from it (ticket 15) are what the service does with
        # retrieval and the Tool surface *alone* — the Error analysis the Gate's
        # signals were chosen from. A re-run must reproduce that, not the
        # post-Gate behaviour.
        runner = TurnRunner.from_settings(
            model_client=ModelClient(transport), disable_gate=True
        )
        assert runner is not None  # the assistant-token check above already returned
        return runner

    done = 0
    for turn in run_traffic(
        cases, runner_factory, api_base_url=settings.api_base_url, checkpoint_path=DEFAULT_TURNS_PATH
    ):
        done += 1
        print(f"[{done}] {turn.case_id} ({turn.set}) -> {turn.trace.outcome}")

    total = len(cases)
    print(f"\n{total} case(s) selected; {DEFAULT_TURNS_PATH} now holds the run.")
    print("Next, by hand: read the Traces, describe each failure, open-code traffic/taxonomy.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
