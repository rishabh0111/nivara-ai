#!/usr/bin/env python
"""Record run over a named slice of the eval set (ticket 18, ADR-0004).

`scripts/record_turn.py` captures the Recordings for one fixture question. This
captures them for a whole slice — the hand-authored **sensitive** slice and the
**regression** cases on every model-facing change, the **all** set on a release
cadence:

    NIVARA_MODEL_TRANSPORT=live \\
    NIVARA_GROQ_API_KEY=... \\
    NIVARA_GEMINI_API_KEY=... \\
    NIVARA_ASSISTANT_TOKEN=nvk_live_... \\
    python scripts/record_eval.py --slice sensitive --slice regression

Each rung of the committed failover chain (`nivara_ai.model.chain.CHAIN`) whose
key is set is recorded in turn — the Turn is driven once per rung through a
single-rung `FailoverChain`, so each rung's response lands in its own per-rung
Recording (`recordings/turn/<key>/step-N/<rung>.json`). `--rung <name>` narrows
it to named rungs. The model router (ticket 24) needs rung 0 and rung 1, which
one Groq key covers.

`NIVARA_GROQ_API_KEYS` (comma-separated) adds more Groq keys; a rung rotates to
the next when the current one hits its daily cap, so N keys ≈ N× the throughput.
The run exits 3 — "cap hit, rerun to resume" — only once every key is spent.

Each Recording is written the moment it is captured and a case already recorded
is skipped, so an interrupted run resumes without re-spending what it has.
Requires the compose stack up with the Corpus indexed; never wired into CI.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

#: Retry-After above this is treated as a rough ceiling — a genuine daily-cap
#: 429 can name hours, and cycling keys every ~90s beats sleeping on one.
_RETRY_AFTER_CEILING_SECONDS = 90
#: Fallback back-off for a failure that named no wait (a timeout, a header-less
#: 429, a truncated payload).
_DEFAULT_BACKOFF_SECONDS = 30
#: Attempts (key rotations, with a sleep after each full cycle of the pool) for
#: one request before it is given up on — the case escalates unrecorded and the
#: run moves on.
_MAX_ATTEMPTS_PER_REQUEST = 9
#: Consecutive cases that captured nothing before the run concludes the whole
#: pool is blocked (daily cap, or a Groq outage) and stops for the wrapper to
#: resume later.
_DEAD_CASES_BEFORE_STOP = 4

from nivara_ai.config import settings
from nivara_ai.harness.ci import regression_case_to_e2e, sensitive_slice
from nivara_ai.harness.endtoend import EndToEndCase, iter_eval_cases, recording_present
from nivara_ai.harness.regression_cases import load_regression_cases
from nivara_ai.model import recording as recording_store
from nivara_ai.model.chain import CHAIN, rung_key_hint, rung_key_pool
from nivara_ai.model.errors import ModelProviderError
from nivara_ai.model.client import ModelClient
from nivara_ai.model.failover import FailoverChain, Rung
from nivara_ai.model.live import LiveTransport
from nivara_ai.model.recording import Recording, RequestSnapshot
from nivara_ai.model.types import ModelRequest, ModelResponse
from nivara_ai.traffic.generate import mint_widget_session, open_conversation
from nivara_ai.traffic.guard import assert_compose_target

#: The slices the two-tier gate names (ADR-0004): `sensitive` + `regression`
#: on a model-facing change, `all` on the release cadence.
SLICES = ("sensitive", "regression", "all")


class _CapturingTransport:
    """Delegates to a live provider and commits each response as a Recording
    the moment it arrives — the file on disk is the checkpoint."""

    def __init__(self, base_url: str, key_pool: list[str], recordings_dir: Path) -> None:
        self._base_url = base_url
        self._pool = key_pool
        self._key = 0
        self._inner = LiveTransport(base_url=base_url, api_key=key_pool[0])
        self._dir = recordings_dir
        self.captured: list[str] = []

    def _next_key(self) -> None:
        self._key = (self._key + 1) % len(self._pool)
        self._inner = LiveTransport(base_url=self._base_url, api_key=self._pool[self._key])

    def complete(self, request: ModelRequest) -> ModelResponse:
        for attempt in range(1, _MAX_ATTEMPTS_PER_REQUEST + 1):
            try:
                response = self._inner.complete(request)
                break
            except ModelProviderError as err:
                if attempt == _MAX_ATTEMPTS_PER_REQUEST:
                    raise
                wait = getattr(err, "retry_after", None) or _DEFAULT_BACKOFF_SECONDS
                wait = min(wait, _RETRY_AFTER_CEILING_SECONDS)
                # Each key has its own per-minute budget, so move on before
                # sleeping — three keys is three times the throughput. Only
                # when a request cannot get through any of them for the whole
                # attempt budget does the run treat the pool as blocked.
                if len(self._pool) > 1:
                    self._next_key()
                if attempt % len(self._pool) == 0 or len(self._pool) == 1:
                    time.sleep(wait)
        recording_store.save(
            self._dir,
            Recording(
                recording_id=request.recording_id,
                captured_at=datetime.now(UTC),
                fingerprint=request.fingerprint(),
                request_snapshot=RequestSnapshot.from_request(request),
                outcome="response",
                response=response,
            ),
        )
        self.captured.append(request.recording_id)
        return response


def _cases_for(slices: set[str]) -> list[EndToEndCase]:
    want_all = "all" in slices
    by_key: dict[str, EndToEndCase] = {}

    if want_all:
        for case in iter_eval_cases():
            by_key[case.recording_key] = case
    if want_all or "regression" in slices:
        for rc in load_regression_cases():
            if (case := regression_case_to_e2e(rc)) is not None:
                by_key[case.recording_key] = case
    if want_all or "sensitive" in slices:
        for case in sensitive_slice():
            by_key[case.recording_key] = case
    return sorted(by_key.values(), key=lambda c: c.case_id)


def _rungs_to_record(names: list[str] | None) -> list[tuple[int, Rung, str, list[str]]]:
    """`(chain_index, rung, base_url, key_pool)` for each chain rung to capture:
    every rung whose key is configured, or just the named ones. The Groq pool
    carries `groq_api_keys` after the primary, to rotate through on a daily cap.
    A named rung with no key is an error — the caller asked for it explicitly."""

    selected: list[tuple[int, Rung, str, list[str]]] = []
    for index, spec in enumerate(CHAIN):
        if names is not None and spec.rung.name not in names:
            continue
        pool = rung_key_pool(spec, settings)
        if not pool:
            if names is not None:
                print(f"rung {spec.rung.name!r} has no key — set {rung_key_hint(spec)}", file=sys.stderr)
                raise SystemExit(2)
            continue
        selected.append((index, spec.rung, spec.base_url, pool))
    return selected


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slice", dest="slices", action="append", choices=SLICES, required=True)
    parser.add_argument(
        "--rung",
        dest="rungs",
        action="append",
        choices=[spec.rung.name for spec in CHAIN],
        help="record only these chain rungs (default: every rung whose key is set)",
    )
    args = parser.parse_args(argv)

    if settings.model_transport != "live":
        print("set NIVARA_MODEL_TRANSPORT=live and the per-rung NIVARA_*_API_KEY vars", file=sys.stderr)
        return 2
    if not settings.assistant_token:
        print("set NIVARA_ASSISTANT_TOKEN — a driven Turn writes the reply / escalation Note", file=sys.stderr)
        return 2

    rungs = _rungs_to_record(args.rungs)
    if not rungs:
        print("no rung to record — set NIVARA_GROQ_API_KEY and/or NIVARA_GEMINI_API_KEY", file=sys.stderr)
        return 2

    assert_compose_target(settings.api_base_url)

    from nivara_ai.turn.service import TurnRunner

    recordings_dir = Path(settings.recordings_dir)
    cases = _cases_for(set(args.slices))
    total_captured = 0

    for index, rung, base_url, key_pool in rungs:
        transport = _CapturingTransport(base_url, key_pool, recordings_dir)
        # A single-rung chain so the request is restamped for this rung — its
        # provider/model land in the fingerprint and its response in a per-rung
        # Recording file, exactly as the deployed multi-rung chain would file it.
        client = ModelClient(FailoverChain([(rung, transport)]))
        runner = TurnRunner.from_settings(model_client=client)
        if runner is None:
            print("no Assistant token configured", file=sys.stderr)
            return 2

        # A rung below the top is only ever reached for a Turn the router would
        # route there (ticket 24) — recording it for the rest is spent quota
        # that nothing replays.
        rung_cases = [
            c for c in cases
            if index == 0 or runner.routing_start_rung(c.subject, c.text) >= index
        ]
        todo = [
            c for c in rung_cases if not recording_present(c, recordings_dir, rung_name=rung.name)
        ]
        routed_note = "" if index == 0 else f" ({len(rung_cases)} of {len(cases)} would route here)"
        keys_note = f", {len(key_pool)} key(s)" if len(key_pool) > 1 else ""
        print(
            f"rung {rung.name}: {len(rung_cases)} case(s) in slice "
            f"{'+'.join(args.slices)}{routed_note}{keys_note}; "
            f"{len(rung_cases) - len(todo)} already recorded, {len(todo)} to capture"
        )
        blocked = False
        dead_streak = 0
        for i, case in enumerate(todo, 1):
            before = len(transport.captured)
            try:
                widget_token = mint_widget_session(settings.api_base_url)
                conversation_id = open_conversation(
                    settings.api_base_url, widget_token, subject=case.subject, message=case.text
                )
                result = runner.run(conversation_id, widget_token)
                outcome = result.outcome
            except Exception as exc:  # noqa: BLE001 — a days-long run must not die on one case
                outcome = f"SKIPPED ({type(exc).__name__}: {exc})"
            print(f"  [{i}/{len(todo)}] {case.case_id}: {outcome}", flush=True)

            # A case that captured a Recording means the pool is alive. A run of
            # cases that captured nothing means every key is blocked — the daily
            # cap, or a Groq outage — so stop and let the wrapper resume later.
            dead_streak = 0 if len(transport.captured) > before else dead_streak + 1
            if dead_streak >= _DEAD_CASES_BEFORE_STOP:
                blocked = True
                break
        total_captured += len(transport.captured)
        if blocked:
            print(
                f"captured {total_captured} Recording(s); {rung.name} captured nothing "
                f"for {dead_streak} cases running — every key looks blocked. Rerun the "
                "same command after the daily reset to resume.",
                flush=True,
            )
            return 3

    print(f"captured {total_captured} Recording(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
