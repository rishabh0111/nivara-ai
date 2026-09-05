# The false-deflection regression baseline

Generated from `eval/harness_results.json` by `python scripts/ci_regression_gate.py --write-baseline`. Do not hand-edit — `eval/regression_baseline.json` is the data and `tests/harness/test_regression_baseline_doc.py` re-renders this file from it.

Every pull request runs `scripts/ci_regression_gate.py`, which replays the deterministic harness levels with no provider key and **fails on any per-category rise** in the counts below (ADR-0004). Zero tolerance is affordable because replay is deterministic: a rise is a real behaviour change, never a sample.

- Baseline taken: 2026-09-04
- Regression cases replayed every run: RC-001, RC-002 (`eval/regression_cases.jsonl`)
- Recordings: Replayed 1228 Recording(s), captured 2026-09-02 to 2026-09-03.
- Recordings: Prompt versions: agent-v1, gate-consistency-v1.
- Recordings: Models: openai/gpt-oss-120b, openai/gpt-oss-20b (groq).
- Recordings: No prompt version here is one this repository stopped building. A Tool-schema or model-choice edit that leaves the version string untouched is caught at pull-request time by `scripts/ci_record_required.py`, not here.

| category | false deflection | of scored |
| --- | --- | --- |
| `component/account-recovery-ownership` | 0 | 30 |
| `component/billing-disputes` | 0 | 30 |
| `component/fraudulent-communications` | 0 | 30 |
| `component/payment-method-changes` | 0 | 30 |
| `component/suspicious-account-activity` | 0 | 30 |

**Total:** 0 false deflection across 5 categories.
