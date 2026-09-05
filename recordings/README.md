# Recordings

Committed, replayable model responses — see ticket 04 and ADR-0004. Every
file here was captured from a real provider by a Record run
(`python -m nivara_ai.model.record`); nothing in this directory is
hand-written.

One JSON file per `recording_id`, under whatever path that id names (a
Scenario, an eval case, a Step within a Turn). A Step whose call went
through the failover chain (ticket 21) files one Recording per rung it
tried, at `<step-id>/<rung-name>` — so a recorded `429` on one rung and a
recorded answer on the next replay the same handoff live provider quota
paid for once. Each file carries the
fingerprint of the inputs it was captured against — model, prompt version,
messages, tools, temperature — so a prompt or model change makes the
existing file stale rather than silently wrong: replay reports it, and a
Record run recaptures it instead of skipping it.

The first Record run against a real provider was the model-router ablation
(ticket 24): the eval set driven against both Groq rungs, 1,228 files —
`turn/<key>/step-0/<rung>.json` for rung 0 on every dispositioned case and
rung 1 on the ~240 the router routes. Regenerating the end-to-end harness
reports and the regression baseline from these — as opposed to the router
ablation's own artifact, which is done — is the release-cadence pass
(`.github/workflows/record-cadence.yml`, ticket 27). The transport and
storage format are what ticket 04 built.

## Age, provenance, and the two-tier gate (ticket 18)

`RecordingInventory` (`nivara_ai.harness.recordings`) folds every file here
into the age-and-provenance stamp each harness report carries, and flags any
Recording captured against a prompt version this repository no longer builds.

The end-to-end harness reports have since been regenerated from these
Recordings (ticket 28): `eval/harness_results.md` carries real per-category
numbers — 595 of 600 cases scored — rather than its pending placeholder. The
two-tier **CI gate** (ADR-0004, `scripts/ci_regression_gate.py`) still
protects the **component** level alone by design, not as a stopgap awaiting
this regen — it runs key-free on every pull request and never drives
end-to-end, so `eval/regression_baseline.md` tracks the component level's
False deflection over the sensitive slice regardless of what
`eval/harness_results.md` shows. A model-facing pull request re-records the
**sensitive slice** plus the **regression cases**
(`scripts/record_eval.py --slice sensitive --slice regression`); the full
set is re-recorded on a release cadence
(`.github/workflows/record-cadence.yml`). `scripts/ci_record_required.py` is
what enforces the first half of that.
