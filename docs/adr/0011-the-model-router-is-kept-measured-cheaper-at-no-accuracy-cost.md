# The model router is kept: measured cheaper at no accuracy cost

Ticket 24. Route a Turn to a cheaper or a stronger rung of the failover chain by what the Turn needs rather than always taking the first available one — and then measure whether it changed anything. The ticket is explicit that this has **two acceptable outcomes, equally good**: ship it with the measurement beside it, or delete it and record the deletion, exactly as a retrieval stage that did not move recall was (decision 12).

## Outcome (2026-09-04): kept

The eval set was recorded against both Groq rungs and `scripts/router_ablation.py --drive` was run. `eval/router_ablation.md`: routing the ~240 easy-looking Turns one rung down (`openai/gpt-oss-120b` → `openai/gpt-oss-20b`) is **26–39% cheaper on every one of the eight routed categories** at modelled list price, with **no accuracy regression** on any of the thirteen and a 2-point gain on `sso-authentication` (the one category past the ablation's materiality bar). `decide` returns keep. So the router **ships on for the deployed service** (`render.yaml` sets `NIVARA_MODEL_ROUTER_ENABLED=true`) with that table committed beside it. `Settings.model_router_enabled` stays `False` — the harness, CI and every replay run keep measuring the strongest-first path, so the committed regression baseline and published numbers are unrouted.

Now-open follow-up: the `_EASY_*` thresholds are still the hand-picked constants below, not fit to anything. They earned a fit the way the Gate's combination did — a separate ticket, since the gain measured here is already real at the conservative constants. The rest of this ADR is the design as it was decided.

## What was built

`nivara_ai.model.router.ConfidenceTieredPolicy` reads the three Free signals a Turn already computes before the loop (retrieval top score, post-rerank margin, sensitive score — `nivara_ai.gate.signals`) off `ModelRequest.routing_features`, and returns a starting rung index for `FailoverChain.complete`. An "easy" Turn — confident, well-separated retrieval, not sensitive — starts one rung down; everything else starts at rung 0, which is the chain's historical behaviour, now named `StrongestFirst`.

It is **over the existing chain, not a parallel path**. `FailoverChain` gained one optional `policy` argument; `complete` starts its existing fall-through loop at the policy's index instead of always at zero. A routed start that fails falls through to the next rung exactly as before; a skipped lower rung is not revisited, because starting higher *is* the routing decision. `routing_features` is not a model input — it never reaches a provider and is excluded from the Recording fingerprint — so turning the router on stales no Recording.

The thresholds (`_EASY_TOP_SCORE` etc.) are hand-picked constants, not learned. Until the ablation says the router is worth keeping there is nothing to fit against; if it survives, the follow-up is to calibrate them the way the Gate's combination is.

## Why it was off until measured

`Settings.model_router_enabled` defaults to `False`, resolving to `StrongestFirst`, so until a decision was made the router changed no behaviour and appeared in no published number.

Deciding it needed the end-to-end eval set driven twice — policy off, then on — for per-category accuracy, latency and modelled cost, which needs a Record run. Until one existed `eval/router_ablation.md` was committed in the **pending a Record run** state, the same honest pending the end-to-end harness level carries, and `nivara_ai.model.router_ablation.decide([])` returned "not yet kept" rather than a guess. Both are still the fallback for a cold clone with no Recordings; the run since landed and `scripts/router_ablation.py --drive` filled the table (see Outcome, above).

## The harness the ablation drives against

The routing policy lives inside `FailoverChain`, which the replay path did not build — replay returned a bare `ReplayTransport`, so `--drive` would have flipped a setting that changed nothing and every arm would have come out identical. Closing that is the second half of ticket 24:

- **Replay goes through the chain.** `build_model_client_from_settings` builds a `FailoverChain` of `ReplayTransport` rungs in replay mode, mirroring the live chain, so the policy runs on the exact path the harness measures. Each rung reads its own per-rung Recording (`restamp_for_rung`, the layout `recordings/README.md` already documents).
- **The Record run captures every rung.** `scripts/record_eval.py` drives the Turn once per chain rung through a single-rung capturing chain, filing `recordings/turn/<key>/step-N/<rung>.json`. It reads the per-rung keys (`NIVARA_GROQ_API_KEY`, `NIVARA_GEMINI_API_KEY`); one Groq key covers rungs 0 and 1, which is all the policy — one rung down at most — can reach.
- **Cost is per-rung.** `ModelResponse` carries the rung that answered; `turn_cost_usd` prices each Step at that rung's list price, so a Turn the router sent to a cheaper rung is cheaper in the table. `decide` reads accuracy and this cost.
- **Latency is not measured here.** Replay latency is harness wall-clock, not provider response time. The table shows it, marked indicative; `decide` reads accuracy and modelled cost only. The ticket's criterion names *"accuracy, latency and modelled cost"* — this reads it as *decidable from* accuracy and cost, with latency reported but not gating, because a replay drive cannot honestly produce a provider-latency number and a live drive of both arms is the quota spend the whole replay design exists to avoid. A keep verdict that later wants a real latency column can get one from a live `--drive`.

**The sample.** `scripts/router_ablation.py --drive` scores a case only if it has a rung-0 Recording and — when the policy would route it down — a rung-1 Recording too, and runs both arms over that same set. So the per-category sample is identical between arms and never silently reweighted by a missing recording. `TurnRunner.routing_start_rung` answers "where would the router send this?" from retrieval and the Free signals alone — no model call — so both `record_eval.py` and the ablation can tell, before spending anything, which cases need rung 1. The Record run captures rung 0 for the whole set and rung 1 only for the routed subset; the free tier's binding limit is 200k tokens/day/model, so skipping the ~non-routed cases (the sensitive slice among them) is the difference between roughly one and two weeks for `--slice all`.

## The cost accepted

This is build effort on a feature that may be deleted in one commit. That is what the ticket signs up for. The alternative — deferring the design entirely — was rejected because a clean, off-by-default seam plus a ready-to-run measurement is what lets the decision be made honestly the moment a Record run exists, rather than months later from a cold start. The alternative of *shipping it on* with a fabricated or hand-waved measurement is exactly the "unexamined feature shipped" the ticket names as the bad outcome.

## The handoff, as executed

With the compose stack up and the Corpus indexed:

1. `scripts/run_router_record_run.sh` — one command: brought the stack up, drove `scripts/record_eval.py --slice all` against the three Groq keys (rotating on each daily cap, resuming across two days), and committed `recordings/` as it went. 1,228 Recordings — rung 0 for all 550 dispositioned cases, rung 1 for the ~240 the policy routes.
2. `python scripts/router_ablation.py --drive` — filled `eval/router_ablation.{json,md}` from the replay, no quota.
3. `decide` returned **keep**. So `render.yaml` sets `NIVARA_MODEL_ROUTER_ENABLED=true`, the table is committed beside the code, and the README, CONTEXT.md, `config.py` and this ADR record it. The delete branch — remove `nivara_ai.model.router`, the `policy` argument, `ModelRequest.routing_features`, `Settings.model_router_enabled`, letting `build_replay_failover_chain` fall back to `StrongestFirst` — was not taken.

**A gap this accepts.** The on-path was measured once, against 549 frozen Recordings. `Settings.model_router_enabled=False` keeps CI, the regression baseline and every published number on the off-path, so the router's on-path has no standing regression gate — a rerun of `router_ablation.py --drive` after a future Record run is what re-checks it. That is the same shape as the Trace sink and the Slack ingress, which are also on only in production and out of the harness.
