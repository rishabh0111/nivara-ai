# The eval harness scores with binary assertions, and the judge is fenced and measured

Ticket 17 builds the evidence behind every accuracy claim: three levels — end to
end, trajectory, component — over the labelled set, reported per category. The
spec (decisions 38–41, 45, 47) is prescriptive about *how* the harness may
score, and this ADR records the shape that took and the two decisions the ticket
left open.

## Every assertion is binary, and no number is allowed to slide

The obvious way to score a generated Answer against a reference is a
text-similarity number — cosine over embeddings, ROUGE, BERTScore — thresholded
into a pass. Decision 38 forbids all of them, and the reason is not taste. A
sliding score with a threshold is two knobs a reviewer cannot check: the metric
and the cut. Move either and the headline moves, and "84% accurate" becomes a
sentence about the harness rather than the service. So a `Check` carries a
`bool`, `Check.__post_init__` refuses anything else, and
`tests/harness/test_no_sliding_scores.py` scans the package's own source for the
name of any overlap metric — the same enforcement-by-parsing
`tests/eval/test_generate.py` uses to keep the eval generator away from the
Corpus.

What is left when text similarity is gone is two kinds of check:

- **Code assertions.** Structural, deterministic, readable. The trajectory level
  is entirely this — Tool names are real, arguments are well-formed, the one
  customer-visible action is last, `read_conversation` happens at most once,
  the loop stayed under the Step ceiling. The end-to-end level's
  `correct-disposition` is this too: the ground truth is the Scenario's
  hand-authored `ordinary`/`sensitive` tag, so "a sensitive question escalated"
  is a fact the harness checks, not a judgement it makes.
- **Judged checks.** "Is this Answer grounded in what retrieval returned?" has no
  structural form and must not be faked with an overlap score. It needs a second
  model's reading — and that is the thing a reviewer is right to distrust, so it
  is fenced.

## The judge: a different family, offline, and demoted if it does not agree

Three rules, from decision 41, and `nivara_ai.harness.judge` is each of them:

1. **A different model family than the answerer.** `assert_different_family`
   refuses a judge whose model string shares a family prefix with the
   answerer's. A model grading its own family's output is not independent. The
   generation does not judge either — the judge is a configured
   model, evaluated independently of whatever generated the inputs it grades.
2. **Offline, through the one model seam.** The judge run goes through
   `ModelClient` with its own `recording_id`, the same seam the agent loop and
   self-consistency use, so a Record run captures it and the harness replays
   it with no provider key (ADR-0004). `judge.py` is the fence around that
   run: `cohens_kappa`, `assert_different_family`, the check specs and the
   demotion rule; `judge_prompt.py`/`judge_replay.py` build the requests and
   replay the committed verdicts.
3. **Measured against ~100 hand labels as Cohen's κ, and demoted below κ ≥ 0.7.**
   `cohens_kappa` is the agreement; `resolve_agreement` applies the floor and
   records the disposition. A demoted check falls back to a code assertion or is
   left human-labelled — it is never reported as a judged number. The committed
   `eval/harness_results.md` lists every check's kind so a reader knows which
   rows rest on a second model.

The hand labels and the judge run are a human, quota-spending step — the
generation may not author ground truth. Both ran: the
judge (the committed Gemini rung, a different family than every Groq-answered
case) against a 100-case sample, and a second human reviewer's independent
hand labels against the same sample. `answer-addresses-question` cleared the
floor (κ = 1.00) and stays a judged check; `answer-grounded` came in at
κ = 0.14 and demoted to human-labelled, exactly the outcome `resolve_agreement`
exists to produce when the judge and an independent human do not actually
agree.

## What is reproducible with no key today, and what waits for a Record run

The harness is one artifact but its levels do not all cost the same to run:

- **Component** replays `eval/gate_calibration.json` — the committed Free-signal
  row per labelled question — against the committed `gate/model.json`. No
  provider key, no Qdrant. It is in the committed results in full: 550 questions,
  per topic, zero sensitive questions auto-answered.
- **Trajectory** scores the 260 committed Traffic Traces
  (`traffic/turns.jsonl`). No key, no stack. Also in the committed results in
  full — and it independently rediscovers the one malformed Tool call the hand
  review of that Traffic already found (`traffic/taxonomy.md`).
- **End to end** drives the whole Turn and needs a Recording for every eval
  case. Before the Record run (ticket 24), `recordings/` was empty and the
  committed results carried it as `pending a Record run` — every case
  counted, none scored, which is honest rather than a silent zero. Ticket 28
  regenerated the committed results from that run: 595 of 600 cases scored,
  the remaining 5 (an unrecorded slice of Real-phrasing) still pending.

The component level's `not-false-escalation` (2.0% on the labelled set) reads
*lower* than `eval/gate_calibration.md`'s operating-point false-escalation
(6.8%). That is not a contradiction: the harness counts only the placements the
Free signals make *outside* the Uncertain band, and the difference is the band —
the cases that get self-consistency and, on a split, one clarifying Turn before
anything is posted. The results file says so beside the number.

## Alternatives rejected

**One combined accuracy figure.** Decision 45 and user story 45 are explicit:
one weak category hidden behind four good ones is the failure mode, so every
level reports per topic (for the labelled set) or per Traffic set (for
trajectory), with the Real-phrasing slice always its own line (decision 20).
The markdown has an "all categories" summary row, but it sits under the
breakdown, not instead of it.

**A judge with no κ, trusted because it is a good model.** That is the "we
prompt it not to" of evaluation. A judged number with no measured agreement
against human labels is not evidence, and decision 41's floor is what makes the
judge a described instrument rather than an appeal to authority.
