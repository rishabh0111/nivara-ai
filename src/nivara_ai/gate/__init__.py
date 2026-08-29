"""The Gate: the ruling on every Turn — answer, clarify, or escalate (ticket 16).

The reason a customer is not confidently told the wrong thing about their money.
`traffic/taxonomy.md` measured the cost of having no Gate: 33 of 70 sensitive
Turns answered rather than escalated. This package is what sits between the agent
loop's candidate and the write:

- `signals` — the three **Free signals**, computed every Turn with no model
  call: retrieval top score, post-rerank margin, and the Sensitive category
  classifier. Their failure modes are independent (ADR-0008, and the table in
  `eval/gate_calibration.md`).
- `sensitive` — that classifier: a Bernoulli NB learned from the labelled eval
  questions, committed as a readable `{term: weight}` file.
- `combine` — the learned combination (`GateModel`): logistic regression on the
  three signals, with the committed operating point and Uncertain band.
- `self_consistency` — the one expensive signal, run **only** inside the band.
- `gate` — `Gate.rule`, which puts it together and returns a `GateRuling`.
- `calibration` — the build-time harness: the signal table, the fit, the swept
  false-escalation/false-deflection curve, and the operating point chosen by a
  committed rule. Reproducible with no provider key.
- `phantom` — Phantom deflection: a clarifying Turn that dwell-resolved
  unanswered, counted separately from False deflection and never summed with it.

Confidence is never a model's statement about its own certainty (decision 32):
that is poorly calibrated and is exactly what an injected instruction targets
(OWASP LLM01).
"""

from nivara_ai.gate.combine import GateModel, load_gate_model
from nivara_ai.gate.gate import Gate, GateRuling
from nivara_ai.gate.phantom import ConversationClose, is_phantom_deflection
from nivara_ai.gate.self_consistency import SelfConsistency, run_self_consistency
from nivara_ai.gate.sensitive import SensitiveClassifier, load_sensitive_classifier
from nivara_ai.gate.signals import FreeSignals, compute

__all__ = [
    "ConversationClose",
    "FreeSignals",
    "Gate",
    "GateModel",
    "GateRuling",
    "SelfConsistency",
    "SensitiveClassifier",
    "compute",
    "is_phantom_deflection",
    "load_gate_model",
    "load_sensitive_classifier",
    "run_self_consistency",
]
