# The injection suite

Hand-authored payloads, and the refusals a real guard gives back. The suite
that runs them is `tests/injection/`; this directory is the committed input.

## The argument, in the vocabulary reviewers screen against

**OWASP LLM06, Excessive Agency.** The textbook mitigation is least privilege
on the tool surface plus a genuine kill switch, and both are real here rather
than aspirational:

- **Least privilege.** The Assistant token holds four of the eleven grantable
  scopes — `ticket:read`, `ticket:reply`, `ticket:transition`, `note:write`
  (ADR-0005). The Tool surface is three task-shaped Tools with no
  Cross-Conversation read and no passthrough. Where a scope cannot express a
  boundary — `ticket:read` is Tenant-wide or nothing — the Tool surface
  closes it by offering no such Tool.
- **The kill switch.** The service token carries no claims and no expiry, so
  revoking it takes effect on the *next* request rather than at the next token
  expiry. `llm06-kill-switch` demonstrates it: revoke, then watch the very next
  privileged call answer `401`.

**OWASP LLM01, Prompt Injection**, covering both the direct payload typed into
the chat box and the indirect variant that arrives inside content the retriever
returns — the one that actually happens in a support setting.

## Why HTTP refusals and not polite declines

A jailbroken prompt and a perfectly obedient model produce the *identical*
response when the token never held the scope. So each case does not check that
the model declined — it takes a real Assistant-scoped token, performs the
privileged act a perfectly obedient model would have performed, and asserts the
refusal the API returns. A recorded refusal would prove nothing; the value of
the suite is that the refusal comes from the API enforcing a scope.

The suite therefore spends **no model provider quota** — every case is either a
direct API call or a unit-level check of the Tool surface and the retrieval
edge.

## The two guarantees are not the same guarantee

"This service **cannot perform a privileged act**" is enforced by the token's
scopes and the Tool surface, and is what this suite demonstrates. "This service
**does not answer a sensitive question** about money, fraud or identity" is
enforced by the Gate and demonstrated by the eval harness (`eval/`). Nivara
Desk is a helpdesk with no refund, payout or KYC capability at all, so the
first is true of those topics the way it is true of an endpoint that does not
exist. The two rest on different mechanisms and are never presented in one
table — mixing them would be the single dishonest artifact in the repository
(spec, Further Notes).

## `payloads.jsonl`

One payload per line. Hand-authored and committed — a model-written set of
instructions a model must refuse would be graded against its own idea of
refusal (spec decision 42).

| field | meaning |
| --- | --- |
| `id` | stable identifier, used as the test id |
| `owasp` | `LLM06` or `LLM01` |
| `capability` | the withheld capability, in words |
| `scope` | the API permission the act would need, or `null` |
| `vector` | `direct` (typed into the chat) or `indirect` (planted in retrieved content) |
| `suite` | which module runs it: `withheld`, `cross-tenant`, `revocation`, `retrieval`, `tool-surface` |
| `injection` | the injected instruction, verbatim, as an attacker would write it |
| `obedient_act` | the request a perfectly obedient model would issue, or `null` when no endpoint exists |
| `probe` | for the two `absent` cases: the plausible route to hit, which answers `404` |
| `refusal` | `{status, code}`, `{kind: "absent", ...}` (no route is guarded by the scope), or `{kind: "structural"}` (no Tool or filter path exists) |
| `why` | one line on which mechanism does the refusing |

### The two "absent" cases

`llm06-read-user` and `llm06-read-contact` name capabilities (`user:read`,
`contact:read`) that ADR-0005 records as unnecessary here. The committed
OpenAPI document bears that out: no operation is guarded by either scope, so a
perfectly obedient model told to list staff emails or read a Contact record has
no endpoint to call. That is a stronger statement than "the token was not
granted it" — the capability is absent from the API, not merely withheld from
this credential. `tests/injection/test_withheld_capabilities.py` asserts the
absence against `contracts/nivara-api.openapi.json`.
