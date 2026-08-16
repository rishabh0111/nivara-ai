# The Assistant token holds four scopes of eleven, and the API's seed is narrowed to match

The API bounds a machine credential at "no more than an agent": `ASSIGNABLE_SCOPES` is defined as `SUPPORT_WORK` rather than re-spelled, `UNGRANTABLE_SCOPES` is the derived complement, and `grantedScopes()` re-narrows on every read so a hand-edited row cannot widen a token. That is the bound. **This decision is about how much of it to take.**

The answer turned out to be four of the eleven — `ticket:read`, `ticket:reply`, `ticket:transition`, `note:write` — and each of the seven omissions is a consequence of a decision made elsewhere rather than a preference:

- `ticket:create` is unnecessary because the Widget opens the Conversation as the Contact (ADR-0001).
- `ticket:assign` and `user:read` are unnecessary because escalation leaves the Conversation unassigned, which files it into the **Unclaimed pool** staff already triage from. The API has no auto-assignment convention, and inventing one to make a demo look finished would be writing product policy this product declined to have.
- `contact:read` is unnecessary because a Widget Contact is unverified, with no name and no email.
- `ticket:priority`, `note:read` and `analytics:read` are withheld on purpose. `note:read` is the one worth naming: a layer that can write an internal Note but never read one cannot be talked into surfacing a colleague's private note into a customer-facing answer.

**Scopes were not sufficient, and that is the part worth recording.** A cross-Conversation search tool — find how similar issues were resolved for other customers — would surface another customer's Messages into an answer, which is the same failure `note:read` was withheld to prevent, one level up. No scope expresses the difference: `ticket:read` is Tenant-wide or it is nothing. **So the boundary lives in the tool surface instead** — no Tool offers a Cross-Conversation read, and the absence is structural rather than a filter that could be got wrong. Least privilege on the credential closed most of the door; the tool surface closed the rest.

**The API's seed is narrowed to match.** It previously minted `{ name: 'Deflection assistant', scopes: SUPPORT_WORK }` — all eleven — with a comment reading *"exactly the support work"*. Left alone it would have contradicted this repository's central claim in a file a reviewer can open, so the seed now grants the four, with the comment rewritten to say that the bound is the support set and this credential takes four of it. That is a stronger statement of the original idea: the bound is what a machine *may* hold, not what it did. A second seeded **Reporter** token holds `analytics:read` alone and lives in a CI secret, so the request path cannot read, quote or be argued into reasoning about its own scoreboard.

Nothing was lost from the demonstration. Seeded Tickets, Messages, Notes and SLA data are written by the seeder under the owner role, not by this token; no front end presents a `nvk_live_` credential; and the only test that reads the row asserts the hash shape and that scopes are non-empty.

The cost accepted is that this couples two repositories. Someone tidying the seed back to `SUPPORT_WORK` would silently widen the deployed credential and falsify a README they never opened — which is why the narrowing is recorded here, and why the comment in `meridian.ts` explains itself rather than merely stating a list.
