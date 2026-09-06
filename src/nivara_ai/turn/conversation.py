"""The two credentials a Widget-ingress Turn acts with (ADR-0001).

`BorrowedReader` reads the Conversation and its thread with the **Visitor's**
forwarded `nvw_` widget session credential — the Borrowed read. A Conversation
that is not that session's answers `404` here for the same reason it answers
`404` to the Visitor's own browser: the credential never had the reach. There
is no ownership check to forget, because there is no check.

`ConversationWriter` writes with the **Assistant token** — the reply, the Note,
the transition — against the staff paths the Tools declare
(`nivara_ai.tools.definitions`). It has no method that closes a Conversation
and no code path that sends `state: "closed"`: `closed` is structurally
unreachable, and the API refuses that destination to `ticket:transition` alone
in any case.

Every write goes through one guard first: the writer re-reads the Conversation
(with the Borrowed credential again, never the Assistant token) and refuses if
a person is now the assignee, so a Turn cannot write into a thread a human has
taken (user story 18). The guard is a constructor argument rather than an
optional step a caller remembers — a writer with no way to check cannot be
built.

Every request this module makes — read or write — honours the API's `Retry-After`
on a `429` and retries a `409 idempotency_in_flight` after a short pause
(`_send`, decision 45). Every write `POST` carries an `Idempotency-Key` derived
from the Turn, so a retried Turn — a Widget that fired `POST /widget/turns`
twice — replays its first reply rather than posting a second (user story
29). The key is stable across retries because it is built from the Conversation
and the customer content being answered, not from a per-request uuid.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx

#: How many times `_send` will re-issue a request that came back `429` or
#: `409 idempotency_in_flight` before giving the last response back to the
#: caller to deal with.
_SEND_ATTEMPTS = 4

#: `Retry-After` from this API is a whole number of seconds and its
#: fixed-window rate limiter never advises more than the ~60s to the next
#: window (`nivara-api-nestjs/src/rate-limit/fixed-window.ts`). The cap is
#: well above that — it only guards against a misconfigured header parking a
#: Turn for minutes, not against honouring a real interval.
_MAX_RETRY_AFTER_S = 120.0


def _retry_after_seconds(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After", "")
    try:
        return min(max(float(raw), 0.0), _MAX_RETRY_AFTER_S)
    except ValueError:
        return 1.0


def _error_code(response: httpx.Response) -> str | None:
    try:
        return response.json().get("error", {}).get("code")
    except (ValueError, AttributeError):
        return None


def _send(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    timeout: float,
    params: dict[str, Any] | None = None,
    json: dict[str, Any] | None = None,
) -> httpx.Response:
    """One API call, with the two retries decision 45 asks for: `Retry-After`
    on a `429`, and a short backoff on a `409 idempotency_in_flight` (the
    first copy of this same request is still running). Anything else — success
    or a terminal error — is returned for the caller to interpret."""

    response = httpx.request(
        method, url, headers=headers, timeout=timeout, params=params, json=json
    )
    for attempt in range(1, _SEND_ATTEMPTS):
        if response.status_code == 429:
            wait = _retry_after_seconds(response)
        elif response.status_code == 409 and _error_code(response) == "idempotency_in_flight":
            wait = min(0.2 * attempt, 2.0)
        else:
            return response
        time.sleep(wait)
        response = httpx.request(
            method, url, headers=headers, timeout=timeout, params=params, json=json
        )
    return response


AuthorKind = Literal["contact", "service", "user", "system"]
ConversationState = Literal["open", "pending", "on_hold", "resolved", "closed"]


class ConversationNotFound(Exception):
    """The Conversation does not exist, or is not this session's — the API
    makes the two indistinguishable, and so does this service."""


class WidgetSessionInvalid(Exception):
    """The forwarded widget session credential is missing, malformed, expired
    or revoked."""


class ConversationNotWritable(Exception):
    """This service holds no write authority over the Conversation.

    The Assistant token is minted for one Tenant. A Visitor on another Tenant's
    site can still reach a Turn — the read is Borrowed, performed with the
    Visitor's own credential, and it succeeds — and then every write is refused,
    because the API will not show one Tenant's Ticket to another's credential.

    It is a distinct exception because the customer-facing consequence is not a
    fault: their Ticket exists, on their own Tenant's queue, carrying their
    question. A person will pick it up. What cannot happen is this service
    answering or annotating it, so the Turn escalates without writing rather
    than reporting that something went wrong.
    """


class HumanHasTakenConversation(Exception):
    """A person is the Conversation's assignee, so this service must not write
    to it (user story 18). Raised by the write guard; the Turn catches it and
    stands down without posting, transitioning, or writing a Note."""

    def __init__(self, assignee_id: str) -> None:
        super().__init__(f"Conversation is assigned to {assignee_id}")
        self.assignee_id = assignee_id


@dataclass(frozen=True)
class ConversationSnapshot:
    """Just the Ticket row a write needs to consult first: its current state
    (does a transition to `open` need to happen?) and who, if anyone, is now
    responsible for it."""

    state: ConversationState
    assignee_id: str | None


@dataclass(frozen=True)
class ThreadMessage:
    author_kind: AuthorKind
    body: str
    #: The API's own id for this Message. Carried so a Turn can say *which*
    #: message it is answering rather than only what that message said — two
    #: customers' identical questions, or one customer asking the same thing
    #: twice, are different Turns and must not be mistaken for a retry of each
    #: other (see `Conversation.latest_customer_message_id`). Optional because
    #: a thread assembled in a test is about the words, not the ids.
    id: str | None = None

    @property
    def role(self) -> str:
        """Chat role for the model: the customer is `user`, everyone else
        (`service`, a human agent, the system) is `assistant`."""

        return "user" if self.author_kind == "contact" else "assistant"


@dataclass(frozen=True)
class Conversation:
    id: str
    subject: str
    state: ConversationState
    assignee_id: str | None
    thread: list[ThreadMessage]

    @property
    def latest_customer_message(self) -> str | None:
        for message in reversed(self.thread):
            if message.author_kind == "contact":
                return message.body
        return None

    @property
    def latest_customer_message_id(self) -> str | None:
        """The id of the Message this Turn is answering.

        What makes one Turn distinguishable from another on the same
        Conversation. A Visitor who asks the same question twice writes two
        Messages with two ids and is owed two answers; the same Turn retried
        after a dropped connection is answering the one Message and is owed
        one. Keyed on the words instead, those two cases are identical — which
        is how a re-asked question came to be answered into silence, the reply
        refused as a duplicate of the first.
        """

        for message in reversed(self.thread):
            if message.author_kind == "contact":
                return message.id
        return None

    def as_messages(self) -> list[dict[str, Any]]:
        return [{"role": message.role, "content": message.body} for message in self.thread]


class ConversationReader(Protocol):
    """What a Turn needs to read the Conversation it is answering: the whole
    thing once, and the Ticket row alone before each write.

    Two implementations, one per Ingress, differing only in *which credential*
    and *which API surface* they read through — which is the whole of what
    separates the two ingresses at the read (ticket 26). `BorrowedReader` uses
    the Visitor's forwarded widget session and the `/widget` surface;
    `AssistantTokenReader` uses the Assistant token and the staff surface,
    because the Slack ingress has no forwardable credential (ADR-0001).
    """

    def read(self, conversation_id: str) -> Conversation: ...

    def snapshot(self, conversation_id: str) -> ConversationSnapshot: ...


class _TicketReader:
    """Shared read mechanics for both ingresses. A subclass supplies only what
    actually differs: the URL prefix for its API surface, and how it maps a
    refused credential (`_raise_for_status`). The `read` / `snapshot` /
    pagination logic is identical and lives here once."""

    #: `"/widget/tickets"` or `"/tickets"`.
    _ticket_prefix: str

    def __init__(self, base_url: str, token: str, *, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout

    def _raise_for_status(self, response: httpx.Response) -> None:
        """Map a refused read to this ingress's exception. `404` is
        `ConversationNotFound` for both; the Widget ingress also maps `401`."""

        if response.status_code == 404:
            raise ConversationNotFound()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        response = _send(
            "GET",
            f"{self._base_url}{path}",
            params=params,
            headers={"Authorization": f"Bearer {self._token}"},
            timeout=self._timeout,
        )
        self._raise_for_status(response)
        response.raise_for_status()
        return response

    def snapshot(self, conversation_id: str) -> ConversationSnapshot:
        """The Ticket row alone — no thread. The write path re-reads this
        immediately before every write, so a person who claimed the
        Conversation mid-Turn is seen before the service writes over them."""

        ticket = self._get(f"{self._ticket_prefix}/{conversation_id}").json()
        return ConversationSnapshot(state=ticket["state"], assignee_id=ticket.get("assigneeId"))

    def read(self, conversation_id: str) -> Conversation:
        ticket = self._get(f"{self._ticket_prefix}/{conversation_id}").json()
        return Conversation(
            id=ticket["id"],
            subject=ticket["subject"],
            state=ticket["state"],
            assignee_id=ticket.get("assigneeId"),
            thread=self._read_thread(conversation_id),
        )

    def _read_thread(self, conversation_id: str) -> list[ThreadMessage]:
        messages: list[ThreadMessage] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"sort": "createdAt"}
            if cursor:
                params["cursor"] = cursor
            page = self._get(
                f"{self._ticket_prefix}/{conversation_id}/messages", params
            ).json()
            messages.extend(
                ThreadMessage(author_kind=row["authorKind"], body=row["body"], id=row.get("id"))
                for row in page["data"]
            )
            cursor = page.get("nextCursor")
            if not cursor:
                return messages


class BorrowedReader(_TicketReader):
    """The Widget ingress's reader: the Conversation and its thread over the
    `/widget` surface, with the **Visitor's forwarded** widget session — the
    Borrowed read (ADR-0001). A Conversation that is not that session's answers
    `404` here for the same reason it does to the Visitor's own browser."""

    _ticket_prefix = "/widget/tickets"

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.status_code == 401:
            raise WidgetSessionInvalid()
        super()._raise_for_status(response)


class AssistantTokenReader(_TicketReader):
    """The Slack ingress's reader: the Conversation and its thread over the
    **staff** ticket surface (`/tickets/...`), with the **Assistant token**.

    This is the entire reason `ticket:read` is on that token (ADR-0001,
    decision 7). The Widget ingress never uses this — it has a Borrowed read —
    and were Slack dropped the token would fall to three scopes. A `404` is
    `ConversationNotFound`; a Ticket on another Tenant answers `404` here too,
    by the API's design.
    """

    _ticket_prefix = "/tickets"


#: Re-reads the Conversation the write is about, with the Borrowed credential.
#: Supplied by the caller so the writer never holds a widget token itself.
WritableCheck = Callable[[], ConversationSnapshot]


class ConversationWriter:
    def __init__(
        self,
        base_url: str,
        assistant_token: str,
        writable_check: WritableCheck,
        *,
        timeout: float = 10.0,
        idempotency_scope: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = assistant_token
        self._writable_check = writable_check
        self._timeout = timeout
        #: Stable per-Turn seed for the `Idempotency-Key` on every write POST
        #: (user story 29). `None` for a writer with no Turn behind it —
        #: the unit tests of the guard — where at-most-once is not in play.
        self._idempotency_scope = idempotency_scope

    def _request(
        self,
        method: str,
        path: str,
        json: dict[str, Any],
        *,
        idempotency_action: str | None = None,
    ) -> httpx.Response:
        headers = {"Authorization": f"Bearer {self._token}"}
        if idempotency_action is not None and self._idempotency_scope is not None:
            # Scoped per caller and per request by the API; a retried Turn
            # sends the same key and the API replays its first result rather
            # than acting twice (contracts/nivara-api.openapi.json, "Safe
            # retries").
            headers["Idempotency-Key"] = f"{self._idempotency_scope}:{idempotency_action}"

        response = _send(
            method,
            f"{self._base_url}{path}",
            json=json,
            headers=headers,
            timeout=self._timeout,
        )
        if (
            response.status_code == 422
            and _error_code(response) == "idempotency_key_reused"
        ):
            # This Turn already wrote here, with a body that differs only
            # because the model is not perfectly deterministic. The first
            # write stands; a second must not be posted. Benign — return it
            # rather than raising.
            return response
        if response.status_code in (403, 404):
            # The Borrowed read let this Turn get here; the write is judged on
            # the Assistant token instead, and a Ticket on another Tenant is
            # `404` to it by the API's design (the same 404-not-403 that stops
            # the surface being a probe). Named rather than left to
            # `raise_for_status`, which would make a foreign Tenant look like
            # an outage.
            raise ConversationNotWritable(
                f"{method} {path} was refused with {response.status_code}"
            )
        response.raise_for_status()
        return response

    def _guard(self) -> ConversationSnapshot:
        """Run before every write. Re-reads the Conversation and refuses if a
        person is now the assignee — the check spec decision 6 requires before
        this service writes anything."""

        snapshot = self._writable_check()
        if snapshot.assignee_id is not None:
            raise HumanHasTakenConversation(snapshot.assignee_id)
        return snapshot

    def post_reply(self, conversation_id: str, message: str) -> None:
        """Post the Answer to the customer (`ticket:reply`). Authorship is
        stamped `service` by the API's trigger from this credential."""

        self._guard()
        self._request(
            "POST",
            f"/tickets/{conversation_id}/messages",
            {"body": message},
            idempotency_action="reply",
        )

    def resolve(self, conversation_id: str) -> None:
        """Resolve the Conversation this service answered (`ticket:transition`),
        rather than leaving it to the dwell sweep. It reopens to `open` on the
        customer's next reply — the API's own reply rule, not this service's.
        `closed` is never a destination here."""

        self._guard()
        self._request("PATCH", f"/tickets/{conversation_id}/state", {"state": "resolved"})

    def escalate(self, conversation_id: str, note: str) -> None:
        """The atomic Escalation (CONTEXT.md): write the reasoning Note
        (`note:write`), ensure the Conversation is `open`, and leave it
        unassigned so it enters the staff Unclaimed pool.

        One method, and the only path to either half — there is no public way
        to transition without a Note or write a Note without escalating, so the
        half the spec names as unacceptable (CONTEXT.md: "transitioned with no
        Note") is impossible by construction rather than caught by an
        assertion. These are still two API calls with no transaction between
        them: the Note is written first, so a failed transition leaves a Note
        with no move — the benign half, and often a no-op anyway because the
        Conversation was already `open`. The current state comes from the
        guard's own re-read, so this spends no extra request on that check.
        """

        snapshot = self._guard()
        self._request(
            "POST",
            f"/tickets/{conversation_id}/notes",
            {"body": note},
            idempotency_action="note",
        )
        if snapshot.state != "open":
            self._request("PATCH", f"/tickets/{conversation_id}/state", {"state": "open"})
