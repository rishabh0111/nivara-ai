"""Failures a `Transport` can raise.

These are the three shapes a real provider fails in that the fallback chain
(ticket 21) needs to fall through on. They are raised identically by
`LiveTransport`, hitting a real provider, and by `ReplayTransport`, replaying
a recorded failure — the fallback chain is tested through the one seam,
never a second one built for testability.
"""

from __future__ import annotations


class ModelProviderError(Exception):
    """Base for every way a model call can fail."""


class ModelRateLimited(ModelProviderError):
    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after
        super().__init__(
            f"rate limited, retry after {retry_after}s"
            if retry_after is not None
            else "rate limited"
        )


class ModelTimeout(ModelProviderError):
    def __init__(self):
        super().__init__("model call timed out")


class MalformedToolCall(ModelProviderError):
    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(f"malformed tool call: {detail}")


class RecordingNotFoundError(ModelProviderError):
    def __init__(self, recording_id: str):
        self.recording_id = recording_id
        super().__init__(f"no Recording captured for {recording_id!r}")


class StaleRecordingError(ModelProviderError):
    """The Recording exists but was captured against different inputs.

    Raised instead of a silent replay — replaying the old model's answer
    under the new prompt's name would be a wrong number reported as a real
    one (ADR-0004).
    """

    def __init__(self, recording_id: str, differences: list[str]):
        self.recording_id = recording_id
        self.differences = differences
        detail = "; ".join(differences)
        super().__init__(f"Recording {recording_id!r} is stale: {detail}")
