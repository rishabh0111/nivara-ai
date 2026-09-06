"""Markdown out of an Answer, because nothing that renders one reads Markdown.

A Message body is plain text on every Surface it reaches: the Widget renders
it `white-space: pre-wrap` inside a shadow root, the Portal and the Dashboard
render the same field, and the Slack ingress posts it into a client whose
emphasis is spelled differently again. So a model that writes
`**Settings → Billing**` is writing four characters the customer sees as four
characters.

It is stripped here rather than forbidden in the prompt, and the distinction is
deliberate. `system_prompt.md` and the messages built from it are hashed into
`ModelRequest.fingerprint`, so editing the prompt makes every committed
Recording stale and costs a full Record run against live providers — a real
price, for a defect that is entirely about presentation. The model goes on
saying what it says; the Trace and the Recordings keep it verbatim, which is
what makes them evidence. Only the copy handed to a customer is normalised, at
the point it becomes one.

Conservative on purpose. Emphasis, inline code, headings and links are markup a
support answer plausibly picks up from a help centre; bullets are left alone
because a list reads as a list in plain text, and an underscore inside a word
is left alone because `snake_case` is not emphasis.
"""

from __future__ import annotations

import re

#: ```lang\n...\n``` — the fence and its language tag go, the code stays.
_FENCE = re.compile(r"^[ \t]*```[^\n]*\n(.*?)^[ \t]*```[ \t]*$", re.S | re.M)

#: `code` → code. Runs before emphasis so an asterisk inside code is not read
#: as markup on the way past.
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

#: **bold** and __bold__, non-greedy so two runs on one line stay two runs.
_STRONG = re.compile(r"\*\*(\S(?:.*?\S)?)\*\*|__(\S(?:.*?\S)?)__", re.S)

#: *italic*, and _italic_ only where the underscores are not inside a word.
_EMPHASIS_STAR = re.compile(r"(?<!\*)\*(\S(?:.*?\S)?)\*(?!\*)", re.S)
_EMPHASIS_UNDERSCORE = re.compile(r"(?<![\w_])_(\S(?:.*?\S)?)_(?![\w_])", re.S)

#: A leading #, ## … heading marker. The words after it are kept as a line.
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.M)

#: [text](url) → "text (url)", or just the url where they say the same thing.
_LINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _link(match: re.Match[str]) -> str:
    text, url = match.group(1).strip(), match.group(2)
    if not text or text == url:
        return url
    return f"{text} ({url})"


def to_plain_text(answer: str) -> str:
    """`answer` with Markdown markup removed and its words left in place."""

    plain = _FENCE.sub(lambda match: match.group(1), answer)
    plain = _INLINE_CODE.sub(lambda match: match.group(1), plain)
    plain = _LINK.sub(_link, plain)
    plain = _STRONG.sub(lambda match: match.group(1) or match.group(2), plain)
    plain = _EMPHASIS_STAR.sub(lambda match: match.group(1), plain)
    plain = _EMPHASIS_UNDERSCORE.sub(lambda match: match.group(1), plain)
    plain = _HEADING.sub("", plain)
    return plain.strip()
