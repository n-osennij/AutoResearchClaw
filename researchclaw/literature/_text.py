"""Markup cleanup shared by the backends that return HTML/JATS fragments.

Europe PMC and Crossref both hand back abstracts with inline markup, and both
used to strip it with ``<[^>]+>``. That blacklist destroys any text where
``<`` is a comparison operator (``p<0.01`` survives only by luck, ``a<b and
c>d`` is eaten whole once ``b`` looks like the ``<b>`` tag). Whitelisting real
tags keeps the math intact.
"""

from __future__ import annotations

import html
import re

# A tag is only stripped when it looks like one: a known name, optional
# *well-formed* attributes (``name="value"``), then ``>``. ``<b and c>`` is a
# comparison, not a tag, and now survives.
_TAG_RE = re.compile(
    r"</?(?:jats:[\w-]+|mml:[\w-]+|h[1-6]|i|b|u|em|strong|italic|bold|sc|"
    r"monospace|underline|named-content|uri|xref|list|list-item|label|caption|"
    r"table|tr|td|th|math|inline-formula|disp-formula|tex-math|sub|sup|p|br|"
    r"span|font|tt|code|pre)"
    r'(?:\s+[a-zA-Z_-]+="[^"<>]*")*\s*/?>',
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")


def strip_markup(text: object) -> str:
    """Remove whitelisted inline markup, unescape entities, collapse spaces."""
    if not text:
        return ""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", str(text)))).strip()
