"""Question answering over the retrieval corpus, and the routing around it.

The important design decision in this package is negative: retrieval does not
answer counting questions. See ``routing.py`` for why, and ``stats.py`` for
what answers them instead.
"""

from __future__ import annotations
