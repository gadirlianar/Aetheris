"""Exception hierarchy.

The orchestrator decides what to do with a failure purely from its type:
``TransientError`` is retried, ``BudgetExceededError`` stops the run gracefully,
anything else fails the run.
"""


class AetherisError(Exception):
    """Base class for all Aetheris errors."""


class TransientError(AetherisError):
    """A retryable failure: rate limits, 5xx responses, timeouts."""


class BudgetExceededError(AetherisError):
    """The run hit one of its hard limits (LLM calls, tokens, wall-clock time)."""


class GraphValidationError(AetherisError):
    """The execution graph is malformed or a router returned an unknown target."""


class ConfigurationError(AetherisError):
    """Missing or invalid configuration (for example, no API key)."""


class LLMOutputError(AetherisError):
    """The model kept returning output that does not match the requested schema."""


class NoEvidenceError(AetherisError):
    """Retrieval produced no usable sources, so there is nothing to ground a report on."""
