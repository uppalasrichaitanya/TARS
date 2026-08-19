"""Errors raised by the provider adapter layer.

Every failure that crosses the adapter boundary is one of these, so loop code
never has to catch `anthropic.APIStatusError` and `openai.APIError` and a bare
`urllib.error.HTTPError` separately. The original exception is always attached
as `__cause__` (and, for API errors, as `.original`).
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for every adapter-layer failure."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model


class UnknownProviderError(ProviderError):
    """A model string named a provider we have no adapter for."""


class ProviderNotInstalledError(ProviderError):
    """The provider's SDK isn't importable.

    Carries `install_hint` so the CLI can tell the user exactly what to run
    instead of surfacing a raw ImportError.
    """

    def __init__(self, message: str, *, provider: str, install_hint: str) -> None:
        super().__init__(message, provider=provider)
        self.install_hint = install_hint


class ProviderNotConfiguredError(ProviderError):
    """Credentials for the provider are missing."""


class ProviderAPIError(ProviderError):
    """The provider was reached but refused or failed the request.

    `retryable` is the adapter's judgment call, derived from the HTTP status:
    429 and 5xx are retryable, 4xx generally are not.
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retryable: bool | None = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.status_code = status_code
        self.retryable = (
            is_retryable_status(status_code) if retryable is None else retryable
        )
        self.original = original


class ProviderResponseError(ProviderError):
    """The provider returned something we could not normalize.

    Distinct from ProviderAPIError on purpose: this one means our adapter and
    their response shape disagree, which is a bug on one side, not a transient
    failure to retry.
    """


def is_retryable_status(status_code: int | None) -> bool:
    if status_code is None:
        return False
    if status_code in (408, 409, 429):
        return True
    return status_code >= 500
