"""Domain exceptions raised by repositories/services and translated at the route layer."""


class DomainException(Exception):
    """Base class for domain-level errors."""


class NotFoundException(DomainException):
    """Raised when a requested resource does not exist."""


class ConflictException(DomainException):
    """Raised when an operation conflicts with current state (e.g. unique violation)."""


class ValidationException(DomainException):
    """Raised when input fails domain validation beyond Pydantic schema checks."""
