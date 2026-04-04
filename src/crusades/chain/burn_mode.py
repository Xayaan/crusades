"""Burn mode state for exploit detection and emergency emission control."""

from datetime import datetime

from pydantic import BaseModel, Field


class BurnMode(BaseModel):
    """Runtime-toggleable burn mode that overrides normal weight distribution.

    When enabled, overrides the hparams burn_rate and optionally blocks
    specific UIDs from receiving emissions.
    """

    enabled: bool = False
    burn_rate_override: float = Field(default=1.0, ge=0.0, le=1.0)
    blocked_uids: list[int] = Field(default_factory=list)
    reason: str = ""
    activated_at: datetime | None = None
    activated_by: str = ""

    @classmethod
    def inactive(cls) -> "BurnMode":
        """Return a default inactive burn mode state."""
        return cls(enabled=False)


BURN_MODE_KEY = "burn_mode"
