"""Common enumerations and constants used across the pipeline."""
from enum import Enum, auto
from typing import Any


class StrEnum(str, Enum):
    """Base class where auto() assigns the name of the member as its value."""

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[Any]
    ) -> str:
        return name


class StatesEnum(StrEnum):
    """Represents the structural states of the JSON generation FSM.

    These states track which part of the JSON function call is currently
    being generated, determining which tokens are valid continuations.
    """
    START = auto()
    NAME_KEY = auto()
    NAME_VALUE = auto()
    ARGS_KEY = auto()
    ARGS_START = auto()
    PARAM_KEY = auto()
    PARAM_VALUE = auto()
    JSON_END = auto()
    END = auto()
