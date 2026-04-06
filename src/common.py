from enum import Enum, auto


class StrEnum(str, Enum):
    """Base class where auto() assigns the name of the member as its value."""

    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name


class StatesEnum(StrEnum):
    START = auto()
    NAME_KEY = auto()
    NAME_VALUE = auto()
    ARGS_KEY = auto()
    ARGS_START = auto()
    PARAM_KEY = auto()
    PARAM_VALUE = auto()
    JSON_END = auto()
