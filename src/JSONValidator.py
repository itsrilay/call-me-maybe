from src.common import StatesEnum
from src.models import FunctionDefinition
import re


class JSONValidator:
    def __init__(self, fn_defs: list[FunctionDefinition]) -> None:
        self.fn_defs = fn_defs
        self.valid_fn_names: list[str] = [fn.name for fn in fn_defs]

        self.prefix_lookups = {
            StatesEnum.START: self._build_prefix_set(['{"name":']),
            StatesEnum.NAME_KEY: self._build_prefix_set(['"name":']),
            StatesEnum.ARGS_KEY: self._build_prefix_set(['"parameters":']),
            StatesEnum.NAME_VALUE: self._build_prefix_set(
                [f'"{name}",' for name in self.valid_fn_names]
            ),
            StatesEnum.JSON_END: {"}"}
        }

        self.param_prefix_lookups = {
            fn.name: self._build_prefix_set(
                [f'"{param}":' for param in fn.parameters]
            )
            for fn in fn_defs
        }

        # Numbers Regex
        self.NUM_FULL_RE = re.compile(
            r"""
                ^               # Start
                -?              # Optional sign
                (0|[1-9]\d*)    # Integer (no leading zeros)
                (\.\d+)?        # Fraction (must have digits)
                ([eE][+-]?\d+)? # Exponent (must have digits)
                $               # End
            """,
            re.VERBOSE
        )

        self.NUM_PART_RE = re.compile(
            r"""
                ^                         # Anchor to the start of the string
                -?                        # Optional leading minus sign
                (?:                       # Start of the core number logic
                    (?:0|[1-9]\d*)        # '0' OR (1-9 followed by digits)
                    (?:                   # Optional branches:
                        \.\d+             # 1. Dot followed by at least a digit
                        (?:[eE][+-]?\d*)? # Can then be followed by an exponent
                        |                 # --- OR ---
                        \.\d*             # 2. Dot and zero or more digits...
                        (?![eE])          # If digits are 0, forbid an 'e'
                        |                 # --- OR ---
                        [eE][+-]?\d*      # 3. Exponent and then integer (1e10)
                    )?                    # End of optional branches
                )?                        # Make group optional (for '-' or '')
                $                         # Anchor to the end of the string
            """,
            re.VERBOSE
        )

        # Strings Regex
        self.STR_FULL_RE = re.compile(
            r"""
                ^
                "                                # Opening quote
                (?:
                    [^"\\\x00-\x1f]              # char: no quote, '\', control
                    | \\ ["\\/bfnrt]             # OR standard escapes
                    | \\ u [0-9a-fA-F]{4}        # OR Unicode escape \uXXXX
                )*                               # Repeat content
                "                                # Closing quote
                $
            """,
            re.VERBOSE
        )

        self.STR_PART_RE = re.compile(
            r"""
                ^
                "                                # Must start with a quote
                (?:
                    [^"\\\x00-\x1f]              # Normal characters
                    | \\ [\"\\/bfnrt]?           # Partial standard escape
                    | \\ u [0-9a-fA-F]{0,4}      # Partial Unicode escape
                )*                               # Repeat content
                "?                               # Optional closing quote
                $
            """,
            re.VERBOSE
        )

    def _build_prefix_set(self, targets: list[str]) -> set[str]:
        prefix_set = set()
        for name in targets:
            for i in range(1, len(name) + 1):
                prefix_set.add(name[:i])

        return prefix_set

    def _validate_fixed(
        self, state: StatesEnum, buffer: str, token: str
    ) -> bool:
        text = buffer + token

        return text in self.prefix_lookups[state]

    def _validate_name(self, buffer: str, token: str) -> bool:
        text = buffer + token

        return text in self.prefix_lookups[StatesEnum.NAME_VALUE]

    def _validate_param_key(
        self,
        buffer: str,
        token: str,
        current_fn: FunctionDefinition | None,
    ) -> bool:
        if not current_fn:
            return False

        text = buffer + token

        return text in self.param_prefix_lookups[current_fn.name]

    def _validate_param_value(
        self,
        buffer: str,
        token: str,
        current_fn: FunctionDefinition | None,
        current_param: str | None
    ) -> bool:
        if not current_fn or not current_param:
            return False

        param_type = current_fn.parameters[current_param].type

        if param_type == "number":
            if token == " ":
                return False

            text = buffer + token

            # Full number validation
            if text.endswith(",") or text.endswith("}"):
                number_part = text[:-1]

                return bool(self.NUM_FULL_RE.match(number_part))

            # Partial number validation
            return bool(self.NUM_PART_RE.match(text))

        if param_type == "string":
            text = buffer + token

            # Full string validation
            if (text.endswith(",") or text.endswith("}")) and text[-2] == '"':
                string_part = text[:-1]

                return bool(self.STR_FULL_RE.match(string_part))

            # Partial string validation
            return bool(self.STR_PART_RE.match(text))

        return False

    def is_token_valid(
        self,
        state: StatesEnum,
        buffer: str,
        token: str,
        current_fn: FunctionDefinition | None = None,
        current_param: str | None = None
    ) -> bool:
        if state == StatesEnum.START:
            return '{"name":'.startswith(buffer + token)
        elif state in [StatesEnum.NAME_KEY, StatesEnum.ARGS_KEY]:
            return self._validate_fixed(state, buffer, token)
        elif state == StatesEnum.NAME_VALUE:
            return self._validate_name(buffer, token)
        elif state == StatesEnum.ARGS_START:
            combined = buffer + token
            if not combined.startswith("{"):
                return False
            return self._validate_param_key("", combined[1:], current_fn)
        elif state == StatesEnum.PARAM_KEY:
            return self._validate_param_key(buffer, token, current_fn)
        elif state == StatesEnum.PARAM_VALUE:
            return self._validate_param_value(
                buffer, token, current_fn, current_param
            )
        elif state == StatesEnum.JSON_END:
            return "}".startswith(token)
        return False

    def validate_buffer(self, buffer: str, param_type: str) -> bool:
        if param_type == "string":
            return bool(self.STR_FULL_RE.match(buffer))
        elif param_type == "number":
            return bool(self.NUM_FULL_RE.match(buffer))
        return False
