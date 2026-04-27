from src.common import StatesEnum
from src.models import FunctionDefinition


class JSONValidator:
    def __init__(
        self,
        fn_defs: list[FunctionDefinition],
        decoded_vocabulary: list[str]
    ) -> None:
        self.fn_defs = fn_defs
        self.valid_fn_names: list[str] = [fn.name for fn in fn_defs]

        self.prefix_lookups = {
            StatesEnum.START: self.build_prefix_set(['{"name":']),
            StatesEnum.NAME_KEY: self.build_prefix_set(['"name":']),
            StatesEnum.ARGS_KEY: self.build_prefix_set(['"parameters":']),
            StatesEnum.NAME_VALUE: self.build_prefix_set(
                [f'"{name}",' for name in self.valid_fn_names]
            ),
            StatesEnum.JSON_END: {"}"}
        }

        self.param_prefix_lookups = {
            fn.name: self.build_prefix_set(
                [f'"{param}":' for param in fn.parameters]
            )
            for fn in fn_defs
        }

        self.structural_id_cache: dict[StatesEnum, list[int]] = {}

        for state in [
            StatesEnum.START,
            StatesEnum.NAME_KEY,
            StatesEnum.ARGS_KEY,
            StatesEnum.JSON_END
        ]:
            allowed_ids = []
            for i, token in enumerate(decoded_vocabulary):
                if self.is_token_valid(state, "", token):
                    allowed_ids.append(i)
            self.structural_id_cache[state] = allowed_ids

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
        text = buffer + token
        stripped_text = text.lstrip()

        # Allow whitespace to make sure the LLM always has options
        if not stripped_text:
            return True

        # Structural boundaries
        forbidden = {"{", "}", ":", ","}

        if param_type == "string":
            if not stripped_text.startswith('"'):
                return False

            is_buffer_closed = False
            stripped_buf = buffer.lstrip()

            # Check if buffer is a closed string
            if len(stripped_buf) >= 2 and stripped_buf.endswith('"'):
                # Check for escaped quotes inside string
                text_before = stripped_buf[:-1]
                bs_count = len(text_before) - len(text_before.rstrip("\\"))
                if bs_count % 2 == 0:
                    is_buffer_closed = True

            if is_buffer_closed:
                # Allow whitespace until LLM transitions
                return token.isspace()
            else:
                # Block newline
                if "\n" in token:
                    return False

                quote_pos = token.find('"')
                if quote_pos != -1:
                    # See if quote is preceded by backslashes
                    text_before = buffer + token[:quote_pos]
                    bs_count = len(text_before) - len(text_before.rstrip("\\"))

                    if bs_count % 2 == 0:
                        # Prevents adding characters after closing quote
                        # Ensures transitions are handled only be the FSM
                        return token[quote_pos + 1:].strip() == ""

                # Allow data, even escaped quotes
                return True

        elif param_type in {"number", "integer"}:
            first_char = stripped_text[0]

            # Must start with a digit or minus sign
            if not (first_char.isdigit() or first_char == "-"):
                return False

            # Block 0 followed by another digit
            if (
                len(stripped_text) >= 2 and
                stripped_text.startswith("0") and
                stripped_text[1].isdigit()
            ):
                return False
            elif (
                len(stripped_text) >= 3 and
                stripped_text.startswith("-0") and
                stripped_text[2].isdigit()
            ):
                return False

            # Reject invalid chars
            if param_type == "number":
                allowed_chars = set("0123456789.-+eE")
            else:
                allowed_chars = set("0123456789-")
            if not all(char in allowed_chars for char in token):
                return False

            return True

        elif param_type == "boolean":
            first_char = stripped_text[0]
            if first_char not in {"t", "f"}:
                return False

            if any(char in token for char in forbidden):
                return False

            # Ensure valid spelling for target
            target = "true" if first_char == "t" else "false"
            if len(stripped_text) <= len(target):
                return target.startswith(stripped_text)
            else:
                return (
                    stripped_text.startswith(target) and
                    stripped_text[len(target):].isspace()
                )

        elif param_type == "null":
            first_char = stripped_text[0]
            if first_char != "n":
                return False

            if any(char in token for char in forbidden):
                return False

            target = "null"
            if len(stripped_text) <= len(target):
                return target.startswith(stripped_text)
            else:
                return (
                    stripped_text.startswith(target) and
                    stripped_text[len(target):].isspace()
                )

        return False

    def build_prefix_set(self, targets: list[str]) -> set[str]:
        prefix_set: set[str] = set()
        for name in targets:
            for i in range(1, len(name) + 1):
                prefix_set.add(name[:i])

        return prefix_set

    def validate_buffer(self, buffer: str, param_type: str) -> bool:
        stripped_buf = buffer.strip()
        if not stripped_buf:
            return False

        if param_type == "string":
            if not (
                stripped_buf.startswith('"') and
                stripped_buf.endswith('"') and
                len(stripped_buf) >= 2
            ):
                return False

            text_before = stripped_buf[:-1]
            bs_count = len(text_before) - len(text_before.rstrip("\\"))
            return bs_count % 2 == 0

        elif param_type == "number":
            # Cannot end in incomplete math symbol
            if stripped_buf[-1] in {".", "-", "+", "e", "E"}:
                return False

            # JSON requires at least one digit
            # after a decimal point before an exponent
            lower_buf = stripped_buf.lower()
            if ".e" in lower_buf:
                return False

            # Force float value
            if "." not in stripped_buf:
                return False

            try:
                float(stripped_buf)
                return True
            except ValueError:
                return False

        elif param_type == "integer":
            # Cannot end in a minus sign
            if stripped_buf == "-":
                return False
            try:
                int(stripped_buf)
                return True
            except ValueError:
                return False

        elif param_type == "boolean":
            return stripped_buf in {"true", "false"}

        elif param_type == "null":
            return stripped_buf == "null"

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
            text = buffer + token
            if not text.startswith("{"):
                return False

            # What comes after the opening brace
            remainder = text[1:].lstrip()

            # Allow '{' or '{ ' to pass
            if not remainder:
                return True

            # If no parameters, continue with '}'
            if current_fn and not current_fn.parameters:
                return "}".startswith(remainder)

            return self._validate_param_key("", text[1:], current_fn)
        elif state == StatesEnum.PARAM_KEY:
            return self._validate_param_key(buffer, token, current_fn)
        elif state == StatesEnum.PARAM_VALUE:
            return self._validate_param_value(
                buffer, token, current_fn, current_param
            )
        elif state == StatesEnum.JSON_END:
            return "}".startswith(buffer + token)
        return False
