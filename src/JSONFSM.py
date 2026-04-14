from src.models import FunctionDefinition
from src.common import StatesEnum
from src.JSONValidator import JSONValidator


class JSONFSM:
    def __init__(self, fn_defs: list[FunctionDefinition]):
        self.fn_map: dict[str, FunctionDefinition] = {
            fn.name: fn for fn in fn_defs
        }
        self.state = StatesEnum.START
        self.current_fn = None
        self.current_param = None
        self.used_params: set[str] = set()
        self.buffer = ""  # Text within a state
        self.full_json = ""  # Complete Output

        # State Transitions
        self.transitions = {
            StatesEnum.START: {"{": StatesEnum.NAME_KEY},
            StatesEnum.NAME_KEY: {":": StatesEnum.NAME_VALUE},
            StatesEnum.NAME_VALUE: {",": StatesEnum.ARGS_KEY},
            StatesEnum.ARGS_KEY: {":": StatesEnum.ARGS_START},
            StatesEnum.ARGS_START: {"{": StatesEnum.PARAM_KEY},
            StatesEnum.PARAM_KEY: {":": StatesEnum.PARAM_VALUE},
            StatesEnum.PARAM_VALUE: {
                ",": StatesEnum.PARAM_KEY,
                "}": StatesEnum.JSON_END,
            },
            StatesEnum.JSON_END: {"}": StatesEnum.END},
        }

        self.STATE_DEFINITIONS: dict[StatesEnum, str | None] = {
            StatesEnum.START: None,
            StatesEnum.NAME_KEY: "string",
            StatesEnum.NAME_VALUE: "string",
            StatesEnum.ARGS_KEY: "string",
            StatesEnum.ARGS_START: None,
            StatesEnum.PARAM_KEY: "string",
            StatesEnum.PARAM_VALUE: "dynamic",
            StatesEnum.JSON_END: None,
        }

    def get_allowed_tokens(
        self, vocabulary: list[str], validator: JSONValidator
    ) -> list[str]:
        # Get prefix set for current state
        prefix_set = validator.prefix_lookups.get(self.state)

        # Special case for PARAM_KEY (which is function-specific)
        if (
            not prefix_set
            and self.state == StatesEnum.PARAM_KEY
            and self.current_fn
        ):
            param_names = [
                f'"{param_name}":' for param_name in self.current_fn.parameters
                if param_name not in self.used_params
            ]
            prefix_set = validator._build_prefix_set(param_names)

        if prefix_set is not None:
            return [
                token for token in vocabulary
                if self.buffer + token in prefix_set
            ]
        else:
            char_filter = None
            required_prefix = None
            if (
                self.state == StatesEnum.PARAM_VALUE and
                self.current_fn and
                self.current_param
            ):
                params = self.current_fn.parameters
                exp_type = params[self.current_param].type

                if exp_type == "number":
                    char_filter = {
                        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                        ".", "-", "+", "e", "E", ",", "}"
                    }

                    if len(self.used_params) == len(params):
                        char_filter.discard(",")

                elif exp_type == "string" and not self.buffer:
                    required_prefix = '"'

            # Determine if there are parameters to use
            is_full = False
            if self.current_fn:
                params = self.current_fn.parameters
                is_full = len(self.used_params) == len(params)

            if char_filter:
                # If number value
                return [
                    token for token in vocabulary
                    if all(c in char_filter for c in token) and
                    validator.is_token_valid(
                        self.state,
                        self.buffer,
                        token,
                        self.current_fn,
                        self.current_param
                    )
                ]
            elif required_prefix:
                # If in the start of a string value
                return [
                    token for token in vocabulary
                    if token.startswith(required_prefix) and
                    validator.is_token_valid(
                        self.state,
                        self.buffer,
                        token,
                        self.current_fn,
                        self.current_param
                    ) and not (is_full and token.endswith(","))
                ]
            else:
                # Standard path for everything else
                return [
                    token for token in vocabulary
                    if validator.is_token_valid(
                        self.state,
                        self.buffer,
                        token,
                        self.current_fn,
                        self.current_param
                    ) and not (is_full and token.endswith(","))
                ]

    def update_state(
        self, last_token_text: str, validator: JSONValidator
    ) -> None:
        if self.state == StatesEnum.END:
            return

        remaining = last_token_text

        while remaining:
            if self.state not in self.STATE_DEFINITIONS:
                return
            triggers = self.transitions.get(self.state, {})

            exp_type = self.STATE_DEFINITIONS.get(self.state, "")

            if self.state == StatesEnum.PARAM_VALUE:
                if self.current_fn and self.current_param:
                    params = self.current_fn.parameters
                    exp_type = params[self.current_param].type

            # Find triggers that are in remaining string
            present_triggers = [char for char in triggers if char in remaining]

            if present_triggers:
                # Get earliest trigger
                first_trigger = min(present_triggers, key=remaining.find)

                # Partition on earliest trigger
                before, trigger, remaining = remaining.partition(first_trigger)

                combined = self.buffer + before
                is_structural = exp_type is None
                is_valid = validator.validate_buffer(combined, exp_type or "")

                # Check if it's an actual trigger
                if is_structural or is_valid:
                    # Transition
                    # Update function info for future reference
                    if self.state == StatesEnum.NAME_VALUE and trigger == ",":
                        fn_name = (self.buffer + before).strip('" ')
                        self.current_fn = self.fn_map.get(fn_name)
                    elif self.state == StatesEnum.PARAM_KEY and trigger == ":":
                        self.current_param = (self.buffer + before).strip('" ')
                        self.used_params.add(self.current_param)

                    # Move to next state
                    self.state = triggers[first_trigger]
                    self.buffer = ""
                    self.full_json += before + trigger
                else:
                    # Decoy
                    self.buffer += before + trigger
                    self.full_json += before + trigger
            else:
                # Update current state
                self.buffer += remaining
                self.full_json += remaining
                break
