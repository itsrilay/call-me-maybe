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
            StatesEnum.PARAM_KEY: {
                ":": StatesEnum.PARAM_VALUE,
                "}": StatesEnum.JSON_END
            },
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
        self, decoded_vocabulary: list[str], validator: JSONValidator
    ) -> list[int]:
        allowed_tokens: list[int] = []

        if self.state == StatesEnum.END:
            return []

        # Fast path for structural states (cache)
        if not self.buffer and self.state in validator.structural_id_cache:
            return validator.structural_id_cache[self.state]

        # Handle structural states
        if self.state != StatesEnum.PARAM_VALUE:
            # Check if all parameters have been used
            is_full = False
            if self.state == StatesEnum.PARAM_KEY and self.current_fn:
                params = self.current_fn.parameters
                is_full = len(self.used_params) == len(params)

            for i, token in enumerate(decoded_vocabulary):
                # If no params left, only closing brace is valid
                if is_full:
                    if "}".startswith(self.buffer + token):
                        allowed_tokens.append(i)
                    continue

                if validator.is_token_valid(
                    self.state,
                    self.buffer,
                    token,
                    self.used_params,
                    self.current_fn,
                    self.current_param
                ):
                    allowed_tokens.append(i)
            return allowed_tokens

        # Handle PARAM_VALUE
        is_full = False
        curr_type = None
        if self.current_fn and self.current_param:
            params = self.current_fn.parameters
            is_full = len(self.used_params) == len(params)
            curr_type = params[self.current_param].type

        req_trigger = "}" if is_full else ","

        # Check if the buffer is complete to transition
        is_buffer_complete = False
        if curr_type:
            is_buffer_complete = validator.validate_buffer(
                self.buffer, curr_type
            )

        for i, token in enumerate(decoded_vocabulary):
            token_added = False
            stripped_token = token.lstrip()

            # Transition state, token has the trigger
            if is_buffer_complete and stripped_token.startswith(req_trigger):
                allowed_tokens.append(i)
                token_added = True

            # Transition state, token completes buffer and has the trigger
            elif not is_buffer_complete and req_trigger in token:
                before, _, remaining = token.partition(req_trigger)
                combined_value = self.buffer + before

                # Check if part before trigger correctly closes the value
                if (
                    curr_type and
                    validator.validate_buffer(combined_value, curr_type) and
                    remaining.strip() == ""
                ):
                    allowed_tokens.append(i)
                    token_added = True

            if not token_added:
                if validator.is_token_valid(
                    self.state,
                    self.buffer,
                    token,
                    self.used_params,
                    self.current_fn,
                    self.current_param
                ):
                    allowed_tokens.append(i)

        return allowed_tokens

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
                is_structural = (
                    exp_type is None or
                    (self.state == StatesEnum.PARAM_KEY and trigger == "}")
                )
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
                # Update buffer if the resulting text is valid for the state
                if validator.is_token_valid(
                    self.state,
                    self.buffer,
                    remaining,
                    self.used_params,
                    self.current_fn,
                    self.current_param
                ):
                    self.buffer += remaining
                    self.full_json += remaining
                break
