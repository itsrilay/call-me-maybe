from src.models import FunctionDefinition
from src.common import StatesEnum


class JSONFSM:
    def __init__(self, fn_defs: list[FunctionDefinition]):
        self.fn_defs = fn_defs
        self.state = StatesEnum.START
        self.current_fn = None
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
            StatesEnum.JSON_END: {"}": StatesEnum.JSON_END},
        }

    def get_allowed_tokens(self, vocabulary):
        # Logit mask logic
        # Based on self.state, return which tokens are okay
        pass

    def update_state(self, last_token_text: str):
        # Look up what triggers are valid for where we are right now
        triggers = self.transitions.get(self.state, {})

        for char, next_state in triggers.items():
            if char in last_token_text:
                # Split the token
                _, _, remaining = last_token_text.partition(char)
                self.state = next_state
                self.buffer = remaining
                break
        else:
            self.buffer += last_token_text
