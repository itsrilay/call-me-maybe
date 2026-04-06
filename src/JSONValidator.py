from src.common import StatesEnum
from src.models import FunctionDefinition


class JSONValidator:
    def __init__(self, fn_defs: list[FunctionDefinition]):
        self.fn_defs = fn_defs
        self.valid_fn_names: list[str] = [fn.name for fn in fn_defs]

    def _validate_fixed(
        self, state: StatesEnum, buffer: str, token: str
    ) -> bool:
        targets: dict[StatesEnum, str] = {
            StatesEnum.NAME_KEY: '"name":',
            StatesEnum.ARGS_KEY: '"arguments":'
        }

        target = targets.get(state)
        if not target or not target.startswith(buffer + token):
            return False

        return True

    def _validate_name(self, buffer: str, token: str) -> bool:
        new_buffer = buffer + token

        if any(
            f'"{name}",'.startswith(new_buffer) for name in self.valid_fn_names
        ):
            return True

        return False

    def _validate_param_key(
        self,
        buffer: str,
        token: str,
        current_fn: FunctionDefinition | None,
    ) -> bool:
        if not current_fn:
            return False

        if any(
            f'"{param}":'.startswith(buffer + token)
            for param in current_fn.parameters.keys()
        ):
            return True

        return False

    def is_token_valid(
        self,
        state: StatesEnum,
        buffer: str,
        token: str,
        current_fn: FunctionDefinition | None = None,
    ):
        if state in [StatesEnum.NAME_KEY, StatesEnum.ARGS_KEY]:
            return self._validate_fixed(state, buffer, token)
        elif state == StatesEnum.NAME_VALUE:
            return self._validate_name(buffer, token)
        elif state == StatesEnum.PARAM_KEY:
            return self._validate_param_key(buffer, token, current_fn)
