"""Unit tests for the JSONFSM class."""
import pytest
from src.json_fsm import JSONFSM
from src.json_validator import JSONValidator
from src.common import StatesEnum
from src.models import FunctionDefinition, ParameterDetail


@pytest.fixture
def sample_fn() -> FunctionDefinition:
    """Provides a basic function for testing transitions."""
    return FunctionDefinition(
        name="fn_test",
        description="A test function",
        parameters={"val": ParameterDetail(type="number")},
        returns={"type": "number"}
    )


@pytest.fixture
def validator(sample_fn: FunctionDefinition) -> JSONValidator:
    """Initializes a validator with a tiny vocabulary."""
    vocab = [
        "{", "}", ":", ",", '"name"', '"parameters"', '"fn_test"', '"val"',
        "1.0"
    ]
    return JSONValidator([sample_fn], vocab)


def test_fsm_initial_state(sample_fn: FunctionDefinition) -> None:
    """Verify FSM starts at the correct point."""
    fsm = JSONFSM([sample_fn])
    assert fsm.state == StatesEnum.START


def test_fsm_state_transitions(
        sample_fn: FunctionDefinition, validator: JSONValidator
) -> None:
    """Simulate a valid JSON generation sequence and verify state changes."""
    fsm = JSONFSM([sample_fn])

    # sequence: { -> "name" -> : -> "fn_test" -> ,
    fsm.update_state("{", validator)
    assert fsm.state == StatesEnum.NAME_KEY

    fsm.update_state('"name"', validator)
    fsm.update_state(":", validator)
    assert fsm.state == StatesEnum.NAME_VALUE

    fsm.update_state('"fn_test"', validator)
    fsm.update_state(",", validator)
    assert fsm.state == StatesEnum.ARGS_KEY


def test_fsm_context_updates(
    sample_fn: FunctionDefinition, validator: JSONValidator
) -> None:
    """Verify FSM correctly identifies function and parameter from tokens."""
    fsm = JSONFSM([sample_fn])

    # Transition through name
    fsm.state = StatesEnum.NAME_VALUE
    fsm.update_state('"fn_test",', validator)

    if fsm.current_fn:
        assert fsm.current_fn.name == "fn_test"

    # Transition through parameter key
    fsm.state = StatesEnum.PARAM_KEY
    fsm.update_state('"val":', validator)

    assert fsm.current_param == "val"
    assert "val" in fsm.used_params


def test_fsm_prevents_duplicate_params() -> None:
    """Ensure FSM does not allow the same parameter to be generated twice."""
    fn_multi = FunctionDefinition(
        name="fn_multi",
        description="Multiple params",
        parameters={
            "a": ParameterDetail(type="number"),
            "b": ParameterDetail(type="number")
        },
        returns={"type": "number"}
    )

    validator = JSONValidator([fn_multi], ['"a"', '"b"'])
    fsm = JSONFSM([fn_multi])
    fsm.current_fn = fn_multi
    fsm.state = StatesEnum.PARAM_KEY

    # Simulate 'a' already being used
    fsm.used_params.add("a")

    # Validator should reject 'a' as a valid key now
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_KEY,
        buffer='"',
        token='a"',
        used_params=fsm.used_params,
        current_fn=fn_multi
    ) is False

    # Validator should still allow 'b'
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_KEY,
        buffer='"',
        token='b"',
        used_params=fsm.used_params,
        current_fn=fn_multi
    ) is True


def test_fsm_zero_parameters(
        sample_fn: FunctionDefinition, validator: JSONValidator
) -> None:
    """Verify FSM transitions correctly for functions with no arguments."""
    sample_fn.parameters = {}
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.ARGS_START

    fsm.update_state("{", validator)
    assert fsm.state == StatesEnum.PARAM_KEY
    fsm.update_state("}", validator)
    assert fsm.state == StatesEnum.JSON_END


def test_fsm_update_state_decoy(
    sample_fn: FunctionDefinition, validator: JSONValidator
) -> None:
    """Verify commas inside strings do not trigger a state transition."""
    sample_fn.parameters["val"].type = "string"
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.PARAM_VALUE
    fsm.current_fn = sample_fn
    fsm.current_param = "val"

    # String with a comma: "Hello, world"
    fsm.update_state('"Hello, world"', validator)

    # Should stay in PARAM_VALUE because the string is not closed
    assert fsm.state == StatesEnum.PARAM_VALUE
    assert fsm.buffer == '"Hello, world"'


def test_fsm_decoy_trigger_at_start(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify trigger character at the start of a value is treated as data."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.PARAM_VALUE
    fsm.current_fn = sample_fn
    fsm.current_param = "val"
    sample_fn.parameters["val"].type = "string"

    # Token is a comma at the start of a string: ","
    fsm.update_state('","', validator)

    assert fsm.state == StatesEnum.PARAM_VALUE
    assert fsm.buffer == '","'


def test_fsm_uses_structural_cache(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify get_allowed_tokens uses validator cache for structural states."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.START
    fsm.buffer = ""  # Cache is only used when buffer is empty

    allowed = fsm.get_allowed_tokens(["{", "}", "a"], validator)

    # Based on our validator fixture, only "{" is allowed in START
    assert len(allowed) == 1
    # Check that it matches the cached IDs
    assert allowed == validator.structural_id_cache[StatesEnum.START]


def test_fsm_get_allowed_tokens_completion_trigger(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify tokens completing a value and including a trigger are allowed."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.PARAM_VALUE
    fsm.current_fn = sample_fn
    fsm.current_param = "val"
    fsm.buffer = ""

    # Vocab has a token that completes the number AND adds a comma
    vocab = ["1.0,"]
    allowed = fsm.get_allowed_tokens(vocab, validator)

    assert len(allowed) == 1
    assert allowed[0] == 0


def test_fsm_buffer_accumulation(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify that partial tokens accumulate correctly in the buffer."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.NAME_VALUE
    fsm.current_fn = sample_fn

    # Step-by-step name value generation: "fn_test"
    fsm.update_state('"fn_', validator)
    assert fsm.buffer == '"fn_'

    fsm.update_state('test"', validator)
    # The comma is the trigger for NAME_VALUE -> ARGS_KEY
    fsm.update_state(",", validator)

    assert fsm.state == StatesEnum.ARGS_KEY
    assert fsm.buffer == ""


def test_fsm_parameter_to_parameter_transition(
    validator: JSONValidator
) -> None:
    """Verify FSM transitions from value back to key on comma."""
    fn_multi = FunctionDefinition(
        name="fn_multi",
        description="...",
        parameters={
            "a": ParameterDetail(type="number"),
            "b": ParameterDetail(type="number")
        },
        returns={"type": "number"}
    )

    fsm = JSONFSM([fn_multi])
    fsm.state = StatesEnum.PARAM_VALUE
    fsm.current_fn = fn_multi
    fsm.current_param = "a"
    fsm.used_params.add("a")
    fsm.buffer = "1.0"

    # Receive comma to transition to the next parameter key
    fsm.update_state(",", validator)
    assert fsm.state == StatesEnum.PARAM_KEY
    assert fsm.buffer == ""


def test_fsm_multi_transition_token(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify a single token can trigger multiple state transitions."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.START

    # This token should trigger transitions for ':' and then start the args '{'
    fsm.update_state('{"name":', validator)

    assert fsm.state == StatesEnum.NAME_VALUE


def test_fsm_get_allowed_tokens_full_params(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify only closing brace is allowed when all parameters are filled."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.PARAM_VALUE
    fsm.current_fn = sample_fn
    fsm.current_param = "val"
    fsm.used_params.add("val")  # Mark as full
    fsm.buffer = "1.0"  # Complete buffer

    # Vocab has comma and closing brace
    vocab = [",", "}"]
    allowed = fsm.get_allowed_tokens(vocab, validator)

    # Only index 1 ("}") should be allowed because is_full is True
    assert len(allowed) == 1
    assert allowed[0] == 1


def test_fsm_accumulation_full_json(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify that full_json correctly accumulates characters over time."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.NAME_VALUE

    fsm.update_state('"fn_', validator)
    fsm.update_state('test"', validator)
    fsm.update_state(",", validator)

    # Verify the internal state of the total string
    assert fsm.full_json == '"fn_test",'


def test_fsm_final_transition_to_end(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify FSM transitions to END state on final closing brace."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.JSON_END

    fsm.update_state("}", validator)
    assert fsm.state == StatesEnum.END


def test_fsm_terminal_logic(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify FSM becomes inactive once it reaches the END state."""
    fsm = JSONFSM([sample_fn])
    fsm.state = StatesEnum.END

    # No tokens should be allowed once finished
    assert fsm.get_allowed_tokens(["{", "}"], validator) == []

    # update_state should do nothing
    fsm.update_state("anything", validator)
    assert fsm.state == StatesEnum.END
    assert fsm.buffer == ""
