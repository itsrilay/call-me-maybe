"""Unit tests for the JSONValidator class."""
import pytest
from src.json_validator import JSONValidator
from src.models import FunctionDefinition, ParameterDetail
from src.common import StatesEnum


@pytest.fixture
def sample_fn() -> FunctionDefinition:
    """Provides a basic function for testing types.

    Returns:
        FunctionDefinition: A function definition used to test type-specific
            constraints.
    """
    return FunctionDefinition(
        name="fn_test",
        description="A test function",
        parameters={"val": ParameterDetail(type="number")},
        returns={"type": "number"}
    )


@pytest.fixture
def validator(sample_fn: FunctionDefinition) -> JSONValidator:
    """Initializes a validator with a limited testing vocabulary.

    Args:
        sample_fn (FunctionDefinition): The function definition used for
            schema validation.

    Returns:
        JSONValidator: A validator instance initialized with a specific set
            of structural characters and schema-specific tokens.
    """
    vocab = [
        "{", "}", ":", ",", '"name"', '"parameters"', '"fn_test"', '"val"',
        "1.0"
    ]
    return JSONValidator([sample_fn], vocab)


def test_validator_args_start_no_params(validator: JSONValidator) -> None:
    """Verify ARGS_START allows immediate closing brace if fn has no params."""
    fn_empty = FunctionDefinition(
        name="fn_empty",
        description="...",
        parameters={},
        returns={"type": "null"}
    )
    assert validator.is_token_valid(
        state=StatesEnum.ARGS_START,
        buffer="{",
        token="}",
        used_params=set(),
        current_fn=fn_empty
    ) is True


def test_validator_type_restriction(
        validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify validator blocks invalid types (e.g., string to number field)."""
    # Simulate being inside a number parameter 'val'
    is_valid = validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token='"not_a_number"',
        used_params=set(),
        current_fn=sample_fn,
        current_param="val"
    )
    assert is_valid is False


def test_validator_number_edge_cases(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify strict number formatting (leading zeros, scientific notation)."""
    # Block leading zeros (e.g., 05)
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="0",
        token="5",
        used_params=set(),
        current_fn=sample_fn,
        current_param="val"
    ) is False

    # Allow scientific notation in buffer validation
    assert validator.validate_buffer("1.5e10", "number") is True
    assert validator.validate_buffer("0.0", "number") is True


def test_validator_number_requires_dot(validator: JSONValidator) -> None:
    """Verify that 'number' type strictly requires a decimal point per code."""
    assert validator.validate_buffer("10", "number") is False
    assert validator.validate_buffer("10.0", "number") is True


def test_validator_invalid_scientific_decimal(
    validator: JSONValidator
) -> None:
    """Verify that a decimal point immediately before 'e' is blocked."""
    # 1.e10 is invalid in JSON; it must be 1.0e10
    assert validator.validate_buffer("1.e10", "number") is False


def test_validator_scientific_notation_signs(validator: JSONValidator) -> None:
    """Verify handling of positive and negative scientific exponents."""
    assert validator.validate_buffer("1.0e+10", "number") is True
    assert validator.validate_buffer("1.0e-10", "number") is True


def test_validator_integer_rejection(validator: JSONValidator) -> None:
    """Verify integer type strictly rejects floats."""
    assert validator.validate_buffer("10", "integer") is True
    assert validator.validate_buffer("10.5", "integer") is False


def test_validator_incomplete_negative_integer(
    validator: JSONValidator
) -> None:
    """Verify a standalone minus sign is rejected as a complete integer."""
    assert validator.validate_buffer("-", "integer") is False


def test_validator_negative_leading_zero(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify that negative leading zeros are blocked (JSON standard)."""
    # -05 is invalid in JSON; it should be -5 or -0.5
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="-0",
        token="5",
        used_params=set(),
        current_fn=sample_fn,
        current_param="val"
    ) is False


def test_validator_string_escaping(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify escaped quotes are allowed but unescaped ones close string."""
    # Define a string parameter for testing
    sample_fn.parameters["text"] = ParameterDetail(type="string")

    # An escaped quote \" should not close the string
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer='"Hello \\"',
        token='"',
        used_params=set(),
        current_fn=sample_fn,
        current_param="text"
    ) is True

    # Unescaped quote closes the string
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer='"Hello"',
        token='extra',
        used_params=set(),
        current_fn=sample_fn,
        current_param="text"
    ) is False


def test_validator_boolean_and_null(validator: JSONValidator) -> None:
    """Verify strict literal enforcement for boolean and null types."""
    fn = FunctionDefinition(
        name="fn_types",
        description="...",
        parameters={
            "b": ParameterDetail(type="boolean"),
            "n": ParameterDetail(type="null")
        },
        returns={"type": "str"}
    )

    # Test boolean
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token="true",
        used_params=set(),
        current_fn=fn,
        current_param="b") is True
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token="True",
        used_params=set(),
        current_fn=fn,
        current_param="b") is False

    # Test null
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token="null",
        used_params=set(),
        current_fn=fn,
        current_param="n"
    ) is True
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token="None",
        used_params=set(),
        current_fn=fn,
        current_param="n"
    ) is False


def test_validator_boolean_trailing_whitespace(
    validator: JSONValidator
) -> None:
    """Verify trailing whitespace is permitted for boolean literals."""
    fn = FunctionDefinition(
        name="fn", description=".",
        parameters={"b": ParameterDetail(type="boolean")},
        returns={"type": "s"}
    )
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE, buffer="tru", token="e   ",
        used_params=set(), current_fn=fn, current_param="b"
    ) is True


def test_validator_literal_garbage(validator: JSONValidator) -> None:
    """Verify that literals with trailing garbage are strictly rejected."""
    fn = FunctionDefinition(
        name="fn", description=".",
        parameters={"b": ParameterDetail(type="boolean")},
        returns={"type": "s"}
    )
    # 'trueX' is not a valid prefix or a valid boolean
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE, buffer="", token="trueX",
        used_params=set(), current_fn=fn, current_param="b"
    ) is False


def test_build_prefix_set(validator: JSONValidator) -> None:
    """Verify incremental substring generation for structural matching."""
    target = '{"name":'
    prefixes = validator.build_prefix_set([target])

    # Verify each prefix in prefixes list
    assert '{' in prefixes
    assert '{"' in prefixes
    assert '{"n' in prefixes
    assert '{"na' in prefixes
    assert '{"nam' in prefixes
    assert '{"name' in prefixes
    assert '{"name"' in prefixes
    assert '{"name":' in prefixes

    assert len(prefixes) == len(target)


def test_validator_key_prefix_rejection(validator: JSONValidator) -> None:
    """Verify tokens starting with allowed keys and garbage are blocked."""
    # "name" is allowed, but "name_extra" is not a valid prefix of "name"
    assert validator.is_token_valid(
        state=StatesEnum.NAME_KEY,
        buffer='"',
        token='name_extra"',
        used_params=set(),
        current_fn=None
    ) is False


def test_structural_id_cache_initialization(validator: JSONValidator) -> None:
    """Verify that structural state token IDs are pre-cached for speed."""
    # Ensure all required structural states have entries in the cache
    expected_states = {
        StatesEnum.START,
        StatesEnum.NAME_KEY,
        StatesEnum.ARGS_KEY,
        StatesEnum.JSON_END
    }
    assert set(validator.structural_id_cache.keys()) == expected_states

    # Verify START cache contains the '{' token ID
    start_ids = validator.structural_id_cache[StatesEnum.START]
    assert len(start_ids) > 0


def test_validate_buffer_incomplete_types(validator: JSONValidator) -> None:
    """Verify incomplete types are rejected as finished values."""
    # Numbers ending in signs or decimals are not complete
    assert validator.validate_buffer("12.", "number") is False
    assert validator.validate_buffer("-", "integer") is False
    assert validator.validate_buffer("1e", "number") is False

    # Unclosed strings
    assert validator.validate_buffer('"Hello', "string") is False


def test_validator_whitespace_resilience(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify that leading whitespace is allowed for all types."""
    # Allows whitespace so the LLM always has a valid next token option
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token="  ",
        used_params=set(),
        current_fn=sample_fn,
        current_param="val"
    ) is True


def test_validator_string_newline_rejection(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify that literal newlines are rejected within strings."""
    sample_fn.parameters["text"] = ParameterDetail(type="string")

    # Escaped newline is valid data
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer='"Hello',
        token="\\nworld",
        used_params=set(),
        current_fn=sample_fn,
        current_param="text"
    ) is True

    # Literal newline is invalid in a JSON string
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer='"Hello',
        token="\nworld",
        used_params=set(),
        current_fn=sample_fn,
        current_param="text"
    ) is False


def test_validator_unicode_and_special_chars(
    validator: JSONValidator, sample_fn: FunctionDefinition
) -> None:
    """Verify that Unicode and control characters are allowed in strings."""
    sample_fn.parameters["text"] = ParameterDetail(type="string")

    # Emojis and Tabs are valid inside JSON strings
    unicode_token = ' "Hello 🌍\\t"'
    assert validator.is_token_valid(
        state=StatesEnum.PARAM_VALUE,
        buffer="",
        token=unicode_token,
        used_params=set(),
        current_fn=sample_fn,
        current_param="text"
    ) is True
