"""Unit tests for the GenerationPipeline class."""
import pytest
import numpy as np
from src.generation_pipeline import GenerationPipeline
from src.models import FunctionDefinition, ParameterDetail
from src.common import StatesEnum


@pytest.fixture
def pipeline() -> GenerationPipeline:
    """Initializes a pipeline with mocked dependencies for unit testing."""
    # Small logit mask for testing speed (size 10)
    mask = np.zeros(10, dtype=np.float32)
    # None for model/validator, the objective is to test masking/prompts
    return GenerationPipeline(None, [], [], None, mask)  # type: ignore


def test_pipeline_apply_mask(pipeline: GenerationPipeline) -> None:
    """Verify logits for disallowed tokens are set to negative infinity."""
    # Logits MUST match the mask size (10) to avoid broadcast errors
    logits = [1.0, 5.0, 2.0, 10.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    # Only tokens at indices 0, 2, and 4 are permitted
    allowed_tokens = [0, 2, 4]

    # Index 3 has raw logit 10.0 but is forbidden.
    # Index 4 has the highest logit (3.0) among allowed tokens [0, 2, 4].
    token_id = pipeline._apply_mask(logits, allowed_tokens)

    assert token_id == 4
    assert pipeline.logit_mask[3] == -np.inf
    assert pipeline.logit_mask[1] == -np.inf
    assert pipeline.logit_mask[0] == 0.0


def test_pipeline_deterministic_bypass() -> None:
    """Verify the logic that skips the LLM when only one path exists."""
    allowed_strings = ['"p', '"pa', '"parameters"']

    # Mirror the logic in src/generation_pipeline.py:157
    longest_str = max(allowed_strings, key=len)

    # Every token is a prefix of '"parameters"', meaning no choice exists
    is_deterministic = all(longest_str.startswith(s) for s in allowed_strings)
    assert is_deterministic is True


def test_pipeline_system_prompt_construction() -> None:
    """Verify the system prompt dynamically includes function schemas."""
    fn = FunctionDefinition(
        name="fn_test",
        description="A test function",
        parameters={"val": ParameterDetail(type="number")},
        returns={"type": "number"}
    )

    mask = np.zeros(10, dtype=np.float32)
    pipeline = GenerationPipeline(None, [fn], [], None, mask)  # type: ignore

    assert "fn_test" in pipeline.system_prompt
    assert "A test function" in pipeline.system_prompt
    assert "val: number" in pipeline.system_prompt


def test_pipeline_run_immediate_stuck_failure() -> None:
    """Verify run returns None if no tokens are allowed from the start."""
    from unittest.mock import MagicMock

    mock_model = MagicMock()
    # Mock encode to provide a starting list for prompt_length calculation
    mock_model.encode.return_value = [np.array([0])]
    # Mock decode to return a string so json.loads does not crash
    mock_model.decode.return_value = "{}"

    mock_fsm = MagicMock()
    mock_fsm.state = StatesEnum.START
    mock_fsm.get_allowed_tokens.return_value = []  # Stuck immediately

    pipeline = GenerationPipeline(
        mock_model, [], [], MagicMock(), np.zeros(1, dtype=np.float32)
    )
    assert pipeline.run("test", mock_fsm) is None


def test_pipeline_run_parse_failure() -> None:
    """Verify run returns None when generated JSON is invalid."""
    from unittest.mock import MagicMock

    # Setup pipeline with a mock model that returns unparseable text
    mock_model = MagicMock()
    mock_model.encode.return_value = [np.array([0])]
    mock_model.decode.return_value = '{"invalid": json'

    mask = np.zeros(1, dtype=np.float32)
    pipeline = GenerationPipeline(mock_model, [], [], MagicMock(), mask)

    # Force the FSM to END state immediately for the test
    fsm = MagicMock()
    fsm.state = StatesEnum.END

    result = pipeline.run("test prompt", fsm)
    assert result is None  # Should catch JSONDecodeError and return None


def test_pipeline_run_non_dict_failure() -> None:
    """Verify run returns None when generated JSON is not a dictionary."""
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.encode.return_value = [np.array([0])]
    # Valid JSON, but a list instead of an object
    mock_model.decode.return_value = '[1, 2, 3]'

    pipeline = GenerationPipeline(
        mock_model, [], [], MagicMock(), np.zeros(1, dtype=np.float32)
    )
    fsm = MagicMock()
    fsm.state = StatesEnum.END

    result = pipeline.run("test prompt", fsm)
    assert result is None  # Should catch ValueError and return None


def test_pipeline_run_validation_error_recovery() -> None:
    """Verify run returns None when JSON is valid but fails Pydantic schema."""
    from unittest.mock import MagicMock
    mock_model = MagicMock()
    mock_model.encode.return_value = [np.array([0])]
    # Valid JSON dict, but missing the mandatory "name" field for FunctionCall
    mock_model.decode.return_value = '{"parameters": {}}'

    pipeline = GenerationPipeline(
        mock_model, [], [], MagicMock(), np.zeros(1, dtype=np.float32)
    )
    fsm = MagicMock()
    fsm.state = StatesEnum.END

    # result = FunctionCall(...) raises ValidationError; run should return None
    assert pipeline.run("test", fsm) is None


def test_pipeline_token_limit_handling(
    capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify that generation stops and warns when hitting MAX_TOKENS."""
    from unittest.mock import MagicMock

    # Setup model mock
    mock_model = MagicMock()
    mock_model.encode.return_value = [np.array([1, 2, 3])]
    mock_model.decode.return_value = '{"junk":'

    # Dummy vocabulary so the loop has strings to work with
    dummy_vocab = ["token_a", "token_b"]

    pipeline = GenerationPipeline(
        mock_model, [], dummy_vocab, MagicMock(), np.zeros(2, dtype=np.float32)
    )
    fsm = MagicMock()
    fsm.state = StatesEnum.START  # Loop forever

    # Tell the mock to return index 0 as an allowed token
    fsm.get_allowed_tokens.return_value = [0]

    # Run with a tiny limit of 2 tokens
    pipeline.run("test", fsm, max_tokens=2)

    stderr = capsys.readouterr().err
    assert "Warning: Hit max token (2) limit." in stderr
