import pytest
from pathlib import Path
from src.io_handler import IOHandler
from src.common import IOHandlerError, NoPromptsFound
from pydantic import ValidationError


def test_io_format_error() -> None:
    """Verify that Pydantic errors are formatted into readable strings."""
    io = IOHandler()
    try:
        # Trigger a validation error by passing wrong types to FunctionCall
        from src.models import FunctionCall
        FunctionCall(prompt=123, name=None, parameters="wrong")  # type: ignore
    except ValidationError as e:
        error_msg = io._format_error(e)
        assert "Field 'prompt'" in error_msg
        assert "Field 'name'" in error_msg
        assert "Received: 123" in error_msg


def test_io_read_permission_denied(tmp_path: Path) -> None:
    """Verify that OS-level read errors raise IOHandlerError."""
    io = IOHandler()
    p = tmp_path / "secret.json"
    p.write_text("{}")
    p.chmod(0o000)  # Remove all permissions

    with pytest.raises(IOHandlerError, match="Error while loading"):
        io.load_prompts(str(p))


def test_is_path_safe_hidden_files() -> None:
    """Verify that hidden files are blocked."""
    io = IOHandler()
    assert io.is_path_safe(".env") is False
    assert io.is_path_safe("data/output/.hidden") is False


def test_is_path_safe_py_extension() -> None:
    """Verify that any file with a .py extension is blocked."""
    io = IOHandler()
    assert io.is_path_safe("legit_output.py") is False
    assert io.is_path_safe("data/output/script.py") is False


def test_is_path_safe_protected_files() -> None:
    """Verify that critical project files cannot be overwritten."""
    io = IOHandler()
    assert io.is_path_safe("Makefile") is False
    assert io.is_path_safe("README.md") is False
    assert io.is_path_safe("pyproject.toml") is False


def test_is_path_safe_directories() -> None:
    """Verify that source and system directories are protected."""
    io = IOHandler()
    assert io.is_path_safe("src/__main__.py") is False
    assert io.is_path_safe("llm_sdk/model.py") is False
    assert io.is_path_safe(".git/config") is False


def test_is_path_safe_relative() -> None:
    """Verify that relative navigation (..) cannot bypass security."""
    io = IOHandler()
    assert io.is_path_safe("data/output/../../src/__main__.py") is False
    assert io.is_path_safe("data/input/.././../Makefile") is False


def test_is_path_safe_exact_directory_match() -> None:
    """Verify writing a file with same name as a protected dir is blocked."""
    io = IOHandler()
    # Check match for 'src' and '.venv' directly
    assert io.is_path_safe("src") is False
    assert io.is_path_safe(".venv") is False


def test_is_path_safe_allowed() -> None:
    """Verify that standard output paths are permitted."""
    io = IOHandler()
    assert io.is_path_safe("data/output/results.json") is True
    assert io.is_path_safe("results.json") is True


def test_load_functions_invalid_json(tmp_path: Path) -> None:
    """Verify that syntax errors in the JSON file trigger a IOHandlerError."""
    io = IOHandler()
    p: Path = tmp_path / "broken.json"
    p.write_text('[{"name": "fn_add" "description": "missing comma"}]')
    with pytest.raises(IOHandlerError):
        io.load_functions(str(p))


def test_load_functions_partial_schema_violation(tmp_path: Path) -> None:
    """Verify that a single invalid entry fails the entire function list."""
    io = IOHandler()
    p = tmp_path / "mixed_functions.json"
    # Second entry is missing the 'returns' key
    p.write_text(
        '[{"name": "fn_good", "description": "d", "parameters": {}, '
        '"returns": {"type": "s"}},'
        ' {"name": "fn_bad", "description": "d", "parameters": {}}]'
    )

    with pytest.raises(
        IOHandlerError, match="Invalid function definition schema"
    ):
        io.load_functions(str(p))


def test_load_functions_not_an_array(tmp_path: Path) -> None:
    """Verify that the program rejects JSON that isn't a list."""
    io = IOHandler()
    p: Path = tmp_path / "object.json"
    p.write_text('{"name": "fn_add"}')
    with pytest.raises(IOHandlerError):
        io.load_functions(str(p))


def test_load_functions_schema_violation(tmp_path: Path) -> None:
    """Verify that Pydantic catches missing required fields like 'returns'."""
    io = IOHandler()
    p: Path = tmp_path / "bad_schema.json"
    # Missing 'returns' key required by FunctionDefinition
    p.write_text(
        '[{"name": "fn_add", "description": "test", "parameters": {}}]'
    )
    with pytest.raises(IOHandlerError):
        io.load_functions(str(p))


def test_load_prompts_file_not_found() -> None:
    """Verify that missing files are handled gracefully."""
    io = IOHandler()
    with pytest.raises(IOHandlerError):
        io.load_prompts("data/input/does_not_exist.json")


def test_load_prompts_empty_list(tmp_path: Path) -> None:
    """Verify that an empty list of prompts triggers an error."""
    io = IOHandler()
    p = tmp_path / "empty_prompts.json"
    p.write_text("[]")
    with pytest.raises(NoPromptsFound):
        io.load_prompts(str(p))


def test_load_prompts_schema_violation(tmp_path: Path) -> None:
    """Verify that prompts with missing keys raise IOHandlerError."""
    io = IOHandler()
    p: Path = tmp_path / "bad_prompts.json"
    # Missing mandatory "prompt" key from the schema
    p.write_text('[{"not_a_prompt": "hello"}]')

    with pytest.raises(IOHandlerError):
        io.load_prompts(str(p))


def test_load_prompts_data_integrity(tmp_path: Path) -> None:
    """Verify load_prompts returns valid Pydantic objects."""
    io = IOHandler()
    p: Path = tmp_path / "valid_prompts.json"
    p.write_text('[{"prompt": "Test request"}]')

    prompts = io.load_prompts(str(p))
    assert len(prompts) == 1
    assert prompts[0].prompt == "Test request"


def test_load_functions_injects_fallback(tmp_path: Path) -> None:
    """Verify fn_unsupported_request is added if not present."""
    io = IOHandler()
    p: Path = tmp_path / "no_fallback.json"
    p.write_text(
        '[{"name": "fn_greet", "description": "...", '
        '"parameters": {}, "returns": {"type": "str"}}]'
    )

    fn_defs = io.load_functions(str(p))
    names = [fn.name for fn in fn_defs]
    assert "fn_unsupported_request" in names


def test_load_functions_no_duplicate_fallback(tmp_path: Path) -> None:
    """Verify fallback is not injected if already present in the file."""
    io = IOHandler()
    p = tmp_path / "fallback_exists.json"
    p.write_text(
        '[{"name": "fn_unsupported_request", "description": "custom", '
        '"parameters": {}, "returns": {"type": "s"}}]'
    )

    fn_defs = io.load_functions(str(p))
    names = [fn.name for fn in fn_defs]
    assert names.count("fn_unsupported_request") == 1


def test_load_functions_success(tmp_path: Path) -> None:
    """Verify valid function definitions are loaded and parsed correctly."""
    io = IOHandler()
    p = tmp_path / "functions.json"
    p.write_text('[{"name": "fn_math", "description": "desc", '
                 '"parameters": {"x": {"type": "number"}}, '
                 '"returns": {"type": "number"}}]')

    fn_defs = io.load_functions(str(p))
    assert len(fn_defs) == 2  # 1 from file + 1 injected fallback
    assert fn_defs[0].name == "fn_math"
    assert fn_defs[0].parameters["x"].type == "number"


def test_load_vocabulary_empty(tmp_path: Path) -> None:
    """Verify that an empty vocabulary is handled correctly"""
    io = IOHandler()
    p: Path = tmp_path / "empty_vocab.json"
    p.write_text("{}")

    with pytest.raises(IOHandlerError):
        io.load_vocabulary(str(p))


def test_load_vocabulary_schema_violation(tmp_path: Path) -> None:
    """Verify that vocabulary values must be integers."""
    io = IOHandler()
    p: Path = tmp_path / "bad_vocab.json"
    # Values are strings, but schema requires integers
    p.write_text('{"token_a": "zero", "token_b": "one"}')

    with pytest.raises(IOHandlerError):
        io.load_vocabulary(str(p))


def test_load_vocabulary_success(tmp_path: Path) -> None:
    """Verify that a valid vocabulary file is loaded correctly."""
    io = IOHandler()
    p: Path = tmp_path / "vocab.json"
    # Valid mapping of tokens to integer IDs
    p.write_text('{"<pad>": 0, "hello": 1, "world": 2}')

    vocab = io.load_vocabulary(str(p))

    assert isinstance(vocab, dict)
    assert vocab["hello"] == 1
    assert len(vocab) == 3


def test_save_results_invalid_paths() -> None:
    """Verify that program exits for invalid paths."""
    io = IOHandler()
    with pytest.raises(IOHandlerError):
        io.save_results("src/io_handler.py", [{}])


def test_save_results_write_error(tmp_path: Path) -> None:
    """Verify that OS-level write errors raise IOHandlerError."""
    from unittest.mock import patch, mock_open
    io = IOHandler()
    p = tmp_path / "output.json"

    # Mock open to raise OSError when trying to write
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = OSError("Disk Full")
        with pytest.raises(IOHandlerError, match="Could not write output"):
            io.save_results(str(p), [{"data": 1}])


def test_save_results_success(tmp_path: Path) -> None:
    """Verify results are correctly saved and directories created."""
    io = IOHandler()
    p = tmp_path / "subdir" / "output.json"
    data = [{"test": "data"}]
    io.save_results(str(p), data)

    assert p.exists()
    assert p.read_text() == '[\n    {\n        "test": "data"\n    }\n]'
