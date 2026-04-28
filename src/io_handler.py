"""Handles all file I/O and data validation for the project.

This module provides the IOHandler class, which encapsulates the logic for
loading JSON files, validating them against Pydantic schemas, and safely
saving results while preventing accidental overwrites of system files.
"""
import json
import sys
import os
from typing import Any
from pydantic import ValidationError, TypeAdapter
from src.models import FunctionDefinition, PromptInput


class IOHandler:
    """Manages loading, validating, and saving JSON data with safety guards.

    Attributes:
        PROTECTED_FILES (set): Filenames in the root directory that are
            forbidden from being overwritten.
        PROTECTED_DIRS (set): Directory names that are off-limits for
            output operations.
    """

    PROTECTED_FILES = {
        "Makefile", "README.md", "LICENSE", "pyproject.toml",
        "uv.lock", ".gitignore"
    }

    PROTECTED_DIRS = {"src", "llm_sdk", ".git", ".venv"}

    @staticmethod
    def _format_error(e: ValidationError) -> str:
        """Formats a Pydantic ValidationError into a readable string.

        Args:
            e (ValidationError): The caught Pydantic exception.

        Returns:
            str: An error message focusing on the missing/invalid fields.
        """
        messages = []
        for error in e.errors():
            # Field path is a tuple, e.g., ('prompt',)
            field = " -> ".join(str(loc) for loc in error["loc"])
            message = error["msg"]
            # Include the input that caused the issue for better debugging
            input_val = error.get("input")

            messages.append(
                f" - Field '{field}': {message} (Received: {input_val})"
            )

        return "\n".join(messages)

    @staticmethod
    def _read_raw_json(path: str) -> list[Any]:
        """Loads and parses a JSON file, ensuring it is a list.

        Args:
            path (str): The path to the JSON file to be loaded.

        Returns:
            list[Any]: The parsed JSON content.

        Raises:
            SystemExit: If the file is not found, contains invalid JSON,
                or is not a JSON array.
        """
        try:
            with open(path, encoding="utf-8") as file:
                data: list[Any] = json.load(file)
                if not isinstance(data, list):
                    print(
                        f"Error: {path} must contain a JSON array.",
                        file=sys.stderr
                    )
                    sys.exit(1)
                return data
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error while loading {path}: {e}", file=sys.stderr)
            sys.exit(1)

    def is_path_safe(self, path: str) -> bool:
        """Determines if a path is safe to write to.

        Prevents overwriting source code, protected project files, or
        writing into forbidden directories.

        Args:
            path (str): The target output path.

        Returns:
            bool: True if the path is safe, False otherwise.
        """
        abs_path = os.path.abspath(path)
        project_root = os.path.abspath(".")
        filename = os.path.basename(abs_path)

        # Block Python files or protected files
        if path.endswith(".py") or filename in self.PROTECTED_FILES:
            return False

        # Block protected directories
        for protected in self.PROTECTED_DIRS:
            # Anchor the protected directory to the project root
            protected_abs = os.path.join(project_root, protected)

            # Check for exact match or if it's a subdirectory
            if (
                abs_path == protected_abs or
                abs_path.startswith(protected_abs + os.sep)
            ):
                return False

        return True

    def load_functions(self, path: str) -> list[FunctionDefinition]:
        """Loads and validates function definitions from a JSON file.

        Args:
            path (str): Path to the function definitions file.

        Returns:
            list[FunctionDefinition]: A list of validated function schemas.

        Raises:
            SystemExit: If the schema is invalid or no functions are found.
        """
        raw_fn_data = self._read_raw_json(path)
        try:
            fn_adapter = TypeAdapter(list[FunctionDefinition])
            fn_defs = fn_adapter.validate_python(raw_fn_data)

            if not fn_defs:
                print("Error: No function definitions found.", file=sys.stderr)
                sys.exit(1)

            return fn_defs
        except ValidationError as e:
            print(
                "Error: Invalid function definition schema:\n",
                self._format_error(e),
                file=sys.stderr
            )
            sys.exit(1)

    def load_prompts(self, path: str) -> list[PromptInput]:
        """Loads and validates prompts from a JSON file.

        Args:
            path (str): Path to the prompts file.

        Returns:
            list[PromptInput]: A list of validated prompt objects.

        Raises:
            SystemExit: If the schema is invalid.
        """
        raw_prompt_data = self._read_raw_json(path)
        try:
            prompt_adapter = TypeAdapter(list[PromptInput])
            return prompt_adapter.validate_python(raw_prompt_data)
        except ValidationError as e:
            print(
                "Error: Invalid prompt input schema:\n",
                self._format_error(e),
                file=sys.stderr
            )
            sys.exit(1)

    def save_results(self, path: str, results: list[dict[str, Any]]) -> None:
        """Safely saves results to a JSON file.

        Ensures the target directory exists and the path is not protected.

        Args:
            path (str): The target output path.
            results (list[dict[str, Any]]): The data to save as JSON.

        Raises:
            SystemExit: If the path is dangerous or a write error occurs.
        """
        # Check for source code or directory overwrite
        if not self.is_path_safe(path):
            print(
                f"Error: Writing to {path} is forbidden. "
                "Target is a protected project file or directory.",
                file=sys.stderr
            )
            sys.exit(1)

        try:
            output_dir = os.path.dirname(path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            with open(path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4)
        except OSError as e:
            print(
                f"Error: Could not write output to {path}: {e}",
                file=sys.stderr
            )
            sys.exit(1)
