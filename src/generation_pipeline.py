"""Pipeline for generating structurally constrained JSON function calls.

This module orchestrates the interaction between the LLM, the Finite
State Machine (FSM), and the JSON Validator to ensure that all generated
tokens conform to a strict JSON schema.
"""
from src.models import FunctionDefinition, FunctionCall
from src.json_validator import JSONValidator
from src.json_fsm import JSONFSM
from src.common import StatesEnum
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
import json
import sys
import numpy as np
import numpy.typing as npt


class Colors:
    """ANSI escape sequences for terminal text coloring.

    Used to visualize the relationship between the LLM and the FSM
    during the generation process.
    """
    BLUE = '\033[94m'    # Deterministic bypass
    GREEN = '\033[92m'   # LLM and FSM agreed
    YELLOW = '\033[93m'  # FSM intervened
    GRAY = '\033[90m'    # State transitions
    RESET = '\033[0m'    # Reset to default terminal color


class GenerationPipeline:
    """Orchestrates the constrained decoding process.

    This class manages the token generation loop, applying logit masks
    based on FSM constraints, and implementing performance optimizations
    like the deterministic bypass.

    Attributes:
        MAX_TOKENS (int): The absolute maximum number of tokens to generate.
        model (Small_LLM_Model): The LLM SDK instance used for inference.
        fn_defs (list[FunctionDefinition]): The available functions.
        decoded_vocabulary (list[str]): The fully decoded token strings.
        validator (JSONValidator): The validation engine for JSON syntax.
        logit_mask (npt.NDArray[np.float32]): Pre-allocated array for masking.
        system_prompt (str): The context and instructions for the LLM.
    """

    MAX_TOKENS = 256

    def __init__(
        self,
        model: Small_LLM_Model,
        fn_defs: list[FunctionDefinition],
        decoded_vocabulary: list[str],
        validator: JSONValidator,
        logit_mask: npt.NDArray[np.float32]
    ) -> None:
        """Initializes the GenerationPipeline with injected dependencies.

        Args:
            model (Small_LLM_Model): The loaded LLM model interface.
            fn_defs (list[FunctionDefinition]): List of valid function schemas.
            decoded_vocabulary (list[str]): The fully decoded token strings.
            validator (JSONValidator): The validation engine for JSON syntax.
            logit_mask (npt.NDArray[np.float32]): Pre-allocated memory for
                logit masking.
        """
        self.model = model
        self.fn_defs = fn_defs
        self.decoded_vocabulary = decoded_vocabulary
        self.validator = validator
        self.logit_mask = logit_mask

        self.system_prompt = (
            "You are a structured data extraction engine. Translate the user's"
            " request directly into a JSON function call.\n\n"
            "Output ONLY valid JSON.\n\n"
            "### EXAMPLES ###\n"
            "User prompt: Execute the script at /usr/local/bin/start.sh "
            "on the primary node\n"
            'Assistant: {"name": "fn_execute", "parameters": '
            '{"path": "/usr/local/bin/start.sh", "target": "primary"}}\n\n'
            'User prompt: Render string: Welcome "{user}" to the team\n'
            'Assistant: {"name": "fn_render", "parameters": '
            '{"template": "Welcome \\"{user}\\" to the team"}}\n\n'
            "### AVAILABLE FUNCTIONS ###\n"
        )

        for fn in self.fn_defs:
            # Tell the model the function definitions
            self.system_prompt += f"{fn.name}:\n{fn.description}\n"
            parameters = "Parameters:\n"
            for param in fn.parameters:
                param_type = fn.parameters[param].type
                parameters += f"{param}: {param_type}\n"
            self.system_prompt += f"{parameters}\n"

        print(self.system_prompt)

    def _apply_mask(
        self, logits: list[float], allowed_tokens: list[int]
    ) -> tuple[int, bool]:
        """Applies the logit mask and detects if FSM intervention was required.

        Converts logits to a NumPy array, applies a mask that sets illegal
        tokens to -inf, and identifies the token with the highest probability.

        Args:
            logits (list[float]): The raw logit output from the LLM.
            allowed_tokens (list[int]): Indices of tokens that are valid
                according to the FSM.

        Returns:
            tuple[int, bool]: Tuple containing the ID of the highest-scoring
                allowed token and confirmation of FSM intervention.
        """
        logits_arr = np.array(logits, dtype=np.float32)

        # Original LLM choice, before FSM intervention
        original_top_choice = int(np.argmax(logits_arr))

        self.logit_mask.fill(-np.inf)
        self.logit_mask[allowed_tokens] = 0.0

        logits_arr += self.logit_mask

        token_id = int(np.argmax(logits_arr))

        intervened = original_top_choice != token_id

        return token_id, intervened

    def _salvage_json(self, fsm: JSONFSM) -> str:
        """Attempts to repair truncated JSON based on FSM state.

        Args:
            fsm (JSONFSM): The state machine at the point of truncation.

        Returns:
            str: The repaired JSON string, or an empty string if unsalvageable.
        """
        # Fail too early
        if fsm.state in [
            StatesEnum.START, StatesEnum.NAME_KEY,
            StatesEnum.NAME_VALUE, StatesEnum.ARGS_KEY, StatesEnum.ARGS_START
        ]:
            print(
                f"\n{Colors.YELLOW}[SALVAGE] Truncated before parameters. "
                f"Abandoning salvage for retry loop...{Colors.RESET}",
                file=sys.stderr
            )
            # Force error
            return ""

        # Reached parameters, salvage
        print(
            f"\n{Colors.YELLOW}[SALVAGE] Truncated output detected. "
            f"Using FSM state to salvage JSON...{Colors.RESET}",
            file=sys.stderr
        )
        healed_json = fsm.full_json

        # Cut off inside a value
        if (
            fsm.state == StatesEnum.PARAM_VALUE and
            fsm.current_fn and
            fsm.current_param
        ):
            param_type = fsm.current_fn.parameters[fsm.current_param].type

            if not self.validator.validate_buffer(fsm.buffer, param_type):
                if param_type == "string":
                    if not fsm.buffer.strip():
                        healed_json += '""'
                    else:
                        # Count backslashes to avoid escaping closing quote
                        trailing_slashes = (
                            len(fsm.buffer) - len(fsm.buffer.rstrip('\\'))
                        )
                        if trailing_slashes % 2 != 0:
                            healed_json += '\\'
                        healed_json += '"'
                else:
                    # Slice off partial garbage ('14.', 'tr')
                    if fsm.buffer:
                        healed_json = healed_json[:-len(fsm.buffer)]

                    # Inject strict type defaults
                    if param_type == "number":
                        healed_json += '0.0'
                    elif param_type == "integer":
                        healed_json += '0'
                    elif param_type == "boolean":
                        healed_json += 'false'
                    elif param_type == "null":
                        healed_json += 'null'

        # Cut off looking for a new key
        elif fsm.state == StatesEnum.PARAM_KEY:
            # Slice off the partial key (e.g., '"b') to remove dangling keys
            if fsm.buffer:
                healed_json = healed_json[:-len(fsm.buffer)]
            healed_json = healed_json.rstrip(" \n\r\t,")

        # Balance structural closing braces
        if fsm.state in [StatesEnum.PARAM_KEY, StatesEnum.PARAM_VALUE]:
            healed_json += "}}"
        elif fsm.state == StatesEnum.JSON_END:
            healed_json += "}"

        return healed_json

    def run(
            self, user_prompt: str, fsm: JSONFSM, max_tokens: int | None = None
    ) -> FunctionCall | None:
        """Executes the generation loop, salvaging data if the limit is hit.

        Args:
            user_prompt (str): The natural language query from the user.
            fsm (JSONFSM): The finite state machine tracking JSON structure.
            max_tokens (int | None): An optional limit on generated tokens.
                Defaults to MAX_TOKENS.

        Returns:
            FunctionCall | None: A Pydantic model containing the parsed JSON
                call, or None if generation failed or timed out.
        """
        full_prompt = (
            f"{self.system_prompt}\n\n" +
            f"User prompt: {user_prompt}\n\n" +
            "Assistant: "
        )

        input_ids_list: list[int] = self.model.encode(full_prompt)[0].tolist()
        prompt_length = len(input_ids_list)

        limit: int = max_tokens if max_tokens else self.MAX_TOKENS
        tokens_generated = 0

        previous_state = None

        print("\n" + "="*45)
        print(
            f"{Colors.BLUE}█ Deterministic{Colors.RESET} | {Colors.GREEN}█ "
            f"Agreed{Colors.RESET} | {Colors.YELLOW}█ "
            f"FSM Intervened{Colors.RESET}"
        )
        print("="*45 + "\n")

        while (
            fsm.state != StatesEnum.END and tokens_generated < limit
        ):
            # Print state changes
            if fsm.state != previous_state:
                print(
                    f"\n{Colors.GRAY}[STATE: {fsm.state.name}]{Colors.RESET} ",
                    end=""
                )
                previous_state = fsm.state

            allowed_tokens: list[int] = fsm.get_allowed_tokens(
                self.decoded_vocabulary, self.validator
            )

            if not allowed_tokens:
                break

            # Convert token ids to strings
            allowed_strings: list[str] = [
                self.decoded_vocabulary[i] for i in allowed_tokens
            ]

            longest_str: str = max(allowed_strings, key=len)

            # If every allowed token is just a prefix of the longest string
            # The LLM has no choice, it must choose the longest string
            is_deterministic = all(
                longest_str.startswith(s) for s in allowed_strings
            )

            if is_deterministic:
                # Bypass LLM, force the longest string
                token_id = allowed_tokens[allowed_strings.index(longest_str)]
                color = Colors.BLUE

            else:
                # Use model inference
                logits: list[float] = self.model.get_logits_from_input_ids(
                    input_ids_list
                )
                token_id, intervened = self._apply_mask(logits, allowed_tokens)
                color = Colors.YELLOW if intervened else Colors.GREEN

            input_ids_list.append(token_id)

            token_string: str = self.model.decode([token_id])
            # Print generated token
            print(f"{color}{token_string}{Colors.RESET}", end="", flush=True)

            tokens_generated += 1

            fsm.update_state(token_string, self.validator)

        if tokens_generated >= limit:
            print(
                f"\n{Colors.YELLOW}Warning: Hit max token ({limit}) limit."
                f"{Colors.RESET}",
                file=sys.stderr
            )
            # Salvage json
            final_json_string = self._salvage_json(fsm)
        else:
            # Extract only the generated tokens normally
            generated_ids = input_ids_list[prompt_length:]
            final_json_string = self.model.decode(generated_ids)

        # Silent return for intentional salvage early fail
        if not final_json_string.strip():
            return None

        try:
            result = json.loads(final_json_string)
            if not isinstance(result, dict):
                raise ValueError("Generated JSON is not a dictionary.")
            if fsm.current_fn:
                required_keys = set(fsm.current_fn.parameters.keys())
                generated_keys = set(result.get("parameters", {}).keys())
                if not required_keys.issubset(generated_keys):
                    missing = required_keys - generated_keys
                    raise ValueError(f"Missing required parameters: {missing}")
            return FunctionCall(prompt=user_prompt, **result)
        except (ValidationError, json.JSONDecodeError, ValueError) as e:
            print(f"\nError processing prompt: {e}", file=sys.stderr)
            return None
