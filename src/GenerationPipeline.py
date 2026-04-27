"""Pipeline for generating structurally constrained JSON function calls.

This module orchestrates the interaction between the LLM, the Finite
State Machine (FSM), and the JSON Validator to ensure that all generated
tokens conform to a strict JSON schema.
"""
from src.models import FunctionDefinition, FunctionCall
from src.JSONValidator import JSONValidator
from src.JSONFSM import JSONFSM
from src.common import StatesEnum
from llm_sdk import Small_LLM_Model
from pydantic import ValidationError
import json
import sys
import numpy as np


class GenerationPipeline:
    """Orchestrates the constrained decoding process.

    This class manages the token generation loop, applying logit masks
    based on FSM constraints, and implementing performance optimizations
    like the deterministic bypass.

    Attributes:
        MAX_TOKENS (int): The absolute maximum number of tokens to generate.
        model (Small_LLM_Model): The LLM SDK instance used for inference.
        fn_defs (list[FunctionDefinition]): The available functions.
        vocabulary (list[str]): The raw string vocabulary loaded from model.
        decoded_vocabulary (list[str]): The fully decoded token strings.
        validator (JSONValidator): The validation engine for JSON syntax.
        logit_mask (np.ndarray): A pre-allocated NumPy array for logit masking.
        token_to_id (dict[str, int]): Mapping of string tokens to their IDs.
        system_prompt (str): The initial context and instructions for the LLM.
    """

    MAX_TOKENS = 256

    def __init__(
        self, model: Small_LLM_Model, fn_defs: list[FunctionDefinition]
    ) -> None:
        """Initializes the GenerationPipeline.

        Sets up the FSM validator, loads the vocabulary, calculates the
        true logit size via a dummy inference, and pre-allocates memory
        for the NumPy mask to maximize generation speed.

        Args:
            model (Small_LLM_Model): The loaded LLM model interface.
            fn_defs (list[FunctionDefinition]): A list of valid function
                schemas.
        """
        self.model = model
        self.fn_defs = fn_defs

        # Get vocabulary
        try:
            with open(f"{model.get_path_to_vocab_file()}") as file:
                self.vocabulary: list[str] = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            print("Couldn't parse LLM vocabulary", file=sys.stderr)
            sys.exit(1)

        self.decoded_vocabulary = [
            self.model.decode([i]) for i in range(len(self.vocabulary))
        ]

        self.validator = JSONValidator(fn_defs, self.decoded_vocabulary)

        dummy_logits = self.model.get_logits_from_input_ids([0])
        logit_size = len(dummy_logits)
        self.logit_mask = np.empty(logit_size, dtype=np.float32)

        self.token_to_id = {
            token: id for id, token in enumerate(self.vocabulary)
        }

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

    def run(self, user_question: str) -> FunctionCall | None:
        """Executes the constrained generation loop for a user prompt.

        Evaluates the prompt token-by-token. It uses a deterministic bypass
        to skip the LLM forward pass when only one structural path is valid,
        and uses constrained logit masking when semantic choices are required.

        Args:
            user_question (str): The natural language query from the user.

        Returns:
            FunctionCall | None: A Pydantic model containing the parsed JSON
                call, or None if generation failed or produced invalid JSON.
        """
        full_prompt = (
            f"{self.system_prompt}\n\n" +
            f"User prompt: {user_question}\n\n" +
            "Assistant: "
        )

        input_ids_list: list[int] = self.model.encode(full_prompt)[0].tolist()
        prompt_length = len(input_ids_list)
        tokens_generated = 0

        fsm = JSONFSM(self.fn_defs)

        while (
            fsm.state != StatesEnum.END and tokens_generated < self.MAX_TOKENS
        ):
            allowed_tokens = fsm.get_allowed_tokens(
                self.decoded_vocabulary, self.validator
            )

            if not allowed_tokens:
                break

            # Convert token ids to strings
            allowed_strings = [
                self.decoded_vocabulary[i] for i in allowed_tokens
            ]

            longest_str = max(allowed_strings, key=len)

            # If every allowed token is just a prefix of the longest string
            # The LLM has no choice, it must choose the longest string
            is_deterministic = all(
                longest_str.startswith(s) for s in allowed_strings
            )

            if is_deterministic:
                # Bypass LLM, force the longest string
                token_id = allowed_tokens[allowed_strings.index(longest_str)]

            else:
                # Use model inference
                logits = self.model.get_logits_from_input_ids(input_ids_list)
                logits_arr = np.array(logits, dtype=np.float32)

                self.logit_mask.fill(-np.inf)
                self.logit_mask[allowed_tokens] = 0.0

                logits_arr += self.logit_mask

                token_id = int(np.argmax(logits_arr))

            input_ids_list.append(token_id)

            token_string = self.model.decode([token_id])
            print(token_string, end="", flush=True)

            tokens_generated += 1

            fsm.update_state(token_string, self.validator)

        # Extract only the generated tokens
        generated_ids = input_ids_list[prompt_length:]

        # Turn them into a single string
        final_json_string = self.model.decode(generated_ids)

        try:
            result = json.loads(final_json_string)
            return FunctionCall(prompt=user_question, **result)
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"\nError processing prompt: {e}")
            return None
