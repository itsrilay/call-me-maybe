*This project has been created as part of the 42 curriculum by ruisilva.*

## Description
Large Language Models (LLMs) are powerful at natural language generation, but notoriously unreliable at producing strict, machine-executable data structures. **Call Me Maybe** is a structured data extraction engine that acts as a bridge between human language and computer functions. 

Using a 0.6B parameter model (Qwen 3), this project translates natural language prompts into perfectly structured JSON function calls. By implementing a custom **Constrained Decoding Engine** and a **Finite State Machine (FSM)**, the system bypasses the unreliability of standard prompting. Instead of hoping the model formats its output correctly, this pipeline mathematically guarantees 100% syntactically valid and schema-compliant JSON, achieving production-grade reliability from a miniature model.

## Instructions

### Prerequisites
The project uses `uv` for dependency management and requires Python 3.10+.

### Installation & Execution
A `Makefile` is provided for standard operations:

1. **Install dependencies:**
   ```bash
   make install
   ```

2. **Run the pipeline:**
    ```bash
    make run
    ```

3. **Run with debugging:**
    ```bash
    make debug
    ```

4. **Code Quality & Testing:**
    ```bash
    make lint         # Runs flake8 and mypy checks
    make lint-strict  # Runs strict type checking
    make test         # Runs the pytest suite
    ```

5. **Clean up environment artifacts:**
    ```bash
    make clean
    ```

## Example Usage
To run the program with specific custom files, use the CLI arguments:

```bash
uv run python -m src \
--functions_definition data/input/functions_definition.json \
--input data/input/function_calling_tests.json \
--output data/output/function_calling_results.json
```

### Input Prompt:
`"What is the sum of 40 and 2?"`

### Output Function Call:
```json
{
  "prompt": "What is the sum of 40 and 2?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 40.0,
    "b": 2.0
  }
}
```

## Algorithm Explanation
The core of this project is **Constrained Decoding** powered by a **Finite State Machine** (FSM).

1. **Logit Generation:** The LLM generates a probability distribution (logits) for every token in its vocabulary.

2. **FSM Tracking:** The FSM tracks the current structural state of the JSON (e.g., `START`, `NAME_KEY`, `PARAM_VALUE`, `JSON_END`).

3. **Validation:** For the current state, the JSON Validator cross-references the vocabulary against the schema rules (e.g., if the state is `PARAM_VALUE` for a boolean, tokens containing `"text"` are blocked).

4. **Logit Masking:** Tokens that violate the JSON syntax or the function schema have their logits set to `-np.inf`.

5. **Token Selection:** The LLM is forced to sample only from mathematically guaranteed valid paths, ensuring 100% compliant structure token-by-token.

## Design Decisions
- **State Machine Architecture:** Instead of complex regex, an FSM was chosen because JSON is fundamentally a state-driven structure (brackets open scopes, colons assign values, commas separate items).

- **Deterministic Bypass:** If the FSM determines that only a single path exists (e.g., completing the spelling of `"parameters":`), the pipeline bypasses the LLM inference entirely and forces the longest valid string prefix.

- **Structural ID Caching:** Valid token IDs for predictable empty-buffer states (like opening braces or quotes) are pre-calculated and cached at initialization.

- **Pre-flight Write Access Check:** To prevent wasting expensive computational resources on generations that cannot be saved, the system verifies OS-level write permissions and path safety for the output file before initiating any LLM inference.

## Performance Analysis
- **Accuracy:** The application guarantees 100% syntactically valid JSON. It prevents the model from voluntarily skipping required schema parameters.

- **Speed:** The Deterministic Bypass and Structural Caching drastically reduce the number of expensive neural network inference calls, noticeably improving tokens-per-second generation speed.

- **Reliability (Advanced Error Recovery):** The implementation boasts a salvager and a retry loop. If the hardware hits the token limit, the salvager inspects the FSM state, slices off partial data, and injects safe type defaults to safely close the JSON. If semantic arguments are missing, the retry loop catches the validation error and reprompts the model with escalating strictness.

## Challenges Faced
- **FSM-Validator Orchestration:** A major hurdle was integrating the `JSONFSM` (state tracking) with the `JSONValidator` (rule enforcement). The system had to distinguish between structural tokens (like `{` or `:`) and data tokens, ensuring the FSM only transitioned once the `JSONValidator` confirmed a value was "complete" based on its schema type.

- **Token Boundary Synchronization:** LLM tokenizers do not respect JSON boundaries. A single token might contain parts of two different states (e.g., `"},"` containing a value closer, a dictionary closer, and a comma for the next key). Implementing an `update_state` loop that could recursively partition a single token into multiple state transitions without losing characters was critical for maintaining sync.

- **Buffer vs. Full JSON Integrity:** Managing the `buffer` (temporary state text) against the `full_json` (committed output) was a delicate balancing act. I had to ensure that partial tokens were stored in the buffer for validation but only committed to the final string once a transition trigger was identified, preventing data loss or "ghost" characters from corrupting the final JSON.

- **String Escaping & Number Precision:** Handling edge cases like escaped quotes (`\"`) or incomplete numbers (e.g., a token ending in a decimal point `.`) required the FSM to "hold" the transition until the next token arrived to confirm the value was legally closed.

- **Hardware Cutoffs:** Managing the strict `MAX_TOKENS = 256` limit severing the data stream required building a deep triage system to determine if a partial generation was salvageable based on the FSM's exact position at the moment of truncation.

## Testing Strategy
The project includes a robust `pytest` suite that heavily tests edge cases without needing full LLM inference:

- **IO Handlers:** Simulates OS-level read/write denials, hidden file blocks, Pydantic schema violations, and OS-level permission errors using mock objects to verify the pre-flight check logic.

- **JSON Validator:** Tests strict parsing constraints, scientific notation (`1.0e-10`), negative leading zeros, and trailing whitespace resilience.

- **Chaos Testing:** The Generation Pipeline tests isolate the salvager by mocking the FSM and injecting artificial token cutoffs. This mathematically proves the system correctly resolves dangling keys, open strings, and partial primitives without crashing `json.loads`.

## Resources
Resources Used:

- Python Documentation (typing, json, argparse)

- Pydantic Documentation

- NumPy Documentation (Logit manipulation)

### AI Usage
AI was utilized as an interactive tutor, sounding board, and architecture reviewer throughout the development process. Specifically, AI helped to:

- **Understand concepts:** Deepen understanding of how LLM logits interact with tokenizer dictionaries and constrained decoding mechanics.

- **Plan architecture:** Review the initial class structures (separating the FSM, Validator, and Pipeline for Single Responsibility).

- **Brainstorm edge cases:** Identify brutal failure points, such as mid-number cutoffs, dangling JSON keys, and escaped quote traps.

- **Review bonus requirements:** Help formulate and review the implementation plan for the testing suite, visualizer, performance optimizations, salvager and retry loop, allowing for confident, rapid development while maximizing learning.

- **Document the project:** Assist in the collaborative drafting and refinement of this `README.md` to ensure it met all specific academic requirements, technical explanations, and performance analysis sections.