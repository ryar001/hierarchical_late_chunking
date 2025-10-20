### AI Coding Assistant Instructions

**Guiding Principle**
- Prioritize readable, reliable, and testable code. If rules conflict, readability for a mid-level developer is the highest priority.

**Project Context & Workflow**
- Always read `PLANNING.md` at the start of a new conversation to understand the project's architecture, goals, style, and constraints. Create one at root if not available
- Then `README.md` to understand the project's current state and functionality. Create one at root if not available
- Then check `UPDATES.md` to understand the latest changes and updates. Create one at root if not available
- Then check `BUGS_LOG.md` to understand the known bugs and their status. Create one at root if not available
- Then check `PROJECT_SPECIFIC_INSTRUCTIONS.md` to understand the project's specific instructions. Create one at root if not available
- Check `TASK.md` before starting a new task. If the task isn’t listed, add it with a brief description and today's date. Create one at root if not available
- Use consistent naming conventions, file structure, and architecture patterns as described in `PLANNING.md`.
- Use the `venv_linux` virtual environment for all Python commands. Ensure dependencies from `requirements.txt` or `pyproject.toml` are installed.

**Code Structure & Modularity**
- Never allow files to exceed 500 lines. Refactor into modules/helpers before hitting the limit.
- Organize code by feature or responsibility.
- Example agent layout:
    - `agent.py`: Main agent definition & execution logic.
    - `tools.py`: Tool functions.
    - `prompts.py`: System prompts.
- Standardize imports; prefer relative imports within packages.
- Load environment variables from a root `.env` file using `python_dotenv`. If required variables are missing, inform the user. Do not commit `.env`.

**Data Handling & Validation**
- Use `TypedDict` for dictionaries with known key/value types.
- Define shared keys or values in a `constants.py` file (e.g., `AK = "AK"`).
- For returning complex or mixed data, create a dedicated class and return an instance of it.
- Use `Pydantic` for data validation and schema enforcement.

**Code Quality & Style**
- Adhere strictly to PEP8 compliance.
- Use type hints for everything.
- Format all code with `black`.
- Do not use single-character variable names.
- Do not use Python keywords as variable names.
- Write testable code using pure functions and dependency injection.

**Error Handling**
- Implement robust error handling using `try...except` blocks for operations that can fail (e.g., API calls, file I/O).
- Define custom exception classes for application-specific errors instead of raising generic `Exception`.
- Log errors with sufficient context to aid in debugging.

**API & Package Development**
- Perform a pre-flight check before implementing API methods:
    - Verify methods against official API documentation to prevent hallucination,search the web if not provided the documentation or the link to the documentation.
    - Understand the expected inputs and outputs.
    - Implement the method, covering all of its parameters.
- Search package documentation to confirm expected data structures for return values.
- Use FastAPI for building APIs.
- NiceGUI for frontend
- Use SQLModel for ORM.
- Use pydantic_ai for llm agent creation
- Use langgraph for llm agent orchestration

**Testing & Reliability**
- Place tests in the `/tests` directory, mirroring the main project structure.
- All tests must use `pytest`.
- For each feature or change, write tests covering:
    - The expected use case.
    - At least one edge case.
    - At least one failure case.
- For API-specific testing:
    - Check official package docs for expected results first.
    if api key needed:
        - create a .env.test file and load the api key from there, add sample placeholder for the keys required,eg GEMINI_API_KEY=YOUR_API_KEY if gemini api key is needed
        - If an API key is required but not provided, mock the test.
        - If a key is provided or not needed, run a live test.
        - If a required key is missing, prompt the user to fill out the .env.test file.
- Always add failure tests to verify error handling.
- Update tests whenever the underlying logic changes.
- when a Error or bug is found add it to a file at root called `BUGS_LOG.md`
  - update the fix tried and the outcome to the bug log

**Documentation & Clarity**
- Write Google-style docstrings for all functions and classes.
- For non-obvious logic, add an inline comment prefixed with `# Reason: ` to explain why.
- Update `README.md` whenever features, dependencies, or setup steps change.
- Write code that is clear and understandable to a mid-level developer.

**AI Behavior & Guardrails**
- Ask clarifying questions if a request is ambiguous; do not assume context.
- Never hallucinate libraries or functions. Only use verified packages.
- Always confirm file paths and module names exist before referencing them.
- Never delete or overwrite code unless explicitly instructed or it is part of a task in `TASK.md`.
- Before any file edit, make a WIP commit: `git commit -m "WIP: Editing {CURRENT_FILE} for {REASON}"`.
