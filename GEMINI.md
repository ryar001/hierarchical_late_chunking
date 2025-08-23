### 1. Core Project Workflow & Context

- **Always check `PLANNING.md` first.** This file is the single source of truth for architecture, style, and goals. If it's missing, create it at the project root.
- **Update `TASK.md` for every task.** Before starting, add a brief description and the current date. Mark it as completed when done. Create this file at the project root if it doesn't exist.
- **For AI-specific tasks, read `AI_TRACKER_GENIE.md`.** This file contains instructions relevant to the `ai-tracker` component.
- **If a closer `.git` repository is detected, use the `GEMINI.md` file from that directory as an additional source of context.**

---

### 2. Code Quality & Consistency

- **Modularity is key.** Refactor any file approaching **500 lines** into smaller, feature- or responsibility-based modules.
- **Standardize data handling.**
  - Use `TypedDict` for all dictionaries with consistent keys.
  - For common API keys or data keys, define them as constants in a dedicated file.
  - When returning complex, arbitrary data (like a list of mixed types), create a structured class for the data and return an instance of that class.
- **Write for testability.** Use **pure functions** and **dependency injection** whenever possible.
- **Follow Python best practices:** Adhere to **PEP8**, use **type hints**, and format with **black**.
- **Avoid single-character variable names** and the use of Python keywords as variable names.

---

### 3. API & Package Development

- When working on new or existing package methods, do a **pre-flight check**:
    1. **Verify method existence** in the official API documentation. Do not hallucinate new methods.
    2. **Map out the method's purpose** by summarizing its inputs and outputs.
    3. **Implement the method**, ensuring all possible API parameters are handled correctly.
- **Always write a unit test for every new method.**
    - Use a **mock test** if an API key is required and the user has not provided one.
    - Use a **live test** if no API key is needed.
    - Add a **failure test** to handle expected errors gracefully.

---

### 4. Testing & Reliability

- **Tests must live in a dedicated `/tests` folder** that mirrors the project's file structure.
- **Cover all use cases.** For every new feature or change, add tests for:
    - **Expected use** (the happy path).
    - At least one **edge case**.
    - At least one **failure case**.
- **Update existing tests** when you modify core logic.

---

### 5. Documentation & Explainability

- **Document your code.** Use **Google-style docstrings** for every function and class.
- **Comment on complex logic.** Use inline comments with a `# Reason:` prefix to explain the "why" behind non-obvious code.
- **Update the `README.md`** whenever new features, dependencies, or setup steps are introduced.

---

### 6. AI Guardrails

- **Do not assume missing context.** If you're unsure about something, **ask for clarification**.
- **Never hallucinate** libraries or functions. Only use known, verified packages.
- **Verify file paths** and module names before referencing them.
- **Never delete or overwrite existing code** unless explicitly instructed by the user or `TASK.md`.
- **Precede all file edits with a WIP commit.** Use the message `git commit -m "WIP: Editing {CURRENT_FILE} for {REASON}"`.