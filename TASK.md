## Task: Implement Jina AI APIs

**Date:** 2025-08-27

**Description:**
Implement all methods mentioned in docs.jina.ai/v8 into @components/llm/jina_llm.py.
- Fetched the docs at https://docs.jina.ai/
- Implemented all the methods in mentioned in the docs
- Used asyncio
- Used the current code and add on the rest of the methods

**Status:** Completed

## Task: Document Project Architecture and Data Flow

**Date:** 2025-10-19

**Description:**
- Thoroughly analyzed the entire project to understand the complete workflow.
- Created a detailed architecture and data flow diagram (ERD style).
- Documented the findings in `PLANNING.md`, covering the journey from data ingestion to vector storage and retrieval.

**Status:** Completed

## Task: Implement Evaluation Module
**Date:** 2025-12-01

**Description:**
- Create an evaluation framework in `evaluate/` folder.
- Ingest `.lamrim.pdf` using `hierarchical_late_chunking.py`.
- Create `evaluate/eval.toml` with 5 test questions based on the PDF.
- Implement `evaluate/run_eval.py` to run queries and evaluate answers using an LLM agent.
- Output results to `evaluate/eval_result.json`.

**Status:** Completed
- Updated `evaluate/run_eval.py` to use RAGAs framework for evaluation metrics (Faithfulness, Answer Relevancy, Context Recall, Answer Correctness).
- Added `ragas` and `langchain` dependencies.

## Task: Refactor Frontend to Chainlit
**Date:** 2025-12-09

**Description:**
- Replace the HTTP server based frontend (`server.py`) with a Chainlit application (`app.py`).
- Implement file upload with ingestion processing and status notifications.
- Replicate duplicate document detection logic.
- Ensure integration with existing backend components and data storage.

**Status:** Completed

## Task: Document Docker Usage
**Date:** 2025-12-10

**Description:**
- Add "Running with Docker" instructions to `README.md`.
- Include build and run commands and link to `DEPLOY_DOCKER.md`.

**Status:** Completed