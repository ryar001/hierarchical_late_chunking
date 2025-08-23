from typing import List
from components.llm_interface import LLMInterface

class DummyLLM(LLMInterface):
    def generate(self, prompt: str) -> str:
        return f"[GEN]: {prompt[:200]}..."

    def summarize(self, text: str, max_tokens: int = 256) -> str:
        return (text[:max_tokens] + "...") if len(text) > max_tokens else text

    def expand_query(self, query: str, max_suggestions: int = 3) -> List[str]:
        base = [
            f"Key definitions related to: {query}",
            f"Worked example for: {query}",
            f"Edge cases / exceptions for: {query}",
        ]
        return base[:max_suggestions]

    def answer(self, question: str, context: str) -> str:
        return f"Q: {question}\n\nA (using {min(200, len(context))} chars of context):\n{context[:200]}..."

