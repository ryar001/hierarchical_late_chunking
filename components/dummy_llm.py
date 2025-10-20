from typing import List
from components.embeddings_llm.llm_interface import LLMInterface
from components.models.llm_model import GenerateOutput, SummarizeOutput, ExpandQueryOutput, AnswerOutput

class DummyLLM(LLMInterface):
    def generate(self, prompt: str) -> GenerateOutput:
        return GenerateOutput(content=f"[GEN]: {prompt[:200]}...")

    def summarize(self, text: str, max_tokens: int = 256) -> SummarizeOutput:
        summary = (text[:max_tokens] + "...") if len(text) > max_tokens else text
        return SummarizeOutput(summary=summary)

    def expand_query(self, query: str, max_suggestions: int = 3) -> ExpandQueryOutput:
        base = [
            f"Key definitions related to: {query}",
            f"Worked example for: {query}",
            f"Edge cases / exceptions for: {query}",
        ]
        return ExpandQueryOutput(expanded_queries=base[:max_suggestions])

    def answer(self, question: str, context: str) -> AnswerOutput:
        answer_text = f"Q: {question}\n\nA (using {min(200, len(context))} chars of context):\n{context[:200]}..."
        return AnswerOutput(answer=answer_text)

