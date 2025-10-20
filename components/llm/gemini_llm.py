
import os
from typing import List
import google.generativeai as genai
from components.embeddings_llm.llm_interface import LLMInterface
from components.models.llm_model import GenerateOutput, SummarizeOutput, ExpandQueryOutput, AnswerOutput

# Prompt Templates
SUMMARY_PROMPT_TEMPLATE = """Summarize the following text concisely, aiming for about {max_tokens} tokens:

---
{text}
---"""

EXPAND_QUERY_PROMPT_TEMPLATE = """Generate {max_suggestions} alternative or related search queries based on the following question. Return them as a plain list separated by newlines, without any numbering or bullet points:

Q: {query}"""

ANSWER_PROMPT_TEMPLATE = """Based on the context below, answer the following question.

---
Context: {context}
---

Question: {question}"""


class GeminiLLM(LLMInterface):
    """
    A wrapper for the Google Gemini LLM.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: str = None):
        """
        Initializes the GeminiLLM.

        Args:
            model_name (str): The name of the Gemini model to use.
            api_key (str): The Google API key. If not provided, it will
                           be read from the GOOGLE_API_KEY environment variable.
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key not provided or found in environment variables.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def _generate_content(self, prompt: str) -> str:
        """Helper to generate content and handle potential errors."""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return f"Error: Could not generate content. Details: {e}"

    def generate(self, prompt: str) -> GenerateOutput:
        """
        Generates content based on a given prompt.
        """
        content = self._generate_content(prompt)
        return GenerateOutput(content=content)

    def summarize(self, text: str, max_tokens: int = 256) -> SummarizeOutput:
        """
        Summarizes a given text. The max_tokens parameter is not directly
        used by the Gemini API in the same way, but we can guide the length
        in the prompt.
        """
        prompt = SUMMARY_PROMPT_TEMPLATE.format(max_tokens=max_tokens, text=text)
        summary = self._generate_content(prompt)
        return SummarizeOutput(summary=summary)

    def expand_query(self, query: str, max_suggestions: int = 3) -> ExpandQueryOutput:
        """
        Expands a query into a list of related search queries.
        """
        prompt = EXPAND_QUERY_PROMPT_TEMPLATE.format(max_suggestions=max_suggestions, query=query)
        response_text = self._generate_content(prompt)
        queries = [line.strip() for line in response_text.split('\n') if line.strip()]
        return ExpandQueryOutput(expanded_queries=queries)

    def answer(self, question: str, context: str) -> AnswerOutput:
        """
        Answers a question based on a given context.
        """
        prompt = ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)
        answer_text = self._generate_content(prompt)
        return AnswerOutput(answer=answer_text)

