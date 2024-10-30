"""
OpenAi LLM module
"""
from langchain_openai import ChatOpenAI

from src.llms.base import BaseLLM
from src.commons.constants import OpenAiModel


class OpenAiLLM(BaseLLM):
    """
    OpenAi LLM class
    """
    def _verify_model_name(self):
        """
        Verify the model name. If the model name is not valid, set the default model name.
        """
        if not OpenAiModel.has_value(self._model_name):
            raise ValueError(f"Invalid model name: {self._model_name}.")

    def _initialize_llm(self):
        """
        Initialize the OpenAi LLM
        """
        self._llm = ChatOpenAI(model=OpenAiModel.GPT_35_TURBO.value)
