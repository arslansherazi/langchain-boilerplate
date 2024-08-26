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
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.__initialize_llm()

    def __verify_model_name(self):
        """
        Verify the model name. If the model name is not valid, set the default model name.
        """
        if not OpenAiModel.has_value(self.__model_name):
            self.__model_name = OpenAiModel.GPT_35_TURBO.value

    def __initialize_llm(self):
        """
        Initialize the OpenAi LLM
        """
        self.__llm = ChatOpenAI(model=OpenAiModel.GPT_35_TURBO.value)
