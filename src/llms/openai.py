from langchain_openai import ChatOpenAI

from src.llms.base import BaseLLM
from src.commons.constants import OpenAiModel


class OpenAiLLM(BaseLLM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.__initialize_llm()

    def __verify_model_name(self):
        if not OpenAiModel.has_value(self.__model_name):
            self.__model_name = OpenAiModel.GPT_35_TURBO.value

    def __initialize_llm(self):
        self.__llm = ChatOpenAI(model=OpenAiModel.GPT_35_TURBO.value)
