from langchain_openai import ChatOpenAI

from src.llms.base import BaseLLM
from src.commons.constants import OpenAiModel


class OpenAiLLM(BaseLLM):
    def __init__(self):
        super().__init__()
        self.__llm = ChatOpenAI(model=OpenAiModel.GPT_35_TURBO.value)
