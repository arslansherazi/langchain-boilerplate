"""
Base module for all LLMs
"""
from langchain_core.messages import HumanMessage


class BaseLLM:
    """
    Base class for all LLMs
    """
    def __init__(self, model_name: str):
        self.__model_name = model_name
        self.__llm = None

    def get_response(self, query: str) -> str:
        """
        Get response from the LLM

        :param query: user query
        :return: llm response
        """
        messages = [
            HumanMessage(content=query)
        ]
        result = self.__llm(messages)
        results_data = result.content
        return results_data

    def get_llm(self):
        """
        Get the llm object

        :return: llm object
        """
        return self.__llm
